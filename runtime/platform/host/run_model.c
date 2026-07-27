#include "ng_model.h"
#include "ng_runtime.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct run_options {
    const char *model_path;
    const char *token_text;
    const char *logits_path;
    const char *generated_path;
    size_t generate_count;
} run_options;

static void usage(void) {
    fputs(
        "usage: run_model --model FILE --tokens ID,ID,... "
        "--logits-out FILE --tokens-out FILE --generate COUNT\n",
        stderr);
}

static int parse_size(const char *text, size_t *result) {
    char *end = NULL;
    unsigned long value;
    errno = 0;
    value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *result = (size_t)value;
    return (unsigned long)*result == value;
}

static int parse_options(
    int argument_count,
    char **arguments,
    run_options *options) {
    int index;
    (void)memset(options, 0, sizeof(*options));
    for (index = 1; index < argument_count; ++index) {
        const char *name = arguments[index];
        const char *value;
        if (index + 1 >= argument_count) {
            return 0;
        }
        value = arguments[++index];
        if (strcmp(name, "--model") == 0) {
            options->model_path = value;
        } else if (strcmp(name, "--tokens") == 0) {
            options->token_text = value;
        } else if (strcmp(name, "--logits-out") == 0) {
            options->logits_path = value;
        } else if (strcmp(name, "--tokens-out") == 0) {
            options->generated_path = value;
        } else if (strcmp(name, "--generate") == 0) {
            if (!parse_size(value, &options->generate_count)) {
                return 0;
            }
        } else {
            return 0;
        }
    }
    return options->model_path != NULL
        && options->token_text != NULL
        && options->logits_path != NULL
        && options->generated_path != NULL;
}

static int parse_tokens(
    const char *text,
    uint32_t *tokens,
    size_t capacity,
    size_t *count) {
    const char *cursor = text;
    size_t length = 0u;
    if (*cursor == '\0') {
        return 0;
    }
    while (*cursor != '\0') {
        char *end = NULL;
        unsigned long value;
        if (length >= capacity) {
            return 0;
        }
        errno = 0;
        value = strtoul(cursor, &end, 10);
        if (errno != 0
            || end == cursor
            || value > (unsigned long)UINT32_MAX) {
            return 0;
        }
        tokens[length++] = (uint32_t)value;
        if (*end == '\0') {
            cursor = end;
        } else if (*end == ',') {
            cursor = end + 1;
            if (*cursor == '\0') {
                return 0;
            }
        } else {
            return 0;
        }
    }
    *count = length;
    return 1;
}

static uint32_t argmax(const float *values, size_t length) {
    uint32_t best = 0u;
    size_t index;
    for (index = 1u; index < length; ++index) {
        if (values[index] > values[best]) {
            best = (uint32_t)index;
        }
    }
    return best;
}

static int write_bytes(
    const char *path,
    const void *bytes,
    size_t element_bytes,
    size_t count) {
    FILE *stream = fopen(path, "wb");
    int ok;
    if (stream == NULL) {
        return 0;
    }
    ok = fwrite(bytes, element_bytes, count, stream) == count;
    if (fclose(stream) != 0) {
        ok = 0;
    }
    return ok;
}

int main(int argument_count, char **arguments) {
    run_options options;
    ng_model model;
    ng_runtime runtime;
    ng_error error;
    uint32_t prompt[NG_MAX_VOCAB_SIZE];
    uint32_t generated[NG_MAX_VOCAB_SIZE];
    size_t prompt_count;
    size_t index;
    uint8_t *arena = NULL;
    const float *logits = NULL;
    clock_t started;
    clock_t finished;
    double elapsed;
    size_t forward_count = 0u;
    int exit_code = 1;
    if (!parse_options(argument_count, arguments, &options)) {
        usage();
        return 2;
    }
    if (!parse_tokens(
            options.token_text,
            prompt,
            (size_t)NG_MAX_VOCAB_SIZE,
            &prompt_count)) {
        fputs("invalid --tokens list\n", stderr);
        return 2;
    }
    if (ng_model_load_file(
            options.model_path,
            (size_t)NG_INFERENCE_MEMORY_LIMIT_BYTES,
            &model,
            &error)
        != NG_STATUS_OK) {
        fprintf(stderr, "model load failed: %s\n", error.message);
        return 1;
    }
    if (prompt_count + options.generate_count > model.spec.block_size
        || options.generate_count > (size_t)NG_MAX_VOCAB_SIZE) {
        fputs("prompt plus generation exceeds fixed context\n", stderr);
        goto cleanup_model;
    }
    arena = (uint8_t *)malloc(model.required_arena_bytes);
    if (arena == NULL) {
        fputs("runtime arena allocation failed\n", stderr);
        goto cleanup_model;
    }
    if (ng_runtime_init(
            &runtime,
            &model,
            arena,
            model.required_arena_bytes,
            &error)
        != NG_STATUS_OK) {
        fprintf(stderr, "runtime init failed: %s\n", error.message);
        goto cleanup_arena;
    }
    started = clock();
    for (index = 0u; index < prompt_count; ++index) {
        if (ng_runtime_forward_token(
                &runtime,
                prompt[index],
                &logits,
                &error)
            != NG_STATUS_OK) {
            fprintf(stderr, "prompt forward failed: %s\n", error.message);
            goto cleanup_runtime;
        }
        forward_count += 1u;
    }
    /*
     * The numerical probe is the last-token distribution for the fixed
     * prompt. Generation continues from it, but must not replace it with a
     * later recurrent state.
     */
    if (!write_bytes(
            options.logits_path,
            logits,
            sizeof(float),
            (size_t)model.spec.vocab_size)) {
        fputs("failed to write prompt logits\n", stderr);
        goto cleanup_runtime;
    }
    for (index = 0u; index < options.generate_count; ++index) {
        uint32_t next = argmax(logits, (size_t)model.spec.vocab_size);
        generated[index] = next;
        if (ng_runtime_forward_token(
                &runtime,
                next,
                &logits,
                &error)
            != NG_STATUS_OK) {
            fprintf(stderr, "generation forward failed: %s\n", error.message);
            goto cleanup_runtime;
        }
        forward_count += 1u;
    }
    finished = clock();
    elapsed = (double)(finished - started) / (double)CLOCKS_PER_SEC;
    if (!write_bytes(
            options.generated_path,
            generated,
            sizeof(uint32_t),
            options.generate_count)) {
        fputs("failed to write probe output\n", stderr);
        goto cleanup_runtime;
    }
    printf(
        "{\"prompt_tokens\":%lu,\"generated_tokens\":%lu,"
        "\"forward_tokens\":%lu,\"elapsed_seconds\":%.9g,"
        "\"tokens_per_second\":%.9g,\"context_length\":%lu,"
        "\"arena_bytes\":%lu,\"vocab_size\":%lu,"
        "\"logits_checkpoint\":\"after_prompt\"}\n",
        (unsigned long)prompt_count,
        (unsigned long)options.generate_count,
        (unsigned long)forward_count,
        elapsed,
        elapsed > 0.0 ? (double)forward_count / elapsed : 0.0,
        (unsigned long)ng_runtime_context_length(&runtime),
        (unsigned long)runtime.arena_bytes,
        (unsigned long)model.spec.vocab_size);
    exit_code = 0;

cleanup_runtime:
    ng_runtime_reset(&runtime);
cleanup_arena:
    free(arena);
cleanup_model:
    ng_model_free(&model);
    return exit_code;
}
