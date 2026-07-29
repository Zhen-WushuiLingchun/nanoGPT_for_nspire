#include "ng_chat.h"
#include "ng_model.h"
#include "ng_runtime.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *role_label(ng_chat_role role) {
    if (role == NG_CHAT_ROLE_USER) {
        return "USER";
    }
    if (role == NG_CHAT_ROLE_THINK) {
        return "THINK";
    }
    if (role == NG_CHAT_ROLE_ASSISTANT) {
        return "AI";
    }
    return "SYS";
}

int main(int argument_count, char **arguments) {
    ng_model model;
    ng_runtime runtime;
    ng_chat chat;
    ng_error error;
    uint8_t *arena = NULL;
    const char *mode;
    const char *prompt;
    size_t index;
    size_t steps = 0u;
    int result = 1;

    if (argument_count != 4) {
        fputs("usage: chat_probe MODEL.ngm direct|think PROMPT\n", stderr);
        return 2;
    }
    mode = arguments[2];
    prompt = arguments[3];
    if (strcmp(mode, "direct") != 0 && strcmp(mode, "think") != 0) {
        fputs("mode must be direct or think\n", stderr);
        return 2;
    }
    if (ng_model_load_file(
            arguments[1],
            (size_t)NG_INFERENCE_MEMORY_LIMIT_BYTES,
            &model,
            &error)
        != NG_STATUS_OK) {
        fprintf(stderr, "model load failed: %s\n", error.message);
        return 1;
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
    ng_chat_init(&chat, &model, &runtime);
    if (strcmp(mode, "think") == 0
        && ng_chat_toggle_mode(&chat) != NG_CHAT_OK) {
        fputs("model does not support think mode\n", stderr);
        goto cleanup_chat;
    }
    for (index = 0u; prompt[index] != '\0'; ++index) {
        if (ng_chat_input_insert(&chat, prompt[index]) != NG_CHAT_OK) {
            fputs("prompt contains unsupported input or is too long\n", stderr);
            goto cleanup_chat;
        }
    }
    if (ng_chat_submit(&chat, 0u) != NG_CHAT_OK) {
        fputs("chat submit failed\n", stderr);
        goto cleanup_chat;
    }
    while (
        (chat.phase == NG_CHAT_PHASE_PREFILL
            || chat.phase == NG_CHAT_PHASE_GENERATING)
        && steps < (size_t)model.spec.block_size * 2u) {
        ng_chat_status status = ng_chat_step(
            &chat,
            (uint32_t)(steps + 1u));
        if (status == NG_CHAT_MODEL_ERROR
            || status == NG_CHAT_FULL
            || status == NG_CHAT_INVALID) {
            fprintf(stderr, "chat step failed: %s\n", chat.error_message);
            goto cleanup_chat;
        }
        steps += 1u;
    }
    if (chat.phase != NG_CHAT_PHASE_DONE) {
        fputs("chat did not reach a clean stop\n", stderr);
        goto cleanup_chat;
    }
    for (index = 0u; index < chat.cell_count; ++index) {
        const ng_chat_cell *cell = &chat.cells[index];
        printf("[%s]\n", role_label(cell->role));
        if (cell->text_length != 0u) {
            (void)fwrite(
                chat.transcript_text + cell->text_offset,
                1u,
                cell->text_length,
                stdout);
        }
        putchar('\n');
    }
    printf(
        "[METRICS]\ncontext=%lu generated=%lu ttft_ms=%lu\n",
        (unsigned long)chat.context_tokens,
        (unsigned long)chat.generated_tokens,
        (unsigned long)chat.ttft_ms);
    result = 0;

cleanup_chat:
    ng_chat_shutdown(&chat);
cleanup_arena:
    free(arena);
cleanup_model:
    ng_model_free(&model);
    return result;
}
