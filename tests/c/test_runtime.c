#include "ng_model.h"
#include "ng_runtime.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

#define CHECK(expression)                                                      \
    do {                                                                       \
        if (!(expression)) {                                                   \
            fprintf(                                                           \
                stderr,                                                        \
                "CHECK failed at %s:%d: %s\n",                                \
                __FILE__,                                                      \
                __LINE__,                                                      \
                #expression);                                                  \
            failures += 1;                                                     \
        }                                                                      \
    } while (0)

static void check_close(float actual, float expected, float tolerance) {
    CHECK(fabsf(actual - expected) <= tolerance);
}

int main(int argument_count, char **arguments) {
    static const uint32_t tokens[] = {0u, 1u, 2u};
    /*
     * Generated independently by the Python full-prefix reference over the
     * deterministic tiny fixture. Each row is the last-token vocabulary
     * logits after appending the corresponding token above.
     */
    static const float expected[][3] = {
        {-0.00576085f, 0.05852219f, 0.12280522f},
        {-0.00576091f, 0.05852218f, 0.12280525f},
        {-0.00576086f, 0.05852218f, 0.12280522f}
    };
    ng_model model;
    ng_runtime runtime;
    ng_error error;
    uint8_t *arena;
    size_t step;
    if (argument_count != 2) {
        fputs("usage: test_runtime MODEL.ngm\n", stderr);
        return 2;
    }
    CHECK(
        ng_model_load_file(
            arguments[1],
            (size_t)NG_INFERENCE_MEMORY_LIMIT_BYTES,
            &model,
            &error)
        == NG_STATUS_OK);
    if (failures != 0) {
        return 1;
    }
    arena = (uint8_t *)malloc(model.required_arena_bytes);
    CHECK(arena != NULL);
    CHECK(
        ng_runtime_init(
            &runtime,
            &model,
            arena,
            model.required_arena_bytes - 1u,
            &error)
        == NG_STATUS_MEMORY);
    CHECK(
        ng_runtime_init(
            &runtime,
            &model,
            arena,
            model.required_arena_bytes,
            &error)
        == NG_STATUS_OK);
    CHECK(ng_runtime_context_length(&runtime) == 0u);
    for (step = 0u; step < 3u; ++step) {
        const float *logits = NULL;
        size_t token;
        CHECK(
            ng_runtime_forward_token(
                &runtime,
                tokens[step],
                &logits,
                &error)
            == NG_STATUS_OK);
        CHECK(logits != NULL);
        CHECK(ng_runtime_context_length(&runtime) == step + 1u);
        for (token = 0u; token < 3u; ++token) {
            check_close(logits[token], expected[step][token], 2.0e-5f);
        }
    }
    ng_runtime_reset(&runtime);
    CHECK(ng_runtime_context_length(&runtime) == 0u);
    {
        const float *logits = NULL;
        size_t token;
        CHECK(
            ng_runtime_forward_token(
                &runtime,
                tokens[0],
                &logits,
                &error)
            == NG_STATUS_OK);
        for (token = 0u; token < 3u; ++token) {
            check_close(logits[token], expected[0][token], 2.0e-5f);
        }
    }
    ng_runtime_reset(&runtime);
    free(arena);
    ng_model_free(&model);
    if (failures != 0) {
        fprintf(stderr, "%d incremental runtime checks failed\n", failures);
        return 1;
    }
    puts("incremental runtime checks passed");
    return 0;
}
