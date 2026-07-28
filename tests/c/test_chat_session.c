#include "ng_chat.h"
#include "ng_model.h"
#include "ng_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void test_incremental_session(const char *model_path) {
    ng_model model;
    ng_runtime runtime;
    ng_chat chat;
    ng_error error;
    uint8_t *arena;
    size_t before;

    CHECK(
        ng_model_load_file(
            model_path,
            (size_t)NG_INFERENCE_MEMORY_LIMIT_BYTES,
            &model,
            &error)
        == NG_STATUS_OK);
    if (failures != 0) {
        return;
    }
    arena = (uint8_t *)malloc(model.required_arena_bytes);
    CHECK(arena != NULL);
    if (arena == NULL) {
        ng_model_free(&model);
        return;
    }
    CHECK(
        ng_runtime_init(
            &runtime,
            &model,
            arena,
            model.required_arena_bytes,
            &error)
        == NG_STATUS_OK);
    ng_chat_init(&chat, &model, &runtime);
    chat.max_generation_tokens = 2u;
    CHECK(ng_chat_input_insert(&chat, 'a') == NG_CHAT_OK);
    CHECK(ng_chat_submit(&chat, 100u) == NG_CHAT_OK);
    CHECK(chat.phase == NG_CHAT_PHASE_PREFILL);
    CHECK(chat.pending_count == 2u);
    CHECK(chat.cell_count == 2u);
    CHECK(chat.cells[0].role == NG_CHAT_ROLE_USER);
    CHECK(chat.cells[1].role == NG_CHAT_ROLE_ASSISTANT);
    CHECK(chat.input_length == 0u);

    before = ng_runtime_context_length(&runtime);
    CHECK(ng_chat_step(&chat, 101u) == NG_CHAT_OK);
    CHECK(ng_runtime_context_length(&runtime) == before + 1u);
    CHECK(chat.phase == NG_CHAT_PHASE_PREFILL);

    before = ng_runtime_context_length(&runtime);
    CHECK(ng_chat_step(&chat, 102u) == NG_CHAT_OK);
    CHECK(ng_runtime_context_length(&runtime) == before + 1u);
    CHECK(chat.phase == NG_CHAT_PHASE_GENERATING);

    before = ng_runtime_context_length(&runtime);
    CHECK(ng_chat_step(&chat, 103u) == NG_CHAT_OK);
    CHECK(ng_runtime_context_length(&runtime) == before + 1u);
    CHECK(chat.generated_tokens == 1u);
    CHECK(chat.ttft_ms == 3u);
    CHECK(chat.decode_milli_tokens_per_second == 0u);

    before = ng_runtime_context_length(&runtime);
    CHECK(ng_chat_step(&chat, 104u) == NG_CHAT_OK);
    CHECK(ng_runtime_context_length(&runtime) == before + 1u);
    CHECK(chat.phase == NG_CHAT_PHASE_DONE);
    CHECK(chat.generated_tokens == 2u);
    CHECK(chat.context_tokens == 4u);
    CHECK(chat.decode_milli_tokens_per_second == 1000000u);
    CHECK(chat.cells[1].text_length == 4u);
    CHECK(
        memcmp(
            chat.transcript_text + chat.cells[1].text_offset,
            "\xc3\xa9\xc3\xa9",
            4u)
        == 0);

    ng_chat_new_chat(&chat);
    CHECK(chat.phase == NG_CHAT_PHASE_IDLE);
    CHECK(chat.cell_count == 0u);
    CHECK(ng_runtime_context_length(&runtime) == 0u);
    ng_chat_shutdown(&chat);
    free(arena);
    ng_model_free(&model);
}

static void test_rejections_and_cancel(const char *model_path) {
    ng_model model;
    ng_runtime runtime;
    ng_chat chat;
    ng_error error;
    uint8_t *arena;

    CHECK(
        ng_model_load_file(
            model_path,
            (size_t)NG_INFERENCE_MEMORY_LIMIT_BYTES,
            &model,
            &error)
        == NG_STATUS_OK);
    if (failures != 0) {
        return;
    }
    arena = (uint8_t *)malloc(model.required_arena_bytes);
    CHECK(arena != NULL);
    if (arena == NULL) {
        ng_model_free(&model);
        return;
    }
    CHECK(
        ng_runtime_init(
            &runtime,
            &model,
            arena,
            model.required_arena_bytes,
            &error)
        == NG_STATUS_OK);
    ng_chat_init(&chat, &model, &runtime);
    CHECK(ng_chat_submit(&chat, 1u) == NG_CHAT_NO_CHANGE);
    CHECK(ng_chat_input_insert(&chat, 'z') == NG_CHAT_OK);
    CHECK(ng_chat_submit(&chat, 2u) == NG_CHAT_UNSUPPORTED);
    CHECK(strcmp(chat.input, "z") == 0);
    CHECK(chat.cell_count == 0u);

    ng_chat_new_chat(&chat);
    CHECK(ng_chat_input_insert(&chat, 'a') == NG_CHAT_OK);
    CHECK(ng_chat_submit(&chat, 10u) == NG_CHAT_OK);
    CHECK(ng_chat_step(&chat, 11u) == NG_CHAT_OK);
    CHECK(ng_chat_step(&chat, 12u) == NG_CHAT_OK);
    CHECK(ng_chat_cancel(&chat) == NG_CHAT_OK);
    CHECK(chat.phase == NG_CHAT_PHASE_DONE);
    CHECK(chat.pending_count == 0u);
    CHECK(chat.last_logits == NULL);

    ng_chat_shutdown(&chat);
    free(arena);
    ng_model_free(&model);
}

int main(int argument_count, char **arguments) {
    if (argument_count != 2) {
        fputs("usage: test_chat_session MODEL.ngm\n", stderr);
        return 2;
    }
    test_incremental_session(arguments[1]);
    test_rejections_and_cancel(arguments[1]);
    if (failures != 0) {
        fprintf(stderr, "%d chat session checks failed\n", failures);
        return 1;
    }
    puts("chat session checks passed");
    return 0;
}
