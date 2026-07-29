#include "ng_chat.h"

#include <stdio.h>
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

static int bytes_are_zero(const void *pointer, size_t length) {
    const unsigned char *bytes = (const unsigned char *)pointer;
    size_t index;
    for (index = 0u; index < length; ++index) {
        if (bytes[index] != 0u) {
            return 0;
        }
    }
    return 1;
}

static void test_input_editing(void) {
    ng_chat chat;

    ng_chat_init(&chat, NULL, NULL);
    CHECK(ng_chat_input_insert(&chat, 'a') == NG_CHAT_OK);
    CHECK(ng_chat_input_insert(&chat, 'c') == NG_CHAT_OK);
    CHECK(ng_chat_input_move(&chat, -1) == NG_CHAT_OK);
    CHECK(ng_chat_input_insert(&chat, 'b') == NG_CHAT_OK);
    CHECK(strcmp(chat.input, "abc") == 0);
    CHECK(chat.input_cursor == 2u);
    CHECK(ng_chat_input_backspace(&chat) == NG_CHAT_OK);
    CHECK(strcmp(chat.input, "ac") == 0);
    CHECK(chat.input_cursor == 1u);
    CHECK(ng_chat_input_delete(&chat) == NG_CHAT_OK);
    CHECK(strcmp(chat.input, "a") == 0);
    CHECK(ng_chat_input_move(&chat, -10) == NG_CHAT_OK);
    CHECK(chat.input_cursor == 0u);
    CHECK(ng_chat_input_backspace(&chat) == NG_CHAT_NO_CHANGE);
    CHECK(ng_chat_input_insert(&chat, '\n') == NG_CHAT_UNSUPPORTED);
}

static void test_input_capacity_is_transactional(void) {
    ng_chat chat;
    size_t index;

    ng_chat_init(&chat, NULL, NULL);
    for (index = 0u; index < NG_CHAT_INPUT_BYTES - 1u; ++index) {
        CHECK(ng_chat_input_insert(&chat, 'x') == NG_CHAT_OK);
    }
    CHECK(chat.input_length == NG_CHAT_INPUT_BYTES - 1u);
    CHECK(ng_chat_input_insert(&chat, 'y') == NG_CHAT_FULL);
    CHECK(chat.input_length == NG_CHAT_INPUT_BYTES - 1u);
    CHECK(chat.input[NG_CHAT_INPUT_BYTES - 1u] == '\0');
}

static void test_cells_and_overflow(void) {
    static const char user_text[] = "To be";
    static const char ai_text[] = " or not";
    ng_chat chat;
    size_t old_length;

    ng_chat_init(&chat, NULL, NULL);
    CHECK(
        ng_chat_append_cell(
            &chat,
            NG_CHAT_ROLE_USER,
            user_text,
            sizeof(user_text) - 1u)
        == NG_CHAT_OK);
    CHECK(
        ng_chat_append_cell(
            &chat,
            NG_CHAT_ROLE_ASSISTANT,
            ai_text,
            sizeof(ai_text) - 1u)
        == NG_CHAT_OK);
    CHECK(chat.cell_count == 2u);
    CHECK(chat.cells[0].role == NG_CHAT_ROLE_USER);
    CHECK(chat.cells[0].text_offset == 0u);
    CHECK(chat.cells[0].text_length == sizeof(user_text) - 1u);
    CHECK(chat.cells[1].text_offset == sizeof(user_text) - 1u);
    CHECK(
        memcmp(
            chat.transcript_text + chat.cells[1].text_offset,
            ai_text,
            sizeof(ai_text) - 1u)
        == 0);
    CHECK(
        ng_chat_append_to_last_cell(&chat, "?", 1u)
        == NG_CHAT_OK);
    CHECK(chat.cells[1].text_length == sizeof(ai_text));

    old_length = chat.transcript_length;
    CHECK(
        ng_chat_append_cell(
            &chat,
            NG_CHAT_ROLE_USER,
            user_text,
            NG_CHAT_TRANSCRIPT_BYTES)
        == NG_CHAT_FULL);
    CHECK(chat.transcript_length == old_length);
    CHECK(chat.cell_count == 2u);
}

static void test_cell_capacity(void) {
    ng_chat chat;
    size_t index;

    ng_chat_init(&chat, NULL, NULL);
    for (index = 0u; index < NG_CHAT_MAX_CELLS; ++index) {
        CHECK(
            ng_chat_append_cell(&chat, NG_CHAT_ROLE_USER, "x", 1u)
            == NG_CHAT_OK);
    }
    CHECK(
        ng_chat_append_cell(&chat, NG_CHAT_ROLE_USER, "x", 1u)
        == NG_CHAT_FULL);
    CHECK(chat.cell_count == NG_CHAT_MAX_CELLS);
}

static void test_reset_and_shutdown_zero_private_state(void) {
    ng_chat chat;

    ng_chat_init(&chat, NULL, NULL);
    CHECK(ng_chat_input_insert(&chat, 's') == NG_CHAT_OK);
    CHECK(
        ng_chat_append_cell(&chat, NG_CHAT_ROLE_USER, "secret", 6u)
        == NG_CHAT_OK);
    chat.pending_tokens[0] = 17u;
    chat.pending_count = 1u;
    chat.generated_tokens = 3u;
    chat.phase = NG_CHAT_PHASE_GENERATING;

    ng_chat_new_chat(&chat);
    CHECK(chat.phase == NG_CHAT_PHASE_IDLE);
    CHECK(chat.input_length == 0u);
    CHECK(chat.cell_count == 0u);
    CHECK(chat.transcript_length == 0u);
    CHECK(chat.pending_count == 0u);
    CHECK(chat.generated_tokens == 0u);
    CHECK(bytes_are_zero(chat.input, sizeof(chat.input)));
    CHECK(bytes_are_zero(chat.transcript_text, sizeof(chat.transcript_text)));
    CHECK(bytes_are_zero(chat.cells, sizeof(chat.cells)));
    CHECK(bytes_are_zero(chat.pending_tokens, sizeof(chat.pending_tokens)));

    CHECK(ng_chat_input_insert(&chat, 'x') == NG_CHAT_OK);
    ng_chat_shutdown(&chat);
    CHECK(bytes_are_zero(&chat, sizeof(chat)));
}

static void test_byte_special_direct_and_think_prompts(void) {
    ng_model model;
    ng_runtime runtime;
    ng_chat chat;

    (void)memset(&model, 0, sizeof(model));
    (void)memset(&runtime, 0, sizeof(runtime));
    model.spec.vocab_size = NG_BYTE_SPECIAL_VOCAB_SIZE;
    model.spec.block_size = 512u;
    model.spec.tokenizer_type = NG_TOKENIZER_BYTE_SPECIAL;

    ng_chat_init(&chat, &model, &runtime);
    CHECK(chat.mode == NG_CHAT_MODE_DIRECT);
    CHECK(ng_chat_input_insert(&chat, 'H') == NG_CHAT_OK);
    CHECK(ng_chat_input_insert(&chat, 'i') == NG_CHAT_OK);
    CHECK(ng_chat_submit(&chat, 10u) == NG_CHAT_OK);
    CHECK(chat.pending_count == 6u);
    CHECK(chat.pending_tokens[0] == NG_TOKEN_BOS);
    CHECK(chat.pending_tokens[1] == NG_TOKEN_USER);
    CHECK(chat.pending_tokens[2] == (uint32_t)'H');
    CHECK(chat.pending_tokens[3] == (uint32_t)'i');
    CHECK(chat.pending_tokens[4] == NG_TOKEN_ASSISTANT);
    CHECK(chat.pending_tokens[5] == NG_TOKEN_FINAL);
    CHECK(chat.cells[1].role == NG_CHAT_ROLE_ASSISTANT);
    CHECK(ng_chat_toggle_mode(&chat) == NG_CHAT_INVALID);

    ng_chat_new_chat(&chat);
    CHECK(ng_chat_toggle_mode(&chat) == NG_CHAT_OK);
    CHECK(chat.mode == NG_CHAT_MODE_THINK);
    CHECK(ng_chat_input_insert(&chat, 'x') == NG_CHAT_OK);
    CHECK(ng_chat_submit(&chat, 20u) == NG_CHAT_OK);
    CHECK(chat.pending_count == 5u);
    CHECK(chat.pending_tokens[3] == NG_TOKEN_ASSISTANT);
    CHECK(chat.pending_tokens[4] == NG_TOKEN_THINK);
    CHECK(chat.cells[1].role == NG_CHAT_ROLE_THINK);

    ng_chat_shutdown(&chat);
}

int main(void) {
    test_input_editing();
    test_input_capacity_is_transactional();
    test_cells_and_overflow();
    test_cell_capacity();
    test_reset_and_shutdown_zero_private_state();
    test_byte_special_direct_and_think_prompts();
    if (failures != 0) {
        fprintf(stderr, "%d chat state checks failed\n", failures);
        return 1;
    }
    puts("chat state checks passed");
    return 0;
}
