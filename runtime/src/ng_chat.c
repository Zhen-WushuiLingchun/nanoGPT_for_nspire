#include "ng_chat.h"

#include <string.h>

static void ng_chat_secure_zero(void *pointer, size_t length) {
    volatile uint8_t *bytes = (volatile uint8_t *)pointer;
    while (length > 0u) {
        *bytes = 0u;
        ++bytes;
        --length;
    }
}

static int ng_chat_role_is_valid(ng_chat_role role) {
    return role == NG_CHAT_ROLE_USER
        || role == NG_CHAT_ROLE_ASSISTANT
        || role == NG_CHAT_ROLE_SYSTEM;
}

static void ng_chat_set_error(ng_chat *chat, const char *message) {
    size_t length;
    if (chat == NULL) {
        return;
    }
    ng_chat_secure_zero(chat->error_message, sizeof(chat->error_message));
    if (message == NULL) {
        return;
    }
    length = strlen(message);
    if (length >= sizeof(chat->error_message)) {
        length = sizeof(chat->error_message) - 1u;
    }
    (void)memcpy(chat->error_message, message, length);
}

static ng_chat_status ng_chat_find_token(
    const ng_model *model,
    const char *bytes,
    size_t length,
    uint32_t *token_id) {
    uint32_t index;
    if (model == NULL || bytes == NULL || token_id == NULL) {
        return NG_CHAT_INVALID;
    }
    for (index = 0u; index < model->spec.vocab_size; ++index) {
        const ng_vocab_token *token = &model->vocabulary[index];
        if ((size_t)token->length == length
            && memcmp(token->bytes, bytes, length) == 0) {
            *token_id = index;
            return NG_CHAT_OK;
        }
    }
    return NG_CHAT_UNSUPPORTED;
}

static uint32_t ng_chat_argmax(const float *values, uint32_t count) {
    uint32_t best = 0u;
    uint32_t index;
    for (index = 1u; index < count; ++index) {
        if (values[index] > values[best]) {
            best = index;
        }
    }
    return best;
}

void ng_chat_init(
    ng_chat *chat,
    const ng_model *model,
    ng_runtime *runtime) {
    if (chat == NULL) {
        return;
    }
    (void)memset(chat, 0, sizeof(*chat));
    chat->model = model;
    chat->runtime = runtime;
    chat->phase = NG_CHAT_PHASE_IDLE;
    chat->max_generation_tokens = NG_CHAT_DEFAULT_GENERATION_TOKENS;
}

ng_chat_status ng_chat_input_insert(ng_chat *chat, char byte) {
    size_t tail_bytes;
    unsigned char value;
    if (chat == NULL) {
        return NG_CHAT_INVALID;
    }
    value = (unsigned char)byte;
    if (value < 0x20u || value > 0x7eu) {
        return NG_CHAT_UNSUPPORTED;
    }
    if (chat->input_length + 1u >= NG_CHAT_INPUT_BYTES) {
        return NG_CHAT_FULL;
    }
    tail_bytes = chat->input_length - chat->input_cursor;
    (void)memmove(
        chat->input + chat->input_cursor + 1u,
        chat->input + chat->input_cursor,
        tail_bytes + 1u);
    chat->input[chat->input_cursor] = byte;
    chat->input_cursor += 1u;
    chat->input_length += 1u;
    return NG_CHAT_OK;
}

ng_chat_status ng_chat_input_move(ng_chat *chat, int delta) {
    size_t magnitude;
    if (chat == NULL) {
        return NG_CHAT_INVALID;
    }
    if (delta < 0) {
        magnitude = (size_t)(-(delta + 1)) + 1u;
        if (magnitude > chat->input_cursor) {
            chat->input_cursor = 0u;
        } else {
            chat->input_cursor -= magnitude;
        }
    } else {
        magnitude = (size_t)delta;
        if (magnitude > chat->input_length - chat->input_cursor) {
            chat->input_cursor = chat->input_length;
        } else {
            chat->input_cursor += magnitude;
        }
    }
    return NG_CHAT_OK;
}

ng_chat_status ng_chat_input_backspace(ng_chat *chat) {
    size_t removed;
    if (chat == NULL) {
        return NG_CHAT_INVALID;
    }
    if (chat->input_cursor == 0u) {
        return NG_CHAT_NO_CHANGE;
    }
    removed = chat->input_cursor - 1u;
    (void)memmove(
        chat->input + removed,
        chat->input + chat->input_cursor,
        chat->input_length - chat->input_cursor + 1u);
    chat->input_cursor -= 1u;
    chat->input_length -= 1u;
    return NG_CHAT_OK;
}

ng_chat_status ng_chat_input_delete(ng_chat *chat) {
    if (chat == NULL) {
        return NG_CHAT_INVALID;
    }
    if (chat->input_cursor >= chat->input_length) {
        return NG_CHAT_NO_CHANGE;
    }
    (void)memmove(
        chat->input + chat->input_cursor,
        chat->input + chat->input_cursor + 1u,
        chat->input_length - chat->input_cursor);
    chat->input_length -= 1u;
    return NG_CHAT_OK;
}

ng_chat_status ng_chat_append_cell(
    ng_chat *chat,
    ng_chat_role role,
    const char *text,
    size_t text_length) {
    ng_chat_cell *cell;
    if (chat == NULL || (text == NULL && text_length != 0u)) {
        return NG_CHAT_INVALID;
    }
    if (!ng_chat_role_is_valid(role)) {
        return NG_CHAT_INVALID;
    }
    if (chat->cell_count >= NG_CHAT_MAX_CELLS
        || text_length > NG_CHAT_TRANSCRIPT_BYTES - chat->transcript_length) {
        return NG_CHAT_FULL;
    }
    cell = &chat->cells[chat->cell_count];
    cell->text_offset = chat->transcript_length;
    cell->text_length = text_length;
    cell->role = role;
    if (text_length != 0u) {
        (void)memcpy(
            chat->transcript_text + chat->transcript_length,
            text,
            text_length);
    }
    chat->transcript_length += text_length;
    chat->cell_count += 1u;
    return NG_CHAT_OK;
}

ng_chat_status ng_chat_append_to_last_cell(
    ng_chat *chat,
    const char *text,
    size_t text_length) {
    ng_chat_cell *cell;
    if (chat == NULL || (text == NULL && text_length != 0u)) {
        return NG_CHAT_INVALID;
    }
    if (chat->cell_count == 0u) {
        return NG_CHAT_INVALID;
    }
    if (text_length > NG_CHAT_TRANSCRIPT_BYTES - chat->transcript_length) {
        return NG_CHAT_FULL;
    }
    cell = &chat->cells[chat->cell_count - 1u];
    if (cell->text_offset + cell->text_length != chat->transcript_length) {
        return NG_CHAT_INVALID;
    }
    if (text_length != 0u) {
        (void)memcpy(
            chat->transcript_text + chat->transcript_length,
            text,
            text_length);
    }
    cell->text_length += text_length;
    chat->transcript_length += text_length;
    return NG_CHAT_OK;
}

ng_chat_status ng_chat_submit(ng_chat *chat, uint32_t now_ms) {
    uint32_t separator_token = 0u;
    size_t separator_count;
    size_t pending_count;
    size_t index;
    int has_separator;
    ng_chat_status status;
    if (chat == NULL || chat->model == NULL || chat->runtime == NULL) {
        return NG_CHAT_INVALID;
    }
    if (chat->phase == NG_CHAT_PHASE_PREFILL
        || chat->phase == NG_CHAT_PHASE_GENERATING) {
        return NG_CHAT_INVALID;
    }
    if (chat->input_length == 0u) {
        return NG_CHAT_NO_CHANGE;
    }
    /*
     * This is a completion model, not a role-conditioned chat model. Ending
     * every prompt with the same newline makes the final conditioning token
     * identical and can collapse greedy decoding onto one common line start.
     * Keep the first prompt exact. On later turns, place the optional line
     * boundary before the new user text so generation still conditions on the
     * user's actual final character.
     */
    has_separator =
        ng_runtime_context_length(chat->runtime) != 0u
        && ng_chat_find_token(chat->model, "\n", 1u, &separator_token)
            == NG_CHAT_OK;
    separator_count = has_separator ? 1u : 0u;
    pending_count = chat->input_length + separator_count;
    if (pending_count > NG_CHAT_MAX_PENDING_TOKENS
        || pending_count > chat->model->spec.block_size
            - ng_runtime_context_length(chat->runtime)
        || chat->cell_count > NG_CHAT_MAX_CELLS - 2u
        || chat->input_length
            > NG_CHAT_TRANSCRIPT_BYTES - chat->transcript_length) {
        return NG_CHAT_FULL;
    }
    if (has_separator) {
        chat->pending_tokens[0] = separator_token;
    }
    for (index = 0u; index < chat->input_length; ++index) {
        status = ng_chat_find_token(
            chat->model,
            chat->input + index,
            1u,
            &chat->pending_tokens[index + separator_count]);
        if (status != NG_CHAT_OK) {
            ng_chat_secure_zero(
                chat->pending_tokens,
                sizeof(chat->pending_tokens));
            return status;
        }
    }
    status = ng_chat_append_cell(
        chat,
        NG_CHAT_ROLE_USER,
        chat->input,
        chat->input_length);
    if (status != NG_CHAT_OK) {
        ng_chat_secure_zero(
            chat->pending_tokens,
            sizeof(chat->pending_tokens));
        return status;
    }
    status = ng_chat_append_cell(
        chat,
        NG_CHAT_ROLE_ASSISTANT,
        NULL,
        0u);
    if (status != NG_CHAT_OK) {
        return status;
    }
    ng_chat_secure_zero(chat->input, sizeof(chat->input));
    chat->input_length = 0u;
    chat->input_cursor = 0u;
    chat->pending_count = pending_count;
    chat->pending_index = 0u;
    chat->last_logits = NULL;
    chat->phase = NG_CHAT_PHASE_PREFILL;
    chat->generated_tokens = 0u;
    chat->consecutive_newlines = 0u;
    chat->submit_ms = now_ms;
    chat->first_token_ms = 0u;
    chat->last_token_ms = 0u;
    chat->ttft_ms = 0u;
    chat->decode_milli_tokens_per_second = 0u;
    chat->context_tokens = ng_runtime_context_length(chat->runtime);
    ng_chat_set_error(chat, NULL);
    return NG_CHAT_OK;
}

ng_chat_status ng_chat_step(ng_chat *chat, uint32_t now_ms) {
    ng_error error;
    ng_status runtime_status;
    if (chat == NULL || chat->model == NULL || chat->runtime == NULL) {
        return NG_CHAT_INVALID;
    }
    if (chat->phase == NG_CHAT_PHASE_PREFILL) {
        if (chat->pending_index >= chat->pending_count) {
            chat->phase = NG_CHAT_PHASE_GENERATING;
            return NG_CHAT_NO_CHANGE;
        }
        runtime_status = ng_runtime_forward_token(
            chat->runtime,
            chat->pending_tokens[chat->pending_index],
            &chat->last_logits,
            &error);
        if (runtime_status != NG_STATUS_OK) {
            chat->phase = NG_CHAT_PHASE_ERROR;
            ng_chat_set_error(chat, error.message);
            return NG_CHAT_MODEL_ERROR;
        }
        chat->pending_index += 1u;
        chat->context_tokens = ng_runtime_context_length(chat->runtime);
        if (chat->pending_index == chat->pending_count) {
            chat->phase = NG_CHAT_PHASE_GENERATING;
        }
        return NG_CHAT_OK;
    }
    if (chat->phase == NG_CHAT_PHASE_GENERATING) {
        uint32_t token_id;
        const uint8_t *token_bytes = NULL;
        uint16_t token_length = 0u;
        uint64_t rate_numerator;
        uint32_t elapsed_ms;
        ng_chat_status append_status;
        if (chat->last_logits == NULL) {
            chat->phase = NG_CHAT_PHASE_ERROR;
            ng_chat_set_error(chat, "missing generation logits");
            return NG_CHAT_MODEL_ERROR;
        }
        if (chat->generated_tokens >= chat->max_generation_tokens
            || ng_runtime_context_length(chat->runtime)
                >= chat->model->spec.block_size) {
            chat->phase = NG_CHAT_PHASE_DONE;
            return NG_CHAT_NO_CHANGE;
        }
        token_id = ng_chat_argmax(
            chat->last_logits,
            chat->model->spec.vocab_size);
        runtime_status = ng_model_token(
            chat->model,
            token_id,
            &token_bytes,
            &token_length);
        if (runtime_status != NG_STATUS_OK) {
            chat->phase = NG_CHAT_PHASE_ERROR;
            ng_chat_set_error(chat, "invalid generated token");
            return NG_CHAT_MODEL_ERROR;
        }
        if ((size_t)token_length
            > NG_CHAT_TRANSCRIPT_BYTES - chat->transcript_length) {
            chat->phase = NG_CHAT_PHASE_ERROR;
            ng_chat_set_error(chat, "transcript capacity full");
            return NG_CHAT_FULL;
        }
        runtime_status = ng_runtime_forward_token(
            chat->runtime,
            token_id,
            &chat->last_logits,
            &error);
        if (runtime_status != NG_STATUS_OK) {
            chat->phase = NG_CHAT_PHASE_ERROR;
            ng_chat_set_error(chat, error.message);
            return NG_CHAT_MODEL_ERROR;
        }
        append_status = ng_chat_append_to_last_cell(
            chat,
            (const char *)token_bytes,
            (size_t)token_length);
        if (append_status != NG_CHAT_OK) {
            chat->phase = NG_CHAT_PHASE_ERROR;
            ng_chat_set_error(chat, "transcript append failed");
            return append_status;
        }
        chat->generated_tokens += 1u;
        chat->context_tokens = ng_runtime_context_length(chat->runtime);
        if (chat->generated_tokens == 1u) {
            chat->first_token_ms = now_ms;
            chat->ttft_ms = now_ms - chat->submit_ms;
        }
        chat->last_token_ms = now_ms;
        if (chat->generated_tokens > 1u) {
            elapsed_ms = now_ms - chat->first_token_ms;
            if (elapsed_ms > 0u) {
                rate_numerator =
                    (uint64_t)(chat->generated_tokens - 1u) * 1000000u;
                chat->decode_milli_tokens_per_second =
                    (uint32_t)(rate_numerator / elapsed_ms);
            }
        }
        if (token_length == 1u && token_bytes[0] == (uint8_t)'\n') {
            chat->consecutive_newlines += 1u;
        } else {
            chat->consecutive_newlines = 0u;
        }
        if (chat->generated_tokens >= chat->max_generation_tokens
            || chat->context_tokens >= chat->model->spec.block_size
            || chat->consecutive_newlines >= 2u) {
            chat->phase = NG_CHAT_PHASE_DONE;
        }
        return NG_CHAT_OK;
    }
    return NG_CHAT_NO_CHANGE;
}

ng_chat_status ng_chat_cancel(ng_chat *chat) {
    if (chat == NULL) {
        return NG_CHAT_INVALID;
    }
    if (chat->phase != NG_CHAT_PHASE_GENERATING) {
        return NG_CHAT_NO_CHANGE;
    }
    ng_chat_secure_zero(chat->pending_tokens, sizeof(chat->pending_tokens));
    chat->pending_count = 0u;
    chat->pending_index = 0u;
    chat->last_logits = NULL;
    chat->phase = NG_CHAT_PHASE_DONE;
    return NG_CHAT_OK;
}

void ng_chat_new_chat(ng_chat *chat) {
    const ng_model *model;
    ng_runtime *runtime;
    size_t max_generation_tokens;
    size_t model_bytes;
    size_t arena_bytes;
    size_t framebuffer_bytes;
    size_t peak_bytes;
    if (chat == NULL) {
        return;
    }
    model = chat->model;
    runtime = chat->runtime;
    max_generation_tokens = chat->max_generation_tokens;
    model_bytes = chat->tracked_model_bytes;
    arena_bytes = chat->tracked_arena_bytes;
    framebuffer_bytes = chat->tracked_framebuffer_bytes;
    peak_bytes = chat->tracked_peak_bytes;
    if (runtime != NULL) {
        ng_runtime_reset(runtime);
    }
    ng_chat_secure_zero(chat, sizeof(*chat));
    chat->model = model;
    chat->runtime = runtime;
    chat->phase = NG_CHAT_PHASE_IDLE;
    chat->max_generation_tokens = max_generation_tokens == 0u
        ? NG_CHAT_DEFAULT_GENERATION_TOKENS
        : max_generation_tokens;
    chat->tracked_model_bytes = model_bytes;
    chat->tracked_arena_bytes = arena_bytes;
    chat->tracked_framebuffer_bytes = framebuffer_bytes;
    chat->tracked_peak_bytes = peak_bytes;
}

void ng_chat_shutdown(ng_chat *chat) {
    if (chat == NULL) {
        return;
    }
    if (chat->runtime != NULL) {
        ng_runtime_reset(chat->runtime);
    }
    ng_chat_secure_zero(chat, sizeof(*chat));
}

const char *ng_chat_status_string(ng_chat_status status) {
    switch (status) {
        case NG_CHAT_OK:
            return "ok";
        case NG_CHAT_NO_CHANGE:
            return "no change";
        case NG_CHAT_FULL:
            return "capacity full";
        case NG_CHAT_UNSUPPORTED:
            return "unsupported input";
        case NG_CHAT_INVALID:
            return "invalid state";
        case NG_CHAT_MODEL_ERROR:
            return "model error";
        default:
            return "unknown";
    }
}
