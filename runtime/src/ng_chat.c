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
