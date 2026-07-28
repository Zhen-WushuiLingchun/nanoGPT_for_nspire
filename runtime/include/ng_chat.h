#ifndef NG_CHAT_H
#define NG_CHAT_H

#include "ng_model.h"
#include "ng_runtime.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NG_CHAT_INPUT_BYTES 192u
#define NG_CHAT_TRANSCRIPT_BYTES 4096u
#define NG_CHAT_MAX_CELLS 24u
#define NG_CHAT_MAX_PENDING_TOKENS 256u
#define NG_CHAT_ERROR_BYTES 96u
#define NG_CHAT_DEFAULT_GENERATION_TOKENS 32u

typedef enum ng_chat_status {
    NG_CHAT_OK = 0,
    NG_CHAT_NO_CHANGE = 1,
    NG_CHAT_FULL = 2,
    NG_CHAT_UNSUPPORTED = 3,
    NG_CHAT_INVALID = 4,
    NG_CHAT_MODEL_ERROR = 5
} ng_chat_status;

typedef enum ng_chat_role {
    NG_CHAT_ROLE_USER = 1,
    NG_CHAT_ROLE_ASSISTANT = 2,
    NG_CHAT_ROLE_SYSTEM = 3
} ng_chat_role;

typedef enum ng_chat_phase {
    NG_CHAT_PHASE_IDLE = 0,
    NG_CHAT_PHASE_PREFILL = 1,
    NG_CHAT_PHASE_GENERATING = 2,
    NG_CHAT_PHASE_DONE = 3,
    NG_CHAT_PHASE_ERROR = 4
} ng_chat_phase;

typedef struct ng_chat_cell {
    size_t text_offset;
    size_t text_length;
    ng_chat_role role;
} ng_chat_cell;

typedef struct ng_chat {
    const ng_model *model;
    ng_runtime *runtime;

    char input[NG_CHAT_INPUT_BYTES];
    size_t input_length;
    size_t input_cursor;

    char transcript_text[NG_CHAT_TRANSCRIPT_BYTES];
    size_t transcript_length;
    ng_chat_cell cells[NG_CHAT_MAX_CELLS];
    size_t cell_count;
    size_t scroll_line;

    uint32_t pending_tokens[NG_CHAT_MAX_PENDING_TOKENS];
    size_t pending_count;
    size_t pending_index;
    const float *last_logits;

    ng_chat_phase phase;
    size_t max_generation_tokens;
    size_t generated_tokens;
    size_t consecutive_newlines;
    size_t context_tokens;
    uint32_t submit_ms;
    uint32_t first_token_ms;
    uint32_t last_token_ms;
    uint32_t ttft_ms;
    uint32_t decode_milli_tokens_per_second;

    size_t tracked_model_bytes;
    size_t tracked_arena_bytes;
    size_t tracked_framebuffer_bytes;
    size_t tracked_peak_bytes;
    char error_message[NG_CHAT_ERROR_BYTES];
} ng_chat;

void ng_chat_init(
    ng_chat *chat,
    const ng_model *model,
    ng_runtime *runtime);

ng_chat_status ng_chat_input_insert(ng_chat *chat, char byte);
ng_chat_status ng_chat_input_move(ng_chat *chat, int delta);
ng_chat_status ng_chat_input_backspace(ng_chat *chat);
ng_chat_status ng_chat_input_delete(ng_chat *chat);

ng_chat_status ng_chat_append_cell(
    ng_chat *chat,
    ng_chat_role role,
    const char *text,
    size_t text_length);

ng_chat_status ng_chat_append_to_last_cell(
    ng_chat *chat,
    const char *text,
    size_t text_length);

ng_chat_status ng_chat_submit(ng_chat *chat, uint32_t now_ms);
ng_chat_status ng_chat_step(ng_chat *chat, uint32_t now_ms);
ng_chat_status ng_chat_cancel(ng_chat *chat);

void ng_chat_new_chat(ng_chat *chat);
void ng_chat_shutdown(ng_chat *chat);

const char *ng_chat_status_string(ng_chat_status status);

#ifdef __cplusplus
}
#endif

#endif
