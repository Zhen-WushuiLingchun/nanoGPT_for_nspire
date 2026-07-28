#include <libndls.h>

#include "chat_platform_ndless.h"
#include "ng_chat.h"
#include "ng_chat_view.h"
#include "ng_gfx.h"
#include "ng_model.h"
#include "ng_runtime.h"

#include <stdlib.h>
#include <string.h>

#define NG_MODEL_PATH_BYTES 256u

typedef struct ng_ndless_app {
    ng_model model;
    ng_runtime runtime;
    ng_chat chat;
    ng_ndless_platform platform;
    uint8_t *arena;
} ng_ndless_app;

static ng_ndless_app ng_app;

static int ng_join_path(
    char *destination,
    size_t capacity,
    const char *root,
    const char *suffix) {
    size_t root_length;
    size_t suffix_length;
    if (destination == NULL || root == NULL || suffix == NULL) {
        return 0;
    }
    root_length = strlen(root);
    suffix_length = strlen(suffix);
    if (root_length + suffix_length + 1u > capacity) {
        return 0;
    }
    (void)memcpy(destination, root, root_length);
    (void)memcpy(destination + root_length, suffix, suffix_length + 1u);
    return 1;
}

static ng_status ng_load_deployment_model(
    int argument_count,
    char **arguments,
    char *loaded_path,
    size_t loaded_path_capacity,
    ng_error *error) {
    static const char *relative_candidates[] = {
        "model.ngm.tns",
        "quantized-small.ngm.tns",
        "distilled-small.ngm.tns",
        "direct-small.ngm.tns"
    };
    static const char *document_candidates[] = {
        "/nanoGPT/model.ngm.tns",
        "/nanoGPT/quantized-small.ngm.tns",
        "/nanoGPT/distilled-small.ngm.tns",
        "/nanoGPT/direct-small.ngm.tns"
    };
    const char *documents = get_documents_dir();
    char candidate[NG_MODEL_PATH_BYTES];
    ng_status status = NG_STATUS_IO;
    size_t index;
    if (argument_count > 1 && arguments[1] != NULL) {
        status = ng_model_load_file(
            arguments[1],
            (size_t)NG_INFERENCE_MEMORY_LIMIT_BYTES,
            &ng_app.model,
            error);
        if (status == NG_STATUS_OK) {
            (void)ng_join_path(
                loaded_path,
                loaded_path_capacity,
                "",
                arguments[1]);
            return status;
        }
    }
    for (index = 0u;
         index
         < sizeof(relative_candidates) / sizeof(relative_candidates[0]);
         ++index) {
        status = ng_model_load_file(
            relative_candidates[index],
            (size_t)NG_INFERENCE_MEMORY_LIMIT_BYTES,
            &ng_app.model,
            error);
        if (status == NG_STATUS_OK) {
            (void)ng_join_path(
                loaded_path,
                loaded_path_capacity,
                "",
                relative_candidates[index]);
            return status;
        }
        if (status != NG_STATUS_IO) {
            return status;
        }
    }
    if (documents != NULL) {
        for (index = 0u;
             index
             < sizeof(document_candidates) / sizeof(document_candidates[0]);
             ++index) {
            if (!ng_join_path(
                    candidate,
                    sizeof(candidate),
                    documents,
                    document_candidates[index])) {
                continue;
            }
            status = ng_model_load_file(
                candidate,
                (size_t)NG_INFERENCE_MEMORY_LIMIT_BYTES,
                &ng_app.model,
                error);
            if (status == NG_STATUS_OK) {
                (void)ng_join_path(
                    loaded_path,
                    loaded_path_capacity,
                    "",
                    candidate);
                return status;
            }
            if (status != NG_STATUS_IO) {
                return status;
            }
        }
    }
    return status;
}

static const char *ng_model_label(const ng_model *model) {
    if (model->spec.model_storage == (uint32_t)NG_MODEL_STORAGE_W4A8) {
        return "QUANT W4A8";
    }
    return "SMALL FP32";
}

static void ng_set_ui_error(ng_chat *chat, const char *message) {
    size_t length;
    if (message == NULL) {
        return;
    }
    length = strlen(message);
    if (length >= sizeof(chat->error_message)) {
        length = sizeof(chat->error_message) - 1u;
    }
    (void)memset(chat->error_message, 0, sizeof(chat->error_message));
    (void)memcpy(chat->error_message, message, length);
    chat->phase = NG_CHAT_PHASE_ERROR;
}

static int ng_handle_event(
    ng_chat *chat,
    ng_ndless_event event,
    uint32_t now_ms) {
    ng_chat_status status = NG_CHAT_NO_CHANGE;
    switch (event.kind) {
        case NG_NDLESS_EVENT_NONE:
            return 0;
        case NG_NDLESS_EVENT_TEXT:
            if (chat->phase != NG_CHAT_PHASE_GENERATING
                && chat->phase != NG_CHAT_PHASE_PREFILL) {
                status = ng_chat_input_insert(chat, event.text);
            }
            break;
        case NG_NDLESS_EVENT_SUBMIT:
            if (chat->phase != NG_CHAT_PHASE_GENERATING
                && chat->phase != NG_CHAT_PHASE_PREFILL) {
                status = ng_chat_submit(chat, now_ms);
            }
            break;
        case NG_NDLESS_EVENT_BACKSPACE:
            status = ng_chat_input_backspace(chat);
            break;
        case NG_NDLESS_EVENT_LEFT:
            status = ng_chat_input_move(chat, -1);
            break;
        case NG_NDLESS_EVENT_RIGHT:
            status = ng_chat_input_move(chat, 1);
            break;
        case NG_NDLESS_EVENT_SCROLL_UP:
            if (chat->scroll_line < 256u) {
                chat->scroll_line += 1u;
            }
            status = NG_CHAT_OK;
            break;
        case NG_NDLESS_EVENT_SCROLL_DOWN:
            if (chat->scroll_line > 0u) {
                chat->scroll_line -= 1u;
            }
            status = NG_CHAT_OK;
            break;
        case NG_NDLESS_EVENT_NEW_CHAT:
            ng_chat_new_chat(chat);
            status = NG_CHAT_OK;
            break;
        case NG_NDLESS_EVENT_CANCEL:
            if (chat->phase == NG_CHAT_PHASE_GENERATING) {
                status = ng_chat_cancel(chat);
            } else if (
                chat->phase != NG_CHAT_PHASE_PREFILL) {
                return -1;
            }
            break;
        case NG_NDLESS_EVENT_EXIT:
            return -1;
        default:
            status = NG_CHAT_INVALID;
            break;
    }
    if (status == NG_CHAT_FULL || status == NG_CHAT_UNSUPPORTED
        || status == NG_CHAT_INVALID || status == NG_CHAT_MODEL_ERROR) {
        ng_set_ui_error(chat, ng_chat_status_string(status));
    }
    return 1;
}

static int ng_run_ui(void) {
    ng_surface surface;
    const char *label = ng_model_label(&ng_app.model);
    int running = 1;
    int dirty = 1;
    if (!ng_surface_init(
            &surface,
            ng_app.platform.framebuffer,
            NG_CHAT_VIEW_WIDTH,
            NG_CHAT_VIEW_HEIGHT,
            NG_CHAT_VIEW_WIDTH)) {
        return 0;
    }
    while (running != 0) {
        ng_ndless_event event =
            ng_ndless_platform_poll(&ng_app.platform);
        uint32_t now_ms = ng_ndless_platform_now_ms();
        int event_result = ng_handle_event(&ng_app.chat, event, now_ms);
        if (event_result < 0) {
            running = 0;
            continue;
        }
        if (event_result > 0) {
            dirty = 1;
        }
        if (ng_app.chat.phase == NG_CHAT_PHASE_PREFILL
            || ng_app.chat.phase == NG_CHAT_PHASE_GENERATING) {
            ng_chat_status step_status =
                ng_chat_step(&ng_app.chat, now_ms);
            if (step_status == NG_CHAT_FULL) {
                ng_set_ui_error(
                    &ng_app.chat,
                    ng_chat_status_string(step_status));
            }
            dirty = 1;
        }
        if (dirty != 0) {
            ng_chat_view_render(&surface, &ng_app.chat, label);
            ng_ndless_platform_present(&ng_app.platform);
            dirty = 0;
        }
        ng_ndless_platform_idle();
    }
    return 1;
}

static void ng_cleanup(void) {
    ng_chat_shutdown(&ng_app.chat);
    ng_ndless_platform_shutdown(&ng_app.platform);
    free(ng_app.arena);
    ng_app.arena = NULL;
    (void)memset(&ng_app.runtime, 0, sizeof(ng_app.runtime));
    ng_model_free(&ng_app.model);
}

int main(int argument_count, char **arguments) {
    char model_path[NG_MODEL_PATH_BYTES] = {0};
    ng_error error;
    ng_status status;
    int result = 1;

    assert_ndless_rev(2022);
    (void)memset(&ng_app, 0, sizeof(ng_app));
    (void)enable_relative_paths(arguments);
    status = ng_load_deployment_model(
        argument_count,
        arguments,
        model_path,
        sizeof(model_path),
        &error);
    if (status != NG_STATUS_OK) {
        show_msgbox(
            "nanoGPT model error",
            status == NG_STATUS_IO
                ? "Place model.ngm.tns beside nanogpt-chat.tns."
                : error.message);
        return 1;
    }
    ng_app.arena =
        (uint8_t *)malloc(ng_app.model.required_arena_bytes);
    if (ng_app.arena == NULL) {
        show_msgbox("nanoGPT memory error", "Could not allocate runtime arena.");
        ng_model_free(&ng_app.model);
        return 1;
    }
    status = ng_runtime_init(
        &ng_app.runtime,
        &ng_app.model,
        ng_app.arena,
        ng_app.model.required_arena_bytes,
        &error);
    if (status != NG_STATUS_OK) {
        show_msgbox("nanoGPT runtime error", error.message);
        ng_cleanup();
        return 1;
    }
    if (!ng_ndless_platform_init(&ng_app.platform)) {
        show_msgbox("nanoGPT display error", "Could not open RGB565 display.");
        ng_cleanup();
        return 1;
    }
    ng_chat_init(&ng_app.chat, &ng_app.model, &ng_app.runtime);
    ng_app.chat.tracked_model_bytes = ng_app.model.file_bytes;
    ng_app.chat.tracked_arena_bytes = ng_app.model.required_arena_bytes;
    ng_app.chat.tracked_framebuffer_bytes = (size_t)NG_NDLESS_SCREEN_BYTES;
    ng_app.chat.tracked_peak_bytes =
        ng_app.chat.tracked_model_bytes
        + ng_app.chat.tracked_arena_bytes
        + ng_app.chat.tracked_framebuffer_bytes
        + sizeof(ng_app.chat);
    result = ng_run_ui() ? 0 : 1;
    ng_cleanup();
    return result;
}
