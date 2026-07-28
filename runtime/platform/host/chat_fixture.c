#include "ng_chat.h"
#include "ng_chat_view.h"
#include "ng_gfx.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t framebuffer_hash(
    const uint16_t *pixels,
    size_t pixel_count) {
    uint32_t hash = 2166136261u;
    size_t index;
    for (index = 0u; index < pixel_count; ++index) {
        hash ^= (uint32_t)(pixels[index] & 0xffu);
        hash *= 16777619u;
        hash ^= (uint32_t)(pixels[index] >> 8u);
        hash *= 16777619u;
    }
    return hash;
}

static int write_ppm(const char *path, const uint16_t *pixels) {
    FILE *output = fopen(path, "wb");
    size_t index;
    if (output == NULL) {
        fprintf(stderr, "error: cannot create %s\n", path);
        return 0;
    }
    if (fprintf(
            output,
            "P6\n%d %d\n255\n",
            NG_CHAT_VIEW_WIDTH,
            NG_CHAT_VIEW_HEIGHT)
        < 0) {
        (void)fclose(output);
        return 0;
    }
    for (index = 0u;
         index
         < (size_t)NG_CHAT_VIEW_WIDTH * (size_t)NG_CHAT_VIEW_HEIGHT;
         ++index) {
        uint16_t pixel = pixels[index];
        unsigned char rgb[3];
        uint8_t red = (uint8_t)((pixel >> 11u) & 0x1fu);
        uint8_t green = (uint8_t)((pixel >> 5u) & 0x3fu);
        uint8_t blue = (uint8_t)(pixel & 0x1fu);
        rgb[0] = (unsigned char)((red << 3u) | (red >> 2u));
        rgb[1] = (unsigned char)((green << 2u) | (green >> 4u));
        rgb[2] = (unsigned char)((blue << 3u) | (blue >> 2u));
        if (fwrite(rgb, sizeof(rgb), 1u, output) != 1u) {
            (void)fclose(output);
            return 0;
        }
    }
    return fclose(output) == 0;
}

static int render_fixture(
    const char *path,
    ng_chat *chat,
    const char *model_label,
    uint16_t *pixels) {
    ng_surface surface;
    uint32_t hash;
    if (!ng_surface_init(
            &surface,
            pixels,
            NG_CHAT_VIEW_WIDTH,
            NG_CHAT_VIEW_HEIGHT,
            NG_CHAT_VIEW_WIDTH)) {
        return 0;
    }
    ng_chat_view_render(&surface, chat, model_label);
    hash = framebuffer_hash(
        pixels,
        (size_t)NG_CHAT_VIEW_WIDTH * (size_t)NG_CHAT_VIEW_HEIGHT);
    if (!write_ppm(path, pixels)) {
        return 0;
    }
    printf("%s %08x\n", path, (unsigned)hash);
    return 1;
}

static void populate_conversation(ng_chat *chat) {
    static const char answer[] =
        "Causal attention lets every position read the past, but never the "
        "future. A triangular mask removes those future attention scores.";
    static const char followup[] = "Why triangular?";
    (void)ng_chat_append_cell(
        chat,
        NG_CHAT_ROLE_USER,
        "Explain causal attention.",
        25u);
    (void)ng_chat_append_cell(
        chat,
        NG_CHAT_ROLE_ASSISTANT,
        answer,
        sizeof(answer) - 1u);
    (void)ng_chat_append_cell(
        chat,
        NG_CHAT_ROLE_USER,
        followup,
        sizeof(followup) - 1u);
}

int main(int argument_count, char **arguments) {
    static uint16_t pixels[
        NG_CHAT_VIEW_WIDTH * NG_CHAT_VIEW_HEIGHT];
    ng_chat chat;

    if (argument_count != 5) {
        fputs(
            "usage: chat_fixture READY.ppm CHAT.ppm RUN.ppm ERROR.ppm\n",
            stderr);
        return 2;
    }

    ng_chat_init(&chat, NULL, NULL);
    chat.tracked_peak_bytes = 8415168u;
    if (!render_fixture(arguments[1], &chat, "Q-V2 W4A8", pixels)) {
        return 1;
    }

    ng_chat_new_chat(&chat);
    populate_conversation(&chat);
    (void)ng_chat_append_cell(
        &chat,
        NG_CHAT_ROLE_ASSISTANT,
        "Because position i may only use keys j <= i.",
        42u);
    (void)ng_chat_input_insert(&chat, 'C');
    (void)ng_chat_input_insert(&chat, 'a');
    (void)ng_chat_input_insert(&chat, 'n');
    chat.context_tokens = 73u;
    chat.ttft_ms = 820u;
    chat.decode_milli_tokens_per_second = 12400u;
    if (!render_fixture(arguments[2], &chat, "Q-V2 W4A8", pixels)) {
        return 1;
    }

    ng_chat_new_chat(&chat);
    populate_conversation(&chat);
    (void)ng_chat_append_cell(
        &chat,
        NG_CHAT_ROLE_ASSISTANT,
        "The mask is lower triangular because",
        36u);
    chat.phase = NG_CHAT_PHASE_GENERATING;
    chat.context_tokens = 97u;
    chat.ttft_ms = 910u;
    chat.decode_milli_tokens_per_second = 8700u;
    if (!render_fixture(arguments[3], &chat, "Q-V2 W4A8", pixels)) {
        return 1;
    }

    ng_chat_new_chat(&chat);
    populate_conversation(&chat);
    chat.phase = NG_CHAT_PHASE_ERROR;
    chat.context_tokens = 128u;
    (void)memcpy(
        chat.error_message,
        "CONTEXT FULL - MENU FOR NEW CHAT",
        32u);
    if (!render_fixture(arguments[4], &chat, "Q-V2 W4A8", pixels)) {
        return 1;
    }

    ng_chat_shutdown(&chat);
    return 0;
}
