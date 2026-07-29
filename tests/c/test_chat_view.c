#include "ng_chat.h"
#include "ng_chat_view.h"
#include "ng_gfx.h"

#include <stdint.h>
#include <stdio.h>

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

static void test_clipping_and_guards(void) {
    uint16_t guarded[18u * 18u + 2u];
    ng_surface surface;
    uint16_t red;
    uint16_t green;
    size_t index;

    guarded[0] = 0xa55au;
    guarded[18u * 18u + 1u] = 0x5aa5u;
    CHECK(ng_surface_init(&surface, guarded + 1u, 18, 18, 18) == 1);
    red = ng_rgb565(255u, 0u, 0u);
    green = ng_rgb565(0u, 255u, 0u);
    ng_gfx_clear(&surface, 0u);
    ng_gfx_fill_rect(&surface, -2, -2, 4, 4, red);
    CHECK(guarded[1] == red);
    CHECK(guarded[2] == red);
    CHECK(guarded[1u + 18u] == red);
    CHECK(guarded[2u + 18u] == red);
    CHECK(guarded[3] == 0u);

    ng_gfx_set_clip(&surface, 4, 4, 4, 4);
    ng_gfx_fill_rect(&surface, 0, 0, 18, 18, green);
    for (index = 0u; index < 18u * 18u; ++index) {
        size_t x = index % 18u;
        size_t y = index / 18u;
        if (x >= 4u && x < 8u && y >= 4u && y < 8u) {
            CHECK(guarded[index + 1u] == green);
        }
    }
    CHECK(guarded[0] == 0xa55au);
    CHECK(guarded[18u * 18u + 1u] == 0x5aa5u);
}

static void test_deterministic_full_view(void) {
    static uint16_t pixels[
        NG_CHAT_VIEW_WIDTH * NG_CHAT_VIEW_HEIGHT];
    static const char response[] =
        "Causal attention lets each position read only the past. "
        "The triangular mask blocks future keys.";
    ng_surface surface;
    ng_chat chat;
    uint32_t hash;

    CHECK(
        ng_surface_init(
            &surface,
            pixels,
            NG_CHAT_VIEW_WIDTH,
            NG_CHAT_VIEW_HEIGHT,
            NG_CHAT_VIEW_WIDTH)
        == 1);
    ng_chat_init(&chat, NULL, NULL);
    CHECK(
        ng_chat_append_cell(
            &chat,
            NG_CHAT_ROLE_USER,
            "Explain causal attention.",
            25u)
        == NG_CHAT_OK);
    CHECK(
        ng_chat_append_cell(
            &chat,
            NG_CHAT_ROLE_ASSISTANT,
            response,
            sizeof(response) - 1u)
        == NG_CHAT_OK);
    CHECK(ng_chat_input_insert(&chat, 'W') == NG_CHAT_OK);
    CHECK(ng_chat_input_insert(&chat, 'h') == NG_CHAT_OK);
    CHECK(ng_chat_input_insert(&chat, 'y') == NG_CHAT_OK);
    chat.phase = NG_CHAT_PHASE_GENERATING;
    chat.context_tokens = 47u;
    chat.ttft_ms = 820u;
    chat.decode_milli_tokens_per_second = 12400u;
    chat.tracked_peak_bytes = 8493465u;
    ng_chat_view_render(&surface, &chat, "Q-V2 W4A8");
    hash = framebuffer_hash(
        pixels,
        (size_t)NG_CHAT_VIEW_WIDTH * (size_t)NG_CHAT_VIEW_HEIGHT);
    fprintf(stderr, "chat view hash: %08x\n", (unsigned)hash);
    CHECK(hash == 0x206105dbu);
    CHECK(pixels[0] == NG_CHAT_COLOR_BORDER);
    CHECK(
        pixels[
            (NG_CHAT_VIEW_HEIGHT - 1) * NG_CHAT_VIEW_WIDTH
            + (NG_CHAT_VIEW_WIDTH - 1)]
        == NG_CHAT_COLOR_BORDER);
}

int main(void) {
    test_clipping_and_guards();
    test_deterministic_full_view();
    if (failures != 0) {
        fprintf(stderr, "%d chat view checks failed\n", failures);
        return 1;
    }
    puts("chat view checks passed");
    return 0;
}
