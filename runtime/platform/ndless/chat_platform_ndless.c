#include "chat_platform_ndless.h"

#include <libndls.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

typedef struct ng_special_binding {
    const t_key *key;
    ng_ndless_event_kind plain;
} ng_special_binding;

typedef struct ng_text_binding {
    const t_key *key;
    char plain;
    char shifted;
} ng_text_binding;

static const ng_special_binding ng_special_bindings[] = {
    {&KEY_NSPIRE_ENTER, NG_NDLESS_EVENT_SUBMIT},
    {&KEY_NSPIRE_DEL, NG_NDLESS_EVENT_BACKSPACE},
    {&KEY_NSPIRE_LEFT, NG_NDLESS_EVENT_LEFT},
    {&KEY_NSPIRE_RIGHT, NG_NDLESS_EVENT_RIGHT},
    {&KEY_NSPIRE_UP, NG_NDLESS_EVENT_SCROLL_UP},
    {&KEY_NSPIRE_DOWN, NG_NDLESS_EVENT_SCROLL_DOWN},
    {&KEY_NSPIRE_MENU, NG_NDLESS_EVENT_NEW_CHAT},
    {&KEY_NSPIRE_ESC, NG_NDLESS_EVENT_CANCEL},
    {&KEY_NSPIRE_TAB, NG_NDLESS_EVENT_NONE}
};

static const ng_text_binding ng_text_bindings[] = {
    {&KEY_NSPIRE_A, 'a', 'A'}, {&KEY_NSPIRE_B, 'b', 'B'},
    {&KEY_NSPIRE_C, 'c', 'C'}, {&KEY_NSPIRE_D, 'd', 'D'},
    {&KEY_NSPIRE_E, 'e', 'E'}, {&KEY_NSPIRE_F, 'f', 'F'},
    {&KEY_NSPIRE_G, 'g', 'G'}, {&KEY_NSPIRE_H, 'h', 'H'},
    {&KEY_NSPIRE_I, 'i', 'I'}, {&KEY_NSPIRE_J, 'j', 'J'},
    {&KEY_NSPIRE_K, 'k', 'K'}, {&KEY_NSPIRE_L, 'l', 'L'},
    {&KEY_NSPIRE_M, 'm', 'M'}, {&KEY_NSPIRE_N, 'n', 'N'},
    {&KEY_NSPIRE_O, 'o', 'O'}, {&KEY_NSPIRE_P, 'p', 'P'},
    {&KEY_NSPIRE_Q, 'q', 'Q'}, {&KEY_NSPIRE_R, 'r', 'R'},
    {&KEY_NSPIRE_S, 's', 'S'}, {&KEY_NSPIRE_T, 't', 'T'},
    {&KEY_NSPIRE_U, 'u', 'U'}, {&KEY_NSPIRE_V, 'v', 'V'},
    {&KEY_NSPIRE_W, 'w', 'W'}, {&KEY_NSPIRE_X, 'x', 'X'},
    {&KEY_NSPIRE_Y, 'y', 'Y'}, {&KEY_NSPIRE_Z, 'z', 'Z'},
    {&KEY_NSPIRE_0, '0', '0'}, {&KEY_NSPIRE_1, '1', '1'},
    {&KEY_NSPIRE_2, '2', '2'}, {&KEY_NSPIRE_3, '3', '3'},
    {&KEY_NSPIRE_4, '4', '4'}, {&KEY_NSPIRE_5, '5', '5'},
    {&KEY_NSPIRE_6, '6', '6'}, {&KEY_NSPIRE_7, '7', '7'},
    {&KEY_NSPIRE_8, '8', '8'}, {&KEY_NSPIRE_9, '9', '9'},
    {&KEY_NSPIRE_SPACE, ' ', ' '},
    {&KEY_NSPIRE_PERIOD, '.', ':'},
    {&KEY_NSPIRE_COMMA, ',', ';'},
    {&KEY_NSPIRE_APOSTROPHE, '\'', '"'},
    {&KEY_NSPIRE_PLUS, '+', '>'},
    {&KEY_NSPIRE_MINUS, '-', '<'},
    {&KEY_NSPIRE_MULTIPLY, '*', '*'},
    {&KEY_NSPIRE_DIVIDE, '/', '\\'},
    {&KEY_NSPIRE_LP, '(', '['},
    {&KEY_NSPIRE_RP, ')', ']'},
    {&KEY_NSPIRE_QUES, '?', '!'}
};

static int ng_key_is_down(const t_key *key) {
    return isKeyPressed(*key) ? 1 : 0;
}

int ng_ndless_platform_init(ng_ndless_platform *platform) {
    if (platform == NULL || platform->initialized) {
        return 0;
    }
    (void)memset(platform, 0, sizeof(*platform));
    platform->framebuffer =
        (uint16_t *)malloc((size_t)NG_NDLESS_SCREEN_BYTES);
    if (platform->framebuffer == NULL) {
        return 0;
    }
    (void)memset(
        platform->framebuffer,
        0,
        (size_t)NG_NDLESS_SCREEN_BYTES);
    if (!lcd_init(SCR_320x240_565)) {
        free(platform->framebuffer);
        platform->framebuffer = NULL;
        return 0;
    }
    platform->display_switched = true;
    platform->initialized = true;
    return 1;
}

void ng_ndless_platform_shutdown(ng_ndless_platform *platform) {
    if (platform == NULL || !platform->initialized) {
        return;
    }
    if (platform->framebuffer != NULL) {
        volatile uint16_t *pixels = platform->framebuffer;
        size_t index;
        for (index = 0u;
             index
             < (size_t)NG_NDLESS_SCREEN_WIDTH
                 * (size_t)NG_NDLESS_SCREEN_HEIGHT;
             ++index) {
            pixels[index] = 0u;
        }
        if (platform->display_switched) {
            lcd_blit(platform->framebuffer, SCR_320x240_565);
        }
    }
    if (platform->display_switched) {
        (void)lcd_init(SCR_TYPE_INVALID);
        platform->display_switched = false;
    }
    free(platform->framebuffer);
    (void)memset(platform, 0, sizeof(*platform));
}

void ng_ndless_platform_present(ng_ndless_platform *platform) {
    if (platform != NULL && platform->initialized
        && platform->framebuffer != NULL) {
        lcd_blit(platform->framebuffer, SCR_320x240_565);
    }
}

ng_ndless_event ng_ndless_platform_poll(ng_ndless_platform *platform) {
    ng_ndless_event event = {NG_NDLESS_EVENT_NONE, '\0'};
    int shift_down;
    int ctrl_down;
    size_t index;
    if (platform == NULL || !platform->initialized) {
        return event;
    }
    shift_down = ng_key_is_down(&KEY_NSPIRE_SHIFT);
    ctrl_down = ng_key_is_down(&KEY_NSPIRE_CTRL);
    for (index = 0u;
         index
         < sizeof(ng_special_bindings) / sizeof(ng_special_bindings[0]);
         ++index) {
        int down = ng_key_is_down(ng_special_bindings[index].key);
        if (down != 0 && !platform->special_down[index]) {
            if (event.kind == NG_NDLESS_EVENT_NONE) {
                event.kind = ng_special_bindings[index].plain;
                if (ng_special_bindings[index].key == &KEY_NSPIRE_ESC
                    && ctrl_down != 0) {
                    event.kind = NG_NDLESS_EVENT_EXIT;
                }
            }
        }
        platform->special_down[index] = down != 0;
    }
    for (index = 0u;
         index < sizeof(ng_text_bindings) / sizeof(ng_text_bindings[0]);
         ++index) {
        int down = ng_key_is_down(ng_text_bindings[index].key);
        if (down != 0 && !platform->text_down[index]
            && event.kind == NG_NDLESS_EVENT_NONE) {
            event.kind = NG_NDLESS_EVENT_TEXT;
            event.text = shift_down != 0
                ? ng_text_bindings[index].shifted
                : ng_text_bindings[index].plain;
        }
        platform->text_down[index] = down != 0;
    }
    return event;
}

uint32_t ng_ndless_platform_now_ms(void) {
    struct timeval time_value;
    if (gettimeofday(&time_value, NULL) != 0 || time_value.tv_sec < 0) {
        return 0u;
    }
    return (uint32_t)time_value.tv_sec * 1000u;
}

void ng_ndless_platform_idle(void) {
    (void)msleep(10u);
}
