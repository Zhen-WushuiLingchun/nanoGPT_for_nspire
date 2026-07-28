#ifndef NG_CHAT_PLATFORM_NDLESS_H
#define NG_CHAT_PLATFORM_NDLESS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define NG_NDLESS_SCREEN_WIDTH 320
#define NG_NDLESS_SCREEN_HEIGHT 240
#define NG_NDLESS_SCREEN_BYTES \
    ((size_t)NG_NDLESS_SCREEN_WIDTH * (size_t)NG_NDLESS_SCREEN_HEIGHT \
     * sizeof(uint16_t))

typedef enum ng_ndless_event_kind {
    NG_NDLESS_EVENT_NONE = 0,
    NG_NDLESS_EVENT_TEXT = 1,
    NG_NDLESS_EVENT_SUBMIT = 2,
    NG_NDLESS_EVENT_BACKSPACE = 3,
    NG_NDLESS_EVENT_LEFT = 4,
    NG_NDLESS_EVENT_RIGHT = 5,
    NG_NDLESS_EVENT_SCROLL_UP = 6,
    NG_NDLESS_EVENT_SCROLL_DOWN = 7,
    NG_NDLESS_EVENT_NEW_CHAT = 8,
    NG_NDLESS_EVENT_CANCEL = 9,
    NG_NDLESS_EVENT_EXIT = 10
} ng_ndless_event_kind;

typedef struct ng_ndless_event {
    ng_ndless_event_kind kind;
    char text;
} ng_ndless_event;

typedef struct ng_ndless_platform {
    uint16_t *framebuffer;
    bool initialized;
    bool display_switched;
    bool special_down[9];
    bool text_down[47];
} ng_ndless_platform;

int ng_ndless_platform_init(ng_ndless_platform *platform);
void ng_ndless_platform_shutdown(ng_ndless_platform *platform);
void ng_ndless_platform_present(ng_ndless_platform *platform);
ng_ndless_event ng_ndless_platform_poll(ng_ndless_platform *platform);
uint32_t ng_ndless_platform_now_ms(void);
void ng_ndless_platform_idle(void);

#endif
