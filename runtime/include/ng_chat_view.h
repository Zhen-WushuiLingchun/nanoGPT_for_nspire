#ifndef NG_CHAT_VIEW_H
#define NG_CHAT_VIEW_H

#include "ng_chat.h"
#include "ng_gfx.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NG_CHAT_VIEW_WIDTH 320
#define NG_CHAT_VIEW_HEIGHT 240

#define NG_CHAT_COLOR_BACKGROUND ((uint16_t)0x0843u)
#define NG_CHAT_COLOR_PANEL ((uint16_t)0x10c5u)
#define NG_CHAT_COLOR_PANEL_ALT ((uint16_t)0x18e7u)
#define NG_CHAT_COLOR_BORDER ((uint16_t)0x31a8u)
#define NG_CHAT_COLOR_TEXT ((uint16_t)0xdedbu)
#define NG_CHAT_COLOR_MUTED ((uint16_t)0x7c10u)
#define NG_CHAT_COLOR_MINT ((uint16_t)0x5f79u)
#define NG_CHAT_COLOR_AMBER ((uint16_t)0xfd20u)
#define NG_CHAT_COLOR_CORAL ((uint16_t)0xfbceu)

void ng_chat_view_render(
    ng_surface *surface,
    const ng_chat *chat,
    const char *model_label);

#ifdef __cplusplus
}
#endif

#endif
