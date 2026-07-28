#ifndef NG_GFX_H
#define NG_GFX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NG_GFX_FONT_WIDTH 5
#define NG_GFX_FONT_HEIGHT 7
#define NG_GFX_GLYPH_ADVANCE 6
#define NG_GFX_LINE_HEIGHT 8

typedef struct ng_surface {
    uint16_t *pixels;
    int width;
    int height;
    int stride;
    int clip_left;
    int clip_top;
    int clip_right;
    int clip_bottom;
} ng_surface;

int ng_surface_init(
    ng_surface *surface,
    uint16_t *pixels,
    int width,
    int height,
    int stride);

uint16_t ng_rgb565(uint8_t red, uint8_t green, uint8_t blue);

void ng_gfx_reset_clip(ng_surface *surface);
void ng_gfx_set_clip(
    ng_surface *surface,
    int x,
    int y,
    int width,
    int height);
void ng_gfx_clear(ng_surface *surface, uint16_t color);
void ng_gfx_fill_rect(
    ng_surface *surface,
    int x,
    int y,
    int width,
    int height,
    uint16_t color);
void ng_gfx_stroke_rect(
    ng_surface *surface,
    int x,
    int y,
    int width,
    int height,
    uint16_t color);
void ng_gfx_draw_char(
    ng_surface *surface,
    int x,
    int y,
    char glyph,
    uint16_t color);
void ng_gfx_draw_text(
    ng_surface *surface,
    int x,
    int y,
    const char *text,
    uint16_t color);
void ng_gfx_draw_text_n(
    ng_surface *surface,
    int x,
    int y,
    const char *text,
    size_t length,
    uint16_t color);

#ifdef __cplusplus
}
#endif

#endif
