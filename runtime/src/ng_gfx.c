#include "ng_gfx.h"

#include "ng_font.h"

#include <string.h>

static int ng_max_int(int left, int right) {
    return left > right ? left : right;
}

static int ng_min_int(int left, int right) {
    return left < right ? left : right;
}

int ng_surface_init(
    ng_surface *surface,
    uint16_t *pixels,
    int width,
    int height,
    int stride) {
    if (surface == NULL || pixels == NULL || width <= 0 || height <= 0
        || stride < width) {
        return 0;
    }
    surface->pixels = pixels;
    surface->width = width;
    surface->height = height;
    surface->stride = stride;
    ng_gfx_reset_clip(surface);
    return 1;
}

uint16_t ng_rgb565(uint8_t red, uint8_t green, uint8_t blue) {
    uint16_t red_bits = (uint16_t)((uint16_t)(red >> 3u) << 11u);
    uint16_t green_bits = (uint16_t)((uint16_t)(green >> 2u) << 5u);
    uint16_t blue_bits = (uint16_t)(blue >> 3u);
    return (uint16_t)(red_bits | green_bits | blue_bits);
}

void ng_gfx_reset_clip(ng_surface *surface) {
    if (surface == NULL) {
        return;
    }
    surface->clip_left = 0;
    surface->clip_top = 0;
    surface->clip_right = surface->width;
    surface->clip_bottom = surface->height;
}

void ng_gfx_set_clip(
    ng_surface *surface,
    int x,
    int y,
    int width,
    int height) {
    if (surface == NULL) {
        return;
    }
    if (width <= 0 || height <= 0) {
        surface->clip_left = 0;
        surface->clip_top = 0;
        surface->clip_right = 0;
        surface->clip_bottom = 0;
        return;
    }
    surface->clip_left = ng_max_int(0, x);
    surface->clip_top = ng_max_int(0, y);
    surface->clip_right = ng_min_int(surface->width, x + width);
    surface->clip_bottom = ng_min_int(surface->height, y + height);
    if (surface->clip_right < surface->clip_left) {
        surface->clip_right = surface->clip_left;
    }
    if (surface->clip_bottom < surface->clip_top) {
        surface->clip_bottom = surface->clip_top;
    }
}

void ng_gfx_clear(ng_surface *surface, uint16_t color) {
    int y;
    if (surface == NULL || surface->pixels == NULL) {
        return;
    }
    for (y = 0; y < surface->height; ++y) {
        int x;
        uint16_t *row = surface->pixels + (size_t)y * (size_t)surface->stride;
        for (x = 0; x < surface->width; ++x) {
            row[x] = color;
        }
    }
}

void ng_gfx_fill_rect(
    ng_surface *surface,
    int x,
    int y,
    int width,
    int height,
    uint16_t color) {
    int left;
    int top;
    int right;
    int bottom;
    int row_index;
    if (surface == NULL || surface->pixels == NULL || width <= 0
        || height <= 0) {
        return;
    }
    left = ng_max_int(x, surface->clip_left);
    top = ng_max_int(y, surface->clip_top);
    right = ng_min_int(x + width, surface->clip_right);
    bottom = ng_min_int(y + height, surface->clip_bottom);
    if (left >= right || top >= bottom) {
        return;
    }
    for (row_index = top; row_index < bottom; ++row_index) {
        int column;
        uint16_t *row =
            surface->pixels
            + (size_t)row_index * (size_t)surface->stride;
        for (column = left; column < right; ++column) {
            row[column] = color;
        }
    }
}

void ng_gfx_stroke_rect(
    ng_surface *surface,
    int x,
    int y,
    int width,
    int height,
    uint16_t color) {
    if (width <= 0 || height <= 0) {
        return;
    }
    ng_gfx_fill_rect(surface, x, y, width, 1, color);
    ng_gfx_fill_rect(surface, x, y + height - 1, width, 1, color);
    ng_gfx_fill_rect(surface, x, y, 1, height, color);
    ng_gfx_fill_rect(surface, x + width - 1, y, 1, height, color);
}

void ng_gfx_draw_char(
    ng_surface *surface,
    int x,
    int y,
    char glyph,
    uint16_t color) {
    unsigned row;
    for (row = 0u; row < 7u; ++row) {
        uint8_t bits = ng_font_row(glyph, row);
        unsigned column;
        for (column = 0u; column < 5u; ++column) {
            if ((bits & (uint8_t)(1u << (4u - column))) != 0u) {
                ng_gfx_fill_rect(
                    surface,
                    x + (int)column,
                    y + (int)row,
                    1,
                    1,
                    color);
            }
        }
    }
}

void ng_gfx_draw_text(
    ng_surface *surface,
    int x,
    int y,
    const char *text,
    uint16_t color) {
    if (text == NULL) {
        return;
    }
    ng_gfx_draw_text_n(surface, x, y, text, strlen(text), color);
}

void ng_gfx_draw_text_n(
    ng_surface *surface,
    int x,
    int y,
    const char *text,
    size_t length,
    uint16_t color) {
    int cursor_x = x;
    int cursor_y = y;
    size_t index = 0u;
    if (surface == NULL || text == NULL) {
        return;
    }
    while (index < length) {
        unsigned char byte = (unsigned char)text[index];
        char glyph;
        if (byte == (unsigned char)'\n') {
            cursor_x = x;
            cursor_y += NG_GFX_LINE_HEIGHT;
            index += 1u;
            continue;
        }
        if (byte < 0x80u) {
            glyph = (char)byte;
            index += 1u;
        } else {
            glyph = '?';
            index += 1u;
            while (index < length
                   && (((unsigned char)text[index] & 0xc0u) == 0x80u)) {
                index += 1u;
            }
        }
        ng_gfx_draw_char(surface, cursor_x, cursor_y, glyph, color);
        cursor_x += NG_GFX_GLYPH_ADVANCE;
    }
}
