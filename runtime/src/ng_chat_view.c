#include "ng_chat_view.h"

#include <stddef.h>
#include <string.h>

#define NG_VIEW_HEADER_HEIGHT 23
#define NG_VIEW_TRANSCRIPT_TOP 24
#define NG_VIEW_TRANSCRIPT_BOTTOM 173
#define NG_VIEW_INPUT_TOP 174
#define NG_VIEW_INPUT_HEIGHT 37
#define NG_VIEW_FOOTER_TOP 212
#define NG_VIEW_CELL_X 7
#define NG_VIEW_CELL_WIDTH 306
#define NG_VIEW_CELL_TEXT_X 13
#define NG_VIEW_CELL_COLUMNS 48u

typedef struct ng_text_builder {
    char *bytes;
    size_t capacity;
    size_t length;
} ng_text_builder;

static void ng_builder_char(ng_text_builder *builder, char byte) {
    if (builder->length + 1u < builder->capacity) {
        builder->bytes[builder->length] = byte;
        builder->length += 1u;
        builder->bytes[builder->length] = '\0';
    }
}

static void ng_builder_text(ng_text_builder *builder, const char *text) {
    if (text == NULL) {
        return;
    }
    while (*text != '\0') {
        ng_builder_char(builder, *text);
        ++text;
    }
}

static void ng_builder_uint(ng_text_builder *builder, size_t value) {
    char reversed[24];
    size_t count = 0u;
    do {
        reversed[count] = (char)('0' + (value % 10u));
        count += 1u;
        value /= 10u;
    } while (value != 0u && count < sizeof(reversed));
    while (count > 0u) {
        count -= 1u;
        ng_builder_char(builder, reversed[count]);
    }
}

static void ng_builder_fixed_milli(
    ng_text_builder *builder,
    uint32_t milli_value) {
    ng_builder_uint(builder, (size_t)(milli_value / 1000u));
    ng_builder_char(builder, '.');
    ng_builder_char(
        builder,
        (char)('0' + (char)((milli_value % 1000u) / 100u)));
}

static void ng_builder_mib_tenths(
    ng_text_builder *builder,
    size_t bytes) {
    size_t tenths = (bytes * 10u + 524288u) / 1048576u;
    ng_builder_uint(builder, tenths / 10u);
    ng_builder_char(builder, '.');
    ng_builder_uint(builder, tenths % 10u);
}

static size_t ng_view_next_glyph(
    const char *text,
    size_t length,
    size_t index,
    char *glyph) {
    unsigned char byte;
    if (index >= length) {
        *glyph = '\0';
        return index;
    }
    byte = (unsigned char)text[index];
    if (byte < 0x80u) {
        *glyph = (char)byte;
        return index + 1u;
    }
    *glyph = '?';
    index += 1u;
    while (index < length
           && (((unsigned char)text[index] & 0xc0u) == 0x80u)) {
        index += 1u;
    }
    return index;
}

static size_t ng_view_line_count(
    const char *text,
    size_t length,
    size_t columns) {
    size_t lines = 1u;
    size_t column = 0u;
    size_t index = 0u;
    while (index < length) {
        char glyph;
        index = ng_view_next_glyph(text, length, index, &glyph);
        if (glyph == '\n') {
            lines += 1u;
            column = 0u;
        } else {
            if (column >= columns) {
                lines += 1u;
                column = 0u;
            }
            column += 1u;
        }
    }
    return lines;
}

static void ng_view_draw_wrapped(
    ng_surface *surface,
    int x,
    int y,
    const char *text,
    size_t length,
    size_t columns,
    uint16_t color) {
    size_t column = 0u;
    size_t index = 0u;
    int cursor_y = y;
    while (index < length) {
        char glyph;
        index = ng_view_next_glyph(text, length, index, &glyph);
        if (glyph == '\n') {
            cursor_y += NG_GFX_LINE_HEIGHT;
            column = 0u;
            continue;
        }
        if (column >= columns) {
            cursor_y += NG_GFX_LINE_HEIGHT;
            column = 0u;
        }
        ng_gfx_draw_char(
            surface,
            x + (int)(column * (size_t)NG_GFX_GLYPH_ADVANCE),
            cursor_y,
            glyph,
            color);
        column += 1u;
    }
}

static int ng_view_cell_height(const ng_chat *chat, size_t cell_index) {
    const ng_chat_cell *cell = &chat->cells[cell_index];
    size_t lines = ng_view_line_count(
        chat->transcript_text + cell->text_offset,
        cell->text_length,
        NG_VIEW_CELL_COLUMNS);
    return 19 + (int)(lines * (size_t)NG_GFX_LINE_HEIGHT);
}

static void ng_view_header(
    ng_surface *surface,
    const ng_chat *chat,
    const char *model_label) {
    char context[24] = {0};
    ng_text_builder builder = {context, sizeof(context), 0u};
    size_t block_size =
        chat->model == NULL ? 128u : (size_t)chat->model->spec.block_size;
    int label_x;

    ng_gfx_fill_rect(
        surface,
        1,
        1,
        NG_CHAT_VIEW_WIDTH - 2,
        NG_VIEW_HEADER_HEIGHT - 1,
        NG_CHAT_COLOR_PANEL_ALT);
    ng_gfx_fill_rect(surface, 1, 1, 3, 21, NG_CHAT_COLOR_MINT);
    ng_gfx_draw_text(
        surface,
        10,
        8,
        "NANOGPT // N-SPIRE",
        NG_CHAT_COLOR_TEXT);
    ng_builder_uint(&builder, chat->context_tokens);
    ng_builder_char(&builder, '/');
    ng_builder_uint(&builder, block_size);
    ng_gfx_draw_text(surface, 183, 8, context, NG_CHAT_COLOR_MUTED);
    if (model_label == NULL) {
        model_label = "MODEL";
    }
    label_x =
        NG_CHAT_VIEW_WIDTH - 8
        - (int)(strlen(model_label) * (size_t)NG_GFX_GLYPH_ADVANCE);
    if (label_x < 228) {
        label_x = 228;
    }
    ng_gfx_draw_text(
        surface,
        label_x,
        8,
        model_label,
        NG_CHAT_COLOR_AMBER);
}

static void ng_view_transcript(
    ng_surface *surface,
    const ng_chat *chat) {
    int total_height = 0;
    int y;
    size_t index;
    for (index = 0u; index < chat->cell_count; ++index) {
        total_height += ng_view_cell_height(chat, index) + 3;
    }
    y = NG_VIEW_TRANSCRIPT_BOTTOM - 4 - total_height
        + (int)(chat->scroll_line * (size_t)NG_GFX_LINE_HEIGHT);
    if (total_height < NG_VIEW_TRANSCRIPT_BOTTOM - NG_VIEW_TRANSCRIPT_TOP - 8) {
        y = NG_VIEW_TRANSCRIPT_TOP + 4;
    }
    ng_gfx_set_clip(
        surface,
        1,
        NG_VIEW_TRANSCRIPT_TOP,
        NG_CHAT_VIEW_WIDTH - 2,
        NG_VIEW_TRANSCRIPT_BOTTOM - NG_VIEW_TRANSCRIPT_TOP);
    if (chat->cell_count == 0u) {
        ng_gfx_stroke_rect(
            surface,
            34,
            55,
            252,
            75,
            NG_CHAT_COLOR_BORDER);
        ng_gfx_fill_rect(
            surface,
            34,
            55,
            3,
            75,
            NG_CHAT_COLOR_MINT);
        ng_gfx_draw_text(
            surface,
            50,
            68,
            "LOCAL MODEL READY",
            NG_CHAT_COLOR_MINT);
        ng_gfx_draw_text(
            surface,
            50,
            88,
            "ENTER SEND  MENU NEW CHAT",
            NG_CHAT_COLOR_TEXT);
        ng_gfx_draw_text(
            surface,
            50,
            100,
            "CTRL+ESC EXIT",
            NG_CHAT_COLOR_MUTED);
        ng_gfx_draw_text(
            surface,
            50,
            116,
            "NO CHAT HISTORY SAVED",
            NG_CHAT_COLOR_AMBER);
    }
    for (index = 0u; index < chat->cell_count; ++index) {
        const ng_chat_cell *cell = &chat->cells[index];
        const char *label;
        uint16_t accent;
        uint16_t fill;
        int height = ng_view_cell_height(chat, index);
        if (cell->role == NG_CHAT_ROLE_USER) {
            label = "USER";
            accent = NG_CHAT_COLOR_AMBER;
            fill = NG_CHAT_COLOR_PANEL_ALT;
        } else if (cell->role == NG_CHAT_ROLE_ASSISTANT) {
            label = "AI";
            accent = NG_CHAT_COLOR_MINT;
            fill = NG_CHAT_COLOR_PANEL;
        } else {
            label = "SYS";
            accent = NG_CHAT_COLOR_CORAL;
            fill = NG_CHAT_COLOR_PANEL;
        }
        ng_gfx_fill_rect(
            surface,
            NG_VIEW_CELL_X,
            y,
            NG_VIEW_CELL_WIDTH,
            height,
            fill);
        ng_gfx_stroke_rect(
            surface,
            NG_VIEW_CELL_X,
            y,
            NG_VIEW_CELL_WIDTH,
            height,
            NG_CHAT_COLOR_BORDER);
        ng_gfx_fill_rect(
            surface,
            NG_VIEW_CELL_X,
            y,
            3,
            height,
            accent);
        ng_gfx_draw_text(
            surface,
            NG_VIEW_CELL_TEXT_X,
            y + 5,
            label,
            accent);
        ng_gfx_fill_rect(
            surface,
            NG_VIEW_CELL_TEXT_X,
            y + 14,
            22,
            1,
            accent);
        ng_view_draw_wrapped(
            surface,
            NG_VIEW_CELL_TEXT_X,
            y + 18,
            chat->transcript_text + cell->text_offset,
            cell->text_length,
            NG_VIEW_CELL_COLUMNS,
            NG_CHAT_COLOR_TEXT);
        if (index + 1u == chat->cell_count
            && cell->role == NG_CHAT_ROLE_ASSISTANT
            && chat->phase == NG_CHAT_PHASE_GENERATING) {
            ng_gfx_fill_rect(
                surface,
                NG_VIEW_CELL_X + NG_VIEW_CELL_WIDTH - 10,
                y + height - 10,
                5,
                7,
                NG_CHAT_COLOR_MINT);
        }
        y += height + 3;
    }
    ng_gfx_reset_clip(surface);
}

static void ng_view_input(ng_surface *surface, const ng_chat *chat) {
    size_t visible_start = 0u;
    size_t visible_length;
    size_t visible_cursor;
    const size_t columns = 48u;

    ng_gfx_fill_rect(
        surface,
        1,
        NG_VIEW_INPUT_TOP,
        NG_CHAT_VIEW_WIDTH - 2,
        NG_VIEW_INPUT_HEIGHT,
        NG_CHAT_COLOR_PANEL_ALT);
    ng_gfx_stroke_rect(
        surface,
        6,
        NG_VIEW_INPUT_TOP + 5,
        NG_CHAT_VIEW_WIDTH - 12,
        26,
        chat->phase == NG_CHAT_PHASE_ERROR
            ? NG_CHAT_COLOR_CORAL
            : NG_CHAT_COLOR_BORDER);
    ng_gfx_draw_text(
        surface,
        12,
        NG_VIEW_INPUT_TOP + 14,
        ">",
        NG_CHAT_COLOR_AMBER);
    if (chat->input_cursor > columns) {
        visible_start = chat->input_cursor - columns;
    }
    visible_length = chat->input_length - visible_start;
    if (visible_length > columns) {
        visible_length = columns;
    }
    ng_gfx_draw_text_n(
        surface,
        24,
        NG_VIEW_INPUT_TOP + 14,
        chat->input + visible_start,
        visible_length,
        NG_CHAT_COLOR_TEXT);
    visible_cursor = chat->input_cursor - visible_start;
    ng_gfx_fill_rect(
        surface,
        24 + (int)(visible_cursor * (size_t)NG_GFX_GLYPH_ADVANCE),
        NG_VIEW_INPUT_TOP + 22,
        5,
        1,
        NG_CHAT_COLOR_AMBER);
}

static void ng_view_footer(ng_surface *surface, const ng_chat *chat) {
    char telemetry[64] = {0};
    ng_text_builder builder = {telemetry, sizeof(telemetry), 0u};

    if (chat->phase == NG_CHAT_PHASE_ERROR) {
        ng_builder_text(&builder, "ERR ");
        ng_builder_text(&builder, chat->error_message);
    } else {
        ng_builder_fixed_milli(
            &builder,
            chat->decode_milli_tokens_per_second);
        ng_builder_text(&builder, " T/S | TTFT:");
        ng_builder_uint(&builder, chat->ttft_ms);
        ng_builder_text(&builder, "MS | RAM:");
        ng_builder_mib_tenths(&builder, chat->tracked_peak_bytes);
        ng_builder_text(&builder, "M");
    }
    ng_gfx_fill_rect(
        surface,
        1,
        NG_VIEW_FOOTER_TOP,
        NG_CHAT_VIEW_WIDTH - 2,
        NG_CHAT_VIEW_HEIGHT - NG_VIEW_FOOTER_TOP - 1,
        NG_CHAT_COLOR_PANEL);
    ng_gfx_fill_rect(
        surface,
        1,
        NG_VIEW_FOOTER_TOP,
        NG_CHAT_VIEW_WIDTH - 2,
        1,
        chat->phase == NG_CHAT_PHASE_ERROR
            ? NG_CHAT_COLOR_CORAL
            : NG_CHAT_COLOR_MINT);
    ng_gfx_draw_text(
        surface,
        8,
        NG_VIEW_FOOTER_TOP + 10,
        telemetry,
        chat->phase == NG_CHAT_PHASE_ERROR
            ? NG_CHAT_COLOR_CORAL
            : NG_CHAT_COLOR_MUTED);
}

void ng_chat_view_render(
    ng_surface *surface,
    const ng_chat *chat,
    const char *model_label) {
    if (surface == NULL || chat == NULL || surface->width != NG_CHAT_VIEW_WIDTH
        || surface->height != NG_CHAT_VIEW_HEIGHT) {
        return;
    }
    ng_gfx_reset_clip(surface);
    ng_gfx_clear(surface, NG_CHAT_COLOR_BACKGROUND);
    ng_gfx_stroke_rect(
        surface,
        0,
        0,
        NG_CHAT_VIEW_WIDTH,
        NG_CHAT_VIEW_HEIGHT,
        NG_CHAT_COLOR_BORDER);
    ng_view_header(surface, chat, model_label);
    ng_view_transcript(surface, chat);
    ng_view_input(surface, chat);
    ng_view_footer(surface, chat);
}
