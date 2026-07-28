#ifndef NG_FONT_H
#define NG_FONT_H

#include <stddef.h>
#include <stdint.h>

/*
 * Independently authored compact 5x7 terminal glyphs. Lowercase input maps to
 * the matching uppercase glyph so the first calculator build stays readable
 * without importing a third-party font or crossing the adjacent GPL boundary.
 */
typedef struct ng_font_glyph {
    char glyph;
    uint8_t rows[7];
} ng_font_glyph;

static uint8_t ng_font_row(char glyph, unsigned row) {
    static const ng_font_glyph glyphs[] = {
        {'A', {14u, 17u, 17u, 31u, 17u, 17u, 17u}},
        {'B', {30u, 17u, 17u, 30u, 17u, 17u, 30u}},
        {'C', {14u, 17u, 16u, 16u, 16u, 17u, 14u}},
        {'D', {30u, 17u, 17u, 17u, 17u, 17u, 30u}},
        {'E', {31u, 16u, 16u, 30u, 16u, 16u, 31u}},
        {'F', {31u, 16u, 16u, 30u, 16u, 16u, 16u}},
        {'G', {14u, 17u, 16u, 23u, 17u, 17u, 15u}},
        {'H', {17u, 17u, 17u, 31u, 17u, 17u, 17u}},
        {'I', {14u, 4u, 4u, 4u, 4u, 4u, 14u}},
        {'J', {7u, 2u, 2u, 2u, 18u, 18u, 12u}},
        {'K', {17u, 18u, 20u, 24u, 20u, 18u, 17u}},
        {'L', {16u, 16u, 16u, 16u, 16u, 16u, 31u}},
        {'M', {17u, 27u, 21u, 21u, 17u, 17u, 17u}},
        {'N', {17u, 25u, 21u, 19u, 17u, 17u, 17u}},
        {'O', {14u, 17u, 17u, 17u, 17u, 17u, 14u}},
        {'P', {30u, 17u, 17u, 30u, 16u, 16u, 16u}},
        {'Q', {14u, 17u, 17u, 17u, 21u, 18u, 13u}},
        {'R', {30u, 17u, 17u, 30u, 20u, 18u, 17u}},
        {'S', {15u, 16u, 16u, 14u, 1u, 1u, 30u}},
        {'T', {31u, 4u, 4u, 4u, 4u, 4u, 4u}},
        {'U', {17u, 17u, 17u, 17u, 17u, 17u, 14u}},
        {'V', {17u, 17u, 17u, 17u, 17u, 10u, 4u}},
        {'W', {17u, 17u, 17u, 21u, 21u, 21u, 10u}},
        {'X', {17u, 17u, 10u, 4u, 10u, 17u, 17u}},
        {'Y', {17u, 17u, 10u, 4u, 4u, 4u, 4u}},
        {'Z', {31u, 1u, 2u, 4u, 8u, 16u, 31u}},
        {'0', {14u, 17u, 19u, 21u, 25u, 17u, 14u}},
        {'1', {4u, 12u, 4u, 4u, 4u, 4u, 14u}},
        {'2', {14u, 17u, 1u, 2u, 4u, 8u, 31u}},
        {'3', {30u, 1u, 1u, 14u, 1u, 1u, 30u}},
        {'4', {2u, 6u, 10u, 18u, 31u, 2u, 2u}},
        {'5', {31u, 16u, 16u, 30u, 1u, 1u, 30u}},
        {'6', {14u, 16u, 16u, 30u, 17u, 17u, 14u}},
        {'7', {31u, 1u, 2u, 4u, 8u, 8u, 8u}},
        {'8', {14u, 17u, 17u, 14u, 17u, 17u, 14u}},
        {'9', {14u, 17u, 17u, 15u, 1u, 1u, 14u}},
        {'!', {4u, 4u, 4u, 4u, 4u, 0u, 4u}},
        {'"', {10u, 10u, 10u, 0u, 0u, 0u, 0u}},
        {'#', {10u, 31u, 10u, 10u, 31u, 10u, 0u}},
        {'$', {4u, 15u, 20u, 14u, 5u, 30u, 4u}},
        {'%', {25u, 26u, 2u, 4u, 8u, 11u, 19u}},
        {'&', {12u, 18u, 20u, 8u, 21u, 18u, 13u}},
        {'\'', {4u, 4u, 8u, 0u, 0u, 0u, 0u}},
        {'(', {2u, 4u, 8u, 8u, 8u, 4u, 2u}},
        {')', {8u, 4u, 2u, 2u, 2u, 4u, 8u}},
        {'*', {0u, 21u, 14u, 31u, 14u, 21u, 0u}},
        {'+', {0u, 4u, 4u, 31u, 4u, 4u, 0u}},
        {',', {0u, 0u, 0u, 0u, 4u, 4u, 8u}},
        {'-', {0u, 0u, 0u, 31u, 0u, 0u, 0u}},
        {'.', {0u, 0u, 0u, 0u, 0u, 4u, 4u}},
        {'/', {1u, 1u, 2u, 4u, 8u, 16u, 16u}},
        {':', {0u, 4u, 4u, 0u, 4u, 4u, 0u}},
        {';', {0u, 4u, 4u, 0u, 4u, 4u, 8u}},
        {'<', {2u, 4u, 8u, 16u, 8u, 4u, 2u}},
        {'=', {0u, 0u, 31u, 0u, 31u, 0u, 0u}},
        {'>', {8u, 4u, 2u, 1u, 2u, 4u, 8u}},
        {'?', {14u, 17u, 1u, 2u, 4u, 0u, 4u}},
        {'@', {14u, 17u, 23u, 21u, 23u, 16u, 14u}},
        {'[', {14u, 8u, 8u, 8u, 8u, 8u, 14u}},
        {'\\', {16u, 16u, 8u, 4u, 2u, 1u, 1u}},
        {']', {14u, 2u, 2u, 2u, 2u, 2u, 14u}},
        {'^', {4u, 10u, 17u, 0u, 0u, 0u, 0u}},
        {'_', {0u, 0u, 0u, 0u, 0u, 0u, 31u}},
        {'|', {4u, 4u, 4u, 4u, 4u, 4u, 4u}}
    };
    size_t index;
    if (row >= 7u || glyph == ' ') {
        return 0u;
    }
    if (glyph >= 'a' && glyph <= 'z') {
        glyph = (char)(glyph - ('a' - 'A'));
    }
    for (index = 0u; index < sizeof(glyphs) / sizeof(glyphs[0]); ++index) {
        if (glyphs[index].glyph == glyph) {
            return glyphs[index].rows[row];
        }
    }
    return row == 0u || row == 6u ? 31u : 17u;
}

#endif
