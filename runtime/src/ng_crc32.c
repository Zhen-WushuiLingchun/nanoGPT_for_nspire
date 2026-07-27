#include "ng_model.h"

uint32_t ng_crc32(const uint8_t *data, size_t length) {
    uint32_t crc = 0xffffffffu;
    size_t index;
    if (data == NULL && length != 0u) {
        return 0u;
    }
    for (index = 0u; index < length; ++index) {
        uint32_t value = crc ^ (uint32_t)data[index];
        uint32_t bit;
        for (bit = 0u; bit < 8u; ++bit) {
            uint32_t mask = 0u - (value & 1u);
            value = (value >> 1u) ^ (0xedb88320u & mask);
        }
        crc = value;
    }
    return crc ^ 0xffffffffu;
}
