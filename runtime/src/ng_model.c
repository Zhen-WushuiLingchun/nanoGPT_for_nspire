#include "ng_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NG_FORMAT_VERSION 1u
#define NG_ENDIAN_MARKER 0x01020304u
#define NG_FLAG_LITTLE_ENDIAN (1u << 0u)
#define NG_FLAG_TIED_EMBEDDING (1u << 1u)
#define NG_FLAG_BIAS (1u << 2u)
#define NG_FLAG_TANH_GELU (1u << 3u)
#define NG_KNOWN_FLAGS                                                       \
    (NG_FLAG_LITTLE_ENDIAN | NG_FLAG_TIED_EMBEDDING | NG_FLAG_BIAS           \
     | NG_FLAG_TANH_GELU)
#define NG_TOKENIZER_CHARACTER_UTF8 1u

static const uint8_t ng_magic[8] = {
    (uint8_t)'N',
    (uint8_t)'G',
    (uint8_t)'N',
    (uint8_t)'S',
    (uint8_t)'P',
    (uint8_t)'0',
    (uint8_t)'0',
    (uint8_t)'1'};

typedef struct ng_region {
    size_t begin;
    size_t end;
} ng_region;

static void ng_error_set(ng_error *error, const char *message) {
    if (error == NULL) {
        return;
    }
    if (message == NULL) {
        error->message[0] = '\0';
        return;
    }
    (void)snprintf(
        error->message,
        (size_t)NG_ERROR_MESSAGE_BYTES,
        "%s",
        message);
}

static uint32_t ng_read_u32(const uint8_t *bytes) {
    return (uint32_t)bytes[0]
        | ((uint32_t)bytes[1] << 8u)
        | ((uint32_t)bytes[2] << 16u)
        | ((uint32_t)bytes[3] << 24u);
}

static uint16_t ng_read_u16(const uint8_t *bytes) {
    return (uint16_t)((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8u));
}

static float ng_read_f32(const uint8_t *bytes) {
    uint32_t bits = ng_read_u32(bytes);
    float value = 0.0f;
    (void)memcpy(&value, &bits, sizeof(value));
    return value;
}

static int ng_checked_add(size_t left, size_t right, size_t *result) {
    if (result == NULL || right > SIZE_MAX - left) {
        return 0;
    }
    *result = left + right;
    return 1;
}

static int ng_checked_multiply(size_t left, size_t right, size_t *result) {
    if (result == NULL || (left != 0u && right > SIZE_MAX / left)) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static int ng_align_size(size_t value, size_t *result) {
    size_t added;
    if (!ng_checked_add(value, (size_t)NG_ALIGNMENT_BYTES - 1u, &added)) {
        return 0;
    }
    *result = added & ~((size_t)NG_ALIGNMENT_BYTES - 1u);
    return 1;
}

static int ng_add_arena_region(size_t *total, size_t bytes) {
    size_t aligned;
    size_t next;
    if (total == NULL || !ng_align_size(*total, &aligned)) {
        return 0;
    }
    if (!ng_checked_add(aligned, bytes, &next)) {
        return 0;
    }
    *total = next;
    return 1;
}

size_t ng_model_estimate_arena_bytes(const ng_model_spec *spec) {
    size_t total = 0u;
    size_t kv_values;
    size_t kv_bytes;
    size_t float_scratch;
    size_t float_scratch_bytes;
    size_t mlp_width;
    size_t activation_scale_count = 0u;
    if (spec == NULL) {
        return 0u;
    }
    if (!ng_checked_multiply(
            (size_t)spec->mlp_ratio,
            (size_t)spec->n_embd,
            &mlp_width)) {
        return 0u;
    }
    if (!ng_checked_multiply(
            (size_t)spec->n_layer,
            (size_t)spec->block_size,
            &kv_values)
        || !ng_checked_multiply(kv_values, (size_t)spec->n_embd, &kv_values)
        || !ng_checked_multiply(kv_values, 2u, &kv_values)
        || !ng_checked_multiply(kv_values, sizeof(float), &kv_bytes)) {
        return 0u;
    }
    /*
     * Shared float scratch: hidden + normalized + fused QKV + attention
     * context + projection = 7*C, followed by MLP, logits and one head's
     * attention scores. Heads are evaluated serially, so scores need T rather
     * than n_head*T values.
     */
    if (!ng_checked_multiply((size_t)spec->n_embd, 7u, &float_scratch)
        || !ng_checked_add(float_scratch, mlp_width, &float_scratch)
        || !ng_checked_add(
            float_scratch,
            (size_t)spec->vocab_size,
            &float_scratch)
        || !ng_checked_add(
            float_scratch,
            (size_t)spec->block_size,
            &float_scratch)) {
        return 0u;
    }
    if (spec->model_storage == (uint32_t)NG_MODEL_STORAGE_W4A8) {
        if (spec->activation_group_size == 0u) {
            return 0u;
        }
        activation_scale_count = (
            mlp_width + (size_t)spec->activation_group_size - 1u)
            / (size_t)spec->activation_group_size;
        if (!ng_checked_add(
                float_scratch,
                activation_scale_count,
                &float_scratch)) {
            return 0u;
        }
    }
    if (!ng_checked_multiply(
            float_scratch,
            sizeof(float),
            &float_scratch_bytes)) {
        return 0u;
    }
    if (!ng_add_arena_region(&total, kv_bytes)
        || !ng_add_arena_region(&total, float_scratch_bytes)) {
        return 0u;
    }
    if (spec->model_storage == (uint32_t)NG_MODEL_STORAGE_W4A8
        && !ng_add_arena_region(&total, mlp_width)) {
        return 0u;
    }
    if (!ng_align_size(total, &total)) {
        return 0u;
    }
    return total;
}

static int ng_region_valid(
    size_t offset,
    size_t bytes,
    size_t lower,
    size_t upper,
    ng_region *region) {
    size_t end;
    if (offset < lower || offset > upper) {
        return 0;
    }
    if (!ng_checked_add(offset, bytes, &end) || end > upper) {
        return 0;
    }
    if (region != NULL) {
        region->begin = offset;
        region->end = end;
    }
    return 1;
}

static int ng_regions_overlap(ng_region left, ng_region right) {
    return left.begin < right.end && right.begin < left.end;
}

static int ng_utf8_one_scalar(const uint8_t *bytes, uint16_t length) {
    if (bytes == NULL) {
        return 0;
    }
    if (length == 1u) {
        return bytes[0] <= 0x7fu;
    }
    if (length == 2u) {
        return bytes[0] >= 0xc2u && bytes[0] <= 0xdfu
            && bytes[1] >= 0x80u && bytes[1] <= 0xbfu;
    }
    if (length == 3u) {
        int continuation = bytes[1] >= 0x80u && bytes[1] <= 0xbfu
            && bytes[2] >= 0x80u && bytes[2] <= 0xbfu;
        if (!continuation || bytes[0] < 0xe0u || bytes[0] > 0xefu) {
            return 0;
        }
        if (bytes[0] == 0xe0u && bytes[1] < 0xa0u) {
            return 0;
        }
        if (bytes[0] == 0xedu && bytes[1] > 0x9fu) {
            return 0;
        }
        return 1;
    }
    if (length == 4u) {
        int continuation = bytes[1] >= 0x80u && bytes[1] <= 0xbfu
            && bytes[2] >= 0x80u && bytes[2] <= 0xbfu
            && bytes[3] >= 0x80u && bytes[3] <= 0xbfu;
        if (!continuation || bytes[0] < 0xf0u || bytes[0] > 0xf4u) {
            return 0;
        }
        if (bytes[0] == 0xf0u && bytes[1] < 0x90u) {
            return 0;
        }
        if (bytes[0] == 0xf4u && bytes[1] > 0x8fu) {
            return 0;
        }
        return 1;
    }
    return 0;
}

static int ng_expected_tensor(
    const ng_model_spec *spec,
    uint32_t index,
    uint32_t *tensor_id,
    uint32_t *rank,
    uint32_t shape[4]) {
    uint32_t final_index;
    if (spec == NULL || tensor_id == NULL || rank == NULL || shape == NULL) {
        return 0;
    }
    shape[0] = 0u;
    shape[1] = 0u;
    shape[2] = 0u;
    shape[3] = 0u;
    final_index = 2u + 6u * spec->n_layer;
    if (index == 0u) {
        *tensor_id = NG_TENSOR_TOKEN_EMBEDDING;
        *rank = 2u;
        shape[0] = spec->vocab_size;
        shape[1] = spec->n_embd;
        return 1;
    }
    if (index == 1u) {
        *tensor_id = NG_TENSOR_POSITION_EMBEDDING;
        *rank = 2u;
        shape[0] = spec->block_size;
        shape[1] = spec->n_embd;
        return 1;
    }
    if (index == final_index) {
        *tensor_id = NG_TENSOR_FINAL_NORM;
        *rank = 1u;
        shape[0] = spec->n_embd;
        return 1;
    }
    if (index > 1u && index < final_index) {
        uint32_t relative = index - 2u;
        uint32_t block = relative / 6u;
        uint32_t slot = relative % 6u;
        *tensor_id = NG_TENSOR_BLOCK_BASE
            + block * NG_TENSOR_BLOCK_STRIDE
            + slot;
        if (slot == 0u || slot == 3u) {
            *rank = 1u;
            shape[0] = spec->n_embd;
        } else {
            *rank = 2u;
            if (slot == 1u) {
                shape[0] = 3u * spec->n_embd;
                shape[1] = spec->n_embd;
            } else if (slot == 2u) {
                shape[0] = spec->n_embd;
                shape[1] = spec->n_embd;
            } else if (slot == 4u) {
                shape[0] = spec->mlp_ratio * spec->n_embd;
                shape[1] = spec->n_embd;
            } else {
                shape[0] = spec->n_embd;
                shape[1] = spec->mlp_ratio * spec->n_embd;
            }
        }
        return 1;
    }
    return 0;
}

static int ng_finite_fp32_values(const uint8_t *bytes, size_t length) {
    size_t offset;
    if (bytes == NULL || length % sizeof(float) != 0u) {
        return 0;
    }
    for (offset = 0u; offset < length; offset += sizeof(float)) {
        if (!isfinite(ng_read_f32(bytes + offset))) {
            return 0;
        }
    }
    return 1;
}

static int ng_positive_fp32_values(const uint8_t *bytes, size_t length) {
    size_t offset;
    if (!ng_finite_fp32_values(bytes, length)) {
        return 0;
    }
    for (offset = 0u; offset < length; offset += sizeof(float)) {
        if (ng_read_f32(bytes + offset) <= 0.0f) {
            return 0;
        }
    }
    return 1;
}

static int ng_int4_values_in_range(const uint8_t *bytes, size_t length) {
    size_t index;
    if (bytes == NULL) {
        return 0;
    }
    for (index = 0u; index < length; ++index) {
        if ((bytes[index] & 0x0fu) == 0x08u
            || ((bytes[index] >> 4u) & 0x0fu) == 0x08u) {
            return 0;
        }
    }
    return 1;
}

static ng_status ng_validate_header(
    const uint8_t *bytes,
    size_t length,
    ng_model *model,
    size_t *tensor_table_offset,
    size_t *vocabulary_offset,
    size_t *vocabulary_bytes,
    size_t *data_offset,
    ng_error *error) {
    uint8_t header_copy[NG_FILE_HEADER_BYTES];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t endian;
    uint32_t flags;
    uint32_t declared_file_bytes;
    uint32_t tensor_entry_bytes;
    uint32_t tensor_table_bytes;
    uint32_t data_bytes;
    uint32_t tokenizer_type;
    int32_t quantized_min;
    uint32_t quantized_max;
    size_t expected_table_bytes;
    size_t expected_vocabulary_offset;
    size_t vocabulary_end;
    size_t expected_data_offset;
    if (length < (size_t)NG_FILE_HEADER_BYTES) {
        ng_error_set(error, "file size is smaller than header");
        return NG_STATUS_FORMAT;
    }
    if (memcmp(bytes, ng_magic, sizeof(ng_magic)) != 0) {
        ng_error_set(error, "model magic does not match");
        return NG_STATUS_FORMAT;
    }
    version = ng_read_u32(bytes + 8u);
    header_bytes = ng_read_u32(bytes + 12u);
    endian = ng_read_u32(bytes + 16u);
    flags = ng_read_u32(bytes + 20u);
    declared_file_bytes = ng_read_u32(bytes + 24u);
    model->payload_crc32 = ng_read_u32(bytes + 28u);
    model->header_crc32 = ng_read_u32(bytes + 32u);
    if (version != NG_FORMAT_VERSION
        || header_bytes != NG_FILE_HEADER_BYTES
        || endian != NG_ENDIAN_MARKER) {
        ng_error_set(error, "unsupported header version or endian marker");
        return NG_STATUS_FORMAT;
    }
    if (length > (size_t)NG_FILE_LIMIT_BYTES
        || declared_file_bytes != (uint32_t)length) {
        ng_error_set(error, "declared file size is invalid");
        return NG_STATUS_FORMAT;
    }
    (void)memcpy(header_copy, bytes, sizeof(header_copy));
    (void)memset(header_copy + NG_HEADER_CRC32_OFFSET, 0, sizeof(uint32_t));
    if (ng_crc32(header_copy, sizeof(header_copy)) != model->header_crc32) {
        ng_error_set(error, "header CRC32 mismatch");
        return NG_STATUS_CHECKSUM;
    }
    if (ng_crc32(
            bytes + NG_FILE_HEADER_BYTES,
            length - (size_t)NG_FILE_HEADER_BYTES)
        != model->payload_crc32) {
        ng_error_set(error, "payload CRC32 mismatch");
        return NG_STATUS_CHECKSUM;
    }
    if ((flags & ~NG_KNOWN_FLAGS) != 0u
        || (flags & NG_FLAG_LITTLE_ENDIAN) == 0u
        || (flags & NG_FLAG_TANH_GELU) == 0u
        || (flags & NG_FLAG_TIED_EMBEDDING) == 0u
        || (flags & NG_FLAG_BIAS) != 0u) {
        ng_error_set(error, "unsupported model flags");
        return NG_STATUS_FORMAT;
    }

    model->tensor_count = ng_read_u32(bytes + 36u);
    tensor_entry_bytes = ng_read_u32(bytes + 40u);
    *tensor_table_offset = (size_t)ng_read_u32(bytes + 44u);
    tensor_table_bytes = ng_read_u32(bytes + 48u);
    *vocabulary_offset = (size_t)ng_read_u32(bytes + 52u);
    *vocabulary_bytes = (size_t)ng_read_u32(bytes + 56u);
    *data_offset = (size_t)ng_read_u32(bytes + 60u);
    data_bytes = ng_read_u32(bytes + 64u);
    model->spec.vocab_size = ng_read_u32(bytes + 68u);
    model->spec.block_size = ng_read_u32(bytes + 72u);
    model->spec.n_layer = ng_read_u32(bytes + 76u);
    model->spec.n_head = ng_read_u32(bytes + 80u);
    model->spec.n_embd = ng_read_u32(bytes + 84u);
    model->spec.mlp_ratio = ng_read_u32(bytes + 88u);
    model->spec.model_storage = ng_read_u32(bytes + 92u);
    model->spec.weight_group_size = ng_read_u32(bytes + 96u);
    quantized_min = (int32_t)ng_read_u32(bytes + 100u);
    quantized_max = ng_read_u32(bytes + 104u);
    tokenizer_type = ng_read_u32(bytes + 108u);
    model->spec.activation_quantization = ng_read_u32(bytes + 112u);
    model->spec.activation_group_size = ng_read_u32(bytes + 116u);
    model->spec.tie_embeddings = 1u;
    model->spec.bias = 0u;

    if (ng_read_u32(bytes + 120u) != 0u
        || ng_read_u32(bytes + 124u) != 0u) {
        ng_error_set(error, "reserved header fields are nonzero");
        return NG_STATUS_FORMAT;
    }
    if (model->tensor_count == 0u
        || model->tensor_count > NG_MAX_TENSORS
        || model->spec.vocab_size == 0u
        || model->spec.vocab_size > NG_MAX_VOCAB_SIZE
        || model->spec.block_size == 0u
        || model->spec.block_size > 128u
        || model->spec.n_layer == 0u
        || model->spec.n_head == 0u
        || model->spec.n_embd == 0u
        || model->spec.mlp_ratio == 0u
        || model->spec.n_embd % model->spec.n_head != 0u) {
        ng_error_set(error, "model dimensions are invalid");
        return NG_STATUS_FORMAT;
    }
    if (model->tensor_count != 2u + 6u * model->spec.n_layer + 1u) {
        ng_error_set(error, "tensor count does not match architecture");
        return NG_STATUS_FORMAT;
    }
    if (model->spec.model_storage == (uint32_t)NG_MODEL_STORAGE_FP32) {
        if (model->spec.weight_group_size != 0u
            || model->spec.activation_quantization
                != (uint32_t)NG_ACTIVATION_NONE
            || model->spec.activation_group_size != 0u
            || quantized_min != 0
            || quantized_max != 0u) {
            ng_error_set(error, "FP32 quantization metadata is invalid");
            return NG_STATUS_FORMAT;
        }
    } else if (
        model->spec.model_storage == (uint32_t)NG_MODEL_STORAGE_W4A8) {
        if (model->spec.weight_group_size == 0u
            || model->spec.weight_group_size % 2u != 0u
            || model->spec.activation_quantization
                != (uint32_t)NG_ACTIVATION_DYNAMIC_INT8_GROUPWISE
            || model->spec.activation_group_size
                != model->spec.weight_group_size
            || quantized_min != -7
            || quantized_max != 7u) {
            ng_error_set(error, "W4A8 quantization metadata is invalid");
            return NG_STATUS_FORMAT;
        }
    } else {
        ng_error_set(error, "model storage route is unsupported");
        return NG_STATUS_FORMAT;
    }
    if (tokenizer_type != NG_TOKENIZER_CHARACTER_UTF8) {
        ng_error_set(error, "tokenizer type is unsupported");
        return NG_STATUS_FORMAT;
    }
    if (tensor_entry_bytes != NG_TENSOR_ENTRY_BYTES
        || *tensor_table_offset != (size_t)NG_FILE_HEADER_BYTES
        || !ng_checked_multiply(
            (size_t)model->tensor_count,
            (size_t)NG_TENSOR_ENTRY_BYTES,
            &expected_table_bytes)
        || expected_table_bytes != (size_t)tensor_table_bytes
        || !ng_checked_add(
            *tensor_table_offset,
            expected_table_bytes,
            &expected_vocabulary_offset)
        || expected_vocabulary_offset != *vocabulary_offset
        || !ng_checked_add(
            *vocabulary_offset,
            *vocabulary_bytes,
            &vocabulary_end)
        || !ng_align_size(vocabulary_end, &expected_data_offset)
        || expected_data_offset != *data_offset
        || *data_offset % (size_t)NG_ALIGNMENT_BYTES != 0u
        || *data_offset > length
        || (size_t)data_bytes != length - *data_offset) {
        ng_error_set(error, "file section layout is invalid");
        return NG_STATUS_FORMAT;
    }
    return NG_STATUS_OK;
}

static ng_status ng_parse_vocabulary(
    const uint8_t *bytes,
    size_t vocabulary_offset,
    size_t vocabulary_bytes,
    ng_model *model,
    ng_error *error) {
    size_t cursor = vocabulary_offset;
    size_t end;
    uint32_t index;
    if (!ng_checked_add(vocabulary_offset, vocabulary_bytes, &end)) {
        ng_error_set(error, "vocabulary length overflows");
        return NG_STATUS_FORMAT;
    }
    for (index = 0u; index < model->spec.vocab_size; ++index) {
        uint16_t token_bytes;
        if (cursor > end || end - cursor < sizeof(uint16_t)) {
            ng_error_set(error, "vocabulary is truncated");
            return NG_STATUS_FORMAT;
        }
        token_bytes = ng_read_u16(bytes + cursor);
        cursor += sizeof(uint16_t);
        if (token_bytes == 0u
            || cursor > end
            || (size_t)token_bytes > end - cursor
            || !ng_utf8_one_scalar(bytes + cursor, token_bytes)) {
            ng_error_set(error, "vocabulary token is invalid UTF-8");
            return NG_STATUS_FORMAT;
        }
        model->vocabulary[index].bytes = bytes + cursor;
        model->vocabulary[index].length = token_bytes;
        cursor += (size_t)token_bytes;
    }
    if (cursor != end) {
        ng_error_set(error, "vocabulary has trailing bytes");
        return NG_STATUS_FORMAT;
    }
    return NG_STATUS_OK;
}

static ng_status ng_parse_tensors(
    const uint8_t *bytes,
    size_t length,
    size_t table_offset,
    size_t data_offset,
    ng_model *model,
    ng_error *error) {
    ng_region regions[2u * NG_MAX_TENSORS];
    size_t region_count = 0u;
    uint32_t index;
    for (index = 0u; index < model->tensor_count; ++index) {
        const uint8_t *entry = bytes
            + table_offset
            + (size_t)index * (size_t)NG_TENSOR_ENTRY_BYTES;
        ng_tensor_view *view = &model->tensors[index];
        uint32_t expected_id;
        uint32_t expected_rank;
        uint32_t expected_shape[4];
        uint32_t entry_flags = ng_read_u32(entry + 12u);
        uint32_t entry_reserved = ng_read_u32(entry + 60u);
        uint32_t element_count = ng_read_u32(entry + 56u);
        uint32_t data_offset_u32 = ng_read_u32(entry + 40u);
        uint32_t data_bytes_u32 = ng_read_u32(entry + 44u);
        uint32_t auxiliary_offset_u32 = ng_read_u32(entry + 48u);
        uint32_t auxiliary_bytes_u32 = ng_read_u32(entry + 52u);
        ng_region data_region;
        ng_region auxiliary_region = {0u, 0u};
        size_t expected_elements = 1u;
        size_t expected_data_bytes;
        size_t expected_auxiliary_bytes = 0u;
        size_t dimension_index;
        size_t prior;
        if (!ng_expected_tensor(
                &model->spec,
                index,
                &expected_id,
                &expected_rank,
                expected_shape)) {
            ng_error_set(error, "tensor index is not part of architecture");
            return NG_STATUS_FORMAT;
        }
        view->tensor_id = ng_read_u32(entry);
        view->storage = ng_read_u32(entry + 4u);
        view->rank = ng_read_u32(entry + 8u);
        view->shape[0] = ng_read_u32(entry + 16u);
        view->shape[1] = ng_read_u32(entry + 20u);
        view->shape[2] = ng_read_u32(entry + 24u);
        view->shape[3] = ng_read_u32(entry + 28u);
        view->group_size = ng_read_u32(entry + 32u);
        view->padded_last_dim = ng_read_u32(entry + 36u);
        if (entry_flags != 0u
            || entry_reserved != 0u
            || view->tensor_id != expected_id
            || view->rank != expected_rank) {
            ng_error_set(error, "tensor ID, rank or reserved fields are invalid");
            return NG_STATUS_FORMAT;
        }
        for (dimension_index = 0u; dimension_index < 4u; ++dimension_index) {
            if (view->shape[dimension_index]
                != expected_shape[dimension_index]) {
                ng_error_set(error, "tensor shape does not match architecture");
                return NG_STATUS_FORMAT;
            }
            if (dimension_index < (size_t)view->rank
                && (!ng_checked_multiply(
                    expected_elements,
                    (size_t)view->shape[dimension_index],
                    &expected_elements))) {
                ng_error_set(error, "tensor element count overflows");
                return NG_STATUS_FORMAT;
            }
        }
        if (expected_elements != (size_t)element_count) {
            ng_error_set(error, "tensor element count is invalid");
            return NG_STATUS_FORMAT;
        }
        if (!ng_region_valid(
                (size_t)data_offset_u32,
                (size_t)data_bytes_u32,
                data_offset,
                length,
                &data_region)
            || data_region.begin % (size_t)NG_ALIGNMENT_BYTES != 0u) {
            ng_error_set(error, "tensor data range is invalid");
            return NG_STATUS_FORMAT;
        }
        if (auxiliary_bytes_u32 != 0u) {
            if (!ng_region_valid(
                    (size_t)auxiliary_offset_u32,
                    (size_t)auxiliary_bytes_u32,
                    data_offset,
                    length,
                    &auxiliary_region)
                || auxiliary_region.begin % (size_t)NG_ALIGNMENT_BYTES != 0u) {
                ng_error_set(error, "tensor auxiliary range is invalid");
                return NG_STATUS_FORMAT;
            }
        } else if (auxiliary_offset_u32 != 0u) {
            ng_error_set(error, "empty auxiliary has nonzero offset");
            return NG_STATUS_FORMAT;
        }

        if (model->spec.model_storage == (uint32_t)NG_MODEL_STORAGE_FP32
            || view->rank == 1u) {
            if (view->storage != (uint32_t)NG_STORAGE_FP32
                || view->group_size != 0u
                || view->padded_last_dim != 0u
                || auxiliary_bytes_u32 != 0u
                || !ng_checked_multiply(
                    expected_elements,
                    sizeof(float),
                    &expected_data_bytes)
                || expected_data_bytes != (size_t)data_bytes_u32
                || !ng_finite_fp32_values(
                    bytes + data_region.begin,
                    data_region.end - data_region.begin)) {
                ng_error_set(error, "FP32 tensor metadata or values are invalid");
                return NG_STATUS_FORMAT;
            }
        } else {
            size_t rows = (size_t)view->shape[0];
            size_t padded_values;
            size_t scale_count;
            uint32_t expected_padded = (
                (
                    view->shape[1]
                    + model->spec.weight_group_size
                    - 1u)
                / model->spec.weight_group_size)
                * model->spec.weight_group_size;
            if (view->storage != (uint32_t)NG_STORAGE_INT4_GROUPWISE
                || view->rank != 2u
                || view->group_size != model->spec.weight_group_size
                || view->padded_last_dim != expected_padded
                || !ng_checked_multiply(
                    rows,
                    (size_t)view->padded_last_dim,
                    &padded_values)) {
                ng_error_set(error, "INT4 tensor metadata is invalid");
                return NG_STATUS_FORMAT;
            }
            expected_data_bytes = (padded_values + 1u) / 2u;
            scale_count = padded_values / (size_t)view->group_size;
            if (!ng_checked_multiply(
                    scale_count,
                    sizeof(float),
                    &expected_auxiliary_bytes)
                || expected_data_bytes != (size_t)data_bytes_u32
                || expected_auxiliary_bytes != (size_t)auxiliary_bytes_u32
                || !ng_int4_values_in_range(
                    bytes + data_region.begin,
                    data_region.end - data_region.begin)
                || !ng_positive_fp32_values(
                    bytes + auxiliary_region.begin,
                    auxiliary_region.end - auxiliary_region.begin)) {
                ng_error_set(error, "INT4 tensor payload or scales are invalid");
                return NG_STATUS_FORMAT;
            }
        }
        for (prior = 0u; prior < region_count; ++prior) {
            if (ng_regions_overlap(regions[prior], data_region)
                || (
                    auxiliary_bytes_u32 != 0u
                    && ng_regions_overlap(regions[prior], auxiliary_region))) {
                ng_error_set(error, "tensor payload regions overlap");
                return NG_STATUS_FORMAT;
            }
        }
        regions[region_count] = data_region;
        region_count += 1u;
        if (auxiliary_bytes_u32 != 0u) {
            regions[region_count] = auxiliary_region;
            region_count += 1u;
        }
        view->data = bytes + data_region.begin;
        view->data_bytes = data_region.end - data_region.begin;
        view->auxiliary = auxiliary_bytes_u32 == 0u
            ? NULL
            : bytes + auxiliary_region.begin;
        view->auxiliary_bytes = auxiliary_region.end - auxiliary_region.begin;
    }
    return NG_STATUS_OK;
}

ng_status ng_model_load_memory(
    const uint8_t *bytes,
    size_t length,
    size_t memory_limit_bytes,
    ng_model *model,
    ng_error *error) {
    ng_status status;
    size_t tensor_table_offset = 0u;
    size_t vocabulary_offset = 0u;
    size_t vocabulary_bytes = 0u;
    size_t data_offset = 0u;
    size_t total_memory;
    if (bytes == NULL || model == NULL) {
        ng_error_set(error, "bytes and model are required");
        return NG_STATUS_ARGUMENT;
    }
    (void)memset(model, 0, sizeof(*model));
    ng_error_set(error, "");
    status = ng_validate_header(
        bytes,
        length,
        model,
        &tensor_table_offset,
        &vocabulary_offset,
        &vocabulary_bytes,
        &data_offset,
        error);
    if (status != NG_STATUS_OK) {
        return status;
    }
    status = ng_parse_vocabulary(
        bytes,
        vocabulary_offset,
        vocabulary_bytes,
        model,
        error);
    if (status != NG_STATUS_OK) {
        return status;
    }
    status = ng_parse_tensors(
        bytes,
        length,
        tensor_table_offset,
        data_offset,
        model,
        error);
    if (status != NG_STATUS_OK) {
        return status;
    }
    model->required_arena_bytes = ng_model_estimate_arena_bytes(&model->spec);
    if (model->required_arena_bytes == 0u
        || !ng_checked_add(length, model->required_arena_bytes, &total_memory)) {
        ng_error_set(error, "inference arena size overflows");
        return NG_STATUS_FORMAT;
    }
    if (total_memory > memory_limit_bytes) {
        ng_error_set(error, "model blob plus inference arena exceeds limit");
        return NG_STATUS_LIMIT;
    }
    model->blob = bytes;
    model->file_bytes = length;
    return NG_STATUS_OK;
}

ng_status ng_model_load_file(
    const char *path,
    size_t memory_limit_bytes,
    ng_model *model,
    ng_error *error) {
    FILE *stream;
    long end;
    uint8_t *blob;
    ng_status status;
    if (path == NULL || model == NULL) {
        ng_error_set(error, "path and model are required");
        return NG_STATUS_ARGUMENT;
    }
    stream = fopen(path, "rb");
    if (stream == NULL) {
        ng_error_set(error, "could not open model file");
        return NG_STATUS_IO;
    }
    if (fseek(stream, 0, SEEK_END) != 0
        || (end = ftell(stream)) <= 0
        || fseek(stream, 0, SEEK_SET) != 0) {
        fclose(stream);
        ng_error_set(error, "could not determine model file size");
        return NG_STATUS_IO;
    }
    if ((unsigned long)end > (unsigned long)NG_FILE_LIMIT_BYTES) {
        fclose(stream);
        ng_error_set(error, "model file exceeds deployment limit");
        return NG_STATUS_LIMIT;
    }
    blob = (uint8_t *)malloc((size_t)end);
    if (blob == NULL) {
        fclose(stream);
        ng_error_set(error, "could not allocate model blob");
        return NG_STATUS_MEMORY;
    }
    if (fread(blob, 1u, (size_t)end, stream) != (size_t)end) {
        free(blob);
        fclose(stream);
        ng_error_set(error, "could not read complete model file");
        return NG_STATUS_IO;
    }
    fclose(stream);
    status = ng_model_load_memory(
        blob,
        (size_t)end,
        memory_limit_bytes,
        model,
        error);
    if (status != NG_STATUS_OK) {
        free(blob);
        return status;
    }
    model->owned_blob = blob;
    return NG_STATUS_OK;
}

void ng_model_free(ng_model *model) {
    if (model == NULL) {
        return;
    }
    free(model->owned_blob);
    (void)memset(model, 0, sizeof(*model));
}

const ng_tensor_view *ng_model_tensor(
    const ng_model *model,
    uint32_t tensor_id) {
    uint32_t index;
    if (model == NULL) {
        return NULL;
    }
    for (index = 0u; index < model->tensor_count; ++index) {
        if (model->tensors[index].tensor_id == tensor_id) {
            return &model->tensors[index];
        }
    }
    return NULL;
}

ng_status ng_model_token(
    const ng_model *model,
    uint32_t token_id,
    const uint8_t **bytes,
    uint16_t *length) {
    if (model == NULL
        || bytes == NULL
        || length == NULL
        || token_id >= model->spec.vocab_size) {
        return NG_STATUS_ARGUMENT;
    }
    *bytes = model->vocabulary[token_id].bytes;
    *length = model->vocabulary[token_id].length;
    return NG_STATUS_OK;
}

const char *ng_status_string(ng_status status) {
    switch (status) {
        case NG_STATUS_OK:
            return "ok";
        case NG_STATUS_ARGUMENT:
            return "argument";
        case NG_STATUS_IO:
            return "io";
        case NG_STATUS_MEMORY:
            return "memory";
        case NG_STATUS_LIMIT:
            return "limit";
        case NG_STATUS_FORMAT:
            return "format";
        case NG_STATUS_CHECKSUM:
            return "checksum";
        default:
            return "unknown";
    }
}
