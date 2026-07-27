#include "ng_model.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_MEMORY_LIMIT (24u * 1024u * 1024u)

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

static uint32_t read_u32_le(const uint8_t *bytes) {
    return (uint32_t)bytes[0]
        | ((uint32_t)bytes[1] << 8u)
        | ((uint32_t)bytes[2] << 16u)
        | ((uint32_t)bytes[3] << 24u);
}

static void write_u32_le(uint8_t *bytes, uint32_t value) {
    bytes[0] = (uint8_t)(value & 0xffu);
    bytes[1] = (uint8_t)((value >> 8u) & 0xffu);
    bytes[2] = (uint8_t)((value >> 16u) & 0xffu);
    bytes[3] = (uint8_t)((value >> 24u) & 0xffu);
}

static void repair_checksums(uint8_t *bytes, size_t length) {
    uint32_t checksum;
    write_u32_le(
        bytes + NG_HEADER_PAYLOAD_CRC32_OFFSET,
        ng_crc32(bytes + NG_FILE_HEADER_BYTES, length - NG_FILE_HEADER_BYTES));
    write_u32_le(bytes + NG_HEADER_CRC32_OFFSET, 0u);
    checksum = ng_crc32(bytes, NG_FILE_HEADER_BYTES);
    write_u32_le(bytes + NG_HEADER_CRC32_OFFSET, checksum);
}

static uint8_t *read_file(const char *path, size_t *length_out) {
    FILE *stream = fopen(path, "rb");
    long end;
    uint8_t *bytes;
    if (stream == NULL) {
        return NULL;
    }
    if (fseek(stream, 0, SEEK_END) != 0) {
        fclose(stream);
        return NULL;
    }
    end = ftell(stream);
    if (end <= 0 || fseek(stream, 0, SEEK_SET) != 0) {
        fclose(stream);
        return NULL;
    }
    bytes = (uint8_t *)malloc((size_t)end);
    if (bytes == NULL) {
        fclose(stream);
        return NULL;
    }
    if (fread(bytes, 1u, (size_t)end, stream) != (size_t)end) {
        free(bytes);
        fclose(stream);
        return NULL;
    }
    fclose(stream);
    *length_out = (size_t)end;
    return bytes;
}

static void check_valid_file(const char *path) {
    ng_model model;
    ng_error error;
    const uint8_t *token;
    uint16_t token_bytes;
    const ng_tensor_view *tensor;
    ng_status status = ng_model_load_file(
        path,
        TEST_MEMORY_LIMIT,
        &model,
        &error);
    CHECK(status == NG_STATUS_OK);
    if (status != NG_STATUS_OK) {
        fprintf(stderr, "load error: %s\n", error.message);
        return;
    }
    CHECK(model.spec.vocab_size == 3u);
    CHECK(model.spec.block_size == 4u);
    CHECK(model.spec.n_layer == 1u);
    CHECK(model.spec.n_head == 2u);
    CHECK(model.spec.n_embd == 4u);
    CHECK(model.tensor_count == 9u);
    CHECK(model.required_arena_bytes > 0u);
    CHECK(model.file_bytes + model.required_arena_bytes <= TEST_MEMORY_LIMIT);
    CHECK(ng_model_token(&model, 2u, &token, &token_bytes) == NG_STATUS_OK);
    CHECK(token_bytes == 2u);
    CHECK(token[0] == 0xc3u && token[1] == 0xa9u);
    tensor = ng_model_tensor(&model, NG_TENSOR_TOKEN_EMBEDDING);
    CHECK(tensor != NULL);
    if (tensor != NULL) {
        CHECK(tensor->rank == 2u);
        CHECK(tensor->shape[0] == 3u);
        CHECK(tensor->shape[1] == 4u);
        CHECK(tensor->storage == NG_STORAGE_FP32);
        CHECK(tensor->data_bytes == 3u * 4u * sizeof(float));
    }
    ng_model_free(&model);
}

static void expect_memory_status(
    const uint8_t *bytes,
    size_t length,
    size_t memory_limit,
    ng_status expected) {
    ng_model model;
    ng_error error;
    ng_status status = ng_model_load_memory(
        bytes,
        length,
        memory_limit,
        &model,
        &error);
    CHECK(status == expected);
    if (status == NG_STATUS_OK) {
        ng_model_free(&model);
    }
}

static void check_malformed_files(const char *path) {
    size_t length = 0u;
    uint8_t *source = read_file(path, &length);
    uint8_t *copy;
    uint32_t first_data_offset;
    CHECK(source != NULL);
    if (source == NULL) {
        return;
    }

    expect_memory_status(
        source,
        length,
        TEST_MEMORY_LIMIT,
        NG_STATUS_OK);
    expect_memory_status(
        source,
        length - 1u,
        TEST_MEMORY_LIMIT,
        NG_STATUS_FORMAT);
    expect_memory_status(source, length, length, NG_STATUS_LIMIT);

    copy = (uint8_t *)malloc(length);
    CHECK(copy != NULL);
    if (copy == NULL) {
        free(source);
        return;
    }

    memcpy(copy, source, length);
    copy[0] = (uint8_t)'X';
    expect_memory_status(copy, length, TEST_MEMORY_LIMIT, NG_STATUS_FORMAT);

    memcpy(copy, source, length);
    copy[length - 1u] ^= 1u;
    expect_memory_status(copy, length, TEST_MEMORY_LIMIT, NG_STATUS_CHECKSUM);

    memcpy(copy, source, length);
    write_u32_le(
        copy + NG_FILE_HEADER_BYTES + NG_TENSOR_DATA_OFFSET_FIELD,
        (uint32_t)(length + NG_ALIGNMENT_BYTES));
    repair_checksums(copy, length);
    expect_memory_status(copy, length, TEST_MEMORY_LIMIT, NG_STATUS_FORMAT);

    memcpy(copy, source, length);
    write_u32_le(
        copy + NG_FILE_HEADER_BYTES + NG_TENSOR_ENTRY_BYTES,
        NG_TENSOR_TOKEN_EMBEDDING);
    repair_checksums(copy, length);
    expect_memory_status(copy, length, TEST_MEMORY_LIMIT, NG_STATUS_FORMAT);

    memcpy(copy, source, length);
    first_data_offset = read_u32_le(
        copy + NG_FILE_HEADER_BYTES + NG_TENSOR_DATA_OFFSET_FIELD);
    write_u32_le(
        copy
            + NG_FILE_HEADER_BYTES
            + NG_TENSOR_ENTRY_BYTES
            + NG_TENSOR_DATA_OFFSET_FIELD,
        first_data_offset);
    repair_checksums(copy, length);
    expect_memory_status(copy, length, TEST_MEMORY_LIMIT, NG_STATUS_FORMAT);

    free(copy);
    free(source);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: test_model_loader MODEL.ngm\n");
        return 2;
    }
    check_valid_file(argv[1]);
    check_malformed_files(argv[1]);
    if (failures != 0) {
        fprintf(stderr, "%d loader checks failed\n", failures);
        return 1;
    }
    printf("model loader checks passed\n");
    return 0;
}
