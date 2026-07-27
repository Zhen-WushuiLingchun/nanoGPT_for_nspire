#ifndef NG_MODEL_H
#define NG_MODEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NG_FILE_HEADER_BYTES 128u
#define NG_TENSOR_ENTRY_BYTES 64u
#define NG_ALIGNMENT_BYTES 64u
#define NG_FILE_LIMIT_BYTES (6u * 1024u * 1024u)
#define NG_INFERENCE_MEMORY_LIMIT_BYTES (24u * 1024u * 1024u)
#define NG_MAX_TENSORS 64u
#define NG_MAX_VOCAB_SIZE 256u
#define NG_ERROR_MESSAGE_BYTES 192u

#define NG_HEADER_PAYLOAD_CRC32_OFFSET 28u
#define NG_HEADER_CRC32_OFFSET 32u
#define NG_TENSOR_DATA_OFFSET_FIELD 40u

#define NG_TENSOR_TOKEN_EMBEDDING 1u
#define NG_TENSOR_POSITION_EMBEDDING 2u
#define NG_TENSOR_BLOCK_BASE 100u
#define NG_TENSOR_BLOCK_STRIDE 10u
#define NG_TENSOR_FINAL_NORM 1000u

typedef enum ng_status {
    NG_STATUS_OK = 0,
    NG_STATUS_ARGUMENT = 1,
    NG_STATUS_IO = 2,
    NG_STATUS_MEMORY = 3,
    NG_STATUS_LIMIT = 4,
    NG_STATUS_FORMAT = 5,
    NG_STATUS_CHECKSUM = 6
} ng_status;

typedef enum ng_model_storage {
    NG_MODEL_STORAGE_FP32 = 1,
    NG_MODEL_STORAGE_W4A8 = 2
} ng_model_storage;

typedef enum ng_tensor_storage {
    NG_STORAGE_FP32 = 1,
    NG_STORAGE_INT4_GROUPWISE = 2
} ng_tensor_storage;

typedef enum ng_activation_quantization {
    NG_ACTIVATION_NONE = 0,
    NG_ACTIVATION_DYNAMIC_INT8_GROUPWISE = 1
} ng_activation_quantization;

typedef struct ng_error {
    char message[NG_ERROR_MESSAGE_BYTES];
} ng_error;

typedef struct ng_model_spec {
    uint32_t vocab_size;
    uint32_t block_size;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_embd;
    uint32_t mlp_ratio;
    uint32_t model_storage;
    uint32_t weight_group_size;
    uint32_t activation_quantization;
    uint32_t activation_group_size;
    uint8_t tie_embeddings;
    uint8_t bias;
} ng_model_spec;

typedef struct ng_tensor_view {
    uint32_t tensor_id;
    uint32_t storage;
    uint32_t rank;
    uint32_t shape[4];
    uint32_t group_size;
    uint32_t padded_last_dim;
    const uint8_t *data;
    size_t data_bytes;
    const uint8_t *auxiliary;
    size_t auxiliary_bytes;
} ng_tensor_view;

typedef struct ng_vocab_token {
    const uint8_t *bytes;
    uint16_t length;
} ng_vocab_token;

typedef struct ng_model {
    const uint8_t *blob;
    uint8_t *owned_blob;
    size_t file_bytes;
    ng_model_spec spec;
    uint32_t tensor_count;
    ng_tensor_view tensors[NG_MAX_TENSORS];
    ng_vocab_token vocabulary[NG_MAX_VOCAB_SIZE];
    size_t required_arena_bytes;
    uint32_t payload_crc32;
    uint32_t header_crc32;
} ng_model;

uint32_t ng_crc32(const uint8_t *data, size_t length);

size_t ng_model_estimate_arena_bytes(const ng_model_spec *spec);

ng_status ng_model_load_memory(
    const uint8_t *bytes,
    size_t length,
    size_t memory_limit_bytes,
    ng_model *model,
    ng_error *error);

ng_status ng_model_load_file(
    const char *path,
    size_t memory_limit_bytes,
    ng_model *model,
    ng_error *error);

void ng_model_free(ng_model *model);

const ng_tensor_view *ng_model_tensor(
    const ng_model *model,
    uint32_t tensor_id);

ng_status ng_model_token(
    const ng_model *model,
    uint32_t token_id,
    const uint8_t **bytes,
    uint16_t *length);

const char *ng_status_string(ng_status status);

#ifdef __cplusplus
}
#endif

#endif
