#include "ng_model.h"

#include <stdio.h>

int main(int argc, char **argv) {
    ng_model model;
    ng_error error;
    ng_status status;
    if (argc != 2) {
        fprintf(stderr, "usage: model_inspect MODEL.ngm\n");
        return 2;
    }
    status = ng_model_load_file(
        argv[1],
        (size_t)NG_INFERENCE_MEMORY_LIMIT_BYTES,
        &model,
        &error);
    if (status != NG_STATUS_OK) {
        fprintf(
            stderr,
            "load failed (%s): %s\n",
            ng_status_string(status),
            error.message);
        return 1;
    }
    printf("file_bytes=%zu\n", model.file_bytes);
    printf("arena_bytes=%zu\n", model.required_arena_bytes);
    printf(
        "total_static_bytes=%zu\n",
        model.file_bytes + model.required_arena_bytes);
    printf("vocab_size=%u\n", model.spec.vocab_size);
    printf("block_size=%u\n", model.spec.block_size);
    printf("n_layer=%u\n", model.spec.n_layer);
    printf("n_head=%u\n", model.spec.n_head);
    printf("n_embd=%u\n", model.spec.n_embd);
    printf("mlp_ratio=%u\n", model.spec.mlp_ratio);
    printf("model_storage=%u\n", model.spec.model_storage);
    printf("tensor_count=%u\n", model.tensor_count);
    printf("payload_crc32=%08x\n", model.payload_crc32);
    printf("header_crc32=%08x\n", model.header_crc32);
    ng_model_free(&model);
    return 0;
}
