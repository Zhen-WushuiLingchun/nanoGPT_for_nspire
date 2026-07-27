#ifndef NG_RUNTIME_H
#define NG_RUNTIME_H

#include "ng_model.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ng_runtime {
    const ng_model *model;
    uint8_t *arena;
    size_t arena_bytes;
    size_t position;
    float *key_cache;
    float *value_cache;
    float *hidden;
    float *normalized;
    float *qkv;
    float *attention;
    float *projection;
    float *mlp;
    float *logits;
    float *scores;
    float *activation_scales;
    int8_t *quantized_activation;
} ng_runtime;

ng_status ng_runtime_init(
    ng_runtime *runtime,
    const ng_model *model,
    void *arena,
    size_t arena_bytes,
    ng_error *error);

void ng_runtime_reset(ng_runtime *runtime);

size_t ng_runtime_context_length(const ng_runtime *runtime);

ng_status ng_runtime_forward_token(
    ng_runtime *runtime,
    uint32_t token_id,
    const float **logits,
    ng_error *error);

#ifdef __cplusplus
}
#endif

#endif
