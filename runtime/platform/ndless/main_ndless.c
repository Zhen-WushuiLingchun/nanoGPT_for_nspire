#include <libndls.h>

#include "ng_model.h"
#include "ng_runtime.h"

#include <string.h>

int main(void) {
    static const uint8_t crc_probe[] = {1u, 2u, 3u, 4u};
    ng_model_spec spec;
    ng_model model;
    ng_runtime runtime;
    ng_error error;
    ng_status status;
    ng_status runtime_status;
    ng_status forward_status;
    size_t arena_bytes;
    const float *logits = NULL;
    union {
        float alignment;
        uint8_t bytes[512];
    } runtime_arena;

    assert_ndless_rev(2022);

    spec.vocab_size = 3u;
    spec.block_size = 4u;
    spec.n_layer = 1u;
    spec.n_head = 2u;
    spec.n_embd = 4u;
    spec.mlp_ratio = 2u;
    spec.model_storage = (uint32_t)NG_MODEL_STORAGE_FP32;
    spec.weight_group_size = 0u;
    spec.activation_quantization = (uint32_t)NG_ACTIVATION_NONE;
    spec.activation_group_size = 0u;
    spec.tie_embeddings = 1u;
    spec.bias = 0u;
    arena_bytes = ng_model_estimate_arena_bytes(&spec);
    (void)memset(&model, 0, sizeof(model));
    model.spec = spec;
    model.required_arena_bytes = arena_bytes;
    runtime_status = ng_runtime_init(
        &runtime,
        &model,
        runtime_arena.bytes,
        sizeof(runtime_arena.bytes),
        &error);
    forward_status = ng_runtime_forward_token(
        &runtime,
        0u,
        &logits,
        &error);
    ng_runtime_reset(&runtime);

    status = ng_model_load_memory(
        NULL,
        0u,
        (size_t)NG_INFERENCE_MEMORY_LIMIT_BYTES,
        &model,
        &error);
    if (ng_crc32(crc_probe, sizeof(crc_probe)) == 0u
        || arena_bytes == 0u
        || runtime_status != NG_STATUS_OK
        || forward_status != NG_STATUS_FORMAT
        || logits != NULL
        || status != NG_STATUS_ARGUMENT) {
        show_msgbox(
            "nanoGPT runtime",
            "Portable runtime compile/link smoke failed.");
        return 1;
    }
    return 0;
}
