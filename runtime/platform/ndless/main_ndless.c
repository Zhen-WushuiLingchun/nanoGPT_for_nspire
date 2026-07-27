#include <libndls.h>

#include "ng_model.h"

int main(void) {
    static const uint8_t crc_probe[] = {1u, 2u, 3u, 4u};
    ng_model_spec spec;
    ng_model model;
    ng_error error;
    ng_status status;
    size_t arena_bytes;

    assert_ndless_rev(2022);

    spec.vocab_size = 65u;
    spec.block_size = 128u;
    spec.n_layer = 4u;
    spec.n_head = 5u;
    spec.n_embd = 160u;
    spec.mlp_ratio = 4u;
    spec.model_storage = (uint32_t)NG_MODEL_STORAGE_FP32;
    spec.weight_group_size = 0u;
    spec.activation_quantization = (uint32_t)NG_ACTIVATION_NONE;
    spec.activation_group_size = 0u;
    spec.tie_embeddings = 1u;
    spec.bias = 0u;
    arena_bytes = ng_model_estimate_arena_bytes(&spec);

    status = ng_model_load_memory(
        NULL,
        0u,
        (size_t)NG_INFERENCE_MEMORY_LIMIT_BYTES,
        &model,
        &error);
    if (ng_crc32(crc_probe, sizeof(crc_probe)) == 0u
        || arena_bytes == 0u
        || status != NG_STATUS_ARGUMENT) {
        show_msgbox(
            "nanoGPT runtime",
            "Portable runtime compile/link smoke failed.");
        return 1;
    }
    return 0;
}
