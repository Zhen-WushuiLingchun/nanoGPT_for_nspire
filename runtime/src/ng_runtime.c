#include "ng_runtime.h"

#include "ng_ops.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define NG_BLOCK_ATTENTION_NORM_SLOT 0u
#define NG_BLOCK_QKV_SLOT 1u
#define NG_BLOCK_ATTENTION_OUTPUT_SLOT 2u
#define NG_BLOCK_MLP_NORM_SLOT 3u
#define NG_BLOCK_MLP_INPUT_SLOT 4u
#define NG_BLOCK_MLP_OUTPUT_SLOT 5u

static void ng_runtime_error(ng_error *error, const char *message) {
    if (error != NULL) {
        (void)snprintf(
            error->message,
            (size_t)NG_ERROR_MESSAGE_BYTES,
            "%s",
            message);
    }
}

static int ng_platform_is_ieee754_little_endian(void) {
    const float one = 1.0f;
    const uint8_t *bytes = (const uint8_t *)&one;
    return sizeof(float) == 4u
        && bytes[0] == 0x00u
        && bytes[1] == 0x00u
        && bytes[2] == 0x80u
        && bytes[3] == 0x3fu;
}

static uint32_t ng_block_tensor_id(
    uint32_t block,
    uint32_t slot) {
    return (
        NG_TENSOR_BLOCK_BASE
        + block * NG_TENSOR_BLOCK_STRIDE
        + slot);
}

static const float *ng_fp32_tensor(
    const ng_model *model,
    uint32_t tensor_id) {
    const ng_tensor_view *view = ng_model_tensor(model, tensor_id);
    if (view == NULL || view->storage != (uint32_t)NG_STORAGE_FP32) {
        return NULL;
    }
    return (const float *)view->data;
}

static size_t ng_align_offset(size_t value) {
    return (
        value + (size_t)NG_ALIGNMENT_BYTES - 1u)
        & ~((size_t)NG_ALIGNMENT_BYTES - 1u);
}

static int ng_all_tensors_are_aligned(const ng_model *model) {
    uint32_t index;
    for (index = 0u; index < model->tensor_count; ++index) {
        const ng_tensor_view *view = &model->tensors[index];
        if (view->storage == (uint32_t)NG_STORAGE_FP32
            && ((uintptr_t)view->data % (uintptr_t)_Alignof(float)) != 0u) {
            return 0;
        }
        if (view->storage == (uint32_t)NG_STORAGE_INT4_GROUPWISE
            && ((uintptr_t)view->auxiliary
                % (uintptr_t)_Alignof(float)) != 0u) {
            return 0;
        }
    }
    return 1;
}

static void ng_runtime_assign_arena(ng_runtime *runtime) {
    const ng_model_spec *spec = &runtime->model->spec;
    size_t cache_values = (
        (size_t)spec->n_layer
        * (size_t)spec->block_size
        * (size_t)spec->n_embd);
    size_t mlp_width = (
        (size_t)spec->mlp_ratio
        * (size_t)spec->n_embd);
    float *cursor = (float *)runtime->arena;
    runtime->key_cache = cursor;
    cursor += cache_values;
    runtime->value_cache = cursor;
    cursor += cache_values;
    runtime->hidden = cursor;
    cursor += spec->n_embd;
    runtime->normalized = cursor;
    cursor += spec->n_embd;
    runtime->qkv = cursor;
    cursor += 3u * spec->n_embd;
    runtime->attention = cursor;
    cursor += spec->n_embd;
    runtime->projection = cursor;
    cursor += spec->n_embd;
    runtime->mlp = cursor;
    cursor += mlp_width;
    runtime->logits = cursor;
    cursor += spec->vocab_size;
    runtime->scores = cursor;
    cursor += spec->block_size;
    if (spec->model_storage == (uint32_t)NG_MODEL_STORAGE_W4A8) {
        size_t padded_mlp_width = (
            (mlp_width + (size_t)spec->activation_group_size - 1u)
            / (size_t)spec->activation_group_size
            * (size_t)spec->activation_group_size);
        size_t activation_scale_count = (
            padded_mlp_width
            / (size_t)spec->activation_group_size);
        size_t quantized_offset;
        runtime->activation_scales = cursor;
        cursor += activation_scale_count;
        quantized_offset = ng_align_offset(
            (size_t)((uint8_t *)cursor - runtime->arena));
        runtime->quantized_activation = (
            (int8_t *)(runtime->arena + quantized_offset));
    }
}

static void ng_runtime_zero_arena(ng_runtime *runtime) {
    volatile uint8_t *bytes = runtime->arena;
    size_t index;
    /*
     * Volatile stores keep the privacy wipe observable even when reset is
     * immediately followed by free() during application shutdown.
     */
    for (index = 0u; index < runtime->arena_bytes; ++index) {
        bytes[index] = 0u;
    }
}

ng_status ng_runtime_init(
    ng_runtime *runtime,
    const ng_model *model,
    void *arena,
    size_t arena_bytes,
    ng_error *error) {
    if (runtime == NULL || model == NULL || arena == NULL) {
        ng_runtime_error(error, "runtime, model and arena are required");
        return NG_STATUS_ARGUMENT;
    }
    (void)memset(runtime, 0, sizeof(*runtime));
    if (!ng_platform_is_ieee754_little_endian()) {
        ng_runtime_error(
            error,
            "runtime requires little-endian IEEE-754 float32");
        return NG_STATUS_FORMAT;
    }
    if (model->spec.model_storage != (uint32_t)NG_MODEL_STORAGE_FP32
        && model->spec.model_storage != (uint32_t)NG_MODEL_STORAGE_W4A8) {
        ng_runtime_error(
            error,
            "runtime model storage is unsupported");
        return NG_STATUS_FORMAT;
    }
    if (model->spec.model_storage == (uint32_t)NG_MODEL_STORAGE_W4A8
        && (model->spec.activation_quantization
                != (uint32_t)NG_ACTIVATION_DYNAMIC_INT8_GROUPWISE
            || model->spec.activation_group_size == 0u
            || model->spec.activation_group_size
                != model->spec.weight_group_size)) {
        ng_runtime_error(error, "W4A8 quantization metadata is invalid");
        return NG_STATUS_FORMAT;
    }
    if (((uintptr_t)arena % (uintptr_t)_Alignof(float)) != 0u
        || !ng_all_tensors_are_aligned(model)) {
        ng_runtime_error(error, "runtime float storage is misaligned");
        return NG_STATUS_FORMAT;
    }
    if (model->required_arena_bytes == 0u
        || arena_bytes < model->required_arena_bytes) {
        ng_runtime_error(error, "runtime arena is smaller than required");
        return NG_STATUS_MEMORY;
    }
    runtime->model = model;
    runtime->arena = (uint8_t *)arena;
    runtime->arena_bytes = model->required_arena_bytes;
    ng_runtime_assign_arena(runtime);
    ng_runtime_reset(runtime);
    ng_runtime_error(error, "");
    return NG_STATUS_OK;
}

void ng_runtime_reset(ng_runtime *runtime) {
    if (runtime == NULL || runtime->arena == NULL) {
        return;
    }
    ng_runtime_zero_arena(runtime);
    runtime->position = 0u;
}

size_t ng_runtime_context_length(const ng_runtime *runtime) {
    return runtime == NULL ? 0u : runtime->position;
}

static ng_status ng_attention(
    ng_runtime *runtime,
    uint32_t layer,
    ng_error *error) {
    const ng_model_spec *spec = &runtime->model->spec;
    size_t width = (size_t)spec->n_embd;
    size_t head_width = width / (size_t)spec->n_head;
    size_t sequence_length = runtime->position + 1u;
    size_t layer_cache = (
        (size_t)layer
        * (size_t)spec->block_size
        * width);
    size_t current_cache = layer_cache + runtime->position * width;
    float scale = 1.0f / sqrtf((float)head_width);
    uint32_t head;
    (void)memcpy(
        runtime->key_cache + current_cache,
        runtime->qkv + width,
        width * sizeof(float));
    (void)memcpy(
        runtime->value_cache + current_cache,
        runtime->qkv + 2u * width,
        width * sizeof(float));
    for (head = 0u; head < spec->n_head; ++head) {
        size_t head_offset = (size_t)head * head_width;
        size_t previous;
        size_t component;
        for (previous = 0u; previous < sequence_length; ++previous) {
            const float *key = (
                runtime->key_cache
                + layer_cache
                + previous * width
                + head_offset);
            const float *query = runtime->qkv + head_offset;
            float score = 0.0f;
            for (component = 0u; component < head_width; ++component) {
                score += query[component] * key[component];
            }
            runtime->scores[previous] = score * scale;
        }
        ng_softmax_f32(
            runtime->scores,
            runtime->scores,
            sequence_length);
        for (component = 0u; component < head_width; ++component) {
            float value = 0.0f;
            for (previous = 0u; previous < sequence_length; ++previous) {
                const float *cached_value = (
                    runtime->value_cache
                    + layer_cache
                    + previous * width
                    + head_offset);
                value += (
                    runtime->scores[previous]
                    * cached_value[component]);
            }
            runtime->attention[head_offset + component] = value;
        }
    }
    ng_runtime_error(error, "");
    return NG_STATUS_OK;
}

static int32_t ng_runtime_unpack_int4(
    const uint8_t *packed,
    size_t value_index) {
    uint8_t byte = packed[value_index / 2u];
    uint8_t nibble = (
        value_index % 2u == 0u
        ? byte & 0x0fu
        : byte >> 4u);
    return nibble >= 8u
        ? (int32_t)nibble - 16
        : (int32_t)nibble;
}

static ng_status ng_embedding_row(
    const ng_tensor_view *view,
    uint32_t row,
    float *output,
    ng_error *error) {
    size_t width = (size_t)view->shape[1];
    if (view->storage == (uint32_t)NG_STORAGE_FP32) {
        const float *values = (const float *)view->data;
        (void)memcpy(
            output,
            values + (size_t)row * width,
            width * sizeof(float));
        return NG_STATUS_OK;
    }
    if (view->storage == (uint32_t)NG_STORAGE_INT4_GROUPWISE) {
        const float *scales = (const float *)view->auxiliary;
        size_t padded_width = (size_t)view->padded_last_dim;
        size_t group_size = (size_t)view->group_size;
        size_t group_count = padded_width / group_size;
        size_t column;
        for (column = 0u; column < width; ++column) {
            size_t value_index = (size_t)row * padded_width + column;
            output[column] = (
                (float)ng_runtime_unpack_int4(view->data, value_index)
                * scales[
                    (size_t)row * group_count
                    + column / group_size]);
        }
        return NG_STATUS_OK;
    }
    ng_runtime_error(error, "embedding tensor storage is unsupported");
    return NG_STATUS_FORMAT;
}

static ng_status ng_runtime_matvec(
    ng_runtime *runtime,
    float *output,
    const ng_tensor_view *view,
    const float *input,
    ng_error *error) {
    size_t rows;
    size_t columns;
    if (view == NULL || view->rank != 2u) {
        ng_runtime_error(error, "matrix tensor is missing");
        return NG_STATUS_FORMAT;
    }
    rows = (size_t)view->shape[0];
    columns = (size_t)view->shape[1];
    if (view->storage == (uint32_t)NG_STORAGE_FP32) {
        ng_matvec_f32(
            output,
            (const float *)view->data,
            input,
            rows,
            columns);
        return NG_STATUS_OK;
    }
    if (view->storage == (uint32_t)NG_STORAGE_INT4_GROUPWISE
        && runtime->quantized_activation != NULL
        && runtime->activation_scales != NULL) {
        ng_matvec_w4a8(
            output,
            view->data,
            (const float *)view->auxiliary,
            input,
            rows,
            columns,
            (size_t)view->padded_last_dim,
            (size_t)view->group_size,
            runtime->quantized_activation,
            runtime->activation_scales);
        return NG_STATUS_OK;
    }
    ng_runtime_error(error, "matrix tensor storage is unsupported");
    return NG_STATUS_FORMAT;
}

ng_status ng_runtime_forward_token(
    ng_runtime *runtime,
    uint32_t token_id,
    const float **logits,
    ng_error *error) {
    const ng_model_spec *spec;
    const ng_tensor_view *token_embedding;
    const ng_tensor_view *position_embedding;
    const float *final_norm;
    size_t width;
    size_t mlp_width;
    size_t index;
    uint32_t layer;
    if (runtime == NULL || runtime->model == NULL || logits == NULL) {
        ng_runtime_error(error, "runtime and logits output are required");
        return NG_STATUS_ARGUMENT;
    }
    *logits = NULL;
    spec = &runtime->model->spec;
    if (token_id >= spec->vocab_size) {
        ng_runtime_error(error, "token ID is outside the vocabulary");
        return NG_STATUS_ARGUMENT;
    }
    if (runtime->position >= spec->block_size) {
        ng_runtime_error(error, "runtime context is full");
        return NG_STATUS_LIMIT;
    }
    width = (size_t)spec->n_embd;
    mlp_width = (size_t)spec->mlp_ratio * width;
    token_embedding = ng_model_tensor(
        runtime->model,
        NG_TENSOR_TOKEN_EMBEDDING);
    position_embedding = ng_model_tensor(
        runtime->model,
        NG_TENSOR_POSITION_EMBEDDING);
    final_norm = ng_fp32_tensor(
        runtime->model,
        NG_TENSOR_FINAL_NORM);
    if (token_embedding == NULL
        || position_embedding == NULL
        || final_norm == NULL) {
        ng_runtime_error(error, "required model tensor is missing");
        return NG_STATUS_FORMAT;
    }
    if (ng_embedding_row(
            token_embedding,
            token_id,
            runtime->hidden,
            error)
        != NG_STATUS_OK
        || ng_embedding_row(
            position_embedding,
            (uint32_t)runtime->position,
            runtime->normalized,
            error)
        != NG_STATUS_OK) {
        return NG_STATUS_FORMAT;
    }
    for (index = 0u; index < width; ++index) {
        runtime->hidden[index] += runtime->normalized[index];
    }
    for (layer = 0u; layer < spec->n_layer; ++layer) {
        const float *attention_norm = ng_fp32_tensor(
            runtime->model,
            ng_block_tensor_id(layer, NG_BLOCK_ATTENTION_NORM_SLOT));
        const ng_tensor_view *qkv_weight = ng_model_tensor(
            runtime->model,
            ng_block_tensor_id(layer, NG_BLOCK_QKV_SLOT));
        const ng_tensor_view *attention_output = ng_model_tensor(
            runtime->model,
            ng_block_tensor_id(
                layer,
                NG_BLOCK_ATTENTION_OUTPUT_SLOT));
        const float *mlp_norm = ng_fp32_tensor(
            runtime->model,
            ng_block_tensor_id(layer, NG_BLOCK_MLP_NORM_SLOT));
        const ng_tensor_view *mlp_input = ng_model_tensor(
            runtime->model,
            ng_block_tensor_id(layer, NG_BLOCK_MLP_INPUT_SLOT));
        const ng_tensor_view *mlp_output = ng_model_tensor(
            runtime->model,
            ng_block_tensor_id(layer, NG_BLOCK_MLP_OUTPUT_SLOT));
        if (attention_norm == NULL
            || qkv_weight == NULL
            || attention_output == NULL
            || mlp_norm == NULL
            || mlp_input == NULL
            || mlp_output == NULL) {
            ng_runtime_error(error, "required block tensor is missing");
            return NG_STATUS_FORMAT;
        }
        ng_layer_norm_f32(
            runtime->normalized,
            runtime->hidden,
            attention_norm,
            width);
        if (ng_runtime_matvec(
                runtime,
                runtime->qkv,
                qkv_weight,
                runtime->normalized,
                error)
            != NG_STATUS_OK) {
            return NG_STATUS_FORMAT;
        }
        if (ng_attention(runtime, layer, error) != NG_STATUS_OK) {
            return NG_STATUS_FORMAT;
        }
        if (ng_runtime_matvec(
                runtime,
                runtime->projection,
                attention_output,
                runtime->attention,
                error)
            != NG_STATUS_OK) {
            return NG_STATUS_FORMAT;
        }
        ng_add_in_place_f32(
            runtime->hidden,
            runtime->projection,
            width);
        ng_layer_norm_f32(
            runtime->normalized,
            runtime->hidden,
            mlp_norm,
            width);
        if (ng_runtime_matvec(
                runtime,
                runtime->mlp,
                mlp_input,
                runtime->normalized,
                error)
            != NG_STATUS_OK) {
            return NG_STATUS_FORMAT;
        }
        ng_gelu_tanh_f32(runtime->mlp, mlp_width);
        if (ng_runtime_matvec(
                runtime,
                runtime->projection,
                mlp_output,
                runtime->mlp,
                error)
            != NG_STATUS_OK) {
            return NG_STATUS_FORMAT;
        }
        ng_add_in_place_f32(
            runtime->hidden,
            runtime->projection,
            width);
    }
    ng_layer_norm_f32(
        runtime->normalized,
        runtime->hidden,
        final_norm,
        width);
    if (ng_runtime_matvec(
            runtime,
            runtime->logits,
            ng_model_tensor(
                runtime->model,
                NG_TENSOR_TOKEN_EMBEDDING),
            runtime->normalized,
            error)
        != NG_STATUS_OK) {
        return NG_STATUS_FORMAT;
    }
    runtime->position += 1u;
    *logits = runtime->logits;
    ng_runtime_error(error, "");
    return NG_STATUS_OK;
}
