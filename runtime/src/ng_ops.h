#ifndef NG_OPS_H
#define NG_OPS_H

#include <stddef.h>
#include <stdint.h>

void ng_matvec_f32(
    float *output,
    const float *matrix,
    const float *input,
    size_t rows,
    size_t columns);

void ng_layer_norm_f32(
    float *output,
    const float *input,
    const float *weight,
    size_t width);

void ng_softmax_f32(
    float *output,
    const float *input,
    size_t length);

void ng_gelu_tanh_f32(float *values, size_t length);

void ng_add_in_place_f32(
    float *destination,
    const float *branch,
    size_t length);

void ng_quantize_activation_int8(
    int8_t *quantized,
    float *scales,
    const float *input,
    size_t columns,
    size_t group_size);

void ng_matvec_w4a8(
    float *output,
    const uint8_t *packed_weights,
    const float *weight_scales,
    const float *input,
    size_t rows,
    size_t columns,
    size_t padded_columns,
    size_t group_size,
    int8_t *quantized_activation,
    float *activation_scales);

#endif
