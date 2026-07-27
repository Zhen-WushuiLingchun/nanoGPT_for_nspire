#ifndef NG_OPS_H
#define NG_OPS_H

#include <stddef.h>

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

#endif
