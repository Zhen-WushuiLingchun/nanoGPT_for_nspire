#include "ng_ops.h"

#include <math.h>

#define NG_LAYER_NORM_EPSILON 1.0e-5f
#define NG_GELU_SCALE 0.7978845608028654f
#define NG_GELU_CUBIC 0.044715f

void ng_matvec_f32(
    float *output,
    const float *matrix,
    const float *input,
    size_t rows,
    size_t columns) {
    size_t row;
    for (row = 0u; row < rows; ++row) {
        const float *weights = matrix + row * columns;
        float sum = 0.0f;
        size_t column;
        for (column = 0u; column < columns; ++column) {
            sum += weights[column] * input[column];
        }
        output[row] = sum;
    }
}

void ng_layer_norm_f32(
    float *output,
    const float *input,
    const float *weight,
    size_t width) {
    float mean = 0.0f;
    float variance = 0.0f;
    float inverse_standard_deviation;
    size_t index;
    for (index = 0u; index < width; ++index) {
        mean += input[index];
    }
    mean /= (float)width;
    for (index = 0u; index < width; ++index) {
        float centered = input[index] - mean;
        variance += centered * centered;
    }
    variance /= (float)width;
    inverse_standard_deviation = 1.0f / sqrtf(
        variance + NG_LAYER_NORM_EPSILON);
    for (index = 0u; index < width; ++index) {
        output[index] = (
            (input[index] - mean)
            * inverse_standard_deviation
            * weight[index]);
    }
}

void ng_softmax_f32(
    float *output,
    const float *input,
    size_t length) {
    float maximum = input[0];
    float denominator = 0.0f;
    size_t index;
    for (index = 1u; index < length; ++index) {
        if (input[index] > maximum) {
            maximum = input[index];
        }
    }
    for (index = 0u; index < length; ++index) {
        output[index] = expf(input[index] - maximum);
        denominator += output[index];
    }
    for (index = 0u; index < length; ++index) {
        output[index] /= denominator;
    }
}

void ng_gelu_tanh_f32(float *values, size_t length) {
    size_t index;
    for (index = 0u; index < length; ++index) {
        float value = values[index];
        float cubic = value * value * value;
        values[index] = 0.5f * value * (
            1.0f
            + tanhf(NG_GELU_SCALE * (value + NG_GELU_CUBIC * cubic)));
    }
}

void ng_add_in_place_f32(
    float *destination,
    const float *branch,
    size_t length) {
    size_t index;
    for (index = 0u; index < length; ++index) {
        destination[index] += branch[index];
    }
}
