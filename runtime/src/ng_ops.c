#include "ng_ops.h"

#include <math.h>
#include <stdint.h>

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

static int32_t ng_round_ties_to_even(float value) {
    float lower_float = floorf(value);
    float fraction = value - lower_float;
    int32_t lower = (int32_t)lower_float;
    if (fraction > 0.5f
        || (fraction == 0.5f && (lower % 2) != 0)) {
        return lower + 1;
    }
    return lower;
}

void ng_quantize_activation_int8(
    int8_t *quantized,
    float *scales,
    const float *input,
    size_t columns,
    size_t group_size) {
    size_t group_count = (
        columns + group_size - 1u) / group_size;
    size_t group;
    for (group = 0u; group < group_count; ++group) {
        size_t begin = group * group_size;
        size_t end = begin + group_size;
        float maximum = 0.0f;
        float scale;
        size_t column;
        if (end > columns) {
            end = columns;
        }
        for (column = begin; column < end; ++column) {
            float magnitude = fabsf(input[column]);
            if (magnitude > maximum) {
                maximum = magnitude;
            }
        }
        scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
        scales[group] = scale;
        for (column = begin; column < end; ++column) {
            int32_t rounded = ng_round_ties_to_even(
                input[column] / scale);
            if (rounded < -127) {
                rounded = -127;
            } else if (rounded > 127) {
                rounded = 127;
            }
            quantized[column] = (int8_t)rounded;
        }
        for (column = end; column < begin + group_size; ++column) {
            quantized[column] = 0;
        }
    }
}

static int32_t ng_unpack_int4(
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
    float *activation_scales) {
    size_t group_count = padded_columns / group_size;
    size_t row;
    ng_quantize_activation_int8(
        quantized_activation,
        activation_scales,
        input,
        columns,
        group_size);
    for (row = 0u; row < rows; ++row) {
        float sum = 0.0f;
        size_t group;
        for (group = 0u; group < group_count; ++group) {
            int32_t integer_dot = 0;
            size_t begin = group * group_size;
            size_t column;
            for (column = begin; column < begin + group_size; ++column) {
                size_t value_index = row * padded_columns + column;
                integer_dot += (
                    ng_unpack_int4(packed_weights, value_index)
                    * (int32_t)quantized_activation[column]);
            }
            sum += (
                (float)integer_dot
                * weight_scales[row * group_count + group]
                * activation_scales[group]);
        }
        output[row] = sum;
    }
}
