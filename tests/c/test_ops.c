#include "ng_ops.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(expression)                                                      \
    do {                                                                       \
        if (!(expression)) {                                                   \
            fprintf(                                                           \
                stderr,                                                        \
                "CHECK failed at %s:%d: %s\n",                                \
                __FILE__,                                                      \
                __LINE__,                                                      \
                #expression);                                                  \
            failures += 1;                                                     \
        }                                                                      \
    } while (0)

static void check_close(float actual, float expected, float tolerance) {
    CHECK(fabsf(actual - expected) <= tolerance);
}

static void check_matvec(void) {
    static const float matrix[] = {
        1.0f, 2.0f, 3.0f,
        -1.0f, 0.5f, 2.0f
    };
    static const float input[] = {2.0f, -1.0f, 0.5f};
    float output[2] = {0.0f, 0.0f};
    ng_matvec_f32(output, matrix, input, 2u, 3u);
    check_close(output[0], 1.5f, 1.0e-6f);
    check_close(output[1], -1.5f, 1.0e-6f);
}

static void check_layer_norm(void) {
    static const float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    static const float weight[] = {1.0f, 0.5f, -1.0f, 2.0f};
    static const float expected[] = {
        -1.3416355f,
        -0.2236059f,
        -0.4472118f,
        2.6832710f
    };
    float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    size_t index;
    ng_layer_norm_f32(output, input, weight, 4u);
    for (index = 0u; index < 4u; ++index) {
        check_close(output[index], expected[index], 2.0e-6f);
    }
}

static void check_softmax(void) {
    static const float input[] = {1.0f, 2.0f, 3.0f};
    static const float expected[] = {
        0.09003057f,
        0.24472848f,
        0.66524094f
    };
    float output[3] = {0.0f, 0.0f, 0.0f};
    size_t index;
    ng_softmax_f32(output, input, 3u);
    for (index = 0u; index < 3u; ++index) {
        check_close(output[index], expected[index], 2.0e-6f);
    }
}

static void check_gelu(void) {
    static const float expected[] = {
        -0.15880801f,
        0.0f,
        0.84119199f,
        1.95459771f
    };
    float values[] = {-1.0f, 0.0f, 1.0f, 2.0f};
    size_t index;
    ng_gelu_tanh_f32(values, 4u);
    for (index = 0u; index < 4u; ++index) {
        check_close(values[index], expected[index], 2.0e-6f);
    }
}

static void check_residual(void) {
    float destination[] = {1.0f, -2.0f, 0.5f};
    static const float branch[] = {0.25f, 3.0f, -1.5f};
    ng_add_in_place_f32(destination, branch, 3u);
    check_close(destination[0], 1.25f, 1.0e-6f);
    check_close(destination[1], 1.0f, 1.0e-6f);
    check_close(destination[2], -1.0f, 1.0e-6f);
}

static void check_dynamic_int8_and_w4a8(void) {
    static const float input[] = {0.5f, -1.0f, 0.0f, 2.0f};
    static const uint8_t packed[] = {0xe1u, 0x03u, 0x79u, 0xf1u};
    static const float weight_scales[] = {
        0.5f, 0.25f,
        0.1f, 2.0f
    };
    int8_t quantized[4] = {0, 0, 0, 0};
    float activation_scales[2] = {0.0f, 0.0f};
    float output[2] = {0.0f, 0.0f};
    ng_quantize_activation_int8(
        quantized,
        activation_scales,
        input,
        4u,
        2u);
    CHECK(quantized[0] == 64);
    CHECK(quantized[1] == -127);
    CHECK(quantized[2] == 0);
    CHECK(quantized[3] == 127);
    check_close(activation_scales[0], 1.0f / 127.0f, 1.0e-8f);
    check_close(activation_scales[1], 2.0f / 127.0f, 1.0e-8f);
    ng_matvec_w4a8(
        output,
        packed,
        weight_scales,
        input,
        2u,
        4u,
        4u,
        2u,
        quantized,
        activation_scales);
    check_close(output[0], 159.0f / 127.0f, 2.0e-6f);
    check_close(
        output[1],
        -4.0f - 133.7f / 127.0f,
        2.0e-6f);
}

static void check_zero_activation_group(void) {
    static const float input[] = {0.0f, 0.0f};
    int8_t quantized[2] = {1, 1};
    float scale[1] = {0.0f};
    ng_quantize_activation_int8(
        quantized,
        scale,
        input,
        2u,
        2u);
    CHECK(quantized[0] == 0);
    CHECK(quantized[1] == 0);
    check_close(scale[0], 1.0f, 0.0f);
}

int main(void) {
    check_matvec();
    check_layer_norm();
    check_softmax();
    check_gelu();
    check_residual();
    check_dynamic_int8_and_w4a8();
    check_zero_activation_group();
    if (failures != 0) {
        fprintf(stderr, "%d scalar operator checks failed\n", failures);
        return 1;
    }
    puts("scalar operator checks passed");
    return 0;
}
