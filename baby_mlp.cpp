#include "gpu.h"
#include <vector>
#include <iostream>

using namespace gpu;

static const char *mlpKernelSrc = R"(
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weights1: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias1: array<f32>;
@group(0) @binding(3) var<storage, read_write> weights2: array<f32>;
@group(0) @binding(4) var<storage, read_write> bias2: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let inputSize = 2u;
    let hiddenSize = 3u;
    let outputSize = 2u;

    // Hidden layer
    var hidden: array<f32, 3>;
    for (var i = 0u; i < hiddenSize; i++) {
        var sum = 0.0;
        for (var j = 0u; j < inputSize; j++) {
            sum += input[j] * weights1[i * inputSize + j];
        }
        hidden[i] = relu(sum + bias1[i]);
    }

    // Output layer
    for (var i = 0u; i < outputSize; i++) {
        var sum = 0.0;
        for (var j = 0u; j < hiddenSize; j++) {
            sum += hidden[j] * weights2[i * hiddenSize + j];
        }
        output[i] = sum + bias2[i];
    }
}
)";

int main(void) {
    Context ctx = createContext();

    const size_t l1_size = 2, l2_size = 3, l3_size = 2;
    std::vector<float> inputs = { 11.0f, 7.0f };
    std::vector<float> weights1 = { 2.0f, 2.0f,
                                    4.0f, 2.0f,
                                    3.0f, 1.0f };
    std::vector<float> bias1 = { 9.0f, 2.0f, 3.0f };
    std::vector<float> weights2 = { 2.0f, 0.5f, 1.0f,
                                    3.0f, 3.0f, 2.0f };
    std::vector<float> bias2 = { 11.0f, 2.0f };

    std::vector<float> results = { 0.0f, 0.0f };

    Tensor input = createTensor(ctx, Shape{l1_size}, kf32, inputs.data());
    Tensor w1 = createTensor(ctx, Shape{l1_size * l2_size}, kf32, weights1.data());
    Tensor w2 = createTensor(ctx, Shape{l2_size * l3_size}, kf32, weights2.data());
    Tensor b1 = createTensor(ctx, Shape{l2_size}, kf32, bias1.data());
    Tensor b2 = createTensor(ctx, Shape{l3_size}, kf32, bias2.data());
    Tensor result_tensor = createTensor(ctx, Shape{l3_size}, kf32, results.data());

    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    Kernel mlp = createKernel(ctx, {mlpKernelSrc, 1, kf32},
                              Bindings{input, w1, b1, w2, b2, result_tensor},
                              {1, 1, 1});
    dispatchKernel(ctx, mlp, promise);
    wait(ctx, future);

    toCPU(ctx, result_tensor, results.data(), sizeof(float) * l3_size);

    printf("Results: %.2f, %.2f", results[0], results[1]);

    return 0;
}
