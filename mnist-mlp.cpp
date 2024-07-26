#include "gpu.h"
#include <vector>
#include <iostream>
#include <random>

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

fn softmax(x: array<f32, 10>) -> array<f32, 10> {
    var max_val = x[0];
    for (var i = 1u; i < 10u; i++) {
        max_val = max(max_val, x[i]);
    }
    
    var sum = 0.0;
    var result: array<f32, 10>;
    for (var i = 0u; i < 10u; i++) {
        result[i] = exp(x[i] - max_val);
        sum += result[i];
    }
    
    for (var i = 0u; i < 10u; i++) {
        result[i] /= sum;
    }
    
    return result;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let inputSize = 1024u;
    let hiddenSize = 128u;
    let outputSize = 10u;

    // Hidden layer (matmul -> relu)
    var hidden: array<f32, 128>;
    for (var i = 0u; i < hiddenSize; i++) {
        var sum = 0.0;
        for (var j = 0u; j < inputSize; j++) {
            sum += input[j] * weights1[i * inputSize + j];
        }
        hidden[i] = relu(sum + bias1[i]);
    }

    // Output layer (matmul -> softmax)
    var outputBefore: array<f32, 10>;
    for (var i = 0u; i < outputSize; i++) {
        var sum = 0.0;
        for (var j = 0u; j < hiddenSize; j++) {
            sum += hidden[j] * weights2[i * hiddenSize + j];
        }
        outputBefore[i] = sum + bias2[i];
    }
    
    // Apply softmax
    let outputAfter = softmax(outputBefore);
    for (var i = 0u; i < outputSize; i++) {
        output[i] = outputAfter[i];
    }
})";

int main(void) {
    Context ctx = createContext();

    const size_t l1_size = 784, l2_size = 128, l3_size = 10;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 0.1);

    std::vector<float> inputs(l1_size);
    for (auto& val : inputs) {
        val = dis(gen);
    }
    
    std::vector<float> weights1(l1_size * l2_size);
    std::vector<float> weights2(l2_size * l3_size);
    std::vector<float> bias1(l2_size);
    std::vector<float> bias2(l2_size);
    std::vector<float> outputs(l3_size, 0.0f);

    for (auto& val : weights1) val = dis(gen);
    for (auto& val : bias1) val = dis(gen);
    for (auto& val : weights2) val = dis(gen);
    for (auto& val : bias2) val = dis(gen);

    Tensor in = createTensor(ctx, Shape{l1_size}, kf32, inputs.data());

    Tensor w1 = createTensor(ctx, Shape{l1_size * l2_size}, kf32, weights1.data());
    Tensor w2 = createTensor(ctx, Shape{l2_size * l3_size}, kf32, weights2.data());

    Tensor b1 = createTensor(ctx, Shape{l2_size}, kf32, bias1.data());
    Tensor b2 = createTensor(ctx, Shape{l3_size}, kf32, bias2.data());

    Tensor out = createTensor(ctx, Shape{l3_size}, kf32, outputs.data());

    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    Kernel mlp = createKernel(ctx, {mlpKernelSrc, 1, kf32},
                              Bindings{in, w1, b1, w2, b2, out},
                              {1, 1, 1});

    dispatchKernel(ctx, mlp, promise);
    wait(ctx, future);

    toCPU(ctx, out, outputs.data(), sizeof(float) * l3_size);

    std::cout << "Output probabilities:\n";
    for (size_t i = 0; i < l3_size; ++i) {
        std::cout << "Class " << i << ": " << outputs[i] << "\n";
    }

    return 0;
}
