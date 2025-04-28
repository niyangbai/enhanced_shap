#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <functional>

namespace py = pybind11;

// Guided ReLU Backward Hook (for Guided Backpropagation)
std::function<torch::Tensor(torch::Tensor, torch::Tensor, torch::Tensor)> guided_relu_backward_hook() {
    return [](torch::Tensor module, torch::Tensor grad_input, torch::Tensor grad_output) -> torch::Tensor {
        // Apply clamp to grad_input and return the result
        return torch::clamp(grad_input, 0.0);
    };
}

// Normalize attention weights
torch::Tensor normalize_attention_weights(torch::Tensor attention, std::string norm_type = "l1") {
    if (norm_type == "l1") {
        torch::Tensor norm = attention.sum(-1, /*keepdim=*/true) + 1e-8;
        return attention / norm;
    } else if (norm_type == "l2") {
        torch::Tensor norm = torch::norm(attention, /*p=*/2, /*dim=*/-1, /*keepdim=*/true) + 1e-8;
        return attention / norm;
    } else {
        throw std::invalid_argument("Unsupported norm type: " + norm_type);
    }
}

// Compute cumulative attention flow across layers
torch::Tensor compute_attention_flow(std::vector<torch::Tensor> attentions) {
    torch::Tensor result = attentions[0];
    for (size_t i = 1; i < attentions.size(); ++i) {
        result = torch::bmm(attentions[i], result);
    }
    return result;
}

// Guided attention masking
torch::Tensor guided_attention_masking(torch::Tensor X, torch::Tensor attention_map, float threshold = 0.5) {
    // Create mask based on attention map
    torch::Tensor mask = (attention_map >= threshold).to(torch::kFloat).unsqueeze(-1);  // (batch, time, 1)
    return X * mask;
}

// Pybind11 module
PYBIND11_MODULE(attention, m) {
    m.def("guided_relu_backward_hook", &guided_relu_backward_hook, "Return a backward hook for Guided Backpropagation");
    m.def("normalize_attention_weights", &normalize_attention_weights, 
          "Normalize attention weights across dimensions (l1 or l2 norm)");
    m.def("compute_attention_flow", &compute_attention_flow, 
          "Compute cumulative attention flow across layers");
    m.def("guided_attention_masking", &guided_attention_masking, 
          "Mask input based on attention scores");
}
