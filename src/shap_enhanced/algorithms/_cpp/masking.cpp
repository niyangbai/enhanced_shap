#include <torch/torch.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Apply binary mask
torch::Tensor apply_binary_mask(torch::Tensor X, torch::Tensor mask, float mask_value = 0.0) {
    return X * mask + (1 - mask) * mask_value;
}

// Apply group-based masking
torch::Tensor apply_group_masking(torch::Tensor X, std::vector<std::vector<int>> groups, 
                                  torch::Tensor group_mask, float mask_value = 0.0) {
    torch::Tensor X_masked = X.clone();  // Create a copy of X to apply masking
    int64_t batch_size = X.size(0);  // Get batch size

    for (int64_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < groups.size(); ++j) {
            // Compare the group_mask element to 0 using .item<int>()
            if (group_mask[i][j].item<int>() == 0) {
                // Apply mask to the group indices
                for (int group_idx : groups[j]) {
                    X_masked[i][group_idx] = mask_value;
                }
            }
        }
    }

    return X_masked;
}

// Pybind11 bindings
PYBIND11_MODULE(masking, m) {
    m.def("apply_binary_mask", &apply_binary_mask, "Apply a binary mask to a tensor");
    m.def("apply_group_masking", &apply_group_masking, 
          "Apply group-based masking");
}
