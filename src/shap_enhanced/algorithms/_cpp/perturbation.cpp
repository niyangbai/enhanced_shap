#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>

namespace py = pybind11;

// Mask specified features
torch::Tensor mask_features(torch::Tensor X, std::vector<int> features, float mask_value = 0.0) {
    torch::Tensor X_masked = X.clone();
    for (int f : features) {
        X_masked.index({torch::indexing::Slice(), f}) = mask_value;
    }
    return X_masked;
}

// Mask specified timesteps
torch::Tensor mask_timesteps(const torch::Tensor& X, const std::vector<int>& timesteps, float mask_value = 0.0) {
    torch::Tensor X_masked = X.clone();
    for (int t : timesteps) {
        X_masked.index({torch::indexing::Slice(), t, torch::indexing::Slice()}) = mask_value;
    }
    return X_masked;
}

// Randomly mask features
torch::Tensor random_mask(torch::Tensor X, float prob = 0.5, float mask_value = 0.0) {
    torch::Tensor mask = (torch::rand_like(X) > prob).to(torch::kFloat);
    return X * mask + (1 - mask) * mask_value;
}

// Mask a continuous window of timesteps
torch::Tensor mask_time_window(const torch::Tensor& X, int center, int window_size, float mask_value) {
    // Clone input tensor
    torch::Tensor X_masked = X.clone();

    // Calculate window bounds
    int time_dim = X.size(1);
    int half_window = window_size / 2;
    int start = std::max(center - half_window, 0);
    int end = std::min(center + half_window + 1, time_dim); // exclusive

    // Fill in the mask
    X_masked.index({torch::indexing::Slice(), torch::indexing::Slice(start, end), torch::indexing::Slice()}) = mask_value;

    return X_masked;
}

// Add structured noise to a sequence
torch::Tensor perturb_sequence_with_noise(torch::Tensor X, float noise_level = 0.1) {
    torch::Tensor noise = torch::randn_like(X) * noise_level;
    return X + noise;
}

// Randomly mask n features for each sample
torch::Tensor mask_random_features(torch::Tensor X, int n_features_to_mask, float mask_value = 0.0) {
    torch::Tensor X_masked = X.clone();
    int64_t batch_size = X.size(0);
    int64_t n_features = X.size(1);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n_features - 1);

    for (int64_t i = 0; i < batch_size; ++i) {
        std::set<int> features_to_mask;
        while (features_to_mask.size() < n_features_to_mask) {
            features_to_mask.insert(dis(gen));
        }

        for (int f : features_to_mask) {
            X_masked.index({i, f}) = mask_value;
        }
    }

    return X_masked;
}

// Mask specific groups of features
torch::Tensor mask_feature_groups(torch::Tensor X, std::vector<std::vector<int>> groups, 
                                  std::vector<int> group_indices, float mask_value = 0.0) {
    std::vector<int> features_to_mask;
    for (int idx : group_indices) {
        for (int f : groups[idx]) {
            features_to_mask.push_back(f);
        }
    }
    return mask_features(X, features_to_mask, mask_value);
}

// Pybind11 bindings
PYBIND11_MODULE(perturbation, m) {
    m.def("mask_features", &mask_features, "Mask specified features");
    m.def("mask_timesteps", &mask_timesteps, "Mask specific timesteps in a tensor");
    m.def("random_mask", &random_mask, "Randomly mask features");
    m.def("mask_time_window", &mask_time_window, "Mask a continuous window of timesteps");
    m.def("perturb_sequence_with_noise", &perturb_sequence_with_noise, "Add structured noise to a sequence");
    m.def("mask_random_features", &mask_random_features, "Randomly mask n features for each sample");
    m.def("mask_feature_groups", &mask_feature_groups, "Mask specific groups of features");
}
