#include <torch/extension.h>
#include <pybind11/functional.h>
#include <cmath>
#include <random>

namespace py = pybind11;

// Default target function: sum of features
at::Tensor default_target_function(const at::Tensor& X) {
    return X.sum(1);  // Sum over features (axis=1)
}

// Main function for generating sparse data
std::tuple<at::Tensor, at::Tensor> generate_sparse_data(
    int n_samples,
    int n_features,
    double sparsity,
    double noise_std,
    std::function<at::Tensor(const at::Tensor&)> target_function,
    int random_seed
) {
    if (sparsity < 0.0 || sparsity > 1.0) {
        throw std::invalid_argument("Sparsity must be between 0 and 1.");
    }

    // Set the random seed for reproducibility
    std::mt19937 gen(random_seed);
    std::normal_distribution<float> dist(0.0, 1.0);  // Normal distribution for noise

    // Generate random data (normally distributed)
    at::Tensor X = torch::empty({n_samples, n_features}, torch::kFloat32);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X[i][j] = dist(gen);  // Fill with random numbers
        }
    }

    // Introduce sparsity (set random elements to 0)
    std::uniform_real_distribution<float> mask_dist(0.0, 1.0);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            if (mask_dist(gen) < sparsity) {
                X[i][j] = 0.0f;  // Set element to 0 based on sparsity
            }
        }
    }

    // Generate target values using the provided target function
    at::Tensor y = target_function(X);

    // Add noise to the target if necessary
    if (noise_std > 0) {
        for (int i = 0; i < n_samples; ++i) {
            y[i] += dist(gen) * noise_std;  // Add Gaussian noise
        }
    }

    return std::make_tuple(X, y);
}

PYBIND11_MODULE(synthetic_sparse, m) {
    // Overload `generate_sparse_data` to handle default functions using lambda
    m.def("generate_sparse_data", 
          [](int n_samples, int n_features, double sparsity, double noise_std, 
             std::function<at::Tensor(const at::Tensor&)> target_function, int random_seed) {
                return generate_sparse_data(n_samples, n_features, sparsity, noise_std, target_function, random_seed);
          },
          py::arg("n_samples"), py::arg("n_features"), py::arg("sparsity"),
          py::arg("noise_std") = 0.0, py::arg("target_function") = [](const at::Tensor& X) { return default_target_function(X); },
          py::arg("random_seed") = 42);

    // Overload the function with no pattern_function and target_function
    m.def("generate_sparse_data", 
          [](int n_samples, int n_features, double sparsity, double noise_std, int random_seed) {
              return generate_sparse_data(n_samples, n_features, sparsity, noise_std, default_target_function, random_seed);
          },
          py::arg("n_samples"), py::arg("n_features"), py::arg("sparsity"),
          py::arg("noise_std") = 0.0, py::arg("random_seed") = 42);
}
