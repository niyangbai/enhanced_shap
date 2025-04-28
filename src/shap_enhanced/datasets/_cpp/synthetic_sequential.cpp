#include <torch/extension.h>
#include <pybind11/functional.h>
#include <cmath>
#include <random>

namespace py = pybind11;

// Default pattern function: sinusoidal pattern
at::Tensor default_pattern_function(int n_timesteps, int n_features, int sample_idx) {
    auto result = torch::empty({n_features}, torch::kFloat32);
    for (int f = 0; f < n_features; ++f) {
        result[f] = std::sin(10.0 * f / n_features);
    }
    return result;
}

// Default target function: sum over timesteps and features
at::Tensor default_target_function(const at::Tensor& data) {
    return data.sum({1, 2}); // Sum over timesteps and features, keep batch
}

// Main function
std::tuple<at::Tensor, at::Tensor> generate_sequential_data(
    int n_samples,
    int n_timesteps,
    int n_features,
    std::function<at::Tensor(int, int, int)> pattern_function,
    double noise_std,
    std::function<at::Tensor(const at::Tensor&)> target_function
) {
    auto X = torch::zeros({n_samples, n_timesteps, n_features}, torch::kFloat32);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0, static_cast<float>(noise_std));

    for (int i = 0; i < n_samples; ++i) {
        at::Tensor pattern = pattern_function(n_timesteps, n_features, i);
        for (int t = 0; t < n_timesteps; ++t) {
            for (int f = 0; f < n_features; ++f) {
                X[i][t][f] = pattern[f] + dist(gen);
            }
        }
    }

    at::Tensor y = target_function(X);

    return std::make_tuple(X, y);
}

PYBIND11_MODULE(synthetic_sequential, m) {
    // Overload `generate_sequential_data` to handle default functions using lambda
    m.def("generate_sequential_data", 
          [](int n_samples, int n_timesteps, int n_features,
             std::function<at::Tensor(int, int, int)> pattern_function,
             double noise_std,
             std::function<at::Tensor(const at::Tensor&)> target_function) {
                return generate_sequential_data(n_samples, n_timesteps, n_features, pattern_function, noise_std, target_function);
          },
          py::arg("n_samples"), py::arg("n_timesteps"), py::arg("n_features"),
          py::arg("pattern_function") = [](int n_timesteps, int n_features, int sample_idx) { 
              return default_pattern_function(n_timesteps, n_features, sample_idx); 
          },
          py::arg("noise_std") = 0.1,
          py::arg("target_function") = [](const at::Tensor& data) { 
              return default_target_function(data); 
          });

    // Overload the function with no pattern_function and target_function
    m.def("generate_sequential_data", 
          [](int n_samples, int n_timesteps, int n_features, double noise_std) {
              return generate_sequential_data(n_samples, n_timesteps, n_features, default_pattern_function, noise_std, default_target_function);
          },
          py::arg("n_samples"), py::arg("n_timesteps"), py::arg("n_features"),
          py::arg("noise_std") = 0.1);
}
