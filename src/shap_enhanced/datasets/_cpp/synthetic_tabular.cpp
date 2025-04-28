#include <torch/extension.h>
#include <pybind11/functional.h>
#include <random>
#include <cmath>

namespace py = pybind11;

// Default target function: sum of squared features
at::Tensor default_target_function(const at::Tensor& X) {
    return X.pow(2).sum(1);  // Sum of squares over features (axis=1)
}

// Main function for generating tabular data
std::tuple<at::Tensor, at::Tensor> generate_tabular_data(
    int n_samples,
    int n_features,
    std::function<at::Tensor(int, int)> feature_distribution,
    std::function<at::Tensor(const at::Tensor&)> target_function,
    double noise_std
) {
    // Generate feature matrix using the provided distribution function
    at::Tensor X = feature_distribution(n_samples, n_features);
    
    // Generate target values using the provided target function
    at::Tensor y = target_function(X);

    // Add noise to the target if necessary
    if (noise_std > 0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0, static_cast<float>(noise_std));

        for (int i = 0; i < n_samples; ++i) {
            y[i] += dist(gen);  // Add Gaussian noise
        }
    }

    return std::make_tuple(X, y);
}

PYBIND11_MODULE(synthetic_tabular, m) {
    // Overload `generate_tabular_data` to handle default functions using lambda
    m.def("generate_tabular_data", 
          [](int n_samples, int n_features, 
             std::function<at::Tensor(int, int)> feature_distribution, 
             std::function<at::Tensor(const at::Tensor&)> target_function,
             double noise_std) {
                return generate_tabular_data(n_samples, n_features, feature_distribution, target_function, noise_std);
          },
          py::arg("n_samples"), py::arg("n_features"),
          py::arg("feature_distribution") = [](int n_samples, int n_features) { 
              return torch::randn({n_samples, n_features}, torch::kFloat32);  // Default to standard normal distribution
          },
          py::arg("target_function") = [](const at::Tensor& X) { 
              return default_target_function(X); 
          },
          py::arg("noise_std") = 0.1);

    // Overload the function with no feature_distribution and target_function
    m.def("generate_tabular_data", 
          [](int n_samples, int n_features, double noise_std) {
              return generate_tabular_data(n_samples, n_features, 
                                            [](int n_samples, int n_features) { 
                                                return torch::randn({n_samples, n_features}, torch::kFloat32); 
                                            },
                                            default_target_function, noise_std);
          },
          py::arg("n_samples"), py::arg("n_features"),
          py::arg("noise_std") = 0.1);
}
