#include <pybind11/pybind11.h>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

double shapley_kernel_weights(int n, int s) {
    if (s == 0 || s == n) {
        return 1000000.0; // Large weight for full/empty sets
    }
    auto comb = [](int n, int k) -> double {
        if (k > n) return 0;
        if (k == 0 || k == n) return 1;
        double result = 1;
        for (int i = 1; i <= k; ++i) {
            result *= (n - i + 1) / static_cast<double>(i);
        }
        return result;
    };
    return (n - 1) / (comb(n, s) * s * (n - s));
}

double entropy_kernel_weights(int n, int s) {
    if (s == 0 || s == n) {
        return 1000000.0;
    }
    double p = static_cast<double>(s) / n;
    double entropy = -(p * std::log(p + 1e-8) + (1 - p) * std::log(1 - p + 1e-8));
    return (entropy > 0) ? (1 / entropy) : 1000000.0;
}

double uniform_kernel_weights(int n, int s) {
    // In uniform kernel, all subsets are given the same weight
    return 1.0; // Equal weight for all subsets
}

PYBIND11_MODULE(shapley_kernel, m) {
    m.doc() = "Kernel weight computations exposed to Python via pybind11";

    m.def("shapley_kernel_weights", &shapley_kernel_weights, "Compute KernelSHAP standard weight");
    m.def("entropy_kernel_weights", &entropy_kernel_weights, "Compute entropy-based kernel weight");
    m.def("uniform_kernel_weights", &uniform_kernel_weights, "Compute uniform kernel weight for a subset");
}
