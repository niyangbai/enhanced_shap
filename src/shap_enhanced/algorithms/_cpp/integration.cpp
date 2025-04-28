#include <torch/torch.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Trapezoidal Integration
torch::Tensor trapezoidal_integrate(torch::Tensor values) {
    // Sum adjacent elements (values[:-1] + values[1:]) and compute the mean along the first dimension
    return (values.slice(0, 0, values.size(0) - 1) + values.slice(0, 1, values.size(0))).mean(0);
}

// Pybind11 bindings
PYBIND11_MODULE(integration, m) {
    m.def("trapezoidal_integrate", &trapezoidal_integrate, "Approximate integral using trapezoidal rule");
}
