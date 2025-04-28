#include <torch/torch.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Linear interpolation
torch::Tensor linear_interpolation(torch::Tensor start, torch::Tensor end, int steps) {
    // Generate alphas from 0 to 1 with 'steps + 1' values
    torch::Tensor alphas = torch::linspace(0, 1, steps + 1, start.device()).view({-1, 1});  // Shape: (steps+1, 1)
    
    // Perform linear interpolation
    return start + alphas * (end - start);
}

// Pybind11 bindings
PYBIND11_MODULE(interpolation, m) {
    m.def("linear_interpolation", &linear_interpolation, "Generate linearly interpolated tensors");
}
