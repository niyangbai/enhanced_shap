#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Function for Monte Carlo Expectation
torch::Tensor monte_carlo_expectation(std::function<torch::Tensor(torch::Tensor)> f, 
                                      torch::Tensor X, 
                                      int nsamples = 100) {
    std::vector<torch::Tensor> outputs;
    auto X_size = X.size(0);  // Get batch size
    for (int i = 0; i < nsamples; ++i) {
        int idx = rand() % X_size;  // Random index in [0, batch size)
        torch::Tensor x_sample = X[idx].unsqueeze(0);  // Single sample, unsqueeze to match shape
        outputs.push_back(f(x_sample));
    }
    return torch::stack(outputs).mean(0);
}

// Function for Joint Marginal Expectation
torch::Tensor joint_marginal_expectation(std::function<torch::Tensor(torch::Tensor)> model, 
                                         torch::Tensor x, 
                                         std::vector<int> S, 
                                         torch::Tensor background, 
                                         int nsamples = 50, 
                                         int target_index = 0) {
    int n_features = x.size(1);  // Number of features in x
    int batch_size = background.size(0);  // Background size
    std::vector<int64_t> idx(nsamples);
    for (int i = 0; i < nsamples; ++i) {
        idx[i] = rand() % batch_size;  // Random indices for sampling
    }

    torch::Tensor sampled = background.index_select(0, torch::tensor(idx, torch::kLong));

    // Overwrite features S in sampled
    for (int s : S) {
        sampled.index({torch::indexing::Slice(), s}) = x.index({0, s}).expand({nsamples});
    }

    torch::Tensor outputs = model(sampled);

    if (outputs.dim() > 1 && outputs.size(1) > 1) {
        if (target_index >= 0) {
            outputs = outputs.index({torch::indexing::Slice(), target_index});
        } else {
            outputs = outputs.index({torch::indexing::Slice(), 0});
        }
    }

    return outputs.mean();
}

// Function for Conditional Marginal Expectation
torch::Tensor conditional_marginal_expectation(std::function<torch::Tensor(torch::Tensor)> model, 
                                              torch::Tensor x, 
                                              std::vector<int> fixed_features, 
                                              torch::Tensor background, 
                                              int nsamples = 50, 
                                              int target_index = 0) {
    return joint_marginal_expectation(model, x, fixed_features, background, nsamples, target_index);
}

// Pybind11 bindings
PYBIND11_MODULE(approximation, m) {
    m.def("monte_carlo_expectation", &monte_carlo_expectation, "Monte Carlo estimate of E[f(X)]");
    m.def("joint_marginal_expectation", &joint_marginal_expectation, 
          "Estimate E[f(X) | X_S = x_S] where S is subset of features");
    m.def("conditional_marginal_expectation", &conditional_marginal_expectation, 
          "Estimate conditional E[f(X) | X_fixed = x_fixed]");
}
