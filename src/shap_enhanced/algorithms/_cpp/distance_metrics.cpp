#include <torch/torch.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Euclidean distance
torch::Tensor euclidean_distance(torch::Tensor x, torch::Tensor y) {
    return torch::norm(x - y, /*dim=*/-1);
}

// Cosine similarity
torch::Tensor cosine_similarity(torch::Tensor x, torch::Tensor y) {
    // Use NormalizeFuncOptions to specify the dimension for normalization
    torch::Tensor x_norm = torch::nn::functional::normalize(x, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    torch::Tensor y_norm = torch::nn::functional::normalize(y, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    return (x_norm * y_norm).sum(-1);
}

// Dynamic Time Warping (DTW) distance
torch::Tensor dynamic_time_warping_distance(torch::Tensor x, torch::Tensor y) {
    int64_t time_x = x.size(0);
    int64_t time_y = y.size(0);

    torch::Tensor dist = torch::full({time_x + 1, time_y + 1}, std::numeric_limits<float>::infinity(), x.device());
    dist[0][0] = 0.0;

    for (int64_t i = 1; i <= time_x; ++i) {
        for (int64_t j = 1; j <= time_y; ++j) {
            torch::Tensor cost = torch::norm(x[i - 1] - y[j - 1]);

            // Find the minimum value from three tensors: dist[i-1][j], dist[i][j-1], dist[i-1][j-1]
            torch::Tensor min_cost = torch::min(torch::min(dist[i - 1][j], dist[i][j - 1]), dist[i - 1][j - 1]);

            dist[i][j] = cost + min_cost;
        }
    }

    return dist[time_x][time_y];
}

// Pybind11 bindings
PYBIND11_MODULE(distance_metrics, m) {
    m.def("euclidean_distance", &euclidean_distance, "Compute Euclidean distance");
    m.def("cosine_similarity", &cosine_similarity, "Compute Cosine similarity");
    m.def("dynamic_time_warping_distance", &dynamic_time_warping_distance, 
          "Compute Dynamic Time Warping (DTW) distance between sequences");
}
