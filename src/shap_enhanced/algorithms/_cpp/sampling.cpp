#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <set>
#include <random>
#include <algorithm>
#include <numeric>

namespace py = pybind11;

std::random_device rd;
std::mt19937 gen(rd());

std::vector<std::vector<int>> sample_subsets(int n_features, int nsamples) {
    std::vector<int> all_features(n_features);
    std::iota(all_features.begin(), all_features.end(), 0);

    std::vector<std::vector<int>> subsets;
    std::uniform_int_distribution<> subset_size_dist(0, n_features - 1);

    for (int i = 0; i < nsamples; ++i) {
        int subset_size = subset_size_dist(gen);
        std::vector<int> temp = all_features;
        std::shuffle(temp.begin(), temp.end(), gen);
        subsets.push_back(std::vector<int>(temp.begin(), temp.begin() + subset_size));
    }
    return subsets;
}

std::vector<std::vector<int>> sample_balanced_subsets(int n_features, int nsamples) {
    std::vector<int> all_features(n_features);
    std::iota(all_features.begin(), all_features.end(), 0);

    std::vector<std::vector<int>> subsets;
    std::vector<int> choices = {1, n_features / 2, n_features - 1};
    std::uniform_int_distribution<> choice_dist(0, choices.size() - 1);

    for (int i = 0; i < nsamples; ++i) {
        int subset_size = choices[choice_dist(gen)];
        std::vector<int> temp = all_features;
        std::shuffle(temp.begin(), temp.end(), gen);
        subsets.push_back(std::vector<int>(temp.begin(), temp.begin() + subset_size));
    }
    return subsets;
}

std::vector<int> sample_timesteps(int total_timesteps, int nsamples) {
    std::vector<int> timesteps;
    std::uniform_int_distribution<> timestep_dist(0, total_timesteps - 1);

    for (int i = 0; i < nsamples; ++i) {
        timesteps.push_back(timestep_dist(gen));
    }
    return timesteps;
}

std::vector<std::vector<int>> sample_feature_subsets(int n_features, int nsamples) {
    std::vector<std::vector<int>> subsets;
    std::uniform_int_distribution<> subset_size_dist(1, n_features);

    for (int i = 0; i < nsamples; ++i) {
        int subset_size = subset_size_dist(gen);
        std::vector<int> features(n_features);
        std::iota(features.begin(), features.end(), 0);
        std::shuffle(features.begin(), features.end(), gen);
        subsets.push_back(std::vector<int>(features.begin(), features.begin() + subset_size));
    }
    return subsets;
}

std::vector<std::vector<int>> sample_feature_groups(const std::vector<std::vector<int>>& groups, int nsamples) {
    int n_groups = groups.size();
    std::vector<std::vector<int>> group_samples;
    std::uniform_int_distribution<> subset_size_dist(1, n_groups);

    for (int i = 0; i < nsamples; ++i) {
        int subset_size = subset_size_dist(gen);
        std::vector<int> group_indices(n_groups);
        std::iota(group_indices.begin(), group_indices.end(), 0);
        std::shuffle(group_indices.begin(), group_indices.end(), gen);
        group_samples.push_back(std::vector<int>(group_indices.begin(), group_indices.begin() + subset_size));
    }
    return group_samples;
}

PYBIND11_MODULE(sampling, m) {
    m.doc() = "Sampling utilities exposed to Python via pybind11";

    m.def("sample_subsets", &sample_subsets, "Random subsets of features (return list of lists)");
    m.def("sample_balanced_subsets", &sample_balanced_subsets, "Random subsets with balanced subset sizes");
    m.def("sample_timesteps", &sample_timesteps, "Sample random timesteps");
    m.def("sample_feature_subsets", &sample_feature_subsets, "Sample random feature subsets");
    m.def("sample_feature_groups", &sample_feature_groups, "Sample random groups from feature groups");
}
