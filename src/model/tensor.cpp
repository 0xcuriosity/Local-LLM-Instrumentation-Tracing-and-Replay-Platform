#include "neuralscope/model/tensor.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace neuralscope {

Tensor::Tensor(const TensorShape& shape) {
    shape_ = shape;
    data_.resize(shape.numElements(), 0.0f);
}

Tensor::Tensor(const TensorShape& shape, std::vector<float> data) 
    : shape_(shape), data_(std::move(data)) {
}

Tensor Tensor::random(const TensorShape& shape, float mean, float stddev) {
    Tensor t(shape);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(mean, stddev);
    
    for (size_t i = 0; i < t.data_.size(); ++i) {
        t.data_[i] = d(gen);
    }
    
    return t;
}

Tensor Tensor::zeros(const TensorShape& shape) {
    return Tensor(shape);
}

Tensor Tensor::ones(const TensorShape& shape) {
    Tensor t(shape);
    std::fill(t.data_.begin(), t.data_.end(), 1.0f);
    return t;
}

ActivationStats Tensor::computeStats() const {
    ActivationStats stats;
    stats.num_elements = data_.size();
    if (data_.empty()) return stats;
    
    double sum = 0.0;
    double sum_sq = 0.0;
    double abs_sum = 0.0;
    
    stats.min_val = data_[0];
    stats.max_val = data_[0];
    
    const double SPARSITY_THRESHOLD = 1e-6;
    
    for (float val : data_) {
        if (std::isnan(val)) stats.has_nan = true;
        if (std::isinf(val)) stats.has_inf = true;
        
        sum += val;
        sum_sq += static_cast<double>(val) * val;
        abs_sum += std::abs(val);
        
        if (val < stats.min_val) stats.min_val = val;
        if (val > stats.max_val) stats.max_val = val;
        
        if (std::abs(val) < SPARSITY_THRESHOLD) {
            stats.num_zeros++;
        }
    }
    
    stats.mean = sum / stats.num_elements;
    double variance = (sum_sq / stats.num_elements) - (stats.mean * stats.mean);
    stats.std_dev = variance > 0 ? std::sqrt(variance) : 0.0;
    stats.sparsity_rate = static_cast<double>(stats.num_zeros) / stats.num_elements;
    stats.l2_norm = std::sqrt(sum_sq);
    stats.abs_mean = abs_sum / stats.num_elements;
    
    return stats;
}

const TensorShape& Tensor::shape() const {
    return shape_;
}

float* Tensor::data() {
    return data_.data();
}

const float* Tensor::data() const {
    return data_.data();
}

} // namespace neuralscope
