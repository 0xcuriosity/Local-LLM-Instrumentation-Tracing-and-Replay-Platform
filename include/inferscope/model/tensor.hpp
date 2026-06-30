#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include <initializer_list>

#include "inferscope/core/types.hpp"

namespace inferscope {

/// @brief Lightweight, row-major tensor with float storage.
///
/// Supports basic operations needed for model simulation and activation
/// statistics computation. No external BLAS dependency — all math is
/// implemented directly for portability and simplicity.
class Tensor {
public:
    // ========================================================================
    // Construction
    // ========================================================================

    /// @brief Construct a zero-initialized tensor of the given shape.
    explicit Tensor(const TensorShape& shape);

    /// @brief Construct from shape and an existing data vector (moved).
    Tensor(const TensorShape& shape, std::vector<float> data);

    /// Default / copy / move
    Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    // ========================================================================
    // Static Factories
    // ========================================================================

    /// @brief Create a tensor filled with samples from N(mean, stddev).
    static Tensor random(const TensorShape& shape,
                         float mean = 0.0f, float stddev = 1.0f);

    /// @brief Create a zero-initialized tensor.
    static Tensor zeros(const TensorShape& shape);

    /// @brief Create a tensor filled with ones.
    static Tensor ones(const TensorShape& shape);

    // ========================================================================
    // Mathematical Operations (each returns a NEW tensor)
    // ========================================================================

    /// @brief Matrix multiply this (…, M, K) by other (…, K, N) → (…, M, N).
    ///        Only the last two dimensions participate in the matmul;
    ///        leading batch dimensions must be broadcastable or equal.
    Tensor matmul(const Tensor& other) const;

    /// @brief Softmax along the given dimension.
    Tensor softmax(int dim = -1) const;

    /// @brief RMS normalization with the given epsilon.
    Tensor rms_norm(float eps = 1e-5f) const;

    /// @brief GELU activation (approximate: 0.5 * x * (1 + tanh(…))).
    Tensor gelu() const;

    /// @brief SiLU (Swish) activation: x * sigmoid(x).
    Tensor silu() const;

    /// @brief Transpose two dimensions.
    Tensor transpose(int dim0, int dim1) const;

    // ========================================================================
    // Element Access
    // ========================================================================

    /// @brief Flat element access by linear index.
    float& at(size_t linear_index);
    const float& at(size_t linear_index) const;

    /// @brief Multi-dimensional element access.
    float& at(std::initializer_list<int64_t> indices);
    const float& at(std::initializer_list<int64_t> indices) const;

    /// @brief Raw data pointer.
    float* data();
    const float* data() const;

    /// @brief Tensor shape.
    const TensorShape& shape() const;

    // ========================================================================
    // Statistics
    // ========================================================================

    /// @brief Compute full activation statistics over all elements.
    ActivationStats computeStats() const;

    // ========================================================================
    // Memory Info
    // ========================================================================

    /// @brief Total storage in bytes (elements × sizeof(float)).
    size_t memoryBytes() const;

    /// @brief Total number of elements.
    int64_t numElements() const;

private:
    TensorShape  shape_;
    std::vector<float> data_;

    /// Convert multi-index to flat linear index (row-major).
    size_t linearIndex(std::initializer_list<int64_t> indices) const;
};

} // namespace inferscope
