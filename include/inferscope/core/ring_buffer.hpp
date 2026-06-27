#pragma once

#include <vector>
#include <optional>
#include <mutex>
#include <cstddef>
#include <stdexcept>
#include <iterator>

#include "inferscope/core/config.hpp"

namespace inferscope {

/// @brief Thread-safe, fixed-capacity ring buffer with circular overwrite semantics.
///
/// When full, pushing a new element overwrites the oldest element.
/// All public methods are mutex-protected for concurrent access.
/// Iteration proceeds from oldest to newest element.
template <typename T>
class RingBuffer {
public:
    // ========================================================================
    // Iterator — non-const forward iterator over logical order (oldest→newest)
    // ========================================================================
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = const T*;
        using reference         = const T&;

        Iterator(const RingBuffer* buf, size_t logical_index)
            : buf_(buf), logical_index_(logical_index) {}

        reference operator*() const {
            return buf_->buffer_[(buf_->head_ + logical_index_) % buf_->capacity_];
        }

        pointer operator->() const {
            return &(**this);
        }

        Iterator& operator++() {
            ++logical_index_;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator& other) const {
            return buf_ == other.buf_ && logical_index_ == other.logical_index_;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

    private:
        const RingBuffer* buf_;
        size_t logical_index_;
    };

    // ========================================================================
    // Construction
    // ========================================================================

    /// @brief Construct a ring buffer with the given capacity.
    /// @param capacity Maximum number of elements. Clamped to [1, MAX_RING_BUFFER_SIZE].
    explicit RingBuffer(size_t capacity = config::DEFAULT_RING_BUFFER_SIZE)
        : capacity_(clampCapacity(capacity))
        , buffer_(capacity_)
        , head_(0)
        , size_(0) {}

    // Non-copyable, movable
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    RingBuffer(RingBuffer&& other) noexcept {
        std::lock_guard<std::mutex> lock(other.mutex_);
        capacity_ = other.capacity_;
        buffer_   = std::move(other.buffer_);
        head_     = other.head_;
        size_     = other.size_;
        other.head_ = 0;
        other.size_ = 0;
    }

    RingBuffer& operator=(RingBuffer&& other) noexcept {
        if (this != &other) {
            std::scoped_lock lock(mutex_, other.mutex_);
            capacity_ = other.capacity_;
            buffer_   = std::move(other.buffer_);
            head_     = other.head_;
            size_     = other.size_;
            other.head_ = 0;
            other.size_ = 0;
        }
        return *this;
    }

    // ========================================================================
    // Push — overwrites oldest element when full
    // ========================================================================

    /// @brief Push an rvalue element. Overwrites the oldest when full.
    void push(T&& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        pushInternal(std::move(value));
    }

    /// @brief Push a const-ref element. Overwrites the oldest when full.
    void push(const T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        pushInternal(value);
    }

    // ========================================================================
    // Pop / Peek
    // ========================================================================

    /// @brief Remove and return the oldest element.
    /// @return The oldest element, or std::nullopt if empty.
    std::optional<T> pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (size_ == 0) {
            return std::nullopt;
        }
        T value = std::move(buffer_[head_]);
        head_ = (head_ + 1) % capacity_;
        --size_;
        return value;
    }

    /// @brief Peek at the oldest element without removing it.
    /// @return A const reference to the oldest element, or std::nullopt if empty.
    std::optional<std::reference_wrapper<const T>> peek() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (size_ == 0) {
            return std::nullopt;
        }
        return std::cref(buffer_[head_]);
    }

    // ========================================================================
    // Element Access
    // ========================================================================

    /// @brief Access element at logical index (0 = oldest).
    /// @throws std::out_of_range if index >= size.
    const T& at(size_t index) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= size_) {
            throw std::out_of_range("RingBuffer::at — index " +
                                    std::to_string(index) +
                                    " out of range (size=" +
                                    std::to_string(size_) + ")");
        }
        return buffer_[(head_ + index) % capacity_];
    }

    /// @brief Return a const reference to the oldest element.
    /// @throws std::out_of_range if the buffer is empty.
    const T& front() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (size_ == 0) {
            throw std::out_of_range("RingBuffer::front — buffer is empty");
        }
        return buffer_[head_];
    }

    /// @brief Return a const reference to the newest element.
    /// @throws std::out_of_range if the buffer is empty.
    const T& back() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (size_ == 0) {
            throw std::out_of_range("RingBuffer::back — buffer is empty");
        }
        size_t tail = (head_ + size_ - 1) % capacity_;
        return buffer_[tail];
    }

    // ========================================================================
    // Capacity / State
    // ========================================================================

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }

    size_t capacity() const {
        // capacity_ is immutable after construction — no lock needed.
        return capacity_;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_ == 0;
    }

    bool full() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_ == capacity_;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        head_ = 0;
        size_ = 0;
        // Re-default-construct elements so held resources are released.
        for (size_t i = 0; i < capacity_; ++i) {
            buffer_[i] = T{};
        }
    }

    // ========================================================================
    // Iterators (oldest → newest)
    //
    // NOTE: The caller must ensure no concurrent modification while iterating.
    //       These are intended for single-threaded consumption or for use
    //       under an external lock the caller already holds.
    // ========================================================================

    Iterator begin() const { return Iterator(this, 0); }
    Iterator end()   const { return Iterator(this, size_); }

private:
    // ========================================================================
    // Internal helpers (caller must hold mutex_)
    // ========================================================================

    template <typename U>
    void pushInternal(U&& value) {
        if (size_ < capacity_) {
            // Not yet full — write to the next available slot.
            size_t write_pos = (head_ + size_) % capacity_;
            buffer_[write_pos] = std::forward<U>(value);
            ++size_;
        } else {
            // Full — overwrite the oldest slot, then advance head.
            buffer_[head_] = std::forward<U>(value);
            head_ = (head_ + 1) % capacity_;
        }
    }

    static size_t clampCapacity(size_t requested) {
        if (requested == 0) return 1;
        if (requested > config::MAX_RING_BUFFER_SIZE) return config::MAX_RING_BUFFER_SIZE;
        return requested;
    }

    // ========================================================================
    // Data
    // ========================================================================
    size_t             capacity_;
    std::vector<T>     buffer_;
    size_t             head_;      // Index of the oldest element.
    size_t             size_;      // Current number of elements.
    mutable std::mutex mutex_;

    // Iterator needs direct access to internal state.
    friend class Iterator;
};

} // namespace inferscope
