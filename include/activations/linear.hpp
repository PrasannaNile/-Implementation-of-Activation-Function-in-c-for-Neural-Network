// header guard: no matter how many times this header file is included, 
// it will only be included once in a single compilation unit
#pragma once 

#include <vector>

namespace activations {

    class Linear {
        // Scalar forward pass (actual output) with respect to input v
        static inline double forward(double v) noexcept {
            return v;
        }

        // Scalar derivative pass with respect to input v
        static inline double derivative([[maybe_unused]] double v) noexcept {
            return 1.0;
        }

        // Vectorized forward pass for a batch/array of values
        static void forward(const std::vector<double>& input, std::vector<double>& output) {
            output.resize(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = forward(input[i]);
            }
        }

        // Vectorized derivative pass for a batch/array of values
        static void derivative(const std::vector<double>& input, std::vector<double>& output) {
            output.resize(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = derivative(input[i]);
            }
        }
    };

}
