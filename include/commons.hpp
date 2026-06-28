#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace commons
{
    /**************************************************************************
     * Weight Initialization
     **************************************************************************/

     /// Python: init_weights()
    void initWeights(
        torch::nn::Module& module,
        double mean = 0.0,
        double std = 0.01);

    /**************************************************************************
     * Utility
     **************************************************************************/

     /// Python: get_padding()
    int64_t getPadding(
        int64_t kernelSize,
        int64_t dilation = 1);

    /// Python: intersperse()
    std::vector<int64_t> intersperse(
        const std::vector<int64_t>& list,
        int64_t item);

    /// Python: kl_divergence()
    torch::Tensor klDivergence(
        const torch::Tensor& mP,
        const torch::Tensor& logsP,
        const torch::Tensor& mQ,
        const torch::Tensor& logsQ);

    /// Python: convert_pad_shape()
    std::vector<int64_t> convertPadShape(
        const std::vector<std::vector<int64_t>>& padShape);

    /**************************************************************************
     * Tensor
     **************************************************************************/

     /// Python: slice_segments()
    torch::Tensor sliceSegments(
        const torch::Tensor& x,
        const torch::Tensor& idsStr,
        int64_t segmentSize = 4);

    /// Python: rand_slice_segments()
    std::pair<torch::Tensor, torch::Tensor> randSliceSegments(
        const torch::Tensor& x,
        const c10::optional<torch::Tensor>& xLengths = c10::nullopt,
        int64_t segmentSize = 4);

    /// Python: rand_gumbel()
    torch::Tensor randGumbel(
        torch::IntArrayRef shape);

    /// Python: rand_gumbel_like()
    torch::Tensor randGumbelLike(
        const torch::Tensor& x);

    /**************************************************************************
     * Positional Encoding
     **************************************************************************/

     /// Python: get_timing_signal_1d()
    torch::Tensor getTimingSignal1D(
        int64_t length,
        int64_t channels,
        double minTimescale = 1.0,
        double maxTimescale = 1.0e4);

    /// Python: add_timing_signal_1d()
    torch::Tensor addTimingSignal1D(
        const torch::Tensor& x,
        double minTimescale = 1.0,
        double maxTimescale = 1.0e4);

    /// Python: cat_timing_signal_1d()
    torch::Tensor catTimingSignal1D(
        const torch::Tensor& x,
        double minTimescale = 1.0,
        double maxTimescale = 1.0e4,
        int64_t axis = 1);

    /**************************************************************************
     * Mask
     **************************************************************************/

     /// Python: subsequent_mask()
    torch::Tensor subsequentMask(
        int64_t length);

    /// Python: sequence_mask()
    torch::Tensor sequenceMask(
        const torch::Tensor& length,
        c10::optional<int64_t> maxLength = c10::nullopt);

    /**************************************************************************
     * Operations
     **************************************************************************/

     /// Python: fused_add_tanh_sigmoid_multiply()
    torch::Tensor fusedAddTanhSigmoidMultiply(
        const torch::Tensor& inputA,
        const torch::Tensor& inputB,
        int64_t nChannels);

    /// Python: shift_1d()
    torch::Tensor shift1D(
        const torch::Tensor& x);

    /// Python: generate_path()
    torch::Tensor generatePath(
        const torch::Tensor& duration,
        const torch::Tensor& mask);

    /// Python: clip_grad_value_()
    double clipGradValue(
        const std::vector<torch::Tensor>& parameters,
        double clipValue,
        double normType = 2.0);
}
