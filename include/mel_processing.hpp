#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace mel_processing
{
    constexpr double MaxWavValue = 32768.0;

    torch::Tensor dynamicRangeCompression(
        const torch::Tensor& x,
        double c = 1.0,
        double clipValue = 1.0e-5);

    torch::Tensor dynamicRangeDecompression(
        const torch::Tensor& x,
        double c = 1.0);

    torch::Tensor spectralNormalize(const torch::Tensor& magnitudes);

    torch::Tensor spectralDenormalize(const torch::Tensor& magnitudes);

    torch::Tensor spectrogram(
        const torch::Tensor& y,
        int64_t nFft,
        int64_t samplingRate,
        int64_t hopSize,
        int64_t winSize,
        bool center = false);

    torch::Tensor specToMel(
        const torch::Tensor& spec,
        int64_t nFft,
        int64_t numMels,
        int64_t samplingRate,
        double fMin,
        double fMax);

    torch::Tensor melSpectrogram(
        const torch::Tensor& y,
        int64_t nFft,
        int64_t numMels,
        int64_t samplingRate,
        int64_t hopSize,
        int64_t winSize,
        double fMin,
        double fMax,
        bool center = false);
}
