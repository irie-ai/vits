#include "mel_processing.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <torch/nn/functional/padding.h>

namespace mel_processing
{
    namespace
    {
        std::map<std::string, torch::Tensor> hannWindowCache;
        std::map<std::string, torch::Tensor> melBasisCache;

        std::string tensorCacheKey(const torch::Tensor& tensor, const std::string& prefix)
        {
            std::ostringstream stream;
            stream << prefix << "_" << static_cast<int>(tensor.scalar_type()) << "_" << tensor.device();
            return stream.str();
        }

        double hzToMel(double freq)
        {
            constexpr double fSp = 200.0 / 3.0;
            constexpr double minLogHz = 1000.0;
            constexpr double minLogMel = minLogHz / fSp;
            const double logstep = std::log(6.4) / 27.0;
            if (freq < minLogHz)
            {
                return freq / fSp;
            }
            return minLogMel + std::log(freq / minLogHz) / logstep;
        }

        double melToHz(double mel)
        {
            constexpr double fSp = 200.0 / 3.0;
            constexpr double minLogHz = 1000.0;
            constexpr double minLogMel = minLogHz / fSp;
            const double logstep = std::log(6.4) / 27.0;
            if (mel < minLogMel)
            {
                return mel * fSp;
            }
            return minLogHz * std::exp(logstep * (mel - minLogMel));
        }

        torch::Tensor createMelBasis(
            int64_t samplingRate,
            int64_t nFft,
            int64_t numMels,
            double fMin,
            double fMax,
            const torch::TensorOptions& options)
        {
            const int64_t fftBins = nFft / 2 + 1;
            std::vector<double> fftFreqs(fftBins);
            for (int64_t i = 0; i < fftBins; ++i)
            {
                fftFreqs[i] = static_cast<double>(i) * samplingRate / nFft;
            }

            const double minMel = hzToMel(fMin);
            const double maxMel = hzToMel(fMax);
            std::vector<double> melFreqs(numMels + 2);
            for (int64_t i = 0; i < numMels + 2; ++i)
            {
                const double mel = minMel + (maxMel - minMel) * static_cast<double>(i) / static_cast<double>(numMels + 1);
                melFreqs[i] = melToHz(mel);
            }

            std::vector<float> weights(static_cast<size_t>(numMels * fftBins), 0.0f);
            for (int64_t mel = 0; mel < numMels; ++mel)
            {
                const double lower = melFreqs[mel];
                const double center = melFreqs[mel + 1];
                const double upper = melFreqs[mel + 2];
                const double enorm = 2.0 / std::max(upper - lower, 1.0e-12);

                for (int64_t bin = 0; bin < fftBins; ++bin)
                {
                    const double freq = fftFreqs[bin];
                    const double lowerSlope = (freq - lower) / std::max(center - lower, 1.0e-12);
                    const double upperSlope = (upper - freq) / std::max(upper - center, 1.0e-12);
                    const double value = std::max(0.0, std::min(lowerSlope, upperSlope)) * enorm;
                    weights[static_cast<size_t>(mel * fftBins + bin)] = static_cast<float>(value);
                }
            }

            return torch::from_blob(weights.data(), {numMels, fftBins}, torch::TensorOptions().dtype(torch::kFloat32)).clone().to(options);
        }

        torch::Tensor hannWindow(int64_t winSize, const torch::Tensor& y)
        {
            auto key = tensorCacheKey(y, std::to_string(winSize));
            auto found = hannWindowCache.find(key);
            if (found == hannWindowCache.end())
            {
                auto window = torch::hann_window(winSize, y.options());
                found = hannWindowCache.emplace(key, window).first;
            }
            return found->second;
        }

        torch::Tensor melBasis(
            const torch::Tensor& spec,
            int64_t nFft,
            int64_t numMels,
            int64_t samplingRate,
            double fMin,
            double fMax)
        {
            std::ostringstream stream;
            stream << nFft << "_" << numMels << "_" << samplingRate << "_" << fMin << "_" << fMax << "_"
                   << static_cast<int>(spec.scalar_type()) << "_" << spec.device();
            auto key = stream.str();
            auto found = melBasisCache.find(key);
            if (found == melBasisCache.end())
            {
                auto basis = createMelBasis(samplingRate, nFft, numMels, fMin, fMax, spec.options());
                found = melBasisCache.emplace(key, basis).first;
            }
            return found->second;
        }
    }

    torch::Tensor dynamicRangeCompression(const torch::Tensor& x, double c, double clipValue)
    {
        return torch::log(torch::clamp_min(x, clipValue) * c);
    }

    torch::Tensor dynamicRangeDecompression(const torch::Tensor& x, double c)
    {
        return torch::exp(x) / c;
    }

    torch::Tensor spectralNormalize(const torch::Tensor& magnitudes)
    {
        return dynamicRangeCompression(magnitudes);
    }

    torch::Tensor spectralDenormalize(const torch::Tensor& magnitudes)
    {
        return dynamicRangeDecompression(magnitudes);
    }

    torch::Tensor spectrogram(
        const torch::Tensor& y,
        int64_t nFft,
        int64_t,
        int64_t hopSize,
        int64_t winSize,
        bool center)
    {
        namespace F = torch::nn::functional;
        const int64_t pad = (nFft - hopSize) / 2;
        auto padded = F::pad(
            y.unsqueeze(1),
            F::PadFuncOptions({pad, pad}).mode(torch::kReflect));
        padded = padded.squeeze(1);

        auto stft = torch::stft(
            padded,
            nFft,
            hopSize,
            winSize,
            hannWindow(winSize, y),
            center,
            "reflect",
            false,
            true,
            false);
        return torch::sqrt(stft.pow(2).sum(-1) + 1.0e-6);
    }

    torch::Tensor specToMel(
        const torch::Tensor& spec,
        int64_t nFft,
        int64_t numMels,
        int64_t samplingRate,
        double fMin,
        double fMax)
    {
        auto mel = torch::matmul(melBasis(spec, nFft, numMels, samplingRate, fMin, fMax), spec);
        return spectralNormalize(mel);
    }

    torch::Tensor melSpectrogram(
        const torch::Tensor& y,
        int64_t nFft,
        int64_t numMels,
        int64_t samplingRate,
        int64_t hopSize,
        int64_t winSize,
        double fMin,
        double fMax,
        bool center)
    {
        auto spec = spectrogram(y, nFft, samplingRate, hopSize, winSize, center);
        return specToMel(spec, nFft, numMels, samplingRate, fMin, fMax);
    }
}
