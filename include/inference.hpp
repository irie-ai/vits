#pragma once

#include <torch/torch.h>

#include "models.hpp"
#include "utils.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace inference
{
    struct PreparedText
    {
        torch::Tensor tokens;
        torch::Tensor lengths;
    };

    struct InferOptions
    {
        double noiseScale = 0.667;
        double lengthScale = 1.0;
        double noiseScaleW = 0.8;
        c10::optional<int64_t> maxLength = c10::nullopt;
        c10::optional<int64_t> speakerId = c10::nullopt;
        c10::optional<torch::Device> device = c10::nullopt;
    };

    std::vector<int64_t> textToSequence(
        const std::string& text,
        const std::vector<std::string>& cleanerNames,
        bool addBlank);

    PreparedText prepareText(
        const std::string& text,
        const std::vector<std::string>& cleanerNames,
        bool addBlank,
        c10::optional<torch::Device> device = c10::nullopt);

    std::vector<std::string> cleanerNamesFromHParams(const utils::HParams& hparams);

    bool addBlankFromHParams(const utils::HParams& hparams);

    models::SynthesizerTrn createSynthesizerFromHParams(const utils::HParams& hparams);

    models::SynthesizerInferResult inferText(
        models::SynthesizerTrn& synthesizer,
        const std::string& text,
        const std::vector<std::string>& cleanerNames,
        bool addBlank,
        const InferOptions& options = {});
}
