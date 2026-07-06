#include "inference.hpp"

#include "commons.hpp"
#include "text_processing.hpp"

#include <stdexcept>

namespace inference
{
    namespace
    {
        const utils::HParamValue& section(
            const utils::HParams& hparams,
            const std::string& name)
        {
            if (!hparams.contains(name))
            {
                throw std::invalid_argument("HParams is missing section: " + name);
            }
            return hparams.at(name);
        }

        int64_t optionalInt(
            const utils::HParamValue& object,
            const std::string& key,
            int64_t fallback)
        {
            return object.contains(key) ? object.at(key).asInt() : fallback;
        }

        bool optionalBool(
            const utils::HParamValue& object,
            const std::string& key,
            bool fallback)
        {
            return object.contains(key) ? object.at(key).asBool() : fallback;
        }

        std::vector<int64_t> intVector(const utils::HParamValue& value)
        {
            std::vector<int64_t> result;
            for (const auto& item : value.asArray())
            {
                result.push_back(item.asInt());
            }
            return result;
        }

        std::vector<std::vector<int64_t>> nestedIntVector(const utils::HParamValue& value)
        {
            std::vector<std::vector<int64_t>> result;
            for (const auto& item : value.asArray())
            {
                result.push_back(intVector(item));
            }
            return result;
        }

        c10::optional<torch::Tensor> speakerTensor(
            const c10::optional<int64_t>& speakerId,
            c10::optional<torch::Device> device)
        {
            if (!speakerId.has_value())
            {
                return c10::nullopt;
            }

            auto tensor = torch::tensor(
                {speakerId.value()},
                torch::TensorOptions().dtype(torch::kLong));
            if (device.has_value())
            {
                tensor = tensor.to(device.value());
            }
            return tensor;
        }
    }

    std::vector<int64_t> textToSequence(
        const std::string& text,
        const std::vector<std::string>& cleanerNames,
        bool addBlank)
    {
        auto sequence = text_processing::textToSequence(text, cleanerNames);
        if (addBlank)
        {
            sequence = commons::intersperse(sequence, 0);
        }
        if (sequence.empty())
        {
            throw std::invalid_argument("textToSequence produced an empty sequence.");
        }
        return sequence;
    }

    PreparedText prepareText(
        const std::string& text,
        const std::vector<std::string>& cleanerNames,
        bool addBlank,
        c10::optional<torch::Device> device)
    {
        auto sequence = textToSequence(text, cleanerNames, addBlank);
        auto tokens = torch::tensor(sequence, torch::TensorOptions().dtype(torch::kLong)).unsqueeze(0);
        auto lengths = torch::tensor(
            {static_cast<int64_t>(sequence.size())},
            torch::TensorOptions().dtype(torch::kLong));

        if (device.has_value())
        {
            tokens = tokens.to(device.value());
            lengths = lengths.to(device.value());
        }

        return {tokens, lengths};
    }

    std::vector<std::string> cleanerNamesFromHParams(const utils::HParams& hparams)
    {
        const auto& data = section(hparams, "data");
        if (!data.contains("text_cleaners"))
        {
            throw std::invalid_argument("HParams data section is missing text_cleaners.");
        }

        std::vector<std::string> result;
        for (const auto& cleaner : data.at("text_cleaners").asArray())
        {
            result.push_back(cleaner.asString());
        }
        return result;
    }

    bool addBlankFromHParams(const utils::HParams& hparams)
    {
        const auto& data = section(hparams, "data");
        return optionalBool(data, "add_blank", false);
    }

    models::SynthesizerTrn createSynthesizerFromHParams(const utils::HParams& hparams)
    {
        const auto& train = section(hparams, "train");
        const auto& data = section(hparams, "data");
        const auto& model = section(hparams, "model");

        const auto filterLength = data.at("filter_length").asInt();
        const auto hopLength = data.at("hop_length").asInt();
        if (hopLength <= 0)
        {
            throw std::invalid_argument("data.hop_length must be positive.");
        }

        const auto segmentSize = train.at("segment_size").asInt() / hopLength;
        const auto specChannels = filterLength / 2 + 1;
        const auto nSpeakers = optionalInt(data, "n_speakers", 0);
        const auto ginChannels = optionalInt(model, "gin_channels", 0);
        const auto useSdp = optionalBool(model, "use_sdp", true);

        return models::SynthesizerTrn(
            static_cast<int64_t>(text_processing::symbols().size()),
            specChannels,
            segmentSize,
            model.at("inter_channels").asInt(),
            model.at("hidden_channels").asInt(),
            model.at("filter_channels").asInt(),
            model.at("n_heads").asInt(),
            model.at("n_layers").asInt(),
            model.at("kernel_size").asInt(),
            model.at("p_dropout").asNumber(),
            model.at("resblock").asString(),
            intVector(model.at("resblock_kernel_sizes")),
            nestedIntVector(model.at("resblock_dilation_sizes")),
            intVector(model.at("upsample_rates")),
            model.at("upsample_initial_channel").asInt(),
            intVector(model.at("upsample_kernel_sizes")),
            nSpeakers,
            ginChannels,
            useSdp);
    }

    models::SynthesizerInferResult inferText(
        models::SynthesizerTrn& synthesizer,
        const std::string& text,
        const std::vector<std::string>& cleanerNames,
        bool addBlank,
        const InferOptions& options)
    {
        torch::NoGradGuard noGrad;
        synthesizer->eval();

        auto prepared = prepareText(text, cleanerNames, addBlank, options.device);
        auto sid = speakerTensor(options.speakerId, options.device);
        return synthesizer->infer(
            prepared.tokens,
            prepared.lengths,
            sid,
            options.noiseScale,
            options.lengthScale,
            options.noiseScaleW,
            options.maxLength);
    }
}
