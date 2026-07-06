#include "data_utils.hpp"

#include "commons.hpp"
#include "mel_processing.hpp"
#include "text_processing.hpp"
#include "utils.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace data_utils
{
    namespace
    {
        std::vector<std::string> splitLine(const std::string& line, const std::string& delimiter)
        {
            std::vector<std::string> result;
            size_t start = 0;
            while (true)
            {
                const auto pos = line.find(delimiter, start);
                if (pos == std::string::npos)
                {
                    result.push_back(line.substr(start));
                    break;
                }
                result.push_back(line.substr(start, pos - start));
                start = pos + delimiter.size();
            }
            return result;
        }

        std::vector<int64_t> sortIndicesBySpecLength(const std::vector<int64_t>& specLengths)
        {
            std::vector<int64_t> indices(specLengths.size());
            for (size_t i = 0; i < indices.size(); ++i)
            {
                indices[i] = static_cast<int64_t>(i);
            }

            std::sort(indices.begin(), indices.end(), [&](int64_t left, int64_t right) {
                return specLengths[static_cast<size_t>(left)] > specLengths[static_cast<size_t>(right)];
            });
            return indices;
        }

        void validateTextAudioItem(const TextAudioItem& item)
        {
            if (item.text.dim() != 1)
            {
                throw std::invalid_argument("TextAudioItem text must have shape [text_length].");
            }
            if (item.spec.dim() != 2)
            {
                throw std::invalid_argument("TextAudioItem spec must have shape [channels, spec_length].");
            }
            if (item.wav.dim() != 2)
            {
                throw std::invalid_argument("TextAudioItem wav must have shape [1, wav_length].");
            }
        }

        TextAudioItem withoutSpeaker(const TextAudioSpeakerItem& item)
        {
            return {item.text, item.spec, item.wav};
        }

        std::string specCachePath(const std::string& audioPath)
        {
            return audioPath + ".spec.pt";
        }
    }

    std::vector<FilepathAndText> loadFilepathsAndText(
        const std::string& filename,
        const std::string& split)
    {
        std::ifstream file(filename);
        if (!file)
        {
            throw std::runtime_error("Could not open filelist: " + filename);
        }

        std::vector<FilepathAndText> result;
        std::string line;
        while (std::getline(file, line))
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }
            if (!line.empty())
            {
                result.push_back(splitLine(line, split));
            }
        }
        return result;
    }

    std::vector<FilepathAndText> filterByTextLength(
        const std::vector<FilepathAndText>& filepathsAndText,
        size_t textIndex,
        int64_t minTextLen,
        int64_t maxTextLen)
    {
        std::vector<FilepathAndText> result;
        for (const auto& row : filepathsAndText)
        {
            if (textIndex >= row.size())
            {
                continue;
            }

            const auto textLength = static_cast<int64_t>(row[textIndex].size());
            if (minTextLen <= textLength && textLength <= maxTextLen)
            {
                result.push_back(row);
            }
        }
        return result;
    }

    torch::Tensor textToTensor(
        const std::string& text,
        const std::vector<std::string>& cleaners,
        bool cleanedText,
        bool addBlank)
    {
        auto sequence = cleanedText
            ? text_processing::cleanedTextToSequence(text)
            : text_processing::textToSequence(text, cleaners);

        if (addBlank)
        {
            sequence = commons::intersperse(sequence, 0);
        }

        return torch::tensor(sequence, torch::TensorOptions().dtype(torch::kLong));
    }

    AudioData getAudio(
        const std::string& audioPath,
        const AudioConfig& config)
    {
        const auto [audio, samplingRate] = utils::loadWavToTorch(audioPath);
        if (samplingRate != config.samplingRate)
        {
            throw std::runtime_error("Sampling rate mismatch for " + audioPath);
        }

        auto wav = audio.to(torch::kFloat32) / mel_processing::MaxWavValue;
        if (wav.dim() == 2)
        {
            wav = wav.mean(1);
        }
        wav = wav.unsqueeze(0);

        torch::Tensor spec;
        const auto cachePath = specCachePath(audioPath);
        if (config.useSpecCache && std::filesystem::exists(cachePath))
        {
            torch::load(spec, cachePath);
            if (!spec.defined() || spec.dim() != 2)
            {
                throw std::runtime_error("Invalid spectrogram cache: " + cachePath);
            }
        }
        else
        {
            spec = mel_processing::spectrogram(
                wav,
                config.filterLength,
                config.samplingRate,
                config.hopLength,
                config.winLength,
                false);
            spec = spec.squeeze(0);

            if (config.useSpecCache)
            {
                torch::save(spec, cachePath);
            }
        }
        return {spec, wav};
    }

    TextAudioItem makeTextAudioItem(
        const std::string& audioPath,
        const std::string& text,
        const AudioConfig& config,
        const std::vector<std::string>& cleaners,
        bool cleanedText,
        bool addBlank)
    {
        const auto textTensor = textToTensor(text, cleaners, cleanedText, addBlank);
        const auto audio = getAudio(audioPath, config);
        return {textTensor, audio.spec, audio.wav};
    }

    TextAudioSpeakerItem makeTextAudioSpeakerItem(
        const std::string& audioPath,
        const std::string& text,
        int64_t speakerId,
        const AudioConfig& config,
        const std::vector<std::string>& cleaners,
        bool cleanedText,
        bool addBlank)
    {
        const auto item = makeTextAudioItem(audioPath, text, config, cleaners, cleanedText, addBlank);
        return {item.text, item.spec, item.wav, speakerId};
    }

    TextAudioBatch collateTextAudio(const std::vector<TextAudioItem>& batch)
    {
        if (batch.empty())
        {
            throw std::invalid_argument("collateTextAudio requires a non-empty batch.");
        }

        std::vector<int64_t> specLengths;
        specLengths.reserve(batch.size());
        int64_t maxTextLen = 0;
        int64_t maxSpecLen = 0;
        int64_t maxWavLen = 0;
        const auto specChannels = batch.front().spec.size(0);

        for (const auto& item : batch)
        {
            validateTextAudioItem(item);
            if (item.spec.size(0) != specChannels)
            {
                throw std::invalid_argument("All specs in a batch must have the same channel count.");
            }
            specLengths.push_back(item.spec.size(1));
            maxTextLen = std::max(maxTextLen, item.text.size(0));
            maxSpecLen = std::max(maxSpecLen, item.spec.size(1));
            maxWavLen = std::max(maxWavLen, item.wav.size(1));
        }

        const auto sorted = sortIndicesBySpecLength(specLengths);
        const auto batchSize = static_cast<int64_t>(batch.size());
        auto textPadded = torch::zeros({batchSize, maxTextLen}, torch::kLong);
        auto specPadded = torch::zeros({batchSize, specChannels, maxSpecLen}, batch.front().spec.options());
        auto wavPadded = torch::zeros({batchSize, 1, maxWavLen}, batch.front().wav.options());
        auto textLengths = torch::zeros({batchSize}, torch::kLong);
        auto specLengthsTensor = torch::zeros({batchSize}, torch::kLong);
        auto wavLengths = torch::zeros({batchSize}, torch::kLong);

        for (int64_t i = 0; i < batchSize; ++i)
        {
            const auto sourceIndex = sorted[static_cast<size_t>(i)];
            const auto& item = batch[static_cast<size_t>(sourceIndex)];
            const auto textLength = item.text.size(0);
            const auto specLength = item.spec.size(1);
            const auto wavLength = item.wav.size(1);

            textPadded.index_put_({i, torch::indexing::Slice(0, textLength)}, item.text);
            specPadded.index_put_({i, torch::indexing::Slice(), torch::indexing::Slice(0, specLength)}, item.spec);
            wavPadded.index_put_({i, torch::indexing::Slice(), torch::indexing::Slice(0, wavLength)}, item.wav);
            textLengths.index_put_({i}, textLength);
            specLengthsTensor.index_put_({i}, specLength);
            wavLengths.index_put_({i}, wavLength);
        }

        return {
            textPadded,
            textLengths,
            specPadded,
            specLengthsTensor,
            wavPadded,
            wavLengths,
            torch::tensor(sorted, torch::TensorOptions().dtype(torch::kLong))};
    }

    TextAudioSpeakerBatch collateTextAudioSpeaker(const std::vector<TextAudioSpeakerItem>& batch)
    {
        if (batch.empty())
        {
            throw std::invalid_argument("collateTextAudioSpeaker requires a non-empty batch.");
        }

        std::vector<TextAudioItem> baseBatch;
        baseBatch.reserve(batch.size());
        for (const auto& item : batch)
        {
            baseBatch.push_back(withoutSpeaker(item));
        }

        const auto collated = collateTextAudio(baseBatch);
        auto speakerIds = torch::zeros({static_cast<int64_t>(batch.size())}, torch::kLong);
        for (int64_t i = 0; i < collated.idsSortedDecreasing.size(0); ++i)
        {
            const auto sourceIndex = collated.idsSortedDecreasing.index({i}).item<int64_t>();
            speakerIds.index_put_({i}, batch[static_cast<size_t>(sourceIndex)].speakerId);
        }

        return {
            collated.textPadded,
            collated.textLengths,
            collated.specPadded,
            collated.specLengths,
            collated.wavPadded,
            collated.wavLengths,
            speakerIds,
            collated.idsSortedDecreasing};
    }

    int64_t bucketIndex(
        int64_t length,
        const std::vector<int64_t>& boundaries)
    {
        if (boundaries.size() < 2)
        {
            throw std::invalid_argument("bucketIndex requires at least two boundaries.");
        }

        int64_t low = 0;
        int64_t high = static_cast<int64_t>(boundaries.size()) - 1;
        while (high > low)
        {
            const int64_t mid = (high + low) / 2;
            if (boundaries[static_cast<size_t>(mid)] < length &&
                length <= boundaries[static_cast<size_t>(mid + 1)])
            {
                return mid;
            }
            if (length <= boundaries[static_cast<size_t>(mid)])
            {
                high = mid;
            }
            else
            {
                low = mid + 1;
            }
        }
        return -1;
    }

    BucketSamplerState createBucketSamplerState(
        const std::vector<int64_t>& lengths,
        const std::vector<int64_t>& boundaries,
        int64_t batchSize,
        int64_t numReplicas)
    {
        if (batchSize <= 0 || numReplicas <= 0)
        {
            throw std::invalid_argument("createBucketSamplerState requires positive batchSize and numReplicas.");
        }
        if (boundaries.size() < 2)
        {
            throw std::invalid_argument("createBucketSamplerState requires at least two boundaries.");
        }

        BucketSamplerState state;
        state.boundaries = boundaries;
        state.buckets.resize(boundaries.size() - 1);

        for (size_t i = 0; i < lengths.size(); ++i)
        {
            const auto bucket = bucketIndex(lengths[i], state.boundaries);
            if (bucket != -1)
            {
                state.buckets[static_cast<size_t>(bucket)].push_back(static_cast<int64_t>(i));
            }
        }

        for (int64_t i = static_cast<int64_t>(state.buckets.size()) - 1; i > 0; --i)
        {
            if (state.buckets[static_cast<size_t>(i)].empty())
            {
                state.buckets.erase(state.buckets.begin() + i);
                state.boundaries.erase(state.boundaries.begin() + i + 1);
            }
        }

        const auto totalBatchSize = numReplicas * batchSize;
        for (const auto& bucket : state.buckets)
        {
            const auto lenBucket = static_cast<int64_t>(bucket.size());
            if (lenBucket == 0)
            {
                state.numSamplesPerBucket.push_back(0);
                continue;
            }
            const auto remainder = (totalBatchSize - (lenBucket % totalBatchSize)) % totalBatchSize;
            const auto padded = lenBucket + remainder;
            state.numSamplesPerBucket.push_back(padded);
            state.totalSize += padded;
        }
        state.numSamples = state.totalSize / numReplicas;
        return state;
    }

    std::vector<std::vector<int64_t>> createBucketBatches(
        const BucketSamplerState& state,
        int64_t batchSize,
        int64_t numReplicas,
        int64_t rank,
        int64_t epoch,
        bool shuffle)
    {
        if (batchSize <= 0 || numReplicas <= 0)
        {
            throw std::invalid_argument("createBucketBatches requires positive batchSize and numReplicas.");
        }
        if (rank < 0 || rank >= numReplicas)
        {
            throw std::invalid_argument("createBucketBatches rank must be in [0, numReplicas).");
        }
        if (state.buckets.size() != state.numSamplesPerBucket.size())
        {
            throw std::invalid_argument("createBucketBatches received inconsistent sampler state.");
        }

        std::mt19937 generator(static_cast<uint32_t>(epoch));
        std::vector<std::vector<int64_t>> batches;

        for (size_t bucketIndexValue = 0; bucketIndexValue < state.buckets.size(); ++bucketIndexValue)
        {
            const auto& bucket = state.buckets[bucketIndexValue];
            if (bucket.empty())
            {
                continue;
            }

            std::vector<int64_t> order(bucket.size());
            std::iota(order.begin(), order.end(), 0);
            if (shuffle)
            {
                std::shuffle(order.begin(), order.end(), generator);
            }

            const auto numSamplesBucket = state.numSamplesPerBucket[bucketIndexValue];
            const auto originalSize = static_cast<int64_t>(order.size());
            const auto extra = numSamplesBucket - originalSize;
            for (int64_t i = 0; i < extra; ++i)
            {
                order.push_back(order[static_cast<size_t>(i % originalSize)]);
            }

            std::vector<int64_t> rankOrder;
            for (int64_t i = rank; i < static_cast<int64_t>(order.size()); i += numReplicas)
            {
                rankOrder.push_back(order[static_cast<size_t>(i)]);
            }

            for (int64_t start = 0; start + batchSize <= static_cast<int64_t>(rankOrder.size()); start += batchSize)
            {
                std::vector<int64_t> batch;
                batch.reserve(static_cast<size_t>(batchSize));
                for (int64_t j = 0; j < batchSize; ++j)
                {
                    const auto localIndex = rankOrder[static_cast<size_t>(start + j)];
                    batch.push_back(bucket[static_cast<size_t>(localIndex)]);
                }
                batches.push_back(batch);
            }
        }

        if (shuffle)
        {
            std::shuffle(batches.begin(), batches.end(), generator);
        }
        return batches;
    }
}
