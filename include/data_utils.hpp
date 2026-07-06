#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

namespace data_utils
{
    using FilepathAndText = std::vector<std::string>;

    struct TextAudioItem
    {
        torch::Tensor text;
        torch::Tensor spec;
        torch::Tensor wav;
    };

    struct TextAudioSpeakerItem
    {
        torch::Tensor text;
        torch::Tensor spec;
        torch::Tensor wav;
        int64_t speakerId;
    };

    struct TextAudioBatch
    {
        torch::Tensor textPadded;
        torch::Tensor textLengths;
        torch::Tensor specPadded;
        torch::Tensor specLengths;
        torch::Tensor wavPadded;
        torch::Tensor wavLengths;
        torch::Tensor idsSortedDecreasing;
    };

    struct TextAudioSpeakerBatch
    {
        torch::Tensor textPadded;
        torch::Tensor textLengths;
        torch::Tensor specPadded;
        torch::Tensor specLengths;
        torch::Tensor wavPadded;
        torch::Tensor wavLengths;
        torch::Tensor speakerIds;
        torch::Tensor idsSortedDecreasing;
    };

    struct BucketSamplerState
    {
        std::vector<int64_t> boundaries;
        std::vector<std::vector<int64_t>> buckets;
        std::vector<int64_t> numSamplesPerBucket;
        int64_t numSamples = 0;
        int64_t totalSize = 0;
    };

    struct AudioConfig
    {
        int64_t samplingRate = 22050;
        int64_t filterLength = 1024;
        int64_t hopLength = 256;
        int64_t winLength = 1024;
        bool useSpecCache = false;
    };

    struct AudioData
    {
        torch::Tensor spec;
        torch::Tensor wav;
    };

    std::vector<FilepathAndText> loadFilepathsAndText(
        const std::string& filename,
        const std::string& split = "|");

    std::vector<FilepathAndText> filterByTextLength(
        const std::vector<FilepathAndText>& filepathsAndText,
        size_t textIndex,
        int64_t minTextLen,
        int64_t maxTextLen);

    torch::Tensor textToTensor(
        const std::string& text,
        const std::vector<std::string>& cleaners,
        bool cleanedText,
        bool addBlank);

    AudioData getAudio(
        const std::string& audioPath,
        const AudioConfig& config);

    TextAudioItem makeTextAudioItem(
        const std::string& audioPath,
        const std::string& text,
        const AudioConfig& config,
        const std::vector<std::string>& cleaners,
        bool cleanedText,
        bool addBlank);

    TextAudioSpeakerItem makeTextAudioSpeakerItem(
        const std::string& audioPath,
        const std::string& text,
        int64_t speakerId,
        const AudioConfig& config,
        const std::vector<std::string>& cleaners,
        bool cleanedText,
        bool addBlank);

    TextAudioBatch collateTextAudio(
        const std::vector<TextAudioItem>& batch);

    TextAudioSpeakerBatch collateTextAudioSpeaker(
        const std::vector<TextAudioSpeakerItem>& batch);

    int64_t bucketIndex(
        int64_t length,
        const std::vector<int64_t>& boundaries);

    BucketSamplerState createBucketSamplerState(
        const std::vector<int64_t>& lengths,
        const std::vector<int64_t>& boundaries,
        int64_t batchSize,
        int64_t numReplicas);

    std::vector<std::vector<int64_t>> createBucketBatches(
        const BucketSamplerState& state,
        int64_t batchSize,
        int64_t numReplicas,
        int64_t rank,
        int64_t epoch,
        bool shuffle);
}
