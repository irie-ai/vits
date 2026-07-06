#include "data_utils.hpp"

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    void expectTrue(bool condition, const std::string& message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    void writeU16(std::ofstream& file, uint16_t value)
    {
        file.put(static_cast<char>(value & 0xff));
        file.put(static_cast<char>((value >> 8) & 0xff));
    }

    void writeU32(std::ofstream& file, uint32_t value)
    {
        file.put(static_cast<char>(value & 0xff));
        file.put(static_cast<char>((value >> 8) & 0xff));
        file.put(static_cast<char>((value >> 16) & 0xff));
        file.put(static_cast<char>((value >> 24) & 0xff));
    }

    void writeTestWav(const std::filesystem::path& path, uint32_t sampleRate)
    {
        const uint16_t channels = 1;
        const uint16_t bitsPerSample = 16;
        std::vector<int16_t> samples(64);
        for (size_t i = 0; i < samples.size(); ++i)
        {
            samples[i] = static_cast<int16_t>((static_cast<int64_t>(i) % 16 - 8) * 1024);
        }
        const uint32_t dataBytes = static_cast<uint32_t>(samples.size() * sizeof(int16_t));

        std::ofstream file(path, std::ios::binary);
        file.write("RIFF", 4);
        writeU32(file, 36 + dataBytes);
        file.write("WAVE", 4);
        file.write("fmt ", 4);
        writeU32(file, 16);
        writeU16(file, 1);
        writeU16(file, channels);
        writeU32(file, sampleRate);
        writeU32(file, sampleRate * channels * bitsPerSample / 8);
        writeU16(file, channels * bitsPerSample / 8);
        writeU16(file, bitsPerSample);
        file.write("data", 4);
        writeU32(file, dataBytes);
        for (const auto sample : samples)
        {
            writeU16(file, static_cast<uint16_t>(sample));
        }
    }

    void testLoadAndFilterFilelist()
    {
        const auto path = std::filesystem::temp_directory_path() / "vits_test_filelist.txt";
        {
            std::ofstream file(path);
            file << "a.wav|hello\n";
            file << "b.wav|x\n";
            file << "c.wav|this is long\n";
        }

        const auto rows = data_utils::loadFilepathsAndText(path.string());
        expectTrue(rows.size() == 3, "loadFilepathsAndText row count");
        expectTrue(rows[0].size() == 2, "loadFilepathsAndText column count");
        expectTrue(rows[0][0] == "a.wav" && rows[0][1] == "hello", "loadFilepathsAndText content");

        const auto filtered = data_utils::filterByTextLength(rows, 1, 2, 5);
        expectTrue(filtered.size() == 1, "filterByTextLength keeps matching rows");
        expectTrue(filtered[0][0] == "a.wav", "filterByTextLength row identity");
        std::filesystem::remove(path);
    }

    void testTextToTensor()
    {
        const auto plain = data_utils::textToTensor("Hi", {"basic_cleaners"}, false, false);
        const auto roundtrip = data_utils::textToTensor("hi", {}, true, true);
        expectTrue(plain.scalar_type() == torch::kLong, "textToTensor dtype");
        expectTrue(plain.size(0) == 2, "textToTensor cleaned length");
        expectTrue(roundtrip.size(0) == 5, "textToTensor addBlank intersperses");
        expectTrue(roundtrip.index({0}).item<int64_t>() == 0, "textToTensor addBlank starts with blank");
    }

    void testGetAudioAndItems()
    {
        const auto path = std::filesystem::temp_directory_path() / "vits_data_utils_audio.wav";
        writeTestWav(path, 22050);

        const data_utils::AudioConfig config{
            22050,
            16,
            4,
            16};
        const auto audio = data_utils::getAudio(path.string(), config);
        expectTrue(audio.wav.sizes() == torch::IntArrayRef({1, 64}), "getAudio wav shape");
        expectTrue(audio.spec.size(0) == 9, "getAudio spec channels");
        expectTrue(audio.spec.size(1) > 0, "getAudio spec frames");
        expectTrue(audio.wav.index({0, 0}).item<float>() == -8192.0f / 32768.0f, "getAudio normalizes wav");

        const auto item = data_utils::makeTextAudioItem(path.string(), "Hi", config, {"basic_cleaners"}, false, true);
        expectTrue(item.text.size(0) == 5, "makeTextAudioItem text shape");
        expectTrue(item.wav.sizes() == torch::IntArrayRef({1, 64}), "makeTextAudioItem wav shape");
        expectTrue(item.spec.size(0) == 9, "makeTextAudioItem spec shape");

        const auto speakerItem = data_utils::makeTextAudioSpeakerItem(path.string(), "Hi", 3, config, {"basic_cleaners"}, false, false);
        expectTrue(speakerItem.speakerId == 3, "makeTextAudioSpeakerItem speaker id");
        expectTrue(speakerItem.text.size(0) == 2, "makeTextAudioSpeakerItem text shape");
        std::filesystem::remove(path);
    }

    void testGetAudioSpecCache()
    {
        const auto path = std::filesystem::temp_directory_path() / "vits_data_utils_cached.wav";
        const auto cachePath = path.string() + ".spec.pt";
        writeTestWav(path, 22050);

        const data_utils::AudioConfig config{
            22050,
            16,
            4,
            16,
            true};

        const auto first = data_utils::getAudio(path.string(), config);
        expectTrue(std::filesystem::exists(cachePath), "getAudio creates spec cache");

        auto cachedSpec = torch::full_like(first.spec, 7.0);
        torch::save(cachedSpec, cachePath);

        const auto second = data_utils::getAudio(path.string(), config);
        expectTrue(torch::allclose(second.spec, cachedSpec), "getAudio loads spec cache");
        expectTrue(second.wav.sizes() == torch::IntArrayRef({1, 64}), "getAudio cache still loads wav");

        std::filesystem::remove(path);
        std::filesystem::remove(cachePath);
    }

    void testTextAudioCollate()
    {
        const std::vector<data_utils::TextAudioItem> batch = {
            {
                torch::tensor({1, 2, 3}, torch::kLong),
                torch::ones({3, 4}, torch::kFloat32),
                torch::ones({1, 8}, torch::kFloat32) * 2.0f,
            },
            {
                torch::tensor({4, 5}, torch::kLong),
                torch::ones({3, 6}, torch::kFloat32) * 3.0f,
                torch::ones({1, 10}, torch::kFloat32) * 4.0f,
            }};

        const auto collated = data_utils::collateTextAudio(batch);
        expectTrue(collated.idsSortedDecreasing.equal(torch::tensor({1, 0}, torch::kLong)), "collate sort ids");
        expectTrue(collated.textPadded.sizes() == torch::IntArrayRef({2, 3}), "collate text shape");
        expectTrue(collated.specPadded.sizes() == torch::IntArrayRef({2, 3, 6}), "collate spec shape");
        expectTrue(collated.wavPadded.sizes() == torch::IntArrayRef({2, 1, 10}), "collate wav shape");
        expectTrue(collated.textLengths.equal(torch::tensor({2, 3}, torch::kLong)), "collate text lengths");
        expectTrue(collated.specLengths.equal(torch::tensor({6, 4}, torch::kLong)), "collate spec lengths");
        expectTrue(collated.wavLengths.equal(torch::tensor({10, 8}, torch::kLong)), "collate wav lengths");
        expectTrue(collated.textPadded.index({0, 0}).item<int64_t>() == 4, "collate sorted text content");
        expectTrue(collated.textPadded.index({0, 2}).item<int64_t>() == 0, "collate text padding");
        expectTrue(collated.specPadded.index({1, 0, 4}).item<float>() == 0.0f, "collate spec padding");
    }

    void testTextAudioSpeakerCollate()
    {
        const std::vector<data_utils::TextAudioSpeakerItem> batch = {
            {
                torch::tensor({1}, torch::kLong),
                torch::ones({2, 3}, torch::kFloat32),
                torch::ones({1, 5}, torch::kFloat32),
                7,
            },
            {
                torch::tensor({2, 3}, torch::kLong),
                torch::ones({2, 5}, torch::kFloat32),
                torch::ones({1, 9}, torch::kFloat32),
                4,
            }};

        const auto collated = data_utils::collateTextAudioSpeaker(batch);
        expectTrue(collated.idsSortedDecreasing.equal(torch::tensor({1, 0}, torch::kLong)), "speaker collate sort ids");
        expectTrue(collated.speakerIds.equal(torch::tensor({4, 7}, torch::kLong)), "speaker collate ids");
        expectTrue(collated.specPadded.sizes() == torch::IntArrayRef({2, 2, 5}), "speaker collate spec shape");
    }

    void testBucketSampler()
    {
        const std::vector<int64_t> boundaries = {0, 5, 10, 20};
        expectTrue(data_utils::bucketIndex(1, boundaries) == 0, "bucketIndex first bucket");
        expectTrue(data_utils::bucketIndex(5, boundaries) == 0, "bucketIndex right inclusive");
        expectTrue(data_utils::bucketIndex(6, boundaries) == 1, "bucketIndex second bucket");
        expectTrue(data_utils::bucketIndex(21, boundaries) == -1, "bucketIndex out of range");

        const std::vector<int64_t> lengths = {3, 4, 6, 9, 12};
        const auto state = data_utils::createBucketSamplerState(lengths, boundaries, 2, 2);
        expectTrue(state.buckets.size() == 3, "createBucketSamplerState bucket count");
        expectTrue(state.numSamplesPerBucket == std::vector<int64_t>({4, 4, 4}), "createBucketSamplerState padded samples");
        expectTrue(state.totalSize == 12, "createBucketSamplerState total size");
        expectTrue(state.numSamples == 6, "createBucketSamplerState rank sample count");

        const auto rank0 = data_utils::createBucketBatches(state, 2, 2, 0, 0, false);
        const auto rank1 = data_utils::createBucketBatches(state, 2, 2, 1, 0, false);
        expectTrue(rank0.size() == 3, "createBucketBatches rank0 batch count");
        expectTrue(rank1.size() == 3, "createBucketBatches rank1 batch count");
        expectTrue(rank0[0] == std::vector<int64_t>({0, 0}), "createBucketBatches rank0 padding");
        expectTrue(rank1[0] == std::vector<int64_t>({1, 1}), "createBucketBatches rank1 padding");

        const auto shuffledA = data_utils::createBucketBatches(state, 2, 1, 0, 123, true);
        const auto shuffledB = data_utils::createBucketBatches(state, 2, 1, 0, 123, true);
        expectTrue(shuffledA == shuffledB, "createBucketBatches deterministic shuffle");
    }
}

int main()
{
    try
    {
        testLoadAndFilterFilelist();
        testTextToTensor();
        testGetAudioAndItems();
        testGetAudioSpecCache();
        testTextAudioCollate();
        testTextAudioSpeakerCollate();
        testBucketSampler();
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_data_utils failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "test_data_utils passed\n";
    return EXIT_SUCCESS;
}
