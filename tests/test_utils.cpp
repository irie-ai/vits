#include "utils.hpp"

#include <torch/torch.h>

#include <cstdint>
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

    void writeTestWav(const std::filesystem::path& path)
    {
        const uint16_t channels = 1;
        const uint32_t sampleRate = 22050;
        const uint16_t bitsPerSample = 16;
        const std::vector<int16_t> samples = {-32768, 0, 32767};
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

    void testParseJson()
    {
        const auto value = utils::parseJson(
            "{\"train\":{\"batch_size\":16,\"fp16\":false},\"symbols\":[\"a\",\"b\"],\"name\":\"vits\"}");
        expectTrue(value.isObject(), "parseJson root object");
        expectTrue(value.at("train").at("batch_size").asInt() == 16, "parseJson nested number");
        expectTrue(!value.at("train").at("fp16").asBool(), "parseJson nested bool");
        expectTrue(value.at("symbols").asArray().size() == 2, "parseJson array");
        expectTrue(value.at("name").asString() == "vits", "parseJson string");
    }

    void testHParamsFromFileAndDir()
    {
        const auto dir = std::filesystem::temp_directory_path() / "vits_utils_hparams";
        std::filesystem::create_directories(dir);
        const auto configPath = dir / "config.json";
        {
            std::ofstream file(configPath);
            file << "{\"data\":{\"sampling_rate\":22050},\"model\":{\"hidden_channels\":192}}";
        }

        const auto fromFile = utils::getHParamsFromFile(configPath.string());
        expectTrue(fromFile.at("data").at("sampling_rate").asInt() == 22050, "getHParamsFromFile reads config");

        const auto fromDir = utils::getHParamsFromDir(dir.string());
        expectTrue(fromDir.modelDir == dir.string(), "getHParamsFromDir sets modelDir");
        expectTrue(fromDir.at("model").at("hidden_channels").asInt() == 192, "getHParamsFromDir reads config");
        std::filesystem::remove_all(dir);
    }

    void testLatestCheckpointPath()
    {
        const auto dir = std::filesystem::temp_directory_path() / "vits_utils_checkpoints";
        std::filesystem::create_directories(dir);
        {
            std::ofstream(dir / "G_10.pth").put('a');
            std::ofstream(dir / "G_2.pth").put('b');
            std::ofstream(dir / "D_100.pth").put('c');
        }

        const auto latest = utils::latestCheckpointPath(dir.string(), "G_*.pth");
        expectTrue(std::filesystem::path(latest).filename().string() == "G_10.pth", "latestCheckpointPath numeric sort");
        std::filesystem::remove_all(dir);
    }

    void testCheckpointRoundtrip()
    {
        const auto path = std::filesystem::temp_directory_path() / "vits_utils_checkpoint.pt";
        torch::nn::Linear model(torch::nn::LinearOptions(2, 1));
        {
            torch::NoGradGuard guard;
            model->weight.fill_(2.0);
            model->bias.fill_(3.0);
        }

        utils::saveCheckpoint(*model, path.string(), 42, 0.0002);

        {
            torch::NoGradGuard guard;
            model->weight.fill_(-1.0);
            model->bias.fill_(-1.0);
        }

        const auto info = utils::loadCheckpoint(*model, path.string());
        expectTrue(info.iteration == 42, "loadCheckpoint iteration");
        expectTrue(info.learningRate == 0.0002, "loadCheckpoint learning rate");
        expectTrue(!info.hasOptimizer, "loadCheckpoint optimizer flag");
        expectTrue(torch::allclose(model->weight, torch::full_like(model->weight, 2.0)), "loadCheckpoint restores weight");
        expectTrue(torch::allclose(model->bias, torch::full_like(model->bias, 3.0)), "loadCheckpoint restores bias");
        std::filesystem::remove(path);
    }

    void testLoadWavToTorch()
    {
        const auto path = std::filesystem::temp_directory_path() / "vits_utils_test.wav";
        writeTestWav(path);

        const auto [wav, samplingRate] = utils::loadWavToTorch(path.string());
        expectTrue(samplingRate == 22050, "loadWavToTorch sampling rate");
        expectTrue(wav.scalar_type() == torch::kFloat32, "loadWavToTorch dtype");
        expectTrue(wav.sizes() == torch::IntArrayRef({3}), "loadWavToTorch mono shape");
        expectTrue(wav.index({0}).item<float>() == -32768.0f, "loadWavToTorch first sample");
        expectTrue(wav.index({2}).item<float>() == 32767.0f, "loadWavToTorch last sample");
        std::filesystem::remove(path);
    }

    void testSaveWavFromTorch()
    {
        const auto path = std::filesystem::temp_directory_path() / "vits_utils_saved.wav";
        const auto audio = torch::tensor({-1.0f, 0.0f, 1.0f}, torch::TensorOptions().dtype(torch::kFloat32));

        utils::saveWavFromTorch(audio, path.string(), 16000, 32767.0);

        const auto [wav, samplingRate] = utils::loadWavToTorch(path.string());
        expectTrue(samplingRate == 16000, "saveWavFromTorch sampling rate");
        expectTrue(wav.sizes() == torch::IntArrayRef({3}), "saveWavFromTorch mono shape");
        expectTrue(wav.index({0}).item<float>() == -32767.0f, "saveWavFromTorch first sample");
        expectTrue(wav.index({1}).item<float>() == 0.0f, "saveWavFromTorch middle sample");
        expectTrue(wav.index({2}).item<float>() == 32767.0f, "saveWavFromTorch last sample");
        std::filesystem::remove(path);
    }

    void testPlotSpectrogramToTensor()
    {
        const auto spectrogram = torch::tensor(
            {{0.0f, 1.0f, 2.0f}, {3.0f, 4.0f, 5.0f}},
            torch::TensorOptions().dtype(torch::kFloat32));

        const auto image = utils::plotSpectrogramToTensor(spectrogram);

        expectTrue(image.scalar_type() == torch::kUInt8, "plotSpectrogramToTensor dtype");
        expectTrue(image.sizes() == torch::IntArrayRef({2, 3, 3}), "plotSpectrogramToTensor HWC shape");
        expectTrue(image.min().item<uint8_t>() >= 0, "plotSpectrogramToTensor min range");
        expectTrue(image.max().item<uint8_t>() <= 255, "plotSpectrogramToTensor max range");
    }

    void testPlotAlignmentToTensor()
    {
        const auto alignment = torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32));

        const auto image = utils::plotAlignmentToTensor(alignment);

        expectTrue(image.scalar_type() == torch::kUInt8, "plotAlignmentToTensor dtype");
        expectTrue(image.sizes() == torch::IntArrayRef({3, 3, 3}), "plotAlignmentToTensor HWC shape");
        expectTrue(image.max().item<uint8_t>() > image.min().item<uint8_t>(), "plotAlignmentToTensor has contrast");
    }

    void testFileSummaryWriter()
    {
        const auto dir = std::filesystem::temp_directory_path() / "vits_summary_writer";
        std::filesystem::remove_all(dir);

        utils::FileSummaryWriter writer(dir.string());
        utils::SummaryData data;
        data.scalars["loss/gen"] = 1.25;
        data.images["spec/train"] = utils::plotSpectrogramToTensor(torch::ones({2, 3}, torch::kFloat32));
        data.audios["audio/train"] = {
            torch::tensor({0.0f, 0.5f, -0.5f}, torch::TensorOptions().dtype(torch::kFloat32)),
            22050};

        utils::summarize(writer, 7, data);

        const auto scalarPath = dir / "scalars.csv";
        const auto imagePath = dir / "images" / "spec_train_7.ppm";
        const auto audioPath = dir / "audios" / "audio_train_7.wav";

        expectTrue(std::filesystem::exists(scalarPath), "summarize writes scalar csv");
        expectTrue(std::filesystem::exists(imagePath), "summarize writes image ppm");
        expectTrue(std::filesystem::exists(audioPath), "summarize writes audio wav");

        {
            std::ifstream scalarFile(scalarPath);
            std::string scalarLine;
            std::getline(scalarFile, scalarLine);
            expectTrue(scalarLine == "7,loss_gen,1.25", "summarize scalar content");
        }

        {
            std::ifstream imageFile(imagePath, std::ios::binary);
            std::string imageMagic;
            std::getline(imageFile, imageMagic);
            expectTrue(imageMagic == "P6", "summarize image ppm header");
        }

        const auto [audio, samplingRate] = utils::loadWavToTorch(audioPath.string());
        expectTrue(samplingRate == 22050, "summarize audio sample rate");
        expectTrue(audio.sizes() == torch::IntArrayRef({3}), "summarize audio shape");

        std::filesystem::remove_all(dir);
    }
}

int main()
{
    try
    {
        testParseJson();
        testHParamsFromFileAndDir();
        testLatestCheckpointPath();
        testCheckpointRoundtrip();
        testLoadWavToTorch();
        testSaveWavFromTorch();
        testPlotSpectrogramToTensor();
        testPlotAlignmentToTensor();
        testFileSummaryWriter();
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_utils failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "test_utils passed\n";
    return EXIT_SUCCESS;
}
