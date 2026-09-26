#include "commons.hpp"
#include "data_utils.hpp"
#include "inference.hpp"
#include "losses.hpp"
#include "mel_processing.hpp"
#include "models.hpp"
#include "utils.hpp"

#include <torch/torch.h>

#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace
{
    struct Options
    {
        std::string datasetDir;
        std::string metadataPath;
        std::string configPath;
        int64_t batchSize = 0;
        int64_t maxItems = 1;
        int64_t steps = 1;
        int64_t epochs = 0;
        int64_t startEpoch = 1;
        int64_t logInterval = 1;
        int64_t inferMaxLength = 64;
        int64_t torchThreads = 0;
        double learningRate = 0.0;
        double lrDecay = 0.0;
        std::string inferText = "This is a VITS training smoke inference.";
        std::string inferOutputPath;
        std::string saveModelPath;
        std::string saveConfigPath;
        bool saveOptimizer = false;
        bool useSpecCache = false;
        bool phaseLog = false;
        bool useBucketSampler = false;
        bool shuffleBuckets = true;
        bool stepsProvided = false;
        std::vector<int64_t> bucketBoundaries;
    };

    struct TrainStepResult
    {
        float dLoss = 0.0f;
        float gLoss = 0.0f;
        float melLoss = 0.0f;
        float waveformL1 = 0.0f;
        float klLoss = 0.0f;
    };

    struct LjsEntry
    {
        std::string audioPath;
        std::string text;
        int64_t specLength = 0;
    };

    void printUsage()
    {
        std::cerr
            << "usage: ljs_train_smoke --dataset PATH [--metadata PATH] [--config PATH]\n"
            << "                       [--batch-size N] [--max-items N] [--steps N]\n"
            << "                       [--epochs N] [--start-epoch N]\n"
            << "                       [--log-interval N] [--lr VALUE] [--lr-decay VALUE]\n"
            << "                       [--torch-threads N] [--phase-log]\n"
            << "                       [--save-model PATH] [--save-config PATH]\n"
            << "                       [--save-optimizer]\n"
            << "                       [--use-bucket-sampler] [--no-shuffle-buckets]\n"
            << "                       [--bucket-boundaries CSV]\n"
            << "                       [--infer-output PATH] [--infer-text TEXT]\n"
            << "                       [--infer-max-length N]\n"
            << "                       [--use-spec-cache]\n";
    }

    int64_t parseInt(const char* value, const std::string& name)
    {
        try
        {
            return std::stoll(value);
        }
        catch (const std::exception&)
        {
            throw std::invalid_argument("Invalid integer for " + name + ": " + value);
        }
    }

    double parseDouble(const char* value, const std::string& name)
    {
        try
        {
            return std::stod(value);
        }
        catch (const std::exception&)
        {
            throw std::invalid_argument("Invalid number for " + name + ": " + value);
        }
    }

    std::vector<int64_t> parseIntList(const std::string& value, const std::string& name)
    {
        std::vector<int64_t> result;
        std::stringstream stream(value);
        std::string token;
        while (std::getline(stream, token, ','))
        {
            token.erase(
                std::remove_if(token.begin(), token.end(), [](unsigned char c) { return std::isspace(c) != 0; }),
                token.end());
            if (!token.empty())
            {
                result.push_back(parseInt(token.c_str(), name));
            }
        }
        if (result.size() < 2)
        {
            throw std::invalid_argument(name + " requires at least two comma-separated boundaries.");
        }
        for (size_t i = 1; i < result.size(); ++i)
        {
            if (result[i - 1] >= result[i])
            {
                throw std::invalid_argument(name + " must be strictly increasing.");
            }
        }
        return result;
    }

    Options parseArgs(int argc, char** argv)
    {
        Options options;
        for (int index = 1; index < argc; ++index)
        {
            const std::string arg = argv[index];
            if (arg == "--help" || arg == "-h")
            {
                printUsage();
                std::exit(EXIT_SUCCESS);
            }
            if (arg == "--use-spec-cache")
            {
                options.useSpecCache = true;
                continue;
            }
            if (arg == "--save-optimizer")
            {
                options.saveOptimizer = true;
                continue;
            }
            if (arg == "--phase-log")
            {
                options.phaseLog = true;
                continue;
            }
            if (arg == "--use-bucket-sampler")
            {
                options.useBucketSampler = true;
                continue;
            }
            if (arg == "--no-shuffle-buckets")
            {
                options.shuffleBuckets = false;
                continue;
            }
            if (index + 1 >= argc)
            {
                throw std::invalid_argument("Missing value for argument: " + arg);
            }

            const auto value = argv[++index];
            if (arg == "--dataset")
            {
                options.datasetDir = value;
            }
            else if (arg == "--metadata")
            {
                options.metadataPath = value;
            }
            else if (arg == "--config")
            {
                options.configPath = value;
            }
            else if (arg == "--batch-size")
            {
                options.batchSize = parseInt(value, arg);
            }
            else if (arg == "--max-items")
            {
                options.maxItems = parseInt(value, arg);
            }
            else if (arg == "--steps")
            {
                options.steps = parseInt(value, arg);
                options.stepsProvided = true;
            }
            else if (arg == "--epochs")
            {
                options.epochs = parseInt(value, arg);
            }
            else if (arg == "--start-epoch")
            {
                options.startEpoch = parseInt(value, arg);
            }
            else if (arg == "--log-interval")
            {
                options.logInterval = parseInt(value, arg);
            }
            else if (arg == "--infer-max-length")
            {
                options.inferMaxLength = parseInt(value, arg);
            }
            else if (arg == "--torch-threads")
            {
                options.torchThreads = parseInt(value, arg);
            }
            else if (arg == "--lr")
            {
                options.learningRate = parseDouble(value, arg);
            }
            else if (arg == "--lr-decay")
            {
                options.lrDecay = parseDouble(value, arg);
            }
            else if (arg == "--infer-text")
            {
                options.inferText = value;
            }
            else if (arg == "--infer-output")
            {
                options.inferOutputPath = value;
            }
            else if (arg == "--save-model")
            {
                options.saveModelPath = value;
            }
            else if (arg == "--save-config")
            {
                options.saveConfigPath = value;
            }
            else if (arg == "--bucket-boundaries")
            {
                options.bucketBoundaries = parseIntList(value, arg);
            }
            else
            {
                throw std::invalid_argument("Unknown argument: " + arg);
            }
        }

        if (options.datasetDir.empty())
        {
            throw std::invalid_argument("--dataset is required.");
        }
        if (options.metadataPath.empty())
        {
            options.metadataPath = (std::filesystem::path(options.datasetDir) / "metadata.csv").string();
        }
        if (options.batchSize < 0 ||
            options.maxItems <= 0 ||
            options.steps <= 0 ||
            options.epochs < 0 ||
            options.startEpoch <= 0 ||
            options.logInterval <= 0 ||
            options.inferMaxLength <= 0 ||
            options.torchThreads < 0 ||
            options.learningRate < 0.0 ||
            options.lrDecay < 0.0)
        {
            throw std::invalid_argument("batch-size/lr/lr-decay must be positive when provided; max-items, steps, epochs, start-epoch, log-interval, and infer-max-length must be positive.");
        }
        return options;
    }

    const char* defaultSmokeJson()
    {
        return R"({
            "data": {
                "training_files": "filelists/ljs_audio_text_train_filelist.txt.cleaned",
                "validation_files": "filelists/ljs_audio_text_val_filelist.txt.cleaned",
                "text_cleaners": ["english_cleaners2"],
                "max_wav_value": 32768.0,
                "filter_length": 1024,
                "hop_length": 256,
                "win_length": 1024,
                "sampling_rate": 22050,
                "n_mel_channels": 80,
                "mel_fmin": 0.0,
                "mel_fmax": null,
                "add_blank": true,
                "n_speakers": 0,
                "cleaned_text": true
            },
            "train": {
                "log_interval": 200,
                "eval_interval": 1000,
                "seed": 1234,
                "epochs": 20000,
                "learning_rate": 2e-4,
                "betas": [0.8, 0.99],
                "eps": 1e-9,
                "batch_size": 64,
                "fp16_run": true,
                "lr_decay": 0.999875,
                "segment_size": 8192,
                "init_lr_ratio": 1,
                "warmup_epochs": 0,
                "c_mel": 45.0,
                "c_kl": 1.0
            },
            "model": {
                "inter_channels": 192,
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0.1,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [8, 8, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16, 16, 4, 4],
                "n_layers_q": 3,
                "use_spectral_norm": false
            }
        })";
    }

    utils::HParams defaultSmokeHParams()
    {
        utils::HParams hparams;
        hparams.root = utils::parseJson(defaultSmokeJson());
        return hparams;
    }

    void saveConfig(
        const std::string& outputPath,
        const std::string& sourceConfigPath)
    {
        const auto path = std::filesystem::path(outputPath);
        if (!path.parent_path().empty())
        {
            std::filesystem::create_directories(path.parent_path());
        }

        if (!sourceConfigPath.empty())
        {
            std::filesystem::copy_file(
                sourceConfigPath,
                path,
                std::filesystem::copy_options::overwrite_existing);
            return;
        }

        std::ofstream file(path);
        if (!file)
        {
            throw std::runtime_error("Could not open config file for writing: " + path.string());
        }
        file << defaultSmokeJson() << '\n';
    }

    int64_t getInt(
        const utils::HParamValue& object,
        const std::string& key,
        int64_t fallback)
    {
        return object.contains(key) ? object.at(key).asInt() : fallback;
    }

    double getNumber(
        const utils::HParamValue& object,
        const std::string& key,
        double fallback)
    {
        if (!object.contains(key) || object.at(key).isNull())
        {
            return fallback;
        }
        return object.at(key).asNumber();
    }

    std::tuple<double, double> getBetas(
        const utils::HParamValue& object,
        const std::string& key,
        std::tuple<double, double> fallback)
    {
        if (!object.contains(key))
        {
            return fallback;
        }
        const auto& values = object.at(key).asArray();
        if (values.size() != 2)
        {
            throw std::invalid_argument("train.betas must contain exactly two numbers.");
        }
        return {values[0].asNumber(), values[1].asNumber()};
    }

    data_utils::AudioConfig audioConfigFromHParams(
        const utils::HParams& hparams,
        bool useSpecCache)
    {
        const auto& data = hparams.at("data");
        return {
            getInt(data, "sampling_rate", 22050),
            data.at("filter_length").asInt(),
            data.at("hop_length").asInt(),
            getInt(data, "win_length", data.at("filter_length").asInt()),
            useSpecCache};
    }

    int64_t segmentFramesFromHParams(const utils::HParams& hparams)
    {
        const auto segmentSize = hparams.at("train").at("segment_size").asInt();
        const auto hopLength = hparams.at("data").at("hop_length").asInt();
        if (hopLength <= 0 || segmentSize <= 0)
        {
            throw std::invalid_argument("train.segment_size and data.hop_length must be positive.");
        }
        return segmentSize / hopLength;
    }

    int64_t segmentSamplesFromHParams(const utils::HParams& hparams)
    {
        return hparams.at("train").at("segment_size").asInt();
    }

    std::vector<int64_t> defaultBucketBoundaries(int64_t minSpecFrames)
    {
        const std::vector<int64_t> base = {32, 300, 400, 500, 600, 700, 800, 900, 1000};
        std::vector<int64_t> boundaries = {std::max<int64_t>(32, minSpecFrames)};
        for (const auto value : base)
        {
            if (value > boundaries.back())
            {
                boundaries.push_back(value);
            }
        }
        if (boundaries.size() < 2)
        {
            boundaries.push_back(boundaries.back() + 100);
        }
        return boundaries;
    }

    uint32_t readU32(std::ifstream& file)
    {
        unsigned char bytes[4] = {};
        file.read(reinterpret_cast<char*>(bytes), 4);
        if (!file)
        {
            throw std::runtime_error("Unexpected end of wav header.");
        }
        return static_cast<uint32_t>(bytes[0]) |
            (static_cast<uint32_t>(bytes[1]) << 8) |
            (static_cast<uint32_t>(bytes[2]) << 16) |
            (static_cast<uint32_t>(bytes[3]) << 24);
    }

    uint16_t readU16(std::ifstream& file)
    {
        unsigned char bytes[2] = {};
        file.read(reinterpret_cast<char*>(bytes), 2);
        if (!file)
        {
            throw std::runtime_error("Unexpected end of wav header.");
        }
        return static_cast<uint16_t>(bytes[0]) |
            (static_cast<uint16_t>(bytes[1]) << 8);
    }

    void skipBytes(std::ifstream& file, uint32_t bytes)
    {
        file.seekg(bytes, std::ios::cur);
        if (!file)
        {
            throw std::runtime_error("Could not skip wav chunk.");
        }
    }

    int64_t readWavSampleCount(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Could not open wav file: " + path);
        }

        char riff[4] = {};
        char wave[4] = {};
        file.read(riff, 4);
        readU32(file);
        file.read(wave, 4);
        if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE")
        {
            throw std::runtime_error("Unsupported wav file: " + path);
        }

        int64_t numChannels = 0;
        int64_t bitsPerSample = 0;
        int64_t dataBytes = -1;
        while (file && dataBytes < 0)
        {
            char chunkId[4] = {};
            file.read(chunkId, 4);
            if (!file)
            {
                break;
            }
            const auto chunkSize = readU32(file);
            const std::string id(chunkId, 4);
            if (id == "fmt ")
            {
                if (chunkSize < 16)
                {
                    throw std::runtime_error("Invalid wav fmt chunk: " + path);
                }
                readU16(file);
                numChannels = readU16(file);
                readU32(file);
                readU32(file);
                readU16(file);
                bitsPerSample = readU16(file);
                if (chunkSize > 16)
                {
                    skipBytes(file, chunkSize - 16);
                }
            }
            else if (id == "data")
            {
                dataBytes = chunkSize;
                skipBytes(file, chunkSize);
            }
            else
            {
                skipBytes(file, chunkSize);
            }

            if (chunkSize % 2 == 1)
            {
                skipBytes(file, 1);
            }
        }

        if (numChannels <= 0 || bitsPerSample <= 0 || dataBytes < 0)
        {
            throw std::runtime_error("Could not read wav length from: " + path);
        }
        const auto bytesPerSample = (bitsPerSample / 8) * numChannels;
        if (bytesPerSample <= 0)
        {
            throw std::runtime_error("Invalid wav sample size for: " + path);
        }
        return dataBytes / bytesPerSample;
    }

    torch::Tensor gradNorm(const std::vector<torch::Tensor>& parameters)
    {
        auto total = torch::zeros({}, torch::kFloat32);
        for (const auto& parameter : parameters)
        {
            if (parameter.grad().defined())
            {
                total = total + parameter.grad().detach().abs().sum().to(torch::kFloat32);
            }
        }
        return total;
    }

    void setOptimizerLearningRate(torch::optim::Optimizer& optimizer, double learningRate)
    {
        for (auto& group : optimizer.param_groups())
        {
            auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
            options.lr(learningRate);
        }
    }

    double optimizerLearningRate(const torch::optim::Optimizer& optimizer)
    {
        if (optimizer.param_groups().empty())
        {
            throw std::runtime_error("optimizer has no parameter groups.");
        }
        const auto& options = static_cast<const torch::optim::AdamWOptions&>(
            optimizer.param_groups().front().options());
        return options.lr();
    }

    void setRequiresGrad(torch::nn::Module& module, bool requiresGrad)
    {
        for (auto& parameter : module.parameters())
        {
            parameter.set_requires_grad(requiresGrad);
        }
    }

    std::vector<LjsEntry> loadLjsEntries(
        const Options& options,
        int64_t batchSize,
        int64_t hopLength)
    {
        if (hopLength <= 0)
        {
            throw std::invalid_argument("hopLength must be positive.");
        }
        const auto metadata = data_utils::loadFilepathsAndText(options.metadataPath, "|");
        const auto datasetDir = std::filesystem::path(options.datasetDir);
        std::vector<LjsEntry> entries;
        entries.reserve(static_cast<size_t>(options.maxItems));

        for (const auto& row : metadata)
        {
            if (row.size() < 2)
            {
                continue;
            }

            const auto audioPath = datasetDir / "wavs" / (row[0] + ".wav");
            const auto& text = row.size() >= 3 ? row[2] : row[1];
            if (!std::filesystem::exists(audioPath))
            {
                throw std::runtime_error("LJSpeech wav was not found: " + audioPath.string());
            }

            const auto sampleCount = readWavSampleCount(audioPath.string());
            entries.push_back({audioPath.string(), text, sampleCount / hopLength});
            if (static_cast<int64_t>(entries.size()) >= options.maxItems)
            {
                break;
            }
        }

        if (static_cast<int64_t>(entries.size()) < batchSize)
        {
            throw std::runtime_error("Not enough LJSpeech entries for the requested batch.");
        }
        return entries;
    }

    data_utils::TextAudioBatch makeBatchForStep(
        const std::vector<LjsEntry>& entries,
        const data_utils::AudioConfig& audioConfig,
        const std::vector<std::string>& cleaners,
        bool cleanedText,
        bool addBlank,
        int64_t minSpecFrames,
        int64_t batchSize,
        int64_t step)
    {
        if (entries.empty())
        {
            throw std::invalid_argument("makeBatchForStep requires at least one item.");
        }

        std::vector<data_utils::TextAudioItem> batchItems;
        batchItems.reserve(static_cast<size_t>(batchSize));
        const auto itemCount = static_cast<int64_t>(entries.size());
        const auto start = ((step - 1) * batchSize) % itemCount;
        for (int64_t offset = 0; static_cast<int64_t>(batchItems.size()) < batchSize && offset < itemCount; ++offset)
        {
            const auto& entry = entries[static_cast<size_t>((start + offset) % itemCount)];
            auto item = data_utils::makeTextAudioItem(
                entry.audioPath,
                entry.text,
                audioConfig,
                cleaners,
                cleanedText,
                addBlank);
            if (item.spec.size(1) >= minSpecFrames)
            {
                batchItems.push_back(std::move(item));
            }
        }
        if (static_cast<int64_t>(batchItems.size()) < batchSize)
        {
            throw std::runtime_error("Not enough usable LJSpeech items for the requested batch.");
        }
        return data_utils::collateTextAudio(batchItems);
    }

    data_utils::TextAudioBatch makeBatchFromIndices(
        const std::vector<LjsEntry>& entries,
        const std::vector<int64_t>& indices,
        const data_utils::AudioConfig& audioConfig,
        const std::vector<std::string>& cleaners,
        bool cleanedText,
        bool addBlank,
        int64_t minSpecFrames)
    {
        if (indices.empty())
        {
            throw std::invalid_argument("makeBatchFromIndices requires at least one item.");
        }

        std::vector<data_utils::TextAudioItem> batchItems;
        batchItems.reserve(indices.size());
        for (const auto index : indices)
        {
            const auto& entry = entries.at(static_cast<size_t>(index));
            auto item = data_utils::makeTextAudioItem(
                entry.audioPath,
                entry.text,
                audioConfig,
                cleaners,
                cleanedText,
                addBlank);
            if (item.spec.size(1) < minSpecFrames)
            {
                throw std::runtime_error("Bucket sampler selected an item shorter than train.segment_size.");
            }
            batchItems.push_back(std::move(item));
        }
        return data_utils::collateTextAudio(batchItems);
    }

    TrainStepResult runTrainStep(
        models::SynthesizerTrn& synthesizer,
        models::MultiPeriodDiscriminator& discriminator,
        torch::optim::AdamW& optimG,
        torch::optim::AdamW& optimD,
        const data_utils::TextAudioBatch& batch,
        const utils::HParams& hparams,
        int64_t hopLength,
        int64_t segmentSamples,
        bool phaseLog = false,
        int64_t step = 0)
    {
        synthesizer->train();
        discriminator->train();

        if (phaseLog)
        {
            std::cout << "phase: " << step << " generator_forward_start" << std::endl;
        }
        auto gForward = synthesizer->forward(
            batch.textPadded,
            batch.textLengths,
            batch.specPadded,
            batch.specLengths);
        auto idsSliceSamples = gForward.idsSlice * hopLength;
        auto yReal = commons::sliceSegments(batch.wavPadded, idsSliceSamples, segmentSamples);

        if (phaseLog)
        {
            std::cout << "phase: " << step << " d_forward_start" << std::endl;
        }
        optimD.zero_grad();
        auto dOutputs = discriminator->forward(yReal, gForward.audio.detach());
        auto dLoss = std::get<0>(losses::discriminatorLoss(std::get<0>(dOutputs), std::get<1>(dOutputs)));
        if (phaseLog)
        {
            std::cout << "phase: " << step << " d_backward_start" << std::endl;
        }
        dLoss.backward();
        if (gradNorm(discriminator->parameters()).item<float>() <= 0.0f)
        {
            throw std::runtime_error("discriminator did not receive gradients.");
        }
        optimD.step();

        if (phaseLog)
        {
            std::cout << "phase: " << step << " g_discriminator_forward_start" << std::endl;
        }
        setRequiresGrad(*discriminator, false);
        optimG.zero_grad();
        auto gOutputs = discriminator->forward(yReal, gForward.audio);
        auto gAdversarial = losses::generatorLoss(std::get<1>(gOutputs)).first;
        auto fmapLoss = losses::featureLoss(std::get<2>(gOutputs), std::get<3>(gOutputs));
        const auto& data = hparams.at("data");
        const auto& train = hparams.at("train");
        const auto filterLength = data.at("filter_length").asInt();
        const auto winLength = getInt(data, "win_length", filterLength);
        const auto samplingRate = getInt(data, "sampling_rate", 22050);
        const auto nMelChannels = getInt(data, "n_mel_channels", 80);
        const auto melFMin = getNumber(data, "mel_fmin", 0.0);
        const auto melFMax = getNumber(data, "mel_fmax", static_cast<double>(samplingRate) / 2.0);
        const auto cMel = getNumber(train, "c_mel", 45.0);
        const auto cKl = getNumber(train, "c_kl", 1.0);

        auto yMel = mel_processing::specToMel(
            batch.specPadded,
            filterLength,
            nMelChannels,
            samplingRate,
            melFMin,
            melFMax);
        yMel = commons::sliceSegments(yMel, gForward.idsSlice, segmentSamples / hopLength);
        auto yHatMel = mel_processing::melSpectrogram(
            gForward.audio.squeeze(1),
            filterLength,
            nMelChannels,
            samplingRate,
            hopLength,
            winLength,
            melFMin,
            melFMax,
            false);
        auto melLoss = torch::nn::functional::l1_loss(
            yMel,
            yHatMel,
            torch::nn::functional::L1LossFuncOptions().reduction(torch::kMean)) * cMel;
        auto kl = losses::klLoss(gForward.zP, gForward.logsQ, gForward.mP, gForward.logsP, gForward.yMask) * cKl;
        auto lossLength = gForward.lengthLoss.mean();
        auto waveformL1 = torch::nn::functional::l1_loss(
            gForward.audio,
            yReal,
            torch::nn::functional::L1LossFuncOptions().reduction(torch::kMean));
        auto gLoss = gAdversarial + fmapLoss + melLoss + kl + lossLength;
        if (phaseLog)
        {
            std::cout << "phase: " << step << " g_backward_start" << std::endl;
        }
        gLoss.backward();
        if (gradNorm(synthesizer->parameters()).item<float>() <= 0.0f)
        {
            throw std::runtime_error("synthesizer did not receive gradients.");
        }
        optimG.step();
        if (phaseLog)
        {
            std::cout << "phase: " << step << " optim_done" << std::endl;
        }
        setRequiresGrad(*discriminator, true);

        if (!torch::isfinite(dLoss).item<bool>() || !torch::isfinite(gLoss).item<bool>())
        {
            throw std::runtime_error("training loss became non-finite.");
        }

        return {
            dLoss.item<float>(),
            gLoss.item<float>(),
            melLoss.item<float>(),
            waveformL1.item<float>(),
            kl.item<float>()};
    }
}

int main(int argc, char** argv)
{
    try
    {
        const auto options = parseArgs(argc, argv);
        if (options.torchThreads > 0)
        {
            torch::set_num_threads(static_cast<int>(options.torchThreads));
            torch::set_num_interop_threads(1);
        }
        torch::manual_seed(1234);

        auto hparams = options.configPath.empty()
            ? defaultSmokeHParams()
            : utils::getHParamsFromFile(options.configPath);
        auto synthesizer = inference::createSynthesizerFromHParams(hparams);
        auto discriminator = models::MultiPeriodDiscriminator();
        const auto cleaners = inference::cleanerNamesFromHParams(hparams);
        const auto addBlank = inference::addBlankFromHParams(hparams);
        const auto audioConfig = audioConfigFromHParams(hparams, options.useSpecCache);
        const auto& train = hparams.at("train");
        const auto cleanedText = hparams.at("data").contains("cleaned_text")
            ? hparams.at("data").at("cleaned_text").asBool()
            : false;
        const auto effectiveBatchSize = options.batchSize > 0
            ? options.batchSize
            : getInt(train, "batch_size", 1);
        const auto effectiveLearningRate = options.learningRate > 0.0
            ? options.learningRate
            : getNumber(train, "learning_rate", 1.0e-4);
        const auto betas = getBetas(train, "betas", {0.9, 0.999});
        const auto eps = getNumber(train, "eps", 1.0e-8);
        const auto minSpecFrames = segmentFramesFromHParams(hparams);

        auto entries = loadLjsEntries(options, effectiveBatchSize, audioConfig.hopLength);
        std::vector<std::vector<int64_t>> bucketBatches;
        std::vector<int64_t> bucketBoundaries = options.bucketBoundaries.empty()
            ? defaultBucketBoundaries(minSpecFrames)
            : options.bucketBoundaries;
        if (options.useBucketSampler)
        {
            std::vector<int64_t> lengths;
            lengths.reserve(entries.size());
            for (const auto& entry : entries)
            {
                lengths.push_back(entry.specLength);
            }
            const auto sampler = data_utils::createBucketSamplerState(
                lengths,
                bucketBoundaries,
                effectiveBatchSize,
                1);
            bucketBatches = data_utils::createBucketBatches(
                sampler,
                effectiveBatchSize,
                1,
                0,
                options.startEpoch - 1,
                options.shuffleBuckets);
            if (bucketBatches.empty())
            {
                throw std::runtime_error("Bucket sampler did not create any batches. Check --bucket-boundaries.");
            }
        }

        const auto batchesPerEpoch = options.useBucketSampler
            ? static_cast<int64_t>(bucketBatches.size())
            : (static_cast<int64_t>(entries.size()) + effectiveBatchSize - 1) / effectiveBatchSize;
        if (batchesPerEpoch <= 0)
        {
            throw std::runtime_error("batches_per_epoch must be positive.");
        }
        const auto totalSteps = options.epochs > 0 && !options.stepsProvided
            ? options.epochs * batchesPerEpoch
            : options.steps;
        const auto effectiveEpochs = options.epochs > 0
            ? options.epochs
            : (totalSteps + batchesPerEpoch - 1) / batchesPerEpoch;
        const auto lrDecay = options.lrDecay > 0.0
            ? options.lrDecay
            : getNumber(train, "lr_decay", 1.0);
        const auto initialLearningRate = effectiveLearningRate * std::pow(lrDecay, static_cast<double>(options.startEpoch - 1));

        torch::optim::AdamWOptions optimizerOptions(initialLearningRate);
        optimizerOptions.betas(betas);
        optimizerOptions.eps(eps);

        torch::optim::AdamW optimG(synthesizer->parameters(), optimizerOptions);
        torch::optim::AdamW optimD(discriminator->parameters(), optimizerOptions);

        std::cout << "loaded_items: " << entries.size() << '\n';
        std::cout << "steps: " << totalSteps << '\n';
        std::cout << "epochs: " << effectiveEpochs << '\n';
        std::cout << "start_epoch: " << options.startEpoch << '\n';
        std::cout << "batches_per_epoch: " << batchesPerEpoch << '\n';
        std::cout << "batch_size: " << effectiveBatchSize << '\n';
        std::cout << "learning_rate: " << optimizerLearningRate(optimG) << '\n';
        std::cout << "base_learning_rate: " << effectiveLearningRate << '\n';
        std::cout << "lr_decay: " << lrDecay << '\n';
        std::cout << "betas: " << std::get<0>(betas) << ", " << std::get<1>(betas) << '\n';
        std::cout << "eps: " << eps << '\n';
        std::cout << "torch_threads: " << torch::get_num_threads() << '\n';
        std::cout << "cleaned_text: " << (cleanedText ? "true" : "false") << '\n';
        std::cout << "bucket_sampler: " << (options.useBucketSampler ? "true" : "false") << '\n';
        if (options.useBucketSampler)
        {
            std::cout << "bucket_batches: " << bucketBatches.size() << '\n';
        }
        std::cout << std::fixed << std::setprecision(5);

        TrainStepResult lastResult;
        for (int64_t step = 1; step <= totalSteps; ++step)
        {
            const auto currentEpoch = options.startEpoch + (step - 1) / batchesPerEpoch;
            const auto epochStep = (step - 1) % batchesPerEpoch + 1;
            if (options.phaseLog)
            {
                std::cout << "phase: " << step << " batch_start" << std::endl;
            }
            data_utils::TextAudioBatch batch;
            if (options.useBucketSampler)
            {
                const auto batchIndex = (step - 1) % static_cast<int64_t>(bucketBatches.size());
                if (batchIndex == 0 && step != 1)
                {
                    std::vector<int64_t> lengths;
                    lengths.reserve(entries.size());
                    for (const auto& entry : entries)
                    {
                        lengths.push_back(entry.specLength);
                    }
                    const auto sampler = data_utils::createBucketSamplerState(
                        lengths,
                        bucketBoundaries,
                        effectiveBatchSize,
                        1);
                    bucketBatches = data_utils::createBucketBatches(
                        sampler,
                        effectiveBatchSize,
                        1,
                        0,
                        currentEpoch - 1,
                        options.shuffleBuckets);
                }
                batch = makeBatchFromIndices(
                    entries,
                    bucketBatches[static_cast<size_t>(batchIndex)],
                    audioConfig,
                    cleaners,
                    cleanedText,
                    addBlank,
                    minSpecFrames);
            }
            else
            {
                batch = makeBatchForStep(
                    entries,
                    audioConfig,
                    cleaners,
                    cleanedText,
                    addBlank,
                    minSpecFrames,
                    effectiveBatchSize,
                    step);
            }
            if (options.phaseLog)
            {
                std::cout << "phase: " << step << " train_start" << std::endl;
            }
            lastResult = runTrainStep(
                synthesizer,
                discriminator,
                optimG,
                optimD,
                batch,
                hparams,
                audioConfig.hopLength,
                segmentSamplesFromHParams(hparams),
                options.phaseLog,
                step);
            if (options.phaseLog)
            {
                std::cout << "phase: " << step << " train_done" << std::endl;
            }

            if (step == 1 || step == totalSteps || step % options.logInterval == 0)
            {
                std::cout
                    << "step: " << step
                    << " epoch: " << currentEpoch
                    << " epoch_step: " << epochStep << "/" << batchesPerEpoch
                    << " lr: " << optimizerLearningRate(optimG)
                    << " d_loss: " << lastResult.dLoss
                    << " g_loss: " << lastResult.gLoss
                    << " mel: " << lastResult.melLoss
                    << " wav_l1: " << lastResult.waveformL1
                    << " kl: " << lastResult.klLoss
                    << std::endl;
            }

            if (epochStep == batchesPerEpoch)
            {
                const auto nextLearningRate = optimizerLearningRate(optimG) * lrDecay;
                setOptimizerLearningRate(optimG, nextLearningRate);
                setOptimizerLearningRate(optimD, nextLearningRate);
                if (options.phaseLog)
                {
                    std::cout << "phase: " << step << " scheduler_step lr " << nextLearningRate << std::endl;
                }
            }
        }

        if (!options.saveModelPath.empty())
        {
            utils::saveCheckpoint(
                *synthesizer,
                options.saveModelPath,
                totalSteps,
                optimizerLearningRate(optimG),
                options.saveOptimizer ? &optimG : nullptr);
            std::cout << "saved model: " << options.saveModelPath << '\n';
        }

        if (!options.saveConfigPath.empty())
        {
            saveConfig(options.saveConfigPath, options.configPath);
            std::cout << "saved config: " << options.saveConfigPath << '\n';
        }

        if (!options.inferOutputPath.empty())
        {
            inference::InferOptions inferOptions;
            inferOptions.maxLength = options.inferMaxLength;

            const auto inferResult = inference::inferText(
                synthesizer,
                options.inferText,
                cleaners,
                addBlank,
                inferOptions);
            const auto maxWavValue = hparams.at("data").contains("max_wav_value")
                ? hparams.at("data").at("max_wav_value").asNumber()
                : 32767.0;
            utils::saveWavFromTorch(
                inferResult.audio,
                options.inferOutputPath,
                audioConfig.samplingRate,
                maxWavValue);

            std::cout << "infer_text: " << options.inferText << '\n';
            std::cout << "infer_audio_shape: ["
                      << inferResult.audio.size(0) << ", "
                      << inferResult.audio.size(1) << ", "
                      << inferResult.audio.size(2) << "]\n";
            std::cout << "saved wav: " << options.inferOutputPath << '\n';
        }

        std::cout << "ljs_train_smoke passed\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "ljs_train_smoke failed: " << error.what() << '\n';
        printUsage();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
