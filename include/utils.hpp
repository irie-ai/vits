#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace utils
{
    struct HParamValue
    {
        enum class Type
        {
            Null,
            Bool,
            Number,
            String,
            Object,
            Array
        };

        Type type = Type::Null;
        bool boolValue = false;
        double numberValue = 0.0;
        std::string stringValue;
        std::map<std::string, HParamValue> objectValue;
        std::vector<HParamValue> arrayValue;

        bool isNull() const;
        bool isBool() const;
        bool isNumber() const;
        bool isString() const;
        bool isObject() const;
        bool isArray() const;

        bool asBool() const;
        double asNumber() const;
        int64_t asInt() const;
        const std::string& asString() const;
        const std::map<std::string, HParamValue>& asObject() const;
        const std::vector<HParamValue>& asArray() const;

        bool contains(const std::string& key) const;
        const HParamValue& at(const std::string& key) const;
        HParamValue& operator[](const std::string& key);
    };

    struct HParams
    {
        HParamValue root;
        std::string modelDir;

        bool contains(const std::string& key) const;
        const HParamValue& at(const std::string& key) const;
        HParamValue& operator[](const std::string& key);
    };

    struct CheckpointInfo
    {
        int64_t iteration = 0;
        double learningRate = 0.0;
        bool hasOptimizer = false;
    };

    struct SummaryData
    {
        std::map<std::string, double> scalars;
        std::map<std::string, torch::Tensor> images;
        std::map<std::string, std::pair<torch::Tensor, int64_t>> audios;
    };

    class FileSummaryWriter
    {
    public:
        explicit FileSummaryWriter(std::string logDir);

        const std::string& logDir() const;

        void addScalar(
            const std::string& tag,
            double value,
            int64_t step) const;

        void addImage(
            const std::string& tag,
            const torch::Tensor& image,
            int64_t step) const;

        void addAudio(
            const std::string& tag,
            const torch::Tensor& audio,
            int64_t step,
            int64_t sampleRate) const;

    private:
        std::string logDir_;
    };

    HParamValue parseJson(const std::string& text);

    HParams getHParamsFromFile(const std::string& configPath);

    HParams getHParamsFromDir(const std::string& modelDir);

    std::string latestCheckpointPath(
        const std::string& dirPath,
        const std::string& pattern = "G_*.pth");

    void saveCheckpoint(
        torch::nn::Module& model,
        const std::string& checkpointPath,
        int64_t iteration,
        double learningRate,
        const torch::optim::Optimizer* optimizer = nullptr);

    CheckpointInfo loadCheckpoint(
        torch::nn::Module& model,
        const std::string& checkpointPath,
        torch::optim::Optimizer* optimizer = nullptr,
        c10::optional<torch::Device> device = c10::nullopt);

    std::pair<torch::Tensor, int64_t> loadWavToTorch(
        const std::string& fullPath);

    void saveWavFromTorch(
        const torch::Tensor& audio,
        const std::string& fullPath,
        int64_t sampleRate,
        double maxWavValue = 32767.0);

    torch::Tensor plotSpectrogramToTensor(const torch::Tensor& spectrogram);

    torch::Tensor plotAlignmentToTensor(const torch::Tensor& alignment);

    void summarize(
        const FileSummaryWriter& writer,
        int64_t step,
        const SummaryData& data);
}
