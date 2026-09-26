#include "utils.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace utils
{
    namespace
    {
        class JsonParser
        {
        public:
            explicit JsonParser(const std::string& input)
                : input_(input)
            {
            }

            HParamValue parse()
            {
                auto value = parseValue();
                skipWhitespace();
                if (pos_ != input_.size())
                {
                    throw std::runtime_error("Unexpected trailing JSON content.");
                }
                return value;
            }

        private:
            void skipWhitespace()
            {
                while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_])))
                {
                    ++pos_;
                }
            }

            char peek() const
            {
                if (pos_ >= input_.size())
                {
                    throw std::runtime_error("Unexpected end of JSON.");
                }
                return input_[pos_];
            }

            bool consume(char expected)
            {
                skipWhitespace();
                if (pos_ < input_.size() && input_[pos_] == expected)
                {
                    ++pos_;
                    return true;
                }
                return false;
            }

            void expect(char expected)
            {
                if (!consume(expected))
                {
                    throw std::runtime_error(std::string("Expected JSON character: ") + expected);
                }
            }

            HParamValue parseValue()
            {
                skipWhitespace();
                const auto ch = peek();
                if (ch == '{')
                {
                    return parseObject();
                }
                if (ch == '[')
                {
                    return parseArray();
                }
                if (ch == '"')
                {
                    HParamValue value;
                    value.type = HParamValue::Type::String;
                    value.stringValue = parseString();
                    return value;
                }
                if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch)))
                {
                    return parseNumber();
                }
                if (input_.compare(pos_, 4, "true") == 0)
                {
                    pos_ += 4;
                    HParamValue value;
                    value.type = HParamValue::Type::Bool;
                    value.boolValue = true;
                    return value;
                }
                if (input_.compare(pos_, 5, "false") == 0)
                {
                    pos_ += 5;
                    HParamValue value;
                    value.type = HParamValue::Type::Bool;
                    value.boolValue = false;
                    return value;
                }
                if (input_.compare(pos_, 4, "null") == 0)
                {
                    pos_ += 4;
                    return {};
                }
                throw std::runtime_error("Could not parse JSON value.");
            }

            HParamValue parseObject()
            {
                HParamValue value;
                value.type = HParamValue::Type::Object;
                expect('{');
                skipWhitespace();
                if (consume('}'))
                {
                    return value;
                }

                while (true)
                {
                    skipWhitespace();
                    if (peek() != '"')
                    {
                        throw std::runtime_error("Expected JSON object key.");
                    }
                    auto key = parseString();
                    expect(':');
                    value.objectValue.emplace(std::move(key), parseValue());
                    if (consume('}'))
                    {
                        break;
                    }
                    expect(',');
                }
                return value;
            }

            HParamValue parseArray()
            {
                HParamValue value;
                value.type = HParamValue::Type::Array;
                expect('[');
                skipWhitespace();
                if (consume(']'))
                {
                    return value;
                }

                while (true)
                {
                    value.arrayValue.push_back(parseValue());
                    if (consume(']'))
                    {
                        break;
                    }
                    expect(',');
                }
                return value;
            }

            std::string parseString()
            {
                expect('"');
                std::string result;
                while (pos_ < input_.size())
                {
                    const auto ch = input_[pos_++];
                    if (ch == '"')
                    {
                        return result;
                    }
                    if (ch != '\\')
                    {
                        result.push_back(ch);
                        continue;
                    }
                    if (pos_ >= input_.size())
                    {
                        throw std::runtime_error("Invalid JSON string escape.");
                    }
                    const auto escaped = input_[pos_++];
                    switch (escaped)
                    {
                    case '"':
                    case '\\':
                    case '/':
                        result.push_back(escaped);
                        break;
                    case 'b':
                        result.push_back('\b');
                        break;
                    case 'f':
                        result.push_back('\f');
                        break;
                    case 'n':
                        result.push_back('\n');
                        break;
                    case 'r':
                        result.push_back('\r');
                        break;
                    case 't':
                        result.push_back('\t');
                        break;
                    default:
                        throw std::runtime_error("Unsupported JSON string escape.");
                    }
                }
                throw std::runtime_error("Unterminated JSON string.");
            }

            HParamValue parseNumber()
            {
                const auto start = pos_;
                if (input_[pos_] == '-')
                {
                    ++pos_;
                }
                while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_])))
                {
                    ++pos_;
                }
                if (pos_ < input_.size() && input_[pos_] == '.')
                {
                    ++pos_;
                    while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_])))
                    {
                        ++pos_;
                    }
                }
                if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E'))
                {
                    ++pos_;
                    if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-'))
                    {
                        ++pos_;
                    }
                    while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_])))
                    {
                        ++pos_;
                    }
                }

                HParamValue value;
                value.type = HParamValue::Type::Number;
                value.numberValue = std::stod(input_.substr(start, pos_ - start));
                return value;
            }

            const std::string& input_;
            size_t pos_ = 0;
        };

        std::string readTextFile(const std::string& path)
        {
            std::ifstream file(path, std::ios::binary);
            if (!file)
            {
                throw std::runtime_error("Could not open file: " + path);
            }
            std::ostringstream stream;
            stream << file.rdbuf();
            return stream.str();
        }

        std::string sanitizeTag(const std::string& tag)
        {
            std::string result;
            result.reserve(tag.size());
            for (const auto ch : tag)
            {
                if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_')
                {
                    result.push_back(ch);
                }
                else
                {
                    result.push_back('_');
                }
            }
            return result.empty() ? "untagged" : result;
        }

        std::string globToRegex(const std::string& pattern)
        {
            std::string result = "^";
            for (const auto ch : pattern)
            {
                switch (ch)
                {
                case '*':
                    result += ".*";
                    break;
                case '?':
                    result += ".";
                    break;
                case '.':
                case '\\':
                case '+':
                case '^':
                case '$':
                case '(':
                case ')':
                case '[':
                case ']':
                case '{':
                case '}':
                case '|':
                    result.push_back('\\');
                    result.push_back(ch);
                    break;
                default:
                    result.push_back(ch);
                    break;
                }
            }
            result += "$";
            return result;
        }

        int64_t numericSortKey(const std::filesystem::path& path)
        {
            std::string digits;
            const auto name = path.filename().string();
            for (const auto ch : name)
            {
                if (std::isdigit(static_cast<unsigned char>(ch)))
                {
                    digits.push_back(ch);
                }
            }
            if (digits.empty())
            {
                return std::numeric_limits<int64_t>::min();
            }
            return std::stoll(digits);
        }

        uint32_t readU32(std::ifstream& file)
        {
            uint8_t bytes[4] = {};
            file.read(reinterpret_cast<char*>(bytes), 4);
            return static_cast<uint32_t>(bytes[0]) |
                   (static_cast<uint32_t>(bytes[1]) << 8) |
                   (static_cast<uint32_t>(bytes[2]) << 16) |
                   (static_cast<uint32_t>(bytes[3]) << 24);
        }

        uint16_t readU16(std::ifstream& file)
        {
            uint8_t bytes[2] = {};
            file.read(reinterpret_cast<char*>(bytes), 2);
            return static_cast<uint16_t>(bytes[0]) |
                   (static_cast<uint16_t>(bytes[1]) << 8);
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

        std::array<uint8_t, 3> heatColor(double value)
        {
            value = std::max(0.0, std::min(1.0, value));
            const auto r = static_cast<uint8_t>(std::round(255.0 * std::min(1.0, std::max(0.0, 1.5 * value - 0.25))));
            const auto g = static_cast<uint8_t>(std::round(255.0 * std::min(1.0, std::max(0.0, 1.5 - std::abs(2.0 * value - 1.0) * 1.5))));
            const auto b = static_cast<uint8_t>(std::round(255.0 * std::min(1.0, std::max(0.0, 1.25 - 1.5 * value))));
            return {r, g, b};
        }

        torch::Tensor matrixToHeatmapTensor(const torch::Tensor& matrix)
        {
            if (!matrix.defined() || matrix.dim() != 2 || matrix.numel() == 0)
            {
                throw std::invalid_argument("plot tensor input must be a non-empty rank-2 tensor.");
            }

            auto values = matrix.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
            auto finiteMask = torch::isfinite(values);
            auto finiteValues = values.masked_select(finiteMask);
            if (finiteValues.numel() == 0)
            {
                values = torch::zeros_like(values);
                finiteValues = values.reshape({-1});
            }
            else
            {
                values = torch::where(finiteMask, values, torch::zeros_like(values));
            }

            const auto minValue = finiteValues.min().item<float>();
            const auto maxValue = finiteValues.max().item<float>();
            const auto denominator = std::max(1.0e-12f, maxValue - minValue);

            const auto height = values.size(0);
            const auto width = values.size(1);
            auto image = torch::empty({height, width, 3}, torch::TensorOptions().dtype(torch::kUInt8));
            auto valueAccessor = values.accessor<float, 2>();
            auto imageAccessor = image.accessor<uint8_t, 3>();

            for (int64_t row = 0; row < height; ++row)
            {
                const auto sourceRow = height - row - 1;
                for (int64_t col = 0; col < width; ++col)
                {
                    const auto normalized = (valueAccessor[sourceRow][col] - minValue) / denominator;
                    const auto color = heatColor(static_cast<double>(normalized));
                    imageAccessor[row][col][0] = color[0];
                    imageAccessor[row][col][1] = color[1];
                    imageAccessor[row][col][2] = color[2];
                }
            }

            return image;
        }

        void saveRgbTensorAsPpm(
            const torch::Tensor& image,
            const std::string& path)
        {
            if (!image.defined() || image.dim() != 3 || image.size(2) != 3)
            {
                throw std::invalid_argument("saveRgbTensorAsPpm expects an HWC RGB tensor.");
            }

            const auto outputPath = std::filesystem::path(path);
            if (!outputPath.parent_path().empty())
            {
                std::filesystem::create_directories(outputPath.parent_path());
            }

            auto rgb = image.detach().to(torch::kCPU);
            if (rgb.scalar_type() != torch::kUInt8)
            {
                rgb = rgb.clamp(0, 255).to(torch::kUInt8);
            }
            rgb = rgb.contiguous();

            std::ofstream file(path, std::ios::binary);
            if (!file)
            {
                throw std::runtime_error("Could not open image file for writing: " + path);
            }

            file << "P6\n" << rgb.size(1) << ' ' << rgb.size(0) << "\n255\n";
            const auto bytes = rgb.numel();
            file.write(reinterpret_cast<const char*>(rgb.data_ptr<uint8_t>()), bytes);
        }
    }

    FileSummaryWriter::FileSummaryWriter(std::string logDir)
        : logDir_(std::move(logDir))
    {
        std::filesystem::create_directories(logDir_);
    }

    const std::string& FileSummaryWriter::logDir() const
    {
        return logDir_;
    }

    void FileSummaryWriter::addScalar(
        const std::string& tag,
        double value,
        int64_t step) const
    {
        const auto path = std::filesystem::path(logDir_) / "scalars.csv";
        std::ofstream file(path, std::ios::app);
        if (!file)
        {
            throw std::runtime_error("Could not open scalar log file: " + path.string());
        }
        file << step << ',' << sanitizeTag(tag) << ',' << value << '\n';
    }

    void FileSummaryWriter::addImage(
        const std::string& tag,
        const torch::Tensor& image,
        int64_t step) const
    {
        const auto filename = sanitizeTag(tag) + "_" + std::to_string(step) + ".ppm";
        saveRgbTensorAsPpm(image, (std::filesystem::path(logDir_) / "images" / filename).string());
    }

    void FileSummaryWriter::addAudio(
        const std::string& tag,
        const torch::Tensor& audio,
        int64_t step,
        int64_t sampleRate) const
    {
        const auto filename = sanitizeTag(tag) + "_" + std::to_string(step) + ".wav";
        saveWavFromTorch(audio, (std::filesystem::path(logDir_) / "audios" / filename).string(), sampleRate);
    }

    bool HParamValue::isNull() const { return type == Type::Null; }
    bool HParamValue::isBool() const { return type == Type::Bool; }
    bool HParamValue::isNumber() const { return type == Type::Number; }
    bool HParamValue::isString() const { return type == Type::String; }
    bool HParamValue::isObject() const { return type == Type::Object; }
    bool HParamValue::isArray() const { return type == Type::Array; }

    bool HParamValue::asBool() const
    {
        if (!isBool())
        {
            throw std::runtime_error("HParamValue is not a bool.");
        }
        return boolValue;
    }

    double HParamValue::asNumber() const
    {
        if (!isNumber())
        {
            throw std::runtime_error("HParamValue is not a number.");
        }
        return numberValue;
    }

    int64_t HParamValue::asInt() const
    {
        return static_cast<int64_t>(asNumber());
    }

    const std::string& HParamValue::asString() const
    {
        if (!isString())
        {
            throw std::runtime_error("HParamValue is not a string.");
        }
        return stringValue;
    }

    const std::map<std::string, HParamValue>& HParamValue::asObject() const
    {
        if (!isObject())
        {
            throw std::runtime_error("HParamValue is not an object.");
        }
        return objectValue;
    }

    const std::vector<HParamValue>& HParamValue::asArray() const
    {
        if (!isArray())
        {
            throw std::runtime_error("HParamValue is not an array.");
        }
        return arrayValue;
    }

    bool HParamValue::contains(const std::string& key) const
    {
        return isObject() && objectValue.find(key) != objectValue.end();
    }

    const HParamValue& HParamValue::at(const std::string& key) const
    {
        return asObject().at(key);
    }

    HParamValue& HParamValue::operator[](const std::string& key)
    {
        if (!isObject())
        {
            type = Type::Object;
            objectValue.clear();
        }
        return objectValue[key];
    }

    bool HParams::contains(const std::string& key) const
    {
        return root.contains(key);
    }

    const HParamValue& HParams::at(const std::string& key) const
    {
        return root.at(key);
    }

    HParamValue& HParams::operator[](const std::string& key)
    {
        return root[key];
    }

    HParamValue parseJson(const std::string& text)
    {
        return JsonParser(text).parse();
    }

    HParams getHParamsFromFile(const std::string& configPath)
    {
        HParams hparams;
        hparams.root = parseJson(readTextFile(configPath));
        return hparams;
    }

    HParams getHParamsFromDir(const std::string& modelDir)
    {
        HParams hparams = getHParamsFromFile((std::filesystem::path(modelDir) / "config.json").string());
        hparams.modelDir = modelDir;
        return hparams;
    }

    std::string latestCheckpointPath(
        const std::string& dirPath,
        const std::string& pattern)
    {
        const std::filesystem::path dir(dirPath);
        if (!std::filesystem::exists(dir))
        {
            throw std::runtime_error("Checkpoint directory does not exist: " + dirPath);
        }

        const std::regex matcher(globToRegex(pattern));
        std::vector<std::filesystem::path> matches;
        for (const auto& entry : std::filesystem::directory_iterator(dir))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }
            if (std::regex_match(entry.path().filename().string(), matcher))
            {
                matches.push_back(entry.path());
            }
        }
        if (matches.empty())
        {
            throw std::runtime_error("No checkpoint matched pattern: " + pattern);
        }

        std::sort(matches.begin(), matches.end(), [](const auto& left, const auto& right) {
            return numericSortKey(left) < numericSortKey(right);
        });
        return matches.back().string();
    }

    void saveCheckpoint(
        torch::nn::Module& model,
        const std::string& checkpointPath,
        int64_t iteration,
        double learningRate,
        const torch::optim::Optimizer* optimizer)
    {
        const auto outputPath = std::filesystem::path(checkpointPath);
        if (!outputPath.parent_path().empty())
        {
            std::filesystem::create_directories(outputPath.parent_path());
        }

        torch::serialize::OutputArchive archive;

        torch::serialize::OutputArchive modelArchive;
        model.save(modelArchive);
        archive.write("model", modelArchive);

        if (optimizer != nullptr)
        {
            torch::serialize::OutputArchive optimizerArchive;
            optimizer->save(optimizerArchive);
            archive.write("optimizer", optimizerArchive);
            archive.write("has_optimizer", torch::tensor({1}, torch::TensorOptions().dtype(torch::kLong)), true);
        }
        else
        {
            archive.write("has_optimizer", torch::tensor({0}, torch::TensorOptions().dtype(torch::kLong)), true);
        }

        archive.write("iteration", torch::tensor({iteration}, torch::TensorOptions().dtype(torch::kLong)), true);
        archive.write("learning_rate", torch::tensor({learningRate}, torch::TensorOptions().dtype(torch::kFloat64)), true);
        archive.save_to(checkpointPath);
    }

    CheckpointInfo loadCheckpoint(
        torch::nn::Module& model,
        const std::string& checkpointPath,
        torch::optim::Optimizer* optimizer,
        c10::optional<torch::Device> device)
    {
        torch::serialize::InputArchive archive;
        if (device.has_value())
        {
            archive.load_from(checkpointPath, device.value());
        }
        else
        {
            archive.load_from(checkpointPath);
        }

        torch::serialize::InputArchive modelArchive;
        archive.read("model", modelArchive);
        model.load(modelArchive);

        torch::Tensor iterationTensor;
        archive.read("iteration", iterationTensor, true);
        torch::Tensor learningRateTensor;
        archive.read("learning_rate", learningRateTensor, true);

        CheckpointInfo info;
        info.iteration = iterationTensor.index({0}).item<int64_t>();
        info.learningRate = learningRateTensor.index({0}).item<double>();

        torch::Tensor hasOptimizerTensor;
        if (archive.try_read("has_optimizer", hasOptimizerTensor, true))
        {
            info.hasOptimizer = hasOptimizerTensor.index({0}).item<int64_t>() != 0;
        }

        if (optimizer != nullptr && info.hasOptimizer)
        {
            torch::serialize::InputArchive optimizerArchive;
            archive.read("optimizer", optimizerArchive);
            optimizer->load(optimizerArchive);
        }

        return info;
    }

    std::pair<torch::Tensor, int64_t> loadWavToTorch(
        const std::string& fullPath)
    {
        std::ifstream file(fullPath, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Could not open wav file: " + fullPath);
        }

        char riff[4] = {};
        file.read(riff, 4);
        (void)readU32(file);
        char wave[4] = {};
        file.read(wave, 4);
        if (std::string(riff, 4) != "RIFF" || std::string(wave, 4) != "WAVE")
        {
            throw std::runtime_error("Invalid wav header.");
        }

        uint16_t audioFormat = 0;
        uint16_t channels = 0;
        uint32_t sampleRate = 0;
        uint16_t bitsPerSample = 0;
        std::vector<char> data;

        while (file && (!audioFormat || data.empty()))
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
                audioFormat = readU16(file);
                channels = readU16(file);
                sampleRate = readU32(file);
                (void)readU32(file);
                (void)readU16(file);
                bitsPerSample = readU16(file);
                if (chunkSize > 16)
                {
                    file.seekg(static_cast<std::streamoff>(chunkSize - 16), std::ios::cur);
                }
            }
            else if (id == "data")
            {
                data.resize(chunkSize);
                file.read(data.data(), static_cast<std::streamsize>(chunkSize));
            }
            else
            {
                file.seekg(static_cast<std::streamoff>(chunkSize), std::ios::cur);
            }
        }

        if (audioFormat != 1 || bitsPerSample != 16 || channels == 0 || sampleRate == 0 || data.empty())
        {
            throw std::runtime_error("Only non-empty PCM16 wav files are supported.");
        }

        const auto sampleCount = static_cast<int64_t>(data.size() / sizeof(int16_t));
        std::vector<float> samples(static_cast<size_t>(sampleCount));
        for (int64_t i = 0; i < sampleCount; ++i)
        {
            const auto lo = static_cast<uint8_t>(data[static_cast<size_t>(i * 2)]);
            const auto hi = static_cast<uint8_t>(data[static_cast<size_t>(i * 2 + 1)]);
            const auto value = static_cast<int16_t>(static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8));
            samples[static_cast<size_t>(i)] = static_cast<float>(value);
        }

        const auto frames = sampleCount / channels;
        auto tensor = torch::from_blob(samples.data(), {frames, static_cast<int64_t>(channels)}, torch::kFloat32).clone();
        if (channels == 1)
        {
            tensor = tensor.squeeze(1);
        }
        return {tensor, static_cast<int64_t>(sampleRate)};
    }

    void saveWavFromTorch(
        const torch::Tensor& audio,
        const std::string& fullPath,
        int64_t sampleRate,
        double maxWavValue)
    {
        if (!audio.defined() || audio.numel() == 0)
        {
            throw std::invalid_argument("saveWavFromTorch received an empty tensor.");
        }
        if (sampleRate <= 0)
        {
            throw std::invalid_argument("saveWavFromTorch sampleRate must be positive.");
        }
        if (maxWavValue <= 0.0)
        {
            throw std::invalid_argument("saveWavFromTorch maxWavValue must be positive.");
        }

        auto wav = audio.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
        while (wav.dim() > 1 && wav.size(0) == 1)
        {
            wav = wav.squeeze(0);
        }

        int64_t frames = 0;
        int64_t channels = 0;
        if (wav.dim() == 1)
        {
            frames = wav.size(0);
            channels = 1;
            wav = wav.view({frames, channels});
        }
        else if (wav.dim() == 2)
        {
            if (wav.size(0) <= 8 && wav.size(1) > wav.size(0))
            {
                wav = wav.transpose(0, 1).contiguous();
            }
            frames = wav.size(0);
            channels = wav.size(1);
        }
        else
        {
            throw std::invalid_argument("saveWavFromTorch expects [T], [C,T], [T,C], or [1,C,T] audio.");
        }

        if (frames <= 0 || channels <= 0 || channels > 8)
        {
            throw std::invalid_argument("saveWavFromTorch received invalid audio shape.");
        }

        const auto outputPath = std::filesystem::path(fullPath);
        if (!outputPath.parent_path().empty())
        {
            std::filesystem::create_directories(outputPath.parent_path());
        }

        const uint16_t channelCount = static_cast<uint16_t>(channels);
        constexpr uint16_t bitsPerSample = 16;
        const auto sampleCount = frames * channels;
        const auto dataBytes = static_cast<uint32_t>(sampleCount * static_cast<int64_t>(sizeof(int16_t)));
        if (sampleCount > static_cast<int64_t>(std::numeric_limits<uint32_t>::max() / sizeof(int16_t)))
        {
            throw std::invalid_argument("saveWavFromTorch audio is too large for a RIFF wav file.");
        }

        std::ofstream file(fullPath, std::ios::binary);
        if (!file)
        {
            throw std::runtime_error("Could not open wav file for writing: " + fullPath);
        }

        file.write("RIFF", 4);
        writeU32(file, 36 + dataBytes);
        file.write("WAVE", 4);
        file.write("fmt ", 4);
        writeU32(file, 16);
        writeU16(file, 1);
        writeU16(file, channelCount);
        writeU32(file, static_cast<uint32_t>(sampleRate));
        writeU32(file, static_cast<uint32_t>(sampleRate * channelCount * bitsPerSample / 8));
        writeU16(file, static_cast<uint16_t>(channelCount * bitsPerSample / 8));
        writeU16(file, bitsPerSample);
        file.write("data", 4);
        writeU32(file, dataBytes);

        auto accessor = wav.accessor<float, 2>();
        for (int64_t frame = 0; frame < frames; ++frame)
        {
            for (int64_t channel = 0; channel < channels; ++channel)
            {
                const auto scaled = std::round(static_cast<double>(accessor[frame][channel]) * maxWavValue);
                const auto clamped = std::max(-32768.0, std::min(32767.0, scaled));
                writeU16(file, static_cast<uint16_t>(static_cast<int16_t>(clamped)));
            }
        }
    }

    torch::Tensor plotSpectrogramToTensor(const torch::Tensor& spectrogram)
    {
        return matrixToHeatmapTensor(spectrogram);
    }

    torch::Tensor plotAlignmentToTensor(const torch::Tensor& alignment)
    {
        if (!alignment.defined() || alignment.dim() != 2 || alignment.numel() == 0)
        {
            throw std::invalid_argument("plotAlignmentToTensor expects a non-empty rank-2 tensor.");
        }
        return matrixToHeatmapTensor(alignment.transpose(0, 1));
    }

    void summarize(
        const FileSummaryWriter& writer,
        int64_t step,
        const SummaryData& data)
    {
        for (const auto& [tag, value] : data.scalars)
        {
            writer.addScalar(tag, value, step);
        }
        for (const auto& [tag, image] : data.images)
        {
            writer.addImage(tag, image, step);
        }
        for (const auto& [tag, audio] : data.audios)
        {
            writer.addAudio(tag, audio.first, step, audio.second);
        }
    }
}
