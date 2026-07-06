#include "inference.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    struct BenchmarkOptions
    {
        int64_t warmup = 2;
        int64_t runs = 10;
        int64_t maxLength = 8;
        std::string text = "benchmark";
    };

    void printUsage()
    {
        std::cerr << "usage: vits_benchmark [--warmup N] [--runs N] [--max-length N] [--text TEXT]\n";
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

    BenchmarkOptions parseArgs(int argc, char** argv)
    {
        BenchmarkOptions options;
        for (int index = 1; index < argc; ++index)
        {
            const std::string arg = argv[index];
            if (arg == "--help" || arg == "-h")
            {
                printUsage();
                std::exit(EXIT_SUCCESS);
            }
            if (index + 1 >= argc)
            {
                throw std::invalid_argument("Missing value for argument: " + arg);
            }

            const auto value = argv[++index];
            if (arg == "--warmup")
            {
                options.warmup = parseInt(value, arg);
            }
            else if (arg == "--runs")
            {
                options.runs = parseInt(value, arg);
            }
            else if (arg == "--max-length")
            {
                options.maxLength = parseInt(value, arg);
            }
            else if (arg == "--text")
            {
                options.text = value;
            }
            else
            {
                throw std::invalid_argument("Unknown argument: " + arg);
            }
        }

        if (options.warmup < 0 || options.runs <= 0 || options.maxLength <= 0)
        {
            throw std::invalid_argument("warmup must be >= 0 and runs/max-length must be > 0.");
        }
        return options;
    }

    utils::HParams tinyHParams()
    {
        utils::HParams hparams;
        hparams.root = utils::parseJson(R"({
            "train": {
                "segment_size": 8
            },
            "data": {
                "text_cleaners": ["basic_cleaners"],
                "filter_length": 10,
                "hop_length": 2,
                "add_blank": true,
                "n_speakers": 0
            },
            "model": {
                "inter_channels": 4,
                "hidden_channels": 8,
                "filter_channels": 16,
                "n_heads": 2,
                "n_layers": 2,
                "kernel_size": 3,
                "p_dropout": 0.0,
                "resblock": "2",
                "resblock_kernel_sizes": [3],
                "resblock_dilation_sizes": [[1, 3]],
                "upsample_rates": [2],
                "upsample_initial_channel": 8,
                "upsample_kernel_sizes": [4],
                "use_sdp": false
            }
        })");
        return hparams;
    }
}

int main(int argc, char** argv)
{
    try
    {
        const auto options = parseArgs(argc, argv);
        torch::manual_seed(1234);

        auto hparams = tinyHParams();
        auto synthesizer = inference::createSynthesizerFromHParams(hparams);
        auto cleaners = inference::cleanerNamesFromHParams(hparams);

        inference::InferOptions inferOptions;
        inferOptions.noiseScale = 0.5;
        inferOptions.noiseScaleW = 0.5;
        inferOptions.maxLength = options.maxLength;

        for (int64_t index = 0; index < options.warmup; ++index)
        {
            (void)inference::inferText(
                synthesizer,
                options.text,
                cleaners,
                inference::addBlankFromHParams(hparams),
                inferOptions);
        }

        std::vector<double> timings;
        timings.reserve(static_cast<size_t>(options.runs));
        int64_t audioSamples = 0;
        for (int64_t index = 0; index < options.runs; ++index)
        {
            const auto start = std::chrono::steady_clock::now();
            const auto result = inference::inferText(
                synthesizer,
                options.text,
                cleaners,
                inference::addBlankFromHParams(hparams),
                inferOptions);
            const auto end = std::chrono::steady_clock::now();
            audioSamples = result.audio.numel();
            timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }

        const auto totalMs = std::accumulate(timings.begin(), timings.end(), 0.0);
        const auto meanMs = totalMs / static_cast<double>(timings.size());
        const auto minmax = std::minmax_element(timings.begin(), timings.end());

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "runs: " << options.runs << '\n';
        std::cout << "warmup: " << options.warmup << '\n';
        std::cout << "text_length: " << options.text.size() << '\n';
        std::cout << "audio_samples: " << audioSamples << '\n';
        std::cout << "mean_ms: " << meanMs << '\n';
        std::cout << "min_ms: " << *minmax.first << '\n';
        std::cout << "max_ms: " << *minmax.second << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << "vits_benchmark failed: " << error.what() << '\n';
        printUsage();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
