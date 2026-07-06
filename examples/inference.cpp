#include "inference.hpp"
#include "utils.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
    void printUsage()
    {
        std::cerr << "usage: vits_inference <config.json> <checkpoint.pth> <text> <output.wav> [speaker_id]\n";
    }
}

int main(int argc, char** argv)
{
    if (argc < 5 || argc > 6)
    {
        printUsage();
        return EXIT_FAILURE;
    }

    try
    {
        auto hparams = utils::getHParamsFromFile(argv[1]);
        auto synthesizer = inference::createSynthesizerFromHParams(hparams);
        utils::loadCheckpoint(*synthesizer, argv[2]);

        auto cleaners = inference::cleanerNamesFromHParams(hparams);
        inference::InferOptions options;
        if (argc == 6)
        {
            options.speakerId = std::stoll(argv[5]);
        }

        auto result = inference::inferText(
            synthesizer,
            argv[3],
            cleaners,
            inference::addBlankFromHParams(hparams),
            options);

        const auto sampleRate = hparams.at("data").at("sampling_rate").asInt();
        const auto maxWavValue = hparams.at("data").contains("max_wav_value")
                                     ? hparams.at("data").at("max_wav_value").asNumber()
                                     : 32767.0;
        utils::saveWavFromTorch(result.audio, argv[4], sampleRate, maxWavValue);

        std::cout << "audio shape: ["
                  << result.audio.size(0) << ", "
                  << result.audio.size(1) << ", "
                  << result.audio.size(2) << "]\n";
        std::cout << "saved wav: " << argv[4] << '\n';
        std::cout << "attention shape: ["
                  << result.attention.size(0) << ", "
                  << result.attention.size(1) << ", "
                  << result.attention.size(2) << ", "
                  << result.attention.size(3) << "]\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "vits_inference failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
