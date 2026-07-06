#include "inference.hpp"
#include "utils.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3)
    {
        std::cerr << "usage: load_model <config.json> [checkpoint.pth]\n";
        return EXIT_FAILURE;
    }

    try
    {
        auto hparams = utils::getHParamsFromFile(argv[1]);
        auto synthesizer = inference::createSynthesizerFromHParams(hparams);

        std::cout << "model created from config: " << argv[1] << '\n';

        if (argc == 3)
        {
            auto info = utils::loadCheckpoint(*synthesizer, argv[2]);
            std::cout << "checkpoint loaded: " << argv[2] << '\n';
            std::cout << "iteration: " << info.iteration << '\n';
            std::cout << "learning_rate: " << info.learningRate << '\n';
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "load_model failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
