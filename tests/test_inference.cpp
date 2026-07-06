#include "inference.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
    void expectTrue(bool condition, const std::string& message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
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

    void testPrepareText()
    {
        auto prepared = inference::prepareText("Hi", {"basic_cleaners"}, true);

        expectTrue(prepared.tokens.sizes() == torch::IntArrayRef({1, 5}), "prepareText token shape with blanks");
        expectTrue(prepared.lengths.sizes() == torch::IntArrayRef({1}), "prepareText length shape");
        expectTrue(prepared.lengths.item<int64_t>() == 5, "prepareText length value");
        expectTrue(prepared.tokens.scalar_type() == torch::kLong, "prepareText token dtype");
        expectTrue(prepared.tokens[0][0].item<int64_t>() == 0, "prepareText inserts leading blank");
    }

    void testHParamHelpers()
    {
        auto hparams = tinyHParams();
        auto cleaners = inference::cleanerNamesFromHParams(hparams);

        expectTrue(cleaners.size() == 1, "cleanerNamesFromHParams count");
        expectTrue(cleaners[0] == "basic_cleaners", "cleanerNamesFromHParams value");
        expectTrue(inference::addBlankFromHParams(hparams), "addBlankFromHParams value");
    }

    void testCreateSynthesizerFromHParamsAndInferText()
    {
        auto hparams = tinyHParams();
        auto synthesizer = inference::createSynthesizerFromHParams(hparams);
        auto cleaners = inference::cleanerNamesFromHParams(hparams);

        inference::InferOptions options;
        options.noiseScale = 0.5;
        options.noiseScaleW = 0.5;
        options.maxLength = 8;

        auto result = inference::inferText(
            synthesizer,
            "test",
            cleaners,
            inference::addBlankFromHParams(hparams),
            options);

        expectTrue(result.audio.size(0) == 1, "inferText audio batch");
        expectTrue(result.audio.size(1) == 1, "inferText audio channel");
        expectTrue(result.audio.dim() == 3, "inferText audio rank");
        expectTrue(result.attention.size(0) == 1, "inferText attention batch");
        expectTrue(result.z.size(1) == 4, "inferText latent channels from hparams");
        expectTrue(torch::isfinite(result.audio).all().item<bool>(), "inferText audio is finite");
    }

    void testEmptyTextThrows()
    {
        bool threw = false;
        try
        {
            (void)inference::prepareText("@@@", {"basic_cleaners"}, false);
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }
        expectTrue(threw, "prepareText rejects empty symbol sequence");
    }
}

int main()
{
    try
    {
        torch::manual_seed(1234);
        testPrepareText();
        testHParamHelpers();
        testCreateSynthesizerFromHParamsAndInferText();
        testEmptyTextThrows();
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_inference failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "test_inference passed\n";
    return EXIT_SUCCESS;
}
