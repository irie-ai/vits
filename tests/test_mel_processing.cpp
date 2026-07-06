#include "mel_processing.hpp"

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

    void testDynamicRange()
    {
        auto x = torch::tensor({0.25, 1.0, 2.0}, torch::TensorOptions().dtype(torch::kFloat32));
        auto compressed = mel_processing::dynamicRangeCompression(x);
        auto decompressed = mel_processing::dynamicRangeDecompression(compressed);
        expectTrue(torch::allclose(decompressed, x, 1.0e-6, 1.0e-6), "dynamic range compression roundtrip");
    }

    void testSpectrogram()
    {
        auto y = torch::rand({2, 1024}) * 2.0 - 1.0;
        auto spec = mel_processing::spectrogram(y, 256, 22050, 128, 256);
        expectTrue(spec.sizes() == torch::IntArrayRef({2, 129, 8}), "spectrogram shape");
        expectTrue(torch::isfinite(spec).all().item<bool>(), "spectrogram is finite");
        expectTrue(torch::all(spec >= 0.0).item<bool>(), "spectrogram is non-negative");
    }

    void testMelSpectrogram()
    {
        auto y = torch::rand({2, 1024}) * 2.0 - 1.0;
        auto mel = mel_processing::melSpectrogram(y, 256, 40, 22050, 128, 256, 0.0, 8000.0);
        expectTrue(mel.sizes() == torch::IntArrayRef({2, 40, 8}), "mel spectrogram shape");
        expectTrue(torch::isfinite(mel).all().item<bool>(), "mel spectrogram is finite");
    }

    void testSpecToMel()
    {
        auto y = torch::rand({2, 1024}) * 2.0 - 1.0;
        auto spec = mel_processing::spectrogram(y, 256, 22050, 128, 256);
        auto mel = mel_processing::specToMel(spec, 256, 32, 22050, 0.0, 8000.0);
        expectTrue(mel.sizes() == torch::IntArrayRef({2, 32, 8}), "specToMel shape");
    }
}

int main()
{
    try
    {
        torch::manual_seed(1234);
        testDynamicRange();
        testSpectrogram();
        testMelSpectrogram();
        testSpecToMel();
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_mel_processing failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "test_mel_processing passed\n";
    return EXIT_SUCCESS;
}
