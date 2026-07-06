#include "models.hpp"

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

    void testTextEncoder()
    {
        models::TextEncoder encoder(12, 4, 8, 16, 2, 2, 3, 0.0);
        auto x = torch::tensor({{1, 2, 3, 4, 5}, {6, 7, 8, 0, 0}}, torch::TensorOptions().dtype(torch::kLong));
        auto xLengths = torch::tensor({5, 3}, torch::TensorOptions().dtype(torch::kLong));

        auto result = encoder->forward(x, xLengths);
        auto encoded = std::get<0>(result);
        auto m = std::get<1>(result);
        auto logs = std::get<2>(result);
        auto xMask = std::get<3>(result);

        expectTrue(encoded.sizes() == torch::IntArrayRef({2, 8, 5}), "TextEncoder encoded shape");
        expectTrue(m.sizes() == torch::IntArrayRef({2, 4, 5}), "TextEncoder mean shape");
        expectTrue(logs.sizes() == torch::IntArrayRef({2, 4, 5}), "TextEncoder log-scale shape");
        expectTrue(xMask.sizes() == torch::IntArrayRef({2, 1, 5}), "TextEncoder mask shape");
        expectTrue(xMask.scalar_type() == encoded.scalar_type(), "TextEncoder mask dtype follows embeddings");

        auto maskedTail = encoded.index({
            1,
            torch::indexing::Slice(),
            torch::indexing::Slice(3, torch::indexing::None)});
        expectTrue(torch::allclose(maskedTail, torch::zeros_like(maskedTail)), "TextEncoder masks padded encoded values");
    }

    void testPosteriorEncoder()
    {
        models::PosteriorEncoder encoder(6, 4, 8, 5, 1, 3);
        auto x = torch::randn({2, 6, 7});
        auto xLengths = torch::tensor({7, 4}, torch::TensorOptions().dtype(torch::kLong));

        auto result = encoder->forward(x, xLengths);
        auto z = std::get<0>(result);
        auto m = std::get<1>(result);
        auto logs = std::get<2>(result);
        auto xMask = std::get<3>(result);

        expectTrue(z.sizes() == torch::IntArrayRef({2, 4, 7}), "PosteriorEncoder latent shape");
        expectTrue(m.sizes() == torch::IntArrayRef({2, 4, 7}), "PosteriorEncoder mean shape");
        expectTrue(logs.sizes() == torch::IntArrayRef({2, 4, 7}), "PosteriorEncoder log-scale shape");
        expectTrue(xMask.sizes() == torch::IntArrayRef({2, 1, 7}), "PosteriorEncoder mask shape");

        auto maskedTail = z.index({
            1,
            torch::indexing::Slice(),
            torch::indexing::Slice(4, torch::indexing::None)});
        expectTrue(torch::allclose(maskedTail, torch::zeros_like(maskedTail)), "PosteriorEncoder masks padded latent values");
    }

    void testDurationPredictor()
    {
        models::DurationPredictor predictor(8, 16, 3, 0.0);
        auto x = torch::randn({2, 8, 6});
        auto xMask = torch::tensor(
            {{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}}, {{1.0, 1.0, 1.0, 0.0, 0.0, 0.0}}},
            torch::TensorOptions().dtype(torch::kFloat32));

        auto y = predictor->forward(x, xMask);

        expectTrue(y.sizes() == torch::IntArrayRef({2, 1, 6}), "DurationPredictor output shape");
        auto maskedTail = y.index({
            1,
            torch::indexing::Slice(),
            torch::indexing::Slice(3, torch::indexing::None)});
        expectTrue(torch::allclose(maskedTail, torch::zeros_like(maskedTail)), "DurationPredictor masks padded values");
    }

    void testDurationPredictorWithConditioning()
    {
        models::DurationPredictor predictor(8, 16, 3, 0.0, 4);
        auto x = torch::randn({2, 8, 6});
        auto xMask = torch::ones({2, 1, 6});
        auto g = torch::randn({2, 4, 6});

        auto y = predictor->forward(x, xMask, g);

        expectTrue(y.sizes() == torch::IntArrayRef({2, 1, 6}), "DurationPredictor conditioned output shape");
    }

    void testStochasticDurationPredictor()
    {
        models::StochasticDurationPredictor predictor(8, 16, 3, 0.0, 2);
        auto x = torch::randn({2, 8, 6});
        auto xMask = torch::ones({2, 1, 6});
        auto w = torch::rand({2, 1, 6}) + 0.5;

        auto nll = predictor->forward(x, xMask, w);
        auto logw = predictor->forward(x, xMask, c10::nullopt, c10::nullopt, true, 0.5);

        expectTrue(nll.sizes() == torch::IntArrayRef({2}), "StochasticDurationPredictor nll shape");
        expectTrue(torch::isfinite(nll).all().item<bool>(), "StochasticDurationPredictor nll is finite");
        expectTrue(logw.sizes() == torch::IntArrayRef({2, 1, 6}), "StochasticDurationPredictor reverse shape");
        expectTrue(torch::isfinite(logw).all().item<bool>(), "StochasticDurationPredictor reverse is finite");
    }

    void testStochasticDurationPredictorWithConditioning()
    {
        models::StochasticDurationPredictor predictor(8, 16, 3, 0.0, 2, 4);
        auto x = torch::randn({2, 8, 6});
        auto xMask = torch::ones({2, 1, 6});
        auto w = torch::rand({2, 1, 6}) + 0.5;
        auto g = torch::randn({2, 4, 6});

        auto nll = predictor->forward(x, xMask, w, g);
        auto logw = predictor->forward(x, xMask, c10::nullopt, g, true, 0.5);

        expectTrue(nll.sizes() == torch::IntArrayRef({2}), "StochasticDurationPredictor conditioned nll shape");
        expectTrue(torch::isfinite(nll).all().item<bool>(), "StochasticDurationPredictor conditioned nll is finite");
        expectTrue(logw.sizes() == torch::IntArrayRef({2, 1, 6}), "StochasticDurationPredictor conditioned reverse shape");
    }

    void testPosteriorEncoderWithConditioning()
    {
        models::PosteriorEncoder encoder(6, 4, 8, 5, 1, 3, 3);
        auto x = torch::randn({2, 6, 7});
        auto g = torch::randn({2, 3, 7});
        auto xLengths = torch::tensor({7, 5}, torch::TensorOptions().dtype(torch::kLong));

        auto result = encoder->forward(x, xLengths, g);
        auto z = std::get<0>(result);
        auto xMask = std::get<3>(result);

        expectTrue(z.sizes() == torch::IntArrayRef({2, 4, 7}), "PosteriorEncoder conditioned latent shape");
        expectTrue(xMask.sizes() == torch::IntArrayRef({2, 1, 7}), "PosteriorEncoder conditioned mask shape");
    }

    void testResidualCouplingBlock()
    {
        models::ResidualCouplingBlock block(4, 8, 5, 1, 3, 2);
        auto x = torch::randn({2, 4, 6});
        auto xMask = torch::ones({2, 1, 6});

        auto y = block->forward(x, xMask);
        auto xRecovered = block->forward(y, xMask, c10::nullopt, true);

        expectTrue(y.sizes() == x.sizes(), "ResidualCouplingBlock keeps shape");
        expectTrue(torch::allclose(xRecovered, x, 1.0e-4, 1.0e-4), "ResidualCouplingBlock reverse inverts forward");
    }

    void testResidualCouplingBlockWithConditioning()
    {
        models::ResidualCouplingBlock block(4, 8, 5, 1, 3, 2, 3);
        auto x = torch::randn({2, 4, 6});
        auto xMask = torch::ones({2, 1, 6});
        auto g = torch::randn({2, 3, 6});

        auto y = block->forward(x, xMask, g);
        auto xRecovered = block->forward(y, xMask, g, true);

        expectTrue(y.sizes() == x.sizes(), "ResidualCouplingBlock conditioned keeps shape");
        expectTrue(torch::allclose(xRecovered, x, 1.0e-4, 1.0e-4), "ResidualCouplingBlock conditioned reverse inverts forward");
    }

    void testGenerator()
    {
        models::Generator generator(
            4,
            "1",
            std::vector<int64_t>{3, 5},
            std::vector<std::vector<int64_t>>{{1, 3, 5}, {1, 3, 5}},
            std::vector<int64_t>{2, 2},
            16,
            std::vector<int64_t>{4, 4});
        auto x = torch::randn({2, 4, 5});

        auto y = generator->forward(x);

        expectTrue(y.sizes() == torch::IntArrayRef({2, 1, 20}), "Generator output shape");
        expectTrue(torch::all(y <= 1.0).item<bool>(), "Generator tanh upper bound");
        expectTrue(torch::all(y >= -1.0).item<bool>(), "Generator tanh lower bound");
    }

    void testGeneratorWithConditioning()
    {
        models::Generator generator(
            4,
            "2",
            std::vector<int64_t>{3},
            std::vector<std::vector<int64_t>>{{1, 3}},
            std::vector<int64_t>{2},
            8,
            std::vector<int64_t>{4},
            3);
        auto x = torch::randn({2, 4, 5});
        auto g = torch::randn({2, 3, 5});

        auto y = generator->forward(x, g);

        expectTrue(y.sizes() == torch::IntArrayRef({2, 1, 10}), "Generator conditioned output shape");
    }

    void testDiscriminatorS()
    {
        models::DiscriminatorS discriminator;
        auto x = torch::randn({2, 1, 512});

        auto result = discriminator->forward(x);

        expectTrue(result.first.size(0) == 2, "DiscriminatorS batch shape");
        expectTrue(result.first.dim() == 2, "DiscriminatorS flattened output rank");
        expectTrue(result.second.size() == 7, "DiscriminatorS feature map count");
    }

    void testDiscriminatorP()
    {
        models::DiscriminatorP discriminator(3);
        auto x = torch::randn({2, 1, 513});

        auto result = discriminator->forward(x);

        expectTrue(result.first.size(0) == 2, "DiscriminatorP batch shape");
        expectTrue(result.first.dim() == 2, "DiscriminatorP flattened output rank");
        expectTrue(result.second.size() == 6, "DiscriminatorP feature map count");
    }

    void testMultiPeriodDiscriminator()
    {
        models::MultiPeriodDiscriminator discriminator;
        auto y = torch::randn({2, 1, 512});
        auto yHat = torch::randn({2, 1, 512});

        auto result = discriminator->forward(y, yHat);
        const auto& realOutputs = std::get<0>(result);
        const auto& generatedOutputs = std::get<1>(result);
        const auto& realFeatureMaps = std::get<2>(result);
        const auto& generatedFeatureMaps = std::get<3>(result);

        expectTrue(realOutputs.size() == 6, "MultiPeriodDiscriminator real output count");
        expectTrue(generatedOutputs.size() == 6, "MultiPeriodDiscriminator generated output count");
        expectTrue(realFeatureMaps.size() == 6, "MultiPeriodDiscriminator real fmap count");
        expectTrue(generatedFeatureMaps.size() == 6, "MultiPeriodDiscriminator generated fmap count");
    }

    void testSynthesizerTrnInfer()
    {
        models::SynthesizerTrn synthesizer(
            16,
            6,
            8,
            4,
            8,
            16,
            2,
            2,
            3,
            0.0,
            "2",
            std::vector<int64_t>{3},
            std::vector<std::vector<int64_t>>{{1, 3}},
            std::vector<int64_t>{2},
            8,
            std::vector<int64_t>{4},
            0,
            0,
            false);
        auto x = torch::tensor({{1, 2, 3, 4}, {5, 6, 0, 0}}, torch::TensorOptions().dtype(torch::kLong));
        auto xLengths = torch::tensor({4, 2}, torch::TensorOptions().dtype(torch::kLong));

        auto result = synthesizer->infer(x, xLengths, c10::nullopt, 0.5, 1.0, 0.5, 12);

        expectTrue(result.audio.size(0) == 2, "SynthesizerTrn infer audio batch");
        expectTrue(result.audio.size(1) == 1, "SynthesizerTrn infer audio channel");
        expectTrue(result.audio.dim() == 3, "SynthesizerTrn infer audio rank");
        expectTrue(result.attention.size(0) == 2, "SynthesizerTrn infer attention batch");
        expectTrue(result.attention.size(1) == 1, "SynthesizerTrn infer attention head dim");
        expectTrue(result.z.size(1) == 4, "SynthesizerTrn infer latent channels");
        expectTrue(torch::isfinite(result.audio).all().item<bool>(), "SynthesizerTrn infer audio is finite");
    }

    void testSynthesizerTrnForward()
    {
        models::SynthesizerTrn synthesizer(
            16,
            6,
            4,
            4,
            8,
            16,
            2,
            2,
            3,
            0.0,
            "2",
            std::vector<int64_t>{3},
            std::vector<std::vector<int64_t>>{{1, 3}},
            std::vector<int64_t>{2},
            8,
            std::vector<int64_t>{4},
            0,
            0,
            false);
        auto x = torch::tensor({{1, 2, 3}, {4, 5, 0}}, torch::TensorOptions().dtype(torch::kLong));
        auto xLengths = torch::tensor({3, 2}, torch::TensorOptions().dtype(torch::kLong));
        auto y = torch::randn({2, 6, 8});
        auto yLengths = torch::tensor({8, 5}, torch::TensorOptions().dtype(torch::kLong));

        auto result = synthesizer->forward(x, xLengths, y, yLengths);

        expectTrue(result.audio.size(0) == 2, "SynthesizerTrn forward audio batch");
        expectTrue(result.audio.size(1) == 1, "SynthesizerTrn forward audio channel");
        expectTrue(result.audio.size(2) == 8, "SynthesizerTrn forward audio segment length after upsample");
        expectTrue(result.lengthLoss.sizes() == torch::IntArrayRef({2}), "SynthesizerTrn forward length loss shape");
        expectTrue(result.attention.sizes() == torch::IntArrayRef({2, 1, 8, 3}), "SynthesizerTrn forward attention shape");
        expectTrue(result.idsSlice.sizes() == torch::IntArrayRef({2}), "SynthesizerTrn forward ids slice shape");
        expectTrue(torch::isfinite(result.audio).all().item<bool>(), "SynthesizerTrn forward audio is finite");
        expectTrue(torch::isfinite(result.lengthLoss).all().item<bool>(), "SynthesizerTrn forward length loss is finite");
    }
}

int main()
{
    try
    {
        torch::manual_seed(1234);
        testTextEncoder();
        testDurationPredictor();
        testDurationPredictorWithConditioning();
        testStochasticDurationPredictor();
        testStochasticDurationPredictorWithConditioning();
        testPosteriorEncoder();
        testPosteriorEncoderWithConditioning();
        testResidualCouplingBlock();
        testResidualCouplingBlockWithConditioning();
        testGenerator();
        testGeneratorWithConditioning();
        testDiscriminatorS();
        testDiscriminatorP();
        testMultiPeriodDiscriminator();
        testSynthesizerTrnInfer();
        testSynthesizerTrnForward();
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_models failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "test_models passed\n";
    return EXIT_SUCCESS;
}
