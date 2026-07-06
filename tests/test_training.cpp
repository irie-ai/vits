#include "losses.hpp"
#include "models.hpp"

#include <torch/torch.h>

#include <cstdlib>
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

    models::SynthesizerTrn createTinySynthesizer()
    {
        return models::SynthesizerTrn(
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

    void setRequiresGrad(torch::nn::Module& module, bool requiresGrad)
    {
        for (auto& parameter : module.parameters())
        {
            parameter.set_requires_grad(requiresGrad);
        }
    }

    void testTrainingSmokeStep()
    {
        auto synthesizer = createTinySynthesizer();
        auto discriminator = models::MultiPeriodDiscriminator();
        synthesizer->train();
        discriminator->train();

        torch::optim::AdamW optimG(
            synthesizer->parameters(),
            torch::optim::AdamWOptions(1.0e-4));
        torch::optim::AdamW optimD(
            discriminator->parameters(),
            torch::optim::AdamWOptions(1.0e-4));

        auto x = torch::tensor({{1, 2, 3}, {4, 5, 0}}, torch::TensorOptions().dtype(torch::kLong));
        auto xLengths = torch::tensor({3, 2}, torch::TensorOptions().dtype(torch::kLong));
        auto y = torch::randn({2, 6, 8});
        auto yLengths = torch::tensor({8, 5}, torch::TensorOptions().dtype(torch::kLong));
        auto yReal = torch::randn({2, 1, 8});

        optimD.zero_grad();
        auto dForward = synthesizer->forward(x, xLengths, y, yLengths);
        auto dOutputs = discriminator->forward(yReal, dForward.audio.detach());
        auto dLossTuple = losses::discriminatorLoss(std::get<0>(dOutputs), std::get<1>(dOutputs));
        auto dLoss = std::get<0>(dLossTuple);
        dLoss.backward();
        const auto dGradNorm = gradNorm(discriminator->parameters());
        expectTrue(dGradNorm.item<float>() > 0.0f, "discriminator receives gradients");
        optimD.step();

        setRequiresGrad(*discriminator, false);
        optimG.zero_grad();
        auto gForward = synthesizer->forward(x, xLengths, y, yLengths);
        auto gOutputs = discriminator->forward(yReal, gForward.audio);
        auto gAdversarial = losses::generatorLoss(std::get<1>(gOutputs)).first;
        auto fmapLoss = losses::featureLoss(std::get<2>(gOutputs), std::get<3>(gOutputs));
        auto kl = losses::klLoss(gForward.zP, gForward.logsQ, gForward.mP, gForward.logsP, gForward.yMask);
        auto lossLength = gForward.lengthLoss.mean();
        auto reconstruction = torch::nn::functional::l1_loss(
            gForward.audio,
            yReal,
            torch::nn::functional::L1LossFuncOptions().reduction(torch::kMean));
        auto gLoss = gAdversarial + fmapLoss + kl + lossLength + reconstruction;
        gLoss.backward();
        const auto gGradNorm = gradNorm(synthesizer->parameters());
        expectTrue(gGradNorm.item<float>() > 0.0f, "synthesizer receives gradients");
        optimG.step();
        setRequiresGrad(*discriminator, true);

        expectTrue(torch::isfinite(dLoss).item<bool>(), "discriminator loss is finite");
        expectTrue(torch::isfinite(gLoss).item<bool>(), "generator loss is finite");
    }
}

int main()
{
    try
    {
        torch::manual_seed(1234);
        testTrainingSmokeStep();
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_training failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "test_training passed\n";
    return EXIT_SUCCESS;
}
