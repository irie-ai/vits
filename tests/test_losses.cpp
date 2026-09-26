#include "losses.hpp"

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

    void testFeatureLoss()
    {
        losses::FeatureMaps real = {
            {torch::ones({2, 3}), torch::ones({2, 1}) * 2.0},
            {torch::zeros({2, 2})},
        };
        losses::FeatureMaps generated = {
            {torch::zeros({2, 3}), torch::ones({2, 1})},
            {torch::ones({2, 2})},
        };

        auto loss = losses::featureLoss(real, generated);
        expectTrue(torch::allclose(loss, torch::tensor(6.0)), "featureLoss matches expected value");
    }

    void testDiscriminatorLoss()
    {
        std::vector<torch::Tensor> real = {
            torch::ones({2, 2}),
            torch::zeros({1, 2}),
        };
        std::vector<torch::Tensor> generated = {
            torch::zeros({2, 2}),
            torch::ones({1, 2}) * 0.5,
        };

        auto result = losses::discriminatorLoss(real, generated);
        auto loss = std::get<0>(result);
        const auto& realLosses = std::get<1>(result);
        const auto& generatedLosses = std::get<2>(result);

        expectTrue(torch::allclose(loss, torch::tensor(1.25)), "discriminatorLoss matches expected value");
        expectTrue(realLosses.size() == 2, "discriminatorLoss real item count");
        expectTrue(generatedLosses.size() == 2, "discriminatorLoss generated item count");
    }

    void testGeneratorLoss()
    {
        std::vector<torch::Tensor> outputs = {
            torch::zeros({2, 2}),
            torch::ones({1, 2}) * 0.5,
        };

        auto result = losses::generatorLoss(outputs);

        expectTrue(torch::allclose(result.first, torch::tensor(1.25)), "generatorLoss matches expected value");
        expectTrue(result.second.size() == 2, "generatorLoss item count");
    }

    void testKlLoss()
    {
        auto zP = torch::ones({2, 2, 3});
        auto logsQ = torch::zeros({2, 2, 3});
        auto mP = torch::zeros({2, 2, 3});
        auto logsP = torch::zeros({2, 2, 3});
        auto mask = torch::ones({2, 1, 3});

        auto loss = losses::klLoss(zP, logsQ, mP, logsP, mask);
        expectTrue(torch::allclose(loss, torch::tensor(0.0)), "klLoss matches expected value");

        zP = torch::tensor({{{2.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}}});
        logsQ = torch::zeros({1, 2, 3});
        mP = torch::zeros({1, 2, 3});
        logsP = torch::zeros({1, 2, 3});
        mask = torch::tensor({{{1.0f, 0.0f, 1.0f}}});

        loss = losses::klLoss(zP, logsQ, mP, logsP, mask);
        expectTrue(torch::allclose(loss, torch::tensor(0.25f)), "klLoss applies mask and normalizes by valid timesteps");
    }
}

int main()
{
    try
    {
        testFeatureLoss();
        testDiscriminatorLoss();
        testGeneratorLoss();
        testKlLoss();
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_losses failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "test_losses passed\n";
    return EXIT_SUCCESS;
}
