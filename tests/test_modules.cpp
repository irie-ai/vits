#include "modules.hpp"

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

    void testLayerNorm()
    {
        modules::LayerNorm norm(4);
        auto x = torch::randn({2, 4, 5});
        auto y = norm->forward(x);
        expectTrue(y.sizes() == x.sizes(), "LayerNorm keeps shape");
    }

    void testConvReluNorm()
    {
        modules::ConvReluNorm block(4, 8, 4, 3, 2, 0.0);
        auto x = torch::randn({2, 4, 6});
        auto mask = torch::ones({2, 1, 6});
        auto y = block->forward(x, mask);
        expectTrue(y.sizes() == x.sizes(), "ConvReluNorm keeps shape");
    }

    void testDDSConv()
    {
        modules::DDSConv block(4, 3, 2, 0.0);
        auto x = torch::randn({2, 4, 6});
        auto mask = torch::ones({2, 1, 6});
        auto y = block->forward(x, mask);
        expectTrue(y.sizes() == x.sizes(), "DDSConv keeps shape");
    }

    void testWN()
    {
        modules::WN block(4, 3, 2, 2, 0, 0.0);
        auto x = torch::randn({2, 4, 6});
        auto mask = torch::ones({2, 1, 6});
        auto y = block->forward(x, mask);
        expectTrue(y.sizes() == x.sizes(), "WN keeps shape");
    }

    void testResBlocks()
    {
        auto x = torch::randn({2, 4, 6});
        auto mask = torch::ones({2, 1, 6});

        modules::ResBlock1 block1(4);
        auto y1 = block1->forward(x, mask);
        expectTrue(y1.sizes() == x.sizes(), "ResBlock1 keeps shape");

        modules::ResBlock2 block2(4);
        auto y2 = block2->forward(x, mask);
        expectTrue(y2.sizes() == x.sizes(), "ResBlock2 keeps shape");
    }

    void testFlowLikeModules()
    {
        auto x = torch::rand({2, 4, 6}) + 0.1;
        auto mask = torch::ones({2, 1, 6});

        modules::Log log;
        auto [logged, logdet] = log->forward(x, mask);
        auto [unlogged, _] = log->forward(logged, mask, true);
        expectTrue(logged.sizes() == x.sizes(), "Log keeps shape");
        expectTrue(logdet.sizes() == torch::IntArrayRef({2}), "Log logdet shape");
        expectTrue(torch::allclose(unlogged, x, 1.0e-5, 1.0e-5), "Log reverse recovers input");

        modules::Flip flip;
        auto [flipped, flipLogdet] = flip->forward(x);
        auto [unflipped, __] = flip->forward(flipped, true);
        expectTrue(torch::allclose(unflipped, x), "Flip reverse recovers input");
        expectTrue(flipLogdet.sizes() == torch::IntArrayRef({2}), "Flip logdet shape");

        modules::ElementwiseAffine affine(4);
        auto [affined, affineLogdet] = affine->forward(x, mask);
        auto [unaffined, ___] = affine->forward(affined, mask, true);
        expectTrue(torch::allclose(unaffined, x, 1.0e-5, 1.0e-5), "ElementwiseAffine reverse recovers input");
        expectTrue(affineLogdet.sizes() == torch::IntArrayRef({2}), "ElementwiseAffine logdet shape");
    }

    void testResidualCouplingLayer()
    {
        modules::ResidualCouplingLayer layer(4, 8, 3, 2, 2, 0.0);
        auto x = torch::randn({2, 4, 6});
        auto mask = torch::ones({2, 1, 6});
        auto [y, logdet] = layer->forward(x, mask);
        auto [xRecovered, _] = layer->forward(y, mask, c10::nullopt, true);

        expectTrue(y.sizes() == x.sizes(), "ResidualCouplingLayer keeps shape");
        expectTrue(logdet.sizes() == torch::IntArrayRef({2}), "ResidualCouplingLayer logdet shape");
        expectTrue(torch::allclose(xRecovered, x, 1.0e-4, 1.0e-4), "ResidualCouplingLayer reverse recovers input");
    }

    void testConvFlow()
    {
        modules::ConvFlow flow(4, 8, 3, 2, 8, 5.0);
        auto x = torch::randn({2, 4, 6}).clamp(-2.0, 2.0);
        auto mask = torch::ones({2, 1, 6});

        auto [y, logdet] = flow->forward(x, mask);
        auto [xRecovered, _] = flow->forward(y, mask, c10::nullopt, true);

        expectTrue(y.sizes() == x.sizes(), "ConvFlow keeps shape");
        expectTrue(logdet.sizes() == torch::IntArrayRef({2}), "ConvFlow logdet shape");
        expectTrue(torch::isfinite(y).all().item<bool>(), "ConvFlow output is finite");
        expectTrue(torch::isfinite(logdet).all().item<bool>(), "ConvFlow logdet is finite");
        expectTrue(torch::allclose(xRecovered, x, 1.0e-3, 1.0e-3), "ConvFlow reverse recovers input");
    }
}

int main()
{
    try
    {
        torch::manual_seed(1234);
        testLayerNorm();
        testConvReluNorm();
        testDDSConv();
        testWN();
        testResBlocks();
        testFlowLikeModules();
        testResidualCouplingLayer();
        testConvFlow();
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_modules failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "test_modules passed\n";
    return EXIT_SUCCESS;
}
