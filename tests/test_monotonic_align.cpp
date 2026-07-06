#include "monotonic_align.hpp"

#include <torch/torch.h>

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

    void testMaximumPathSimple()
    {
        auto values = torch::tensor(
            {{{5.0f, 0.0f, 0.0f},
              {5.0f, 1.0f, 0.0f},
              {0.0f, 5.0f, 0.0f},
              {0.0f, 0.0f, 5.0f}}},
            torch::kFloat32);
        auto mask = torch::ones_like(values);
        auto path = monotonic_align::maximumPath(values, mask);
        auto expected = torch::tensor(
            {{{1.0f, 0.0f, 0.0f},
              {1.0f, 0.0f, 0.0f},
              {0.0f, 1.0f, 0.0f},
              {0.0f, 0.0f, 1.0f}}},
            torch::kFloat32);

        expectTrue(torch::equal(path, expected), "maximumPath chooses expected monotonic path");
        expectTrue(path.scalar_type() == values.scalar_type(), "maximumPath preserves dtype");
    }

    void testMaximumPathMask()
    {
        auto values = torch::ones({2, 4, 3}, torch::kFloat64);
        auto mask = torch::zeros({2, 4, 3}, torch::kFloat64);
        mask.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice()}, 1.0);
        mask.index_put_({1, torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 2)}, 1.0);

        auto path = monotonic_align::maximumPath(values, mask);
        expectTrue(path.sizes() == values.sizes(), "maximumPath masked shape");
        expectTrue(path.scalar_type() == torch::kFloat64, "maximumPath masked dtype");
        expectTrue(path.index({1, 3, torch::indexing::Slice()}).sum().item<double>() == 0.0, "maximumPath masks inactive y");
        expectTrue(path.index({1, torch::indexing::Slice(), 2}).sum().item<double>() == 0.0, "maximumPath masks inactive x");
        expectTrue(path.index({1}).sum().item<double>() == 3.0, "maximumPath active y count");
    }

    void testInvalidShape()
    {
        bool threw = false;
        try
        {
            (void)monotonic_align::maximumPath(torch::ones({2, 3}), torch::ones({2, 3}));
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }
        expectTrue(threw, "maximumPath rejects non-3D tensors");
    }
}

int main()
{
    try
    {
        testMaximumPathSimple();
        testMaximumPathMask();
        testInvalidShape();
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_monotonic_align failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "test_monotonic_align passed\n";
    return EXIT_SUCCESS;
}
