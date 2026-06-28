#include "commons.hpp"

#include <cmath>
#include <cstdint>
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

    void expectNear(double actual, double expected, double tolerance, const std::string& message)
    {
        if (std::abs(actual - expected) > tolerance)
        {
            throw std::runtime_error(
                message + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
        }
    }

    void testGetPadding()
    {
        expectTrue(commons::getPadding(5, 1) == 2, "getPadding kernel=5 dilation=1");
        expectTrue(commons::getPadding(3, 2) == 2, "getPadding kernel=3 dilation=2");
    }

    void testIntersperse()
    {
        const std::vector<int64_t> values = {1, 2, 3};
        const std::vector<int64_t> expected = {0, 1, 0, 2, 0, 3, 0};
        expectTrue(commons::intersperse(values, 0) == expected, "intersperse inserts item between and around values");
    }

    void testConvertPadShape()
    {
        const std::vector<std::vector<int64_t>> padShape = {{0, 0}, {1, 0}, {2, 3}};
        const std::vector<int64_t> expected = {2, 3, 1, 0, 0, 0};
        expectTrue(commons::convertPadShape(padShape) == expected, "convertPadShape reverses nested pad order");
    }

    void testSliceSegments()
    {
        auto x = torch::arange(24, torch::kFloat32).view({2, 3, 4});
        auto ids = torch::tensor({1, 0}, torch::kLong);
        auto segments = commons::sliceSegments(x, ids, 2);
        auto expected = torch::stack({
            x.index({0, torch::indexing::Slice(), torch::indexing::Slice(1, 3)}),
            x.index({1, torch::indexing::Slice(), torch::indexing::Slice(0, 2)})
        });

        expectTrue(torch::allclose(segments, expected), "sliceSegments extracts expected windows");
        expectTrue(segments.sizes() == torch::IntArrayRef({2, 3, 2}), "sliceSegments output shape");
    }

    void testRandSliceSegments()
    {
        torch::manual_seed(1234);
        auto x = torch::arange(40, torch::kFloat32).view({2, 4, 5});
        auto lengths = torch::tensor({5, 4}, torch::kLong);
        auto [segments, ids] = commons::randSliceSegments(x, lengths, 3);

        expectTrue(segments.sizes() == torch::IntArrayRef({2, 4, 3}), "randSliceSegments segment shape");
        expectTrue(ids.sizes() == torch::IntArrayRef({2}), "randSliceSegments ids shape");
        expectTrue(torch::all(ids >= 0).item<bool>(), "randSliceSegments ids lower bound");
        expectTrue(torch::all(ids < torch::tensor({3, 2}, torch::kLong)).item<bool>(), "randSliceSegments ids upper bound");
    }

    void testGumbel()
    {
        auto samples = commons::randGumbel({2, 3});
        expectTrue(samples.sizes() == torch::IntArrayRef({2, 3}), "randGumbel shape");
        expectTrue(torch::all(torch::isfinite(samples)).item<bool>(), "randGumbel finite values");

        auto x = torch::zeros({2, 3}, torch::kFloat64);
        auto like = commons::randGumbelLike(x);
        expectTrue(like.sizes() == x.sizes(), "randGumbelLike shape");
        expectTrue(like.scalar_type() == x.scalar_type(), "randGumbelLike dtype");
    }

    void testTimingSignal()
    {
        auto signal = commons::getTimingSignal1D(4, 5);
        expectTrue(signal.sizes() == torch::IntArrayRef({1, 5, 4}), "getTimingSignal1D shape");
        expectNear(signal.index({0, 0, 0}).item<double>(), 0.0, 1.0e-6, "timing signal sin starts at zero");
        expectNear(signal.index({0, 2, 0}).item<double>(), 1.0, 1.0e-6, "timing signal cos starts at one");

        auto x = torch::zeros({2, 5, 4}, torch::kFloat32);
        expectTrue(commons::addTimingSignal1D(x).sizes() == x.sizes(), "addTimingSignal1D shape");
        expectTrue(commons::catTimingSignal1D(x).sizes() == torch::IntArrayRef({2, 10, 4}), "catTimingSignal1D shape");
    }

    void testMasks()
    {
        auto subsequent = commons::subsequentMask(3);
        auto expectedSubsequent = torch::tensor({{{{1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 1.0f}}}});
        expectTrue(torch::allclose(subsequent, expectedSubsequent), "subsequentMask lower triangular mask");

        auto lengths = torch::tensor({1, 3}, torch::kLong);
        auto mask = commons::sequenceMask(lengths);
        auto expectedMask = torch::tensor({{true, false, false}, {true, true, true}}, torch::kBool);
        expectTrue(torch::equal(mask, expectedMask), "sequenceMask inferred max length");
    }

    void testFusedActivation()
    {
        auto inputA = torch::zeros({1, 4, 2}, torch::kFloat32);
        auto inputB = torch::zeros({1, 4, 2}, torch::kFloat32);
        inputA.index_put_({0, torch::indexing::Slice(0, 2), torch::indexing::Slice()}, 1.0);
        auto output = commons::fusedAddTanhSigmoidMultiply(inputA, inputB, 2);
        auto expected = torch::tanh(torch::ones({1, 2, 2})) * 0.5;
        expectTrue(torch::allclose(output, expected), "fusedAddTanhSigmoidMultiply output");
    }

    void testShift1D()
    {
        auto x = torch::tensor({{{1.0f, 2.0f, 3.0f}}});
        auto shifted = commons::shift1D(x);
        auto expected = torch::tensor({{{0.0f, 1.0f, 2.0f}}});
        expectTrue(torch::allclose(shifted, expected), "shift1D shifts right with zero padding");
    }

    void testGeneratePath()
    {
        auto duration = torch::tensor({{1.0f, 2.0f, 1.0f}});
        auto mask = torch::ones({1, 3, 4}, torch::kFloat32);
        auto path = commons::generatePath(duration, mask);
        auto expected = torch::tensor({{{1.0f, 0.0f, 0.0f, 0.0f},
                                       {0.0f, 1.0f, 1.0f, 0.0f},
                                       {0.0f, 0.0f, 0.0f, 1.0f}}});
        expectTrue(torch::allclose(path, expected), "generatePath expands durations into alignment path");
    }

    void testKlDivergence()
    {
        auto zeros = torch::zeros({2, 3}, torch::kFloat32);
        auto kl = commons::klDivergence(zeros, zeros, zeros, zeros);
        expectTrue(torch::allclose(kl, zeros), "klDivergence identical distributions are zero");
    }
}

int main()
{
    try
    {
        testGetPadding();
        testIntersperse();
        testConvertPadShape();
        testSliceSegments();
        testRandSliceSegments();
        testGumbel();
        testTimingSignal();
        testMasks();
        testFusedActivation();
        testShift1D();
        testGeneratePath();
        testKlDivergence();
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_commons failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "test_commons passed\n";
    return EXIT_SUCCESS;
}
