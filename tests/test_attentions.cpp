#include "attentions.hpp"

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

    void testFFN()
    {
        attentions::FFN ffn(4, 4, 8, 3, 0.0);
        auto x = torch::randn({2, 4, 6});
        auto mask = torch::ones({2, 1, 6});
        auto y = ffn->forward(x, mask);
        expectTrue(y.sizes() == x.sizes(), "FFN keeps shape");
    }

    void testMultiHeadAttention()
    {
        attentions::MultiHeadAttention attn(4, 4, 2, 0.0);
        auto x = torch::randn({2, 4, 6});
        auto mask = torch::ones({2, 1, 6, 6});
        auto y = attn->forward(x, x, mask);
        expectTrue(y.sizes() == x.sizes(), "MultiHeadAttention keeps shape");
        expectTrue(attn->attentionWeights().sizes() == torch::IntArrayRef({2, 2, 6, 6}), "attention weight shape");

        attentions::MultiHeadAttention relativeAttn(4, 4, 2, 0.0, 2);
        auto relativeY = relativeAttn->forward(x, x, mask);
        expectTrue(relativeY.sizes() == x.sizes(), "Relative MultiHeadAttention keeps shape");
        expectTrue(relativeAttn->attentionWeights().sizes() == torch::IntArrayRef({2, 2, 6, 6}), "relative attention weight shape");
    }

    void testEncoder()
    {
        attentions::Encoder encoder(4, 8, 2, 2, 3, 0.0, c10::nullopt);
        auto x = torch::randn({2, 4, 6});
        auto mask = torch::ones({2, 1, 6});
        auto y = encoder->forward(x, mask);
        expectTrue(y.sizes() == x.sizes(), "Encoder keeps shape");
    }

    void testDecoder()
    {
        attentions::Decoder decoder(4, 8, 2, 2, 3, 0.0);
        auto x = torch::randn({2, 4, 5});
        auto xMask = torch::ones({2, 1, 5});
        auto h = torch::randn({2, 4, 6});
        auto hMask = torch::ones({2, 1, 6});
        auto y = decoder->forward(x, xMask, h, hMask);
        expectTrue(y.sizes() == x.sizes(), "Decoder keeps shape");
    }
}

int main()
{
    try
    {
        torch::manual_seed(1234);
        testFFN();
        testMultiHeadAttention();
        testEncoder();
        testDecoder();
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_attentions failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "test_attentions passed\n";
    return EXIT_SUCCESS;
}
