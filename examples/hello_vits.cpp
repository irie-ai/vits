#include "inference.hpp"
#include "text_processing.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main()
{
    try
    {
        auto prepared = inference::prepareText(
            "Hello VITS.",
            {"basic_cleaners"},
            true);

        std::cout << "VITS C++ example\n";
        std::cout << "symbols: " << text_processing::symbols().size() << '\n';
        std::cout << "tokens shape: [" << prepared.tokens.size(0) << ", " << prepared.tokens.size(1) << "]\n";
        std::cout << "length: " << prepared.lengths.item<int64_t>() << '\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << "hello_vits failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
