#include "text_processing.hpp"

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

    void testCleaners()
    {
        expectTrue(text_processing::lowercase("HeLLo!") == "hello!", "lowercase converts ASCII letters");
        expectTrue(text_processing::collapseWhitespace("a\t  b\nc") == "a b c", "collapseWhitespace collapses spaces");
        expectTrue(text_processing::basicCleaners("HeLLo\t WORLD") == "hello world", "basicCleaners lowercases and collapses");
        expectTrue(text_processing::convertToAscii("caf\xC3\xA9") == "cafe", "convertToAscii handles common accents");
        expectTrue(text_processing::transliterationCleaners("Caf\xC3\xA9\tWORLD") == "cafe world", "transliterationCleaners converts accents");
    }

    void testEnglishCleaners()
    {
        expectTrue(
            text_processing::expandAbbreviations("mr. smith and dr. jones") == "mister smith and doctor jones",
            "expandAbbreviations follows VITS cleaner table");
        expectTrue(
            text_processing::englishCleaners("Mr.\tSmith") == "mister smith",
            "englishCleaners lowercases and expands abbreviations");
        const auto cleaned2 = text_processing::englishCleaners2("Capt.  Jones");
        expectTrue(!cleaned2.empty(), "englishCleaners2 is selectable");
        if (!text_processing::englishPhonemizerAvailable())
        {
            expectTrue(cleaned2 == "captain jones", "englishCleaners2 falls back without espeak");
        }
        else
        {
            const auto sequence = text_processing::cleanedTextToSequence(cleaned2);
            expectTrue(!sequence.empty(), "englishCleaners2 phonemizer output maps to symbols");
        }
    }

    void testPhonemizerDependency()
    {
        const auto available = text_processing::englishPhonemizerAvailable();
        const auto output = text_processing::phonemizeEnglish("hello world");
        expectTrue(!output.empty(), "phonemizeEnglish returns usable text");
        if (!available)
        {
            expectTrue(output == "hello world", "phonemizeEnglish falls back when espeak is unavailable");
        }
        else
        {
            expectTrue(output != "hello world", "phonemizeEnglish uses espeak when available");
            expectTrue(!text_processing::cleanedTextToSequence(output).empty(), "phonemizeEnglish output is symbol-compatible");
        }
    }

    void testNumberExpansion()
    {
        expectTrue(
            text_processing::normalizeNumbers("I have 12 cats.") == "I have twelve cats.",
            "normalizeNumbers expands cardinal numbers");
        expectTrue(
            text_processing::normalizeNumbers("1,234 files") == "one thousand two hundred thirty four files",
            "normalizeNumbers removes commas");
        expectTrue(
            text_processing::normalizeNumbers("version 3.14") == "version three point fourteen",
            "normalizeNumbers expands decimals");
        expectTrue(
            text_processing::normalizeNumbers("21st place") == "twenty first place",
            "normalizeNumbers expands ordinals");
        expectTrue(
            text_processing::normalizeNumbers("$1.05") == "one dollar five cents",
            "normalizeNumbers expands dollars");
        expectTrue(
            text_processing::englishCleaners("Dr. Smith has 2 cats.") == "doctor smith has two cats.",
            "englishCleaners expands numbers before abbreviations");
    }

    void testSequenceRoundtrip()
    {
        const std::string text = "hello, world!";
        auto sequence = text_processing::cleanedTextToSequence(text);
        auto roundtrip = text_processing::sequenceToText(sequence);
        expectTrue(roundtrip == text, "cleanedTextToSequence roundtrip");
        expectTrue(text_processing::symbols().size() > 150, "symbol table includes IPA symbols");

        const std::string vitsPunctuation =
            "\xC2\xA1\xC2\xBF"
            "\xE2\x80\x94\xE2\x80\xA6"
            "\xC2\xAB\xC2\xBB"
            "\xE2\x80\x9C\xE2\x80\x9D";
        auto punctuationSequence = text_processing::cleanedTextToSequence(vitsPunctuation);
        expectTrue(punctuationSequence.size() == 8, "VITS punctuation symbols are mapped");

        const std::string ipaText = u8"həlˈoʊ wɝld ᵻ ̩";
        auto ipaSequence = text_processing::cleanedTextToSequence(ipaText);
        auto ipaRoundtrip = text_processing::sequenceToText(ipaSequence);
        expectTrue(ipaRoundtrip == ipaText, "cleanedTextToSequence keeps IPA symbols");
    }

    void testTextToSequence()
    {
        auto sequence = text_processing::textToSequence("Hello   World!", {"basic_cleaners"});
        auto roundtrip = text_processing::sequenceToText(sequence);
        expectTrue(roundtrip == "hello world!", "textToSequence applies cleaners");
        expectTrue(!sequence.empty(), "textToSequence produces ids");
    }

    void testSpaceId()
    {
        auto sequence = text_processing::cleanedTextToSequence(" ");
        expectTrue(sequence.size() == 1, "space has an id");
        expectTrue(sequence[0] == text_processing::spaceId(), "spaceId matches symbol table");
    }

    void testUnknownCleaner()
    {
        bool threw = false;
        try
        {
            (void)text_processing::cleanText("hello", {"missing_cleaner"});
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }
        expectTrue(threw, "unknown cleaner throws");
    }
}

int main()
{
    try
    {
        testCleaners();
        testEnglishCleaners();
        testPhonemizerDependency();
        testNumberExpansion();
        testSequenceRoundtrip();
        testTextToSequence();
        testSpaceId();
        testUnknownCleaner();
    }
    catch (const std::exception& error)
    {
        std::cerr << "test_text_processing failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "test_text_processing passed\n";
    return EXIT_SUCCESS;
}
