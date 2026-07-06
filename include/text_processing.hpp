#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace text_processing
{
    const std::vector<std::string>& symbols();

    int64_t spaceId();

    std::string collapseWhitespace(const std::string& text);

    std::string lowercase(const std::string& text);

    std::string basicCleaners(const std::string& text);

    std::string transliterationCleaners(const std::string& text);

    std::string englishCleaners(const std::string& text);

    std::string englishCleaners2(const std::string& text);

    std::string expandAbbreviations(const std::string& text);

    std::string expandNumbers(const std::string& text);

    std::string normalizeNumbers(const std::string& text);

    std::string convertToAscii(const std::string& text);

    std::string cleanText(
        const std::string& text,
        const std::vector<std::string>& cleanerNames);

    std::vector<int64_t> cleanedTextToSequence(const std::string& cleanedText);

    std::vector<int64_t> textToSequence(
        const std::string& text,
        const std::vector<std::string>& cleanerNames);

    std::string sequenceToText(const std::vector<int64_t>& sequence);
}
