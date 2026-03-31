#ifndef NETWORK_H
#define NETWORK_H

#include <cstdint>
#include <string>
#include <vector>

std::vector<uint8_t> loadMnistLabels(const std::string& path);

std::vector<std::vector<uint8_t>> loadMnistImages(
	const std::string& path,
	uint32_t& rows,
	uint32_t& cols
);

void printImageAscii(const std::vector<uint8_t>& image, uint32_t rows, uint32_t cols);

#endif
