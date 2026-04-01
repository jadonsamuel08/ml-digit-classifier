#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <limits>

#include "mnist_loader.h"

using namespace std;

int main() {
    try {
        const string labelsPath = "archive/train-labels.idx1-ubyte"; // train-labels path
        const string imagesPath = "archive/train-images.idx3-ubyte"; // train-images path

        uint32_t rows = 0;
        uint32_t cols = 0;

        vector<uint8_t> labels = loadMnistLabels(labelsPath);
        vector<vector<uint8_t>> images = loadMnistImages(imagesPath, rows, cols);

        if (labels.size() != images.size()) {
            throw runtime_error("Labels count does not match images count.");
        }

        cout << "Loaded " << images.size() << " training images." << '\n';
        cout << "Each image size: " << rows << " x " << cols << '\n';

        while (true) {
            cout << "\nEnter image index (0 to " << (images.size() - 1) << ", -1 to quit): ";

            int index = -1;
            cin >> index;

            if (!cin) {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                cout << "Please enter a number." << '\n';
                continue;
            }

            if (index == -1) {
                cout << "Goodbye!" << '\n';
                break;
            }

            if (index < 0 || static_cast<size_t>(index) >= images.size()) {
                cout << "That index is out of range." << '\n';
                continue;
            }

            cout << "Label at index " << index << ": "
                << static_cast<int>(labels[static_cast<size_t>(index)]) << '\n';
            cout << "Image preview:" << '\n';
            printImageAscii(images[static_cast<size_t>(index)], rows, cols);
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}