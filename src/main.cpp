#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

#include "mnist_loader.h"
#include "neural_network.h"

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

        const size_t inputSize = static_cast<size_t>(rows * cols);
        if (inputSize != 784) {
            throw runtime_error("Expected MNIST images to be 28x28 (784 values).");
        }

        const size_t hiddenSize = 128;
        const size_t outputSize = 10;
        const double learningRate = 0.1;
        const size_t warmupCheckSize = min<size_t>(1000, images.size());
        const size_t trainSize = images.size();
        const int epochs = 5;

        NeuralNetwork network(inputSize, hiddenSize, outputSize, learningRate);

        cout << "Loaded " << images.size() << " training images." << '\n';
        cout << "Initial sanity check size: " << warmupCheckSize << " (completed in earlier run)" << '\n';
        cout << "Training on full dataset size: " << trainSize << '\n';

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            double totalLoss = 0.0;
            size_t correct = 0;

            for (size_t idx = 0; idx < trainSize; ++idx) {
                vector<double> input(inputSize, 0.0);
                for (size_t p = 0; p < inputSize; ++p) {
                    input[p] = static_cast<double>(images[idx][p]) / 255.0;
                }

                vector<double> target(outputSize, 0.0);
                target[labels[idx]] = 1.0;

                const vector<double> output = network.forward(input);

                for (size_t k = 0; k < outputSize; ++k) {
                    const double diff = output[k] - target[k];
                    totalLoss += 0.5 * diff * diff;
                }

                const auto best = max_element(output.begin(), output.end());
                const uint8_t predicted = static_cast<uint8_t>(distance(output.begin(), best));
                if (predicted == labels[idx]) {
                    ++correct;
                }

                network.backpropagate(input, target);
            }

            const double avgLoss = totalLoss / static_cast<double>(trainSize);
            const double accuracy = (100.0 * static_cast<double>(correct)) / static_cast<double>(trainSize);

            cout << "Epoch " << epoch
                << " | Avg Loss: " << avgLoss
                << " | Accuracy: " << accuracy << "%" << '\n';
        }

        cout << "Training run complete." << '\n';
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}