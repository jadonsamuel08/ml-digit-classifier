#include "neural_network.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

using namespace std;

NeuralNetwork::NeuralNetwork(size_t inputSize, size_t hiddenSize, size_t outputSize, double learningRate)
	: inputSize_(inputSize),
	  hiddenSize_(hiddenSize),
	  outputSize_(outputSize),
	  learningRate_(learningRate),
	  weightsInputHidden_(inputSize, vector<double>(hiddenSize, 0.0)),
	  weightsHiddenOutput_(hiddenSize, vector<double>(outputSize, 0.0)),
	  biasHidden_(hiddenSize, 0.0),
	  biasOutput_(outputSize, 0.0),
	  lastHiddenActivations_(hiddenSize, 0.0),
	  lastOutputActivations_(outputSize, 0.0) {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> dist(-0.5, 0.5);

	for (size_t i = 0; i < inputSize_; ++i) {
		for (size_t j = 0; j < hiddenSize_; ++j) {
			weightsInputHidden_[i][j] = dist(gen);
		}
	}

	for (size_t j = 0; j < hiddenSize_; ++j) {
		for (size_t k = 0; k < outputSize_; ++k) {
			weightsHiddenOutput_[j][k] = dist(gen);
		}
	}

	for (double& value : biasHidden_) {
		value = dist(gen);
	}

	for (double& value : biasOutput_) {
		value = dist(gen);
	}
}

double NeuralNetwork::sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::sigmoidDerivativeFromActivation(double activation) {
	return activation * (1.0 - activation);
}

vector<double> NeuralNetwork::forward(const vector<double>& input) {
	if (input.size() != inputSize_) {
		throw runtime_error("Input size does not match network input layer size.");
	}

	for (size_t j = 0; j < hiddenSize_; ++j) {
		double sum = biasHidden_[j];
		for (size_t i = 0; i < inputSize_; ++i) {
			sum += input[i] * weightsInputHidden_[i][j];
		}
		lastHiddenActivations_[j] = sigmoid(sum);
	}

	for (size_t k = 0; k < outputSize_; ++k) {
		double sum = biasOutput_[k];
		for (size_t j = 0; j < hiddenSize_; ++j) {
			sum += lastHiddenActivations_[j] * weightsHiddenOutput_[j][k];
		}
		lastOutputActivations_[k] = sigmoid(sum);
	}

	return lastOutputActivations_;
}

void NeuralNetwork::backpropagate(const vector<double>& input, const vector<double>& target) {
	if (target.size() != outputSize_) {
		throw runtime_error("Target size does not match network output layer size.");
	}

	forward(input);

	vector<double> outputDelta(outputSize_, 0.0);
	for (size_t k = 0; k < outputSize_; ++k) {
		const double error = lastOutputActivations_[k] - target[k];
		outputDelta[k] = error * sigmoidDerivativeFromActivation(lastOutputActivations_[k]);
	}

	vector<double> hiddenDelta(hiddenSize_, 0.0);
	for (size_t j = 0; j < hiddenSize_; ++j) {
		double propagatedError = 0.0;
		for (size_t k = 0; k < outputSize_; ++k) {
			propagatedError += outputDelta[k] * weightsHiddenOutput_[j][k];
		}
		hiddenDelta[j] = propagatedError * sigmoidDerivativeFromActivation(lastHiddenActivations_[j]);
	}

	for (size_t j = 0; j < hiddenSize_; ++j) {
		for (size_t k = 0; k < outputSize_; ++k) {
			weightsHiddenOutput_[j][k] -= learningRate_ * lastHiddenActivations_[j] * outputDelta[k];
		}
	}

	for (size_t k = 0; k < outputSize_; ++k) {
		biasOutput_[k] -= learningRate_ * outputDelta[k];
	}

	for (size_t i = 0; i < inputSize_; ++i) {
		for (size_t j = 0; j < hiddenSize_; ++j) {
			weightsInputHidden_[i][j] -= learningRate_ * input[i] * hiddenDelta[j];
		}
	}

	for (size_t j = 0; j < hiddenSize_; ++j) {
		biasHidden_[j] -= learningRate_ * hiddenDelta[j];
	}
}

uint8_t NeuralNetwork::predict(const vector<double>& input) {
	const vector<double> output = forward(input);
	const auto best = max_element(output.begin(), output.end());
	return static_cast<uint8_t>(distance(output.begin(), best));
}