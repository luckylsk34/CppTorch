#ifndef _NN_H_
#define _NN_H_

#include "layers.hpp"
#include <vector>

template <typename T>
class NeuralNet
{
public:
	std::vector<Layer<T> *> layers;

	virtual Tensor<T> forward(Tensor<T> input)
	{
		for (auto &layer : layers)
			input = (*layer)(input);
		return input;
	}

	virtual Tensor<T> backward(Tensor<T> grad)
	{
		for (int i = layers.size() - 1; i >= 0; i--)
			grad = layers[i]->backward(grad);
		return grad;
	}

	NeuralNet &add(Layer<T> *layer)
	{
		layers.push_back(layer);
		return *this;
	}

	NeuralNet &add(Layer<T> &layer)
	{
		layers.push_back(&layer);
		return *this;
	}

	inline auto operator()(Tensor<T> input)
	{
		return forward(input);
	}
};

template <typename T>
NeuralNet<T> &operator<<(NeuralNet<T> &nn, Layer<T> *layer)
{
	return nn.add(layer);
}

template <typename T>
NeuralNet<T> &operator<<(NeuralNet<T> &nn, Layer<T> &layer)
{
	return nn.add(layer);
}

#endif