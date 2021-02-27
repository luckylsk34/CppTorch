#ifndef _NN_H_
#define _NN_H_

#include "../layers/nn.hpp"

template <typename T>
class Optimizer
{
public:
	NeuralNet<T> *nn;
	Optimizer(NeuralNet &nn)
		: nn(&nn) {};
	virtual void step() = 0;
}

template <typename T>
class SGD : Optimizer<T>
{
public:
	int lr;

	SGD(int lr = 0.01)
		: lr(lr) {};

	inline void step()
	{
		for (auto &&layer : (*nn).layers)
			layers.params["w"] += -this->lr * layer->grads["w"];
	}
}

#endif