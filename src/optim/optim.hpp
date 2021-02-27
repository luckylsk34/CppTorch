#ifndef _OPTIM_H_
#define _OPTIM_H_

#include "../layers/nn.hpp"

template <typename T>
class Optimizer
{
public:
	NeuralNet<T> *nn;
	Optimizer(NeuralNet<T> &nn)
		: nn(&nn) {};
	virtual void step() = 0;
};

template <typename T>
class SGD : public Optimizer<T>
{
public:
	double lr;

	SGD(NeuralNet<T> &nn,double lr = 0.01)
		: lr(lr),Optimizer<T>(nn) {};

	void step() override
	{
		for (auto &layer : (this->nn)->layers)
		{
			layer->params["w"] += -this->lr * layer->grads["w"];
			layer->params["b"] += -this->lr * layer->grads["b"];
		}

	}
};

#endif