#ifndef _Loss_H_
#define _Loss_H_

#include "../tensor/tensor.hpp"
#include "../layers/nn.hpp"

template <typename T>
class Loss
{
public:
	NeuralNet<T> *nn;
	Loss(NeuralNet<T> &nn)
		: nn(&nn) {};
	/*loss function*/
	virtual Tensor<T> loss() = 0;
	/*gradiant of the loss function*/
	virtual Tensor<T> grad(Tensor<T> predicted, Tensor<T> actual) = 0;
};

template<typename T>
class MSE : public Loss<T>
{
public:
	MSE(NeuralNet<T> &nn): Loss<T>(nn) {};
	Tensor<T> predicted, actual;
	Tensor<T> loss() override
	{
		auto res = xt::pow(this->predicted - this->actual, 2);
		return xt::sum(res);
	}

	Tensor<T> grad(Tensor<T> predicted, Tensor<T> actual) override
	{
		auto res = predicted - actual;
		return res * 2;
	}
	auto backward()
	{
		auto res = grad(this->predicted, this->actual);
		this->predicted = Tensor<T>();
		this->actual = Tensor<T>();
		return this->nn->backward(res);

	}
	auto operator()(Tensor<T> predicted, Tensor<T> actual)
	{
		this->predicted = predicted;
		this->actual = actual;
		return *this;
	}
};

#endif