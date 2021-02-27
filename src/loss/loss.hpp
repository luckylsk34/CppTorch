#ifndef _Loss_H_
#define _Loss_H_

#include "../tensor/tensor.hpp"

template <typename T>
class Loss
{
public:
	/*loss function*/
	virtual Tensor<T> loss(Tensor<T> predicted, Tensor<T> actual) = 0;
	/*gradiant of the loss function*/
	virtual Tensor<T> grad(Tensor<T> predicted, Tensor<T> actual) = 0;
};

typedef double MSE_Dtype;
class MSE : Loss<MSE_Dtype>
{
public:
	MSE() {};

	Tensor<MSE_Dtype> loss(Tensor<MSE_Dtype> predicted, Tensor<MSE_Dtype> actual)
	{
		auto res = xt::pow(predicted - actual, 2);
		return xt::sum(res);
	}

	Tensor<MSE_Dtype> grad(Tensor<MSE_Dtype> predicted, Tensor<MSE_Dtype> actual)
	{
		auto res = predicted - actual;
		return res * 2;
	}

	auto operator()(Tensor<MSE_Dtype> predicted, Tensor<MSE_Dtype> actual)
	{
		return loss(predicted, actual);
	}
};

#endif