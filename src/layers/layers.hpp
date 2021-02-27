#ifndef _LAYERS_H_
#define _LAYERS_H_

#include "../tensor/tensor.hpp"
#include <map>

template <typename T>
using tensordict = std::map<std::string, Tensor<T>>;

template <typename T>
class Layer
{
public:
	tensordict<T> params;
	tensordict<T> grads;

	virtual Tensor<T> forward(Tensor<T> input) = 0;
	virtual Tensor<T> backward(Tensor<T> grad) = 0;

	auto operator()(Tensor<T> input)
	{
		return forward(input);
	}
};

template <typename T>
class Linear : public Layer<T>
{
public:
	Tensor<T> inputs;
	Linear(int input_size, int output_size)
	{
		this->params["w"] = xt::eval(xt::random::randn<T>({input_size, output_size}));
		this->params["b"] = xt::eval(xt::random::randn<T>({output_size}));
	}

	Tensor<T> forward(Tensor<T> input) override
	{
		this->inputs = input;
		auto res = xt::linalg::dot(input, this->params["w"]);

		return res + this->params["b"];
	}

	Tensor<T> backward(Tensor<T> grad) override
	{
		this->grads["b"] = xt::sum(grad, 0);
		this->grads["w"] = xt::linalg::dot(xt::transpose(this->inputs), grad);

		return xt::linalg::dot(grad, xt::transpose(this->params["w"]));
	}

	~Linear()
	{
		this->params.clear();
		this->grads.clear();
	}
};

template <typename T>
class Activation : public Layer<T>
{
public:
	Tensor<T> inputs;
	std::function<Tensor<T>(Tensor<T>)> f;
	std::function<Tensor<T>(Tensor<T>)> f_prime;
	Activation(std::function<Tensor<T>(Tensor<T>)> f,
			   std::function<Tensor<T>(Tensor<T>)> f_prime)
		: f(f),
		  f_prime(f_prime)
	{
	}

	Tensor<T> forward(Tensor<T> input) override
	{
		this->inputs = input;
		return this->f(input);
	}

	Tensor<T> backward(Tensor<T> grad) override
	{
		return this->f_prime(this->inputs) * grad;
	}

};

template <typename T>
Tensor<T> tanh(Tensor<T> input)
{
	return xt::tanh(input);
}

template <typename T>
Tensor<T> tanh_prime(Tensor<T> input)
{
	input = xt::tanh(input);
	return 1 - xt::pow(input, 2);
}

template <typename T>
class Tanh : public Activation<T>
{
public:
	Tanh() : Activation<T>(tanh<T>, tanh_prime<T>)
	{
	}
};

#endif