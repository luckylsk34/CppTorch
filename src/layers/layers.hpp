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
	bool alt_backward;
	Tensor<T> inputs;
	std::function<Tensor<T>(Tensor<T>)> f;
	std::function<Tensor<T>(Tensor<T>)> f_prime;
	Activation(std::function<Tensor<T>(Tensor<T>)> f,
			   std::function<Tensor<T>(Tensor<T>)> f_prime, bool alt_backward = false)
		: f(f), f_prime(f_prime), alt_backward(alt_backward)
	{
	}

	Tensor<T> forward(Tensor<T> input) override
	{
		this->inputs = input;
		return this->f(input);
	}

	Tensor<T> backward(Tensor<T> grad) override
	{
		if (!alt_backward)
			return this->f_prime(this->inputs) * grad;

		auto f_grad = this->f_prime(this->inputs);
		auto res = Tensor<T>::from_shape(grad.shape());
		for (int batch = 0; batch < f_grad.shape(0); batch++)
			xt::view(res, batch, xt::all()) =
				xt::linalg::dot(xt::view(f_grad, batch, xt::all()),
								xt::view(grad, batch, xt::all()));
		return res;
	}
};

template <typename T>
class Tanh : public Activation<T>
{
	static Tensor<T> tanh(Tensor<T> input)
	{
		return xt::tanh(input);
	}

	static Tensor<T> tanh_prime(Tensor<T> input)
	{
		input = xt::tanh(input);
		return 1 - xt::pow(input, 2);
	}

public:
	Tanh()
		: Activation<T>(tanh, tanh_prime)
	{
	}
};

template <typename T>
class Sigmoid : public Activation<T>
{
	static Tensor<T> sigmoid(Tensor<T> input)
	{
		return 1 / (1 + xt::exp(-input));
	}

	static Tensor<T> sigmoid_prime(Tensor<T> input)
	{
		auto x = sigmoid(input);
		return x * (1 - x);
	}

public:
	Sigmoid()
		: Activation<T>(sigmoid, sigmoid_prime)
	{
	}
};

template <typename T>
class Softmax : public Activation<T>
{
public:
	static Tensor<T> softmax(Tensor<T> input)
	{
		auto m = xt::amax(input);
		auto x = xt::exp(input - m);
		return xt::transpose(xt::transpose(x) / xt::transpose(xt::sum(x, 1)));
	}

	static Tensor<T> softmax_prime(Tensor<T> input)
	{
		auto sm = softmax(input);
		auto eyes = Tensor<T>::from_shape({input.shape(0), input.shape(1), input.shape(1)});
		for (int batch = 0; batch < input.shape(0); ++batch)
		{
			auto eye = xt::view(eyes, batch, xt::all());
			auto s = xt::view(sm, batch, xt::all());
			auto oms = xt::eye(input.shape(1)) - s;
			eye = xt::transpose(s * xt::transpose(oms));
		}
		return eyes;
	}

	Softmax()
		: Activation<T>(softmax, softmax_prime, true)
	{
	}
};

#endif