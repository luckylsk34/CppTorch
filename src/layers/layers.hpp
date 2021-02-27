#ifndef _LAYERS_H_
#define _LAYERS_H_

#include <map>
#include "../tensor/tensor.hpp"

template <typename T>
using tensordict = std::map<std::string, Tensor<T>>;

template <typename T>
class Layer
{
public:
    tensordict<T> params;
    tensordict<T> grads;

    virtual Tensor<T> forward(Tensor<T> inputs) = 0;
    virtual Tensor<T> backward(Tensor<T> grad) = 0;
};

template <typename T>
class Linear : Layer<T>
{
public:
    Tensor<T> inputs;
    Linear(int input_size, int output_size)
    {
        this->params["w"] = xt::random::randn<T>({input_size, output_size});
        this->params["b"] = xt::random::randn<T>({output_size});
    }

    Tensor<T> forward(Tensor<T> input) override
    {
        this->inputs = input;
        auto res = xt::linalg::dot(inputs, this->params["w"]);

        return res + this->params["b"];
    }
    Tensor<T> backward(Tensor<T> grad) override
    {
        this->grads["b"] = xt::sum(grad, 0);
        this->grads["w"] = xt::linalg::dot(this->inputs, grad);

        return xt::linalg::dot(grad, xt::transpose(this->params["w"]));
    }

    ~Linear()
    {
        this->params.clear();
        this->grads.clear();
    }
};

template<typename T>
class Activation: Layer <T>
{
    





};


#endif