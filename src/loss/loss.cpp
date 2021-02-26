#include "loss.hpp"

Tensor<MSE_Dtype> MSE::loss(Tensor<MSE_Dtype> predicted, Tensor<MSE_Dtype> actual)
{
    auto res = xt::pow(predicted - actual, 2);
    return xt::sum(res);
}

Tensor<MSE_Dtype> MSE::grad(Tensor<MSE_Dtype> predicted, Tensor<MSE_Dtype> actual)
{
    auto res = predicted - actual;
    return res * 2;
}