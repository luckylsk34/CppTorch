#include "../src/loss/loss.hpp"

using namespace xt;

int main(int argc, char *argv[])
{
	Tensor<double> arr1{{1.0, 2.0, 3.0},
						{2.0, 5.0, 7.0},
						{2.0, 5.0, 7.0}};
	Tensor<double> arr2{5.0, 6.0, 7.0};
	// std::cout << xt::view(arr1, 0, xt::all()) << std::endl;
	// std::cout << xt::pow(arr1, 2)<< std::endl;
	Tensor<double> pred{5.0, 6.0, 7.0};
	Tensor<double> act{1.0, 6.0, 7.0};
	MSE mse_loss = MSE();
	std::cout << mse_loss(pred,act) << std::endl;
	std::cout <<  xt::transpose(arr1) << std::endl;
	std::cout << xt::linalg::dot(arr1, arr1) +  arr2<< std::endl;
	std::cout << xt::random::randn<double>({3, 3}) << std::endl;

	return 0;
}