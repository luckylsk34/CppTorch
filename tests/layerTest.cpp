#include "../src/layers/layers.hpp"

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
    Linear <double> linear(3,4);

    std::cout << linear.forward(arr1) << std::endl;

	return 0;
}