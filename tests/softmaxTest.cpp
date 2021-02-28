#include "../src/layers/layers.hpp"

using namespace xt;

int main(int argc, char *argv[])
{
	Tensor<double> arr1 { 5.0, 6.0, 7.0 };
    Tensor<double> arr2 {{ 1,2 }};
	Tensor<double> grad { { 1.0, 2.0, 3.0, 4.0 },
		                  { 2.0, 5.0, 7.0, 4.0 },
		                  { 2.0, 5.0, 7.0, 4.0 } };


	Softmax<double> soft;
	std::cout << soft(arr2)<<std::endl;
	std::cout << soft.softmax_prime(arr2) << std::endl;
	// std::cout << tanh.backward(arr2)<<std::endl;


	return 0;
}