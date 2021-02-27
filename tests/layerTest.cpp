#include "../src/layers/layers.hpp"

using namespace xt;

int main(int argc, char *argv[])
{
	Tensor<double> arr1 { { 1.0, 2.0, 3.0 },
		                  { 2.0, 5.0, 7.0 },
		                  { 2.0, 5.0, 7.0 } };
	Tensor<double> arr2 { 5.0, 6.0, 7.0 };

	Linear<double> linear(3, 4);
	std::cout << linear(arr1) << std::endl;
	Tensor<double> grad { { 1.0, 2.0, 3.0, 4.0 },
		                  { 2.0, 5.0, 7.0, 4.0 },
		                  { 2.0, 5.0, 7.0, 4.0 } };
	std::cout << linear.backward(grad) << std::endl;

	return 0;
}