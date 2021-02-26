#include "../src/CppTorch.h"

using namespace xt;

int main(int argc, char *argv[])
{
	Tensor<double> arr1{{1.0, 2.0, 3.0},
						{2.0, 5.0, 7.0},
						{2.0, 5.0, 7.0}};
	Tensor<double> arr2{5.0, 6.0, 7.0};
	std::cout << xt::view(arr1, 0, xt::all()) << std::endl;
	std::cout << arr1*2<< std::endl;
	// auto res = view(arr1, 1) + arr2;
	// std::cout << res << std::endl;
	// Tensor<double> a = {{1., 2., 3.}, {4., 5., 6.}};
	// auto shape = a.shape(1);
	// std::cout << shape << std::endl;

	return 0;
}