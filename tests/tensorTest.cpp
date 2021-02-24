#include "../src/CppTorch.h"

using namespace xt;

int main(int argc, char* argv[])
{
    Tensor<double> arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};

    Tensor<double> arr2
      {5.0, 6.0, 7.0};

    auto res = view(arr1, 1) + arr2;

    std::cout << res << std::endl;

    return 0;
}