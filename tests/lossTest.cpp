#include "../src/loss/loss.hpp"
#include <assert.h>

using namespace xt;

int main(int argc, char *argv[])
{
	Tensor<double> pred { 5.0, 6.0, 7.0 };
	Tensor<double> act { 1.0, 6.0, 7.0 };
	MSE mse_loss = MSE();
	assert(mse_loss(pred, act)[0] == 16);

	return 0;
}