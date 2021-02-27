#include "../src/layers/nn.hpp"

class MyNeuralNet : public NeuralNet<double>
{
public:
	MyNeuralNet()
	{
		add(new Linear<double>(4, 6));
	}
};

int main()
{
	MyNeuralNet nn;
	auto l1 = Linear<double>(6, 6);
	nn << l1
	   << new Linear<double>(6, 6);

	xt::print_options::set_line_width(180);

	auto input = xt::eval(xt::random::randn<double>({ 2, 4 }));
	auto grad = xt::eval(xt::random::randn<double>({ 2, 6 }));
	std::cout << nn(input) << std::endl;
	std::cout << nn.backward(grad) << std::endl;
}
