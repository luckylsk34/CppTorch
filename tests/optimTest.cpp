#include "../src/layers/nn.hpp"
#include "../src/optim/optim.hpp"
#include "../src/loss/loss.hpp"


class MyNeuralNet : public NeuralNet<double>
{
public:
	MyNeuralNet()
	{
		add(new Linear<double>(4, 6));
		add(new Linear<double>(6, 6));
        add(new Tanh<double>());

	}
};

int main()
{
	MyNeuralNet nn;
    SGD <double> sgd(nn);
    MSE mse_loss(nn);
	xt::print_options::set_line_width(180);

	auto input = xt::eval(xt::random::randn<double>({ 2, 4 }));
	auto actual = xt::eval(xt::random::randn<double>({ 2, 6 }));
	// auto grad = xt::eval(xt::random::randn<double>({ 2, 6 }));
    auto pred = nn(input);
	std::cout << actual << std::endl;
	std::cout << pred << std::endl;
    auto loss = mse_loss(pred,actual);
	std::cout <<  loss.loss() << std::endl;
    loss.backward();
    sgd.step();
    pred = nn(input);
    loss = mse_loss(pred,actual);
	std::cout <<  loss.loss() << std::endl;
    loss.backward();
    sgd.step();
}
