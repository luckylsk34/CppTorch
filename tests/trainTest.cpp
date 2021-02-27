#include "../src/layers/nn.hpp"
#include "../src/optim/optim.hpp"
#include "../src/loss/loss.hpp"

class MyNeuralNet : public NeuralNet<double>
{
public:
    MyNeuralNet()
    {
        add(new Linear<double>(2, 2));
        add(new Tanh<double>());
        add(new Linear<double>(2, 2));
    }
};

int main()
{
    MyNeuralNet nn;
    SGD<double> sgd(nn, 0.01);
    MSE mse_loss(nn);
    xt::print_options::set_line_width(180);

    auto input = xt::eval(Tensor<double>{{0, 0}, {1, 0}, {0, 1}, {1, 1}});
    auto target = xt::eval(Tensor<double>{{1, 0}, {0, 1}, {0, 1}, {1, 0}});
    // auto grad = xt::eval(xt::random::randn<double>({ 2, 6 }));
    std::cout << input << std::endl;
    int epochs = 1000;
    int n = epochs;
    while (n--)
    {
        auto pred = nn(input);
        auto loss = mse_loss(pred, target);
        std::cout << "Epoch :" << epochs - n << " loss: " << loss.loss() << std::endl;
        loss.backward();
        sgd.step();
    }
    std::cout << nn(input) << std::endl;
}
