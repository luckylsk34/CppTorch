#include "../src/CppTorch.hpp"

class MNISTClassifier : public NeuralNet<double>
{
public:
	MNISTClassifier()
	{
		auto input_size = 28 * 28;
		add(new Linear<double>(input_size, 256));
		add(new Linear<double>(256, 128));
		add(new Tanh<double>());
		add(new Linear<double>(128, 10));
		add(new Softmax<double>());
	}
};

int main(int argc, char **argv)
{
	std::string images_file = "datasets\\mnist\\train-images.idx3-ubyte";
	std::string labels_file = "datasets\\mnist\\train-labels.idx1-ubyte";
	MNISTDataset mnistDataset(images_file, labels_file);
	DataLoader dataLoader(mnistDataset, 128);

	MNISTClassifier nn;
	SGD sgd(nn, 0.001);
	MSE mse_loss(nn);

	xt::print_options::set_line_width(180);

	int epochs = 2, n = epochs;
	while (n--) {
		for (auto batch : dataLoader) {
			auto [input, target] = batch;
			input.reshape({ 128, -1 });
			auto pred = nn(input);
			auto loss = mse_loss(pred, target);
			std::cout << "Epoch :" << epochs - n << " loss: " << loss.loss() << std::endl;
			loss.backward();
			sgd.step();
		}
	}
}

// int main(int argc, char **argv)
// {

// 	Tensor<double> s {
// 		{ 1, 2, 3 },
// 		{ 2, 3, 4 },
// 		{ 4, 5, 6 }
// 	};
// 	std::cout << xt::transpose(xt::transpose(s) / xt::transpose(xt::sum(s, 1))) << std::endl;
// 	// std::cout << xt::transpose(s).shape(0) << std::endl;
// 	// std::cout << xt::transpose(s) << std::endl;
// 	// std::cout << xt::transpose(xt::sum(s, 1)) << std::endl;

// 	// std::cout << xt::transpose(xt::transpose(s) / xt::transpose(xt::sum(s, 1)))
// 	// 		  << std::endl;
// 	Tensor<double> a { 1, 2, 3 };
// 	// for (int i = 0; i < 2; i++) {
// 	// 	std::cout << xt::view(s, i, xt::all()) * xt::view(a, i, xt::all()) << std::endl
// 	// 			  << std::endl;
// 	// 	std::cout << xt::linalg::dot(xt::view(s, i, xt::all()), xt::view(a, i, xt::all())) << std::endl;
// 	// }
// 	// std::cout << xt::transpose(s) << std::endl;
// 	// std::cout << xt::transpose(a) << std::endl;
// 	// std::cout << xt::linalg::dot(xt::transpose(s), xt::transpose(a)) << std::endl;

// 	// std::cout << xt::linalg::dot(s, a) << std::endl;
// 	// std::cout << s * a << std::endl;
// 	auto oms = xt::eye(3) - s;
// 	std::cout << oms << std::endl;
// 	auto soms = xt::transpose(s * xt::transpose(oms));
// 	std::cout << soms << std::endl;
// 	// Tensor<double> grad {};
// 	// MNISTClassifier nn;
// 	// SGD sgd(nn, 0.0001);
// 	// MSE mse_loss(nn);

// 	// xt::print_options::set_line_width(180);

// 	// int epochs = 2, n = epochs;
// 	// while (n--) {
// 	//     for()
// 	// 	auto pred = nn(input);
// 	// 	auto loss = mse_loss(pred, target);
// 	// 	std::cout << "Epoch :" << epochs - n << " loss: " << loss.loss() << std::endl;
// 	// 	loss.backward();
// 	// 	sgd.step();
// 	// }
// }
