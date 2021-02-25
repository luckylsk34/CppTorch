#include "../src/data/data.hpp"
#include <iostream>
#include <string>

int main()
{
	std::string images_file = "datasets\\mnist\\train-images.idx3-ubyte";
	std::string labels_file = "datasets\\mnist\\train-labels.idx1-ubyte";
	MNISTDataset mnistDataset(images_file, labels_file);
	DataLoader dataLoader(mnistDataset, 16);

	xt::print_options::set_line_width(200);
	for (auto batch : dataLoader) {
		auto [x, y] = batch;
		std::cout << xt::adapt(x.shape())
				  << xt::adapt(y.shape()) << std::endl;
	}
	std::cout << "\n";
	return 0;
}