#include "data.hpp"
#include <assert.h>
#include <fstream>
#include <string>
#include <vector>

//! Byte swap unsigned int
inline uint32_t swap_uint32(uint32_t val)
{
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

//! Byte swap int
inline int32_t swap_int32(int32_t val)
{
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | ((val >> 16) & 0xFFFF);
}

template <typename T>
inline void read(T &n, std::ifstream &file)
{
	file.read((char *) &n, sizeof(T));
	n = swap_int32(n);
}

template <typename T>
inline void read_byte(T &n, std::ifstream &file)
{
	file.read((char *) &n, 1);
}

auto compute(uint8_t *&data, int size)
{
	std::vector<int> shape = { size };
	// return xt::adapt(data, size, xt::acquire_ownership(), shape);
	return xt::adapt(data, size, xt::acquire_ownership(), shape);
}

MNISTDataset::MNISTDataset(std::string images_filename, std::string labels_filename)
{
	std::ifstream images_file(images_filename, std::ios::binary);
	std::ifstream labels_file(labels_filename, std::ios::binary);

	int magic_number;
	read(magic_number, images_file);
	assert(magic_number == 2051);

	read(magic_number, labels_file);
	assert(magic_number == 2049);

	int num_images, num_labels;
	read(num_images, images_file);
	read(num_labels, labels_file);
	assert(num_images == num_labels);
	// num_images = 10;

	int width, height;
	read(height, images_file);
	read(width, images_file);

	images = xt::xarray<uint8_t>::from_shape(std::vector<int>({ num_images, height, width }));
	labels = xt::xarray<uint8_t>::from_shape(std::vector<int>({ num_images, 1 }));

	int size = num_images * height * width;
	std::vector<uint8_t> q = std::vector<uint8_t>(size);
	images_file.read((char *) q.data(), size);
	std::vector<int> shape = { num_images, height, width };
	images = xt::adapt(q, shape);

	size = num_images;
	q = std::vector<uint8_t>(size);
	labels_file.read((char *) q.data(), size);
	shape = { num_images, 1 };
	labels = xt::adapt(q, shape);

	std::cout << xt::sum(labels) << std::endl;

	images_file.close();
	labels_file.close();
}
int MNISTDataset::size()
{
	return 10;
}

// TrainingPair<double> operator[]() {
// };
