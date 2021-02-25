#ifndef _DATA_H_
#define _DATA_H_

#include "../tensor/tensor.hpp"
#include <iterator>
#include <string>
#include <tuple>

template <typename T>
using TrainingPair = std::tuple<Tensor<T>, Tensor<T>>;

template <typename T>
class Dataset
{
public:
	virtual int size() { return 0; };
	virtual TrainingPair<T> operator[](int index) { return make_tuple(Tensor<T>(), Tensor<T>()); };
};

template <typename T>
class DataLoader
{
public:
	Dataset<T> *dataset;

	DataLoader(Dataset<T> &dataset)
		: dataset(&dataset) {};

	template <typename S>
	class iterator
	{
	public:
		int index = 0;
		Dataset<S> *dataset;

		iterator(int index, Dataset<S> *dataset)
			: index(index), dataset(dataset) {};

		inline void operator++() { index++; };
		inline void operator++(int) { index++; };
		inline bool operator==(iterator &rhs) { return index == rhs.index; }
		inline bool operator!=(iterator &rhs) { return index != rhs.index; }
		inline auto operator*() { return (*dataset)[index]; }
	};

	auto begin() { return iterator<T>(0, dataset); }
	auto end() { return iterator<T>(dataset->size(), dataset); }
};

typedef uint8_t MNIST_Dtype;
class MNISTDataset : public Dataset<MNIST_Dtype>
{
private:
	Tensor<MNIST_Dtype> images;
	Tensor<MNIST_Dtype> labels;

public:
	MNISTDataset(std::string images_file, std::string labels_file);
	int size();
	TrainingPair<MNIST_Dtype> operator[](int index)
	{
		return { xt::view(images, index, xt::all()), xt::view(labels, index, xt::all()) };
	};
};
#endif