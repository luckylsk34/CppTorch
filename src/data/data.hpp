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
	virtual int size() = 0;
	virtual TrainingPair<T> operator[](int index) = 0;
};

template <typename T>
class DataLoader
{
public:
	Dataset<T> *dataset;
	int batch_size;

	DataLoader(Dataset<T> &dataset, int batch_size = 1)
		: dataset(&dataset), batch_size(batch_size) {};

	template <typename S>
	class iterator
	{
		int index = 0;
		Dataset<S> *dataset;
		int batch_size;

	public:
		iterator(int index, Dataset<S> *dataset, int batch_size)
			: index(index), dataset(dataset), batch_size(batch_size) {};

		inline void operator++() { index++; };
		inline void operator++(int) { index++; };
		inline bool operator==(iterator &rhs) { return index >= rhs.index; }
		inline bool operator!=(iterator &rhs) { return index < rhs.index; }
		inline auto operator*()
		{
			auto [x, y] = (*dataset)[index];
			x = xt::expand_dims(x, 0);
			y = xt::expand_dims(y, 0);

			for (int i = 1; i < batch_size && ++index < dataset->size(); i++) {
				auto [x1, y1] = (*dataset)[index];
				x1 = xt::expand_dims(x1, 0);
				y1 = xt::expand_dims(y1, 0);
				x = xt::concatenate(xt::xtuple(x, x1), 0);
				y = xt::concatenate(xt::xtuple(y, y1), 0);
			}

			return make_tuple(x, y);
		}
	};

	auto begin() { return iterator<T>(0, dataset, batch_size); }
	auto end() { return iterator<T>(dataset->size(), dataset, batch_size); }
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
		return { (Tensor<MNIST_Dtype>) xt::view(images, index, xt::all()),
			     (Tensor<MNIST_Dtype>) xt::view(labels, index, xt::all()) };
	};
};
#endif