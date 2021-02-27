#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xblas.hpp>
#include <xtensor-blas/xlinalg.hpp>
using namespace xt;

template <class T,
          layout_type L = XTENSOR_DEFAULT_LAYOUT,
          class A = XTENSOR_DEFAULT_ALLOCATOR(T),
          class SA = std::allocator<typename std::vector<T, A>::size_type>>
using Tensor = xarray<T, L, A, SA>;
// template <class T,
//           layout_type L = XTENSOR_DEFAULT_LAYOUT,
//           class A = XTENSOR_DEFAULT_ALLOCATOR(T),
//           class SA = std::allocator<typename std::vector<T, A>::size_type>>
// class Tensor : xarray <T, L, A, SA>
// {
// public:
// 	using xarray<T, L, A, SA>::xarray_container;
// };

#endif