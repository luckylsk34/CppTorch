#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>

template <class T,
          xt::layout_type L = XTENSOR_DEFAULT_LAYOUT,
          class A = XTENSOR_DEFAULT_ALLOCATOR(T),
          class SA = std::allocator<typename std::vector<T, A>::size_type>>
using Tensor = xt::xarray<T, L, A, SA>;

#endif