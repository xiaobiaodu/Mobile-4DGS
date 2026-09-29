/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/adam.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::function<int*(size_t N)> resizeIntFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return t.contiguous().data_ptr<int>();
    };
    return lambda;
}

std::function<float*(size_t N)> resizeFloatFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return t.contiguous().data_ptr<float>();
    };
    return lambda;
}

std::tuple<int, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& metric_map,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
    const float mult,
	const bool prefiltered,
	const bool debug,
	const bool get_flag)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHAFFELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  torch::Tensor sampleBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  std::function<char*(size_t)> sampleFunc = resizeFunctional(sampleBuffer);

  float* accum_metric_counts_ptr = nullptr;

  torch::Tensor metricCount = torch::empty({0}, float_opts);

  if(get_flag)
  {
	metricCount = torch::full({P}, 0, float_opts);
	accum_metric_counts_ptr = metricCount.contiguous().data<float>();
  }
  
  int rendered = 0;
  int num_buckets = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  auto tup = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
		sampleFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(),
		metric_map.contiguous().data<int>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
        mult,
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		debug,
		get_flag,
		accum_metric_counts_ptr);

		rendered = std::get<0>(tup);
		num_buckets = std::get<1>(tup);
  }
  return std::make_tuple(rendered, num_buckets, out_color, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, metricCount);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
BuildStaticGaussianCacheCUDA(
	const torch::Tensor& geomBuffer,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& static_colors,
	const torch::Tensor& static_global_ids,
	const int static_point_count,
	const int static_rendered_count,
	const int total_point_count)
{
	TORCH_CHECK(static_point_count >= 0, "static_point_count must be non-negative");
	TORCH_CHECK(static_rendered_count >= 0, "static_rendered_count must be non-negative");
	TORCH_CHECK(total_point_count >= static_point_count,
		"total_point_count must be at least static_point_count");
	TORCH_CHECK(geomBuffer.is_cuda() && geomBuffer.is_contiguous(),
		"geomBuffer must be a contiguous CUDA tensor");
	TORCH_CHECK(binningBuffer.is_cuda() && binningBuffer.is_contiguous(),
		"binningBuffer must be a contiguous CUDA tensor");
	TORCH_CHECK(geomBuffer.scalar_type() == torch::kUInt8 &&
		binningBuffer.scalar_type() == torch::kUInt8,
		"geomBuffer and binningBuffer must be byte tensors");
	TORCH_CHECK(static_global_ids.is_cuda() && static_global_ids.is_contiguous(),
		"static_global_ids must be a contiguous CUDA tensor");
	TORCH_CHECK(static_global_ids.scalar_type() == torch::kInt64,
		"static_global_ids must have dtype torch.int64");
	TORCH_CHECK(static_global_ids.numel() == static_point_count,
		"static_global_ids length must equal static_point_count");
	TORCH_CHECK(static_point_count == 0 || geomBuffer.numel() > 0,
		"a non-empty static set requires a geometry buffer");
	TORCH_CHECK(static_rendered_count == 0 || binningBuffer.numel() > 0,
		"non-empty static entries require a binning buffer");

	if (static_colors.numel() > 0)
	{
		TORCH_CHECK(static_colors.is_cuda() && static_colors.is_contiguous(),
			"static_colors must be a contiguous CUDA tensor when provided");
		TORCH_CHECK(static_colors.scalar_type() == torch::kFloat32,
			"static_colors must have dtype torch.float32");
		TORCH_CHECK(static_colors.dim() == 2 &&
			static_colors.size(0) == static_point_count &&
			static_colors.size(1) == NUM_CHAFFELS,
			"static_colors must have shape [static_point_count, 3]");
	}

	auto float_options = torch::TensorOptions()
		.dtype(torch::kFloat32)
		.device(geomBuffer.device());
	auto int64_options = torch::TensorOptions()
		.dtype(torch::kInt64)
		.device(geomBuffer.device());
	torch::Tensor static_entries = torch::empty(
		{static_rendered_count, 2}, int64_options);
	torch::Tensor combined_means2D = torch::empty(
		{total_point_count, 2}, float_options);
	torch::Tensor combined_conic = torch::empty(
		{total_point_count, 4}, float_options);
	torch::Tensor combined_features = torch::empty(
		{total_point_count, NUM_CHAFFELS}, float_options);

	const float* static_colors_ptr = static_colors.numel() > 0
		? static_colors.data_ptr<float>() : nullptr;
	CudaRasterizer::Rasterizer::buildStaticCache(
		static_point_count,
		static_rendered_count,
		static_colors_ptr,
		static_global_ids.data_ptr<int64_t>(),
		reinterpret_cast<char*>(geomBuffer.data_ptr()),
		reinterpret_cast<char*>(binningBuffer.data_ptr()),
		static_entries.data_ptr<int64_t>(),
		combined_means2D.data_ptr<float>(),
		combined_conic.data_ptr<float>(),
		combined_features.data_ptr<float>());

	return std::make_tuple(
		static_entries,
		combined_means2D,
		combined_conic,
		combined_features);
}

std::tuple<torch::Tensor, torch::Tensor>
RasterizeGaussiansCachedCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& colors,
	const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& metric_map,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
	const int image_height,
	const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const float mult,
	const bool prefiltered,
	const bool debug,
	const bool get_flag,
	const torch::Tensor& static_entries,
	torch::Tensor& combined_means2D,
	torch::Tensor& combined_conic,
	torch::Tensor& combined_features,
	const int static_point_count,
	const torch::Tensor& dynamic_global_ids)
{
	TORCH_CHECK(!get_flag,
		"rasterize_gaussians_cached is inference-only and does not support get_flag");
	TORCH_CHECK(image_height > 0 && image_width > 0,
		"cached rasterization requires positive image dimensions");
	TORCH_CHECK(means3D.is_cuda() && means3D.is_contiguous(),
		"means3D must be a contiguous CUDA tensor");
	TORCH_CHECK(means3D.scalar_type() == torch::kFloat32 &&
		means3D.dim() == 2 && means3D.size(1) == 3,
		"means3D must have dtype float32 and shape [P, 3]");
	const int P = means3D.size(0);
	TORCH_CHECK(static_point_count >= 0,
		"static_point_count must be non-negative");
	TORCH_CHECK(dynamic_global_ids.is_cuda() && dynamic_global_ids.is_contiguous(),
		"dynamic_global_ids must be a contiguous CUDA tensor");
	TORCH_CHECK(dynamic_global_ids.scalar_type() == torch::kInt64 &&
		dynamic_global_ids.numel() == P,
		"dynamic_global_ids must be int64 with one entry per dynamic Gaussian");
	TORCH_CHECK(static_entries.is_cuda() && static_entries.is_contiguous() &&
		static_entries.scalar_type() == torch::kInt64 &&
		static_entries.dim() == 2 && static_entries.size(1) == 2,
		"static_entries must be a contiguous CUDA int64 tensor of shape [R_s, 2]");

	TORCH_CHECK(combined_means2D.is_cuda() && combined_means2D.is_contiguous() &&
		combined_means2D.scalar_type() == torch::kFloat32 &&
		combined_means2D.dim() == 2 && combined_means2D.size(1) == 2,
		"combined_means2D must be contiguous CUDA float32 [N, 2]");
	TORCH_CHECK(combined_conic.is_cuda() && combined_conic.is_contiguous() &&
		combined_conic.scalar_type() == torch::kFloat32 &&
		combined_conic.dim() == 2 && combined_conic.size(1) == 4,
		"combined_conic must be contiguous CUDA float32 [N, 4]");
	TORCH_CHECK(combined_features.is_cuda() && combined_features.is_contiguous() &&
		combined_features.scalar_type() == torch::kFloat32 &&
		combined_features.dim() == 2 &&
		combined_features.size(1) == NUM_CHAFFELS,
		"combined_features must be contiguous CUDA float32 [N, 3]");
	const int combined_point_count = combined_means2D.size(0);
	TORCH_CHECK(combined_conic.size(0) == combined_point_count &&
		combined_features.size(0) == combined_point_count,
		"combined cache arrays must have the same point count");
	TORCH_CHECK(static_point_count + P == combined_point_count,
		"static and dynamic compact point counts must exactly fill the cache");

	auto require_cuda_float_contiguous = [](
		const torch::Tensor& tensor, const char* name)
	{
		TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
			tensor.scalar_type() == torch::kFloat32,
			name, " must be a contiguous CUDA float32 tensor");
	};
	require_cuda_float_contiguous(background, "background");
	require_cuda_float_contiguous(opacity, "opacity");
	require_cuda_float_contiguous(viewmatrix, "viewmatrix");
	require_cuda_float_contiguous(projmatrix, "projmatrix");
	TORCH_CHECK(viewmatrix.dim() == 2 && viewmatrix.size(0) == 4 &&
		viewmatrix.size(1) == 4,
		"viewmatrix must have shape [4, 4]");
	TORCH_CHECK(projmatrix.dim() == 2 && projmatrix.size(0) == 4 &&
		projmatrix.size(1) == 4,
		"projmatrix must have shape [4, 4]");
	require_cuda_float_contiguous(campos, "campos");
	TORCH_CHECK(opacity.size(0) == P,
		"opacity must have one row per dynamic Gaussian");
	TORCH_CHECK(metric_map.is_cuda() && metric_map.is_contiguous() &&
		metric_map.scalar_type() == torch::kInt32,
		"metric_map must be a contiguous CUDA int32 tensor");

	if (colors.numel() > 0)
	{
		require_cuda_float_contiguous(colors, "colors");
		TORCH_CHECK(colors.dim() == 2 && colors.size(0) == P &&
			colors.size(1) == NUM_CHAFFELS,
			"colors must have shape [P, 3]");
	}
	if (sh.numel() > 0)
	{
		require_cuda_float_contiguous(sh, "sh");
		TORCH_CHECK(sh.size(0) == P,
			"sh must have one row per dynamic Gaussian");
	}
	TORCH_CHECK((colors.numel() > 0) != (sh.numel() > 0) || P == 0,
		"provide exactly one of colors or sh for non-empty dynamic data");

	if (cov3D_precomp.numel() > 0)
	{
		require_cuda_float_contiguous(cov3D_precomp, "cov3D_precomp");
		TORCH_CHECK(cov3D_precomp.size(0) == P,
			"cov3D_precomp must have one row per dynamic Gaussian");
		TORCH_CHECK(scales.numel() == 0 && rotations.numel() == 0,
			"provide covariance or scale/rotation, not both");
	}
	else if (P > 0)
	{
		require_cuda_float_contiguous(scales, "scales");
		require_cuda_float_contiguous(rotations, "rotations");
		TORCH_CHECK(scales.size(0) == P && rotations.size(0) == P,
			"scales and rotations must have one row per dynamic Gaussian");
	}

	TORCH_CHECK(!means3D.requires_grad() && !colors.requires_grad() &&
		!opacity.requires_grad() && !scales.requires_grad() &&
		!rotations.requires_grad() && !cov3D_precomp.requires_grad() &&
		!sh.requires_grad(),
		"rasterize_gaussians_cached is inference-only; call it under torch.no_grad()");

	auto float_options = means3D.options().dtype(torch::kFloat32);
	auto byte_options = torch::TensorOptions()
		.dtype(torch::kUInt8)
		.device(means3D.device());
	torch::Tensor out_color = torch::empty(
		{NUM_CHAFFELS, image_height, image_width}, float_options);
	torch::Tensor radii = torch::zeros(
		{P}, means3D.options().dtype(torch::kInt32));
	torch::Tensor geomBuffer = torch::empty({0}, byte_options);
	torch::Tensor binningBuffer = torch::empty({0}, byte_options);
	torch::Tensor mergeBuffer = torch::empty({0}, byte_options);
	torch::Tensor imgBuffer = torch::empty({0}, byte_options);
	torch::Tensor sampleBuffer = torch::empty({0}, byte_options);

	std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
	std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
	std::function<char*(size_t)> mergeFunc = resizeFunctional(mergeBuffer);
	std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
	std::function<char*(size_t)> sampleFunc = resizeFunctional(sampleBuffer);

	int M = sh.numel() > 0 ? sh.size(1) : 0;
	const float* colors_ptr = colors.numel() > 0
		? colors.data_ptr<float>() : nullptr;
	const float* sh_ptr = sh.numel() > 0 ? sh.data_ptr<float>() : nullptr;
	const float* scales_ptr = scales.numel() > 0
		? scales.data_ptr<float>() : nullptr;
	const float* rotations_ptr = rotations.numel() > 0
		? rotations.data_ptr<float>() : nullptr;
	const float* cov_ptr = cov3D_precomp.numel() > 0
		? cov3D_precomp.data_ptr<float>() : nullptr;

	CudaRasterizer::Rasterizer::forwardCached(
		geomFunc,
		binningFunc,
		mergeFunc,
		imgFunc,
		sampleFunc,
		P, degree, M,
		background.data_ptr<float>(),
		image_width, image_height,
		means3D.data_ptr<float>(),
		sh_ptr,
		colors_ptr,
		opacity.data_ptr<float>(),
		scales_ptr,
		scale_modifier,
		rotations_ptr,
		cov_ptr,
		metric_map.data_ptr<int>(),
		viewmatrix.data_ptr<float>(),
		projmatrix.data_ptr<float>(),
		campos.data_ptr<float>(),
		mult,
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.data_ptr<float>(),
		radii.data_ptr<int>(),
		static_entries.data_ptr<int64_t>(),
		static_entries.size(0),
		combined_means2D.data_ptr<float>(),
		combined_conic.data_ptr<float>(),
		combined_features.data_ptr<float>(),
		combined_point_count,
		static_point_count,
		dynamic_global_ids.data_ptr<int64_t>(),
		debug);

	return std::make_tuple(out_color, radii);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int B,
	const torch::Tensor& sampleBuffer,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 4}, means3D.options());  // abs
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHAFFELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R, B,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(sampleBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug);
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

void adamUpdate(
	torch::Tensor &param,
	torch::Tensor &param_grad,
	torch::Tensor &exp_avg,
	torch::Tensor &exp_avg_sq,
	torch::Tensor &visible,
	const float lr,
	const float b1,
	const float b2,
	const float eps,
	const uint32_t N,
	const uint32_t M
){
	ADAM::adamUpdate(
		param.contiguous().data<float>(),
		param_grad.contiguous().data<float>(),
		exp_avg.contiguous().data<float>(),
		exp_avg_sq.contiguous().data<float>(),
		visible.contiguous().data<bool>(),
		lr,
		b1,
		b2,
		eps,
		N,
		M);
}
