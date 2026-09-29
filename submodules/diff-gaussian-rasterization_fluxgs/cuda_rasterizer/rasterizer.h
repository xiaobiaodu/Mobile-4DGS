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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>
#include <cstdint>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static std::tuple<int,int> forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			std::function<char* (size_t)> sampleBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const int* metric_map,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
            const float mult,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			int* radii = nullptr,
			bool debug = false,
			bool get_flag = false,
			float* metricCount = nullptr);

		// Build the immutable part of an inference cache from buffers produced by
		// a normal static-only forward pass.  Static entries store
		// [tile|depth, global-id|local-id], while the compact static geometry is
		// copied once into the prefix of the persistent combined arrays.
		static void buildStaticCache(
			const int static_point_count,
			const int static_rendered_count,
			const float* static_colors_precomp,
			const int64_t* static_global_ids,
			char* static_geom_buffer,
			char* static_binning_buffer,
			int64_t* static_entries,
			float* combined_means2D,
			float* combined_conic,
			float* combined_features);

		// Inference-only cached forward.  It preprocesses and sorts only the
		// dynamic compact suffix, exactly merges that stream with the immutable
		// static stream, then calls the existing front-to-back renderer.
		static std::tuple<int,int> forwardCached(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> mergeBuffer,
			std::function<char* (size_t)> imageBuffer,
			std::function<char* (size_t)> sampleBuffer,
			const int P, int D, int M,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* opacities,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const int* metric_map,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float mult,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			int* radii,
			const int64_t* static_entries,
			const int static_rendered_count,
			float* combined_means2D,
			float* combined_conic,
			float* combined_features,
			const int combined_point_count,
			const int static_point_count,
			const int64_t* dynamic_global_ids,
			bool debug = false);

		static void backward(
			const int P, int D, int M, int R, int B,
			const float* background,
			const int width, int height,
			const float* means3D,
			const float* shs,
			const float* colors_precomp,
			const float* scales,
			const float scale_modifier,
			const float* rotations,
			const float* cov3D_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			char* sample_buffer,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dconic,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dmean3D,
			float* dL_dcov3D,
			float* dL_dsh,
			float* dL_dscale,
			float* dL_drot,
			bool debug);
	};
};

#endif
