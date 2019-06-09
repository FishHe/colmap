
// Xu Q , Tao W . Multi-View Stereo with Asymmetric Checkerboard Propagation and Multi-Hypothesis Joint View Selection[J]. 2018.
// Paper: https://arxiv.org/abs/1805.07920
// Written by FishHe: https://github.com/FishHe
// Notes: https://github.com/FishHe/SFMVS/blob/master/2-MVS/01-DepthEstimation/ACMH.md
// Email: zhangdengsu@whu.edu.cn

#ifndef COLMAP_SRC_MVS_ACMH_PATCH_MATCH_CUDA_H_
#define COLMAP_SRC_MVS_ACMH_PATCH_MATCH_CUDA_H_

#include <cuda_runtime.h>

#include "mvs/depth_map.h"
#include "mvs/normal_map.h"
#include "mvs/patch_match.h"
#include "mvs/acmh_patch_match.h"
#include "mvs/gpu_mat_prng.h"
#include "mvs/gpu_mat_ref_image.h"

namespace colmap {
  namespace mvs {
    class AcmhPatchMatchCuda {
      AcmhPatchMatchCuda(const AcmhPatchMatchOptions& options,
        const PatchMatch::Problem& problem);
      ~AcmhPatchMatchCuda();

      void Run();

      DepthMap GetDepthMap() const;
      NormalMap GetNormalMap() const;
      Mat<float> GetSelProbMap() const;
      std::vector<int> GetConsistentImageIdxs() const;

    private:

      template <int kWindowSize, int kWindowStep>
      void RunWithWindowSizeAndStep();

      void ComputeCudaConfig();

      void InitRefImage();
      void InitSourceImages();
      void InitTransforms();
      void InitWorkspaceMemory();

      const AcmhPatchMatchOptions options_;
      const PatchMatch::Problem problem_;

      // Dimensions for checkerboard from black to red.
      dim3 checker_block_size_;
      dim3 checker_grid_size_;
      // Dimensions for element-wise operations, i.e. one thread per pixel.
      dim3 elem_wise_block_size_;
      dim3 elem_wise_grid_size_;

      // Original (not rotated) dimension of reference image.
      size_t ref_width_;
      size_t ref_height_;

      // Gpu arrays.
      std::unique_ptr<CudaArrayWrapper<uint8_t>> ref_image_device_;
      std::unique_ptr<CudaArrayWrapper<uint8_t>> src_images_device_;
      std::unique_ptr<CudaArrayWrapper<float>> src_depth_maps_device_;

      // Relative poses from rotated versions of reference image to source images
      // corresponding to _rotationInHalfPi:
      //
      //    [S(1), S(2), S(3), ..., S(n)]
      //
      // where n is the number of source images and:
      //
      //    S(i) = [K_i(0, 0), K_i(0, 2), K_i(1, 1), K_i(1, 2), R_i(:), T_i(:)
      //            C_i(:), P(:), P^-1(:)]
      //
      // where i denotes the index of the source image and K is its calibration.
      // R, T, C, P, P^-1 denote the relative rotation, translation, camera
      // center, projection, and inverse projection from there reference to the
      // i-th source image.
      std::unique_ptr<CudaArrayWrapper<float>> poses_device_[4];

      // Calibration matrix for rotated versions of reference image
      // as {K[0, 0], K[0, 2], K[1, 1], K[1, 2]} corresponding to _rotationInHalfPi.
      float ref_K_host_[4][4];
      float ref_inv_K_host_[4][4];

      // Data for reference image.
      std::unique_ptr<GpuMatRefImage> ref_image_;
      std::unique_ptr<GpuMat<float>> depth_map_;
      std::unique_ptr<GpuMat<float>> normal_map_;
      std::unique_ptr<GpuMat<float>> cost_map_;
      std::unique_ptr<GpuMat<uint8_t>> best_view_map_;
      std::unique_ptr<GpuMatPRNG> rand_state_map_;
    };
  }
}

#endif