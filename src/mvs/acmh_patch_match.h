#ifndef COLMAP_SRC_MVS_ACMH_PATCH_MATCH_H_
#define COLMAP_SRC_MVS_ACMH_PATCH_MATCH_H_

namespace colmap {
  namespace mvs {

    struct AcmhPatchMatchOptions {
      // Index of the GPU used for patch match. For multi-GPU usage,
      // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
      std::string gpu_index = "-1";

      // Half window size to compute NCC photo-consistency cost.
      int window_radius = 5;

      // Number of pixels to skip when computing NCC. For a value of 1, every
      // pixel is used to compute the NCC. For larger values, only every n-th row
      // and column is used and the computation speed thereby increases roughly by
      // a factor of window_step^2. Note that not all combinations of window sizes
      // and steps produce nice results, especially if the step is greather than 2.
      int window_step = 1;

      // Parameters for bilaterally weighted NCC.
      double sigma_spatial = -1;
      double sigma_color = 0.2f;

      // Depth range in which to randomly sample depth hypotheses.
      double depth_min = -1.0f;
      double depth_max = -1.0f;

    };
  }
}




#endif