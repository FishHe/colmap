#include "acmh_patch_match_cuda.h"

#include "util/cuda.h"
#include "util/cudacc.h"
#include "util/logging.h"

// THREADS_PER_BLOCK = BLOCK_W * BLOCK_H
// TILE is the data that will be processed in one checker iteration by the corresponding block's thread.
// SHARED is the area of ref_image that will be used for the current tile.
// With this block structure, kWindowRadius <= min(BLOCK_W,BLOCK_H)

// Change these values only if you're clear with the thread struture.
#define BLOCK_W 32
#define BLOCK_H (BLOCK_W / 2)
// Do not change these values.
#define TILE_W BLOCK_W
#define TILE_H (BLOCK_H * 2)
#define SHARED_W (BLOCK_W * 4)
#define SHARED_H (BLOCK_H * 3)


namespace colmap {
  namespace mvs {
               
    texture<uint8_t, cudaTextureType2D, cudaReadModeNormalizedFloat>
      ref_image_texture;
    texture<uint8_t, cudaTextureType2DLayered, cudaReadModeNormalizedFloat>
      src_images_texture;
    texture<float, cudaTextureType2DLayered, cudaReadModeElementType>
      src_depth_maps_texture;
    texture<float, cudaTextureType2D, cudaReadModeElementType> poses_texture;

    // Calibration of reference image as {fx, cx, fy, cy}.
    __constant__ float ref_K[4];
    // Calibration of reference image as {1/fx, -cx/fx, 1/fy, -cy/fy}.
    __constant__ float ref_inv_K[4];

    __device__ inline void Mat33DotVec3(const float mat[9], const float vec[3],
      float result[3]) {
      result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
      result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
      result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
    }

    __device__ inline void Mat33DotVec3Homogeneous(const float mat[9],
      const float vec[2],
      float result[2]) {
      const float inv_z = 1.0f / (mat[6] * vec[0] + mat[7] * vec[1] + mat[8]);
      result[0] = inv_z * (mat[0] * vec[0] + mat[1] * vec[1] + mat[2]);
      result[1] = inv_z * (mat[3] * vec[0] + mat[4] * vec[1] + mat[5]);
    }

    __device__ inline float DotProduct3(const float vec1[3], const float vec2[3]) {
      return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
    }

    __device__ inline float GenerateRandomDepth(const float depth_min,
      const float depth_max,
      curandState* rand_state) {
      return curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
    }

    // Generate random normal.
    __device__ inline void GenerateRandomNormal(const int row, const int col,
      curandState* rand_state,
      float normal[3]) {
      // Unbiased sampling of normal, according to George Marsaglia, "Choosing a
      // Point from the Surface of a Sphere", 1972.
      float v1 = 0.0f;
      float v2 = 0.0f;
      float s = 2.0f;
      while (s >= 1.0f) {
        v1 = 2.0f * curand_uniform(rand_state) - 1.0f;
        v2 = 2.0f * curand_uniform(rand_state) - 1.0f;
        s = v1 * v1 + v2 * v2;
      }

      const float s_norm = sqrt(1.0f - s);
      normal[0] = 2.0f * v1 * s_norm;
      normal[1] = 2.0f * v2 * s_norm;
      normal[2] = 1.0f - 2.0f * s;

      // Make sure normal is looking away from camera.
      const float view_ray[3] = { ref_inv_K[0] * col + ref_inv_K[1],
        ref_inv_K[2] * row + ref_inv_K[3], 1.0f };
      if (DotProduct3(normal, view_ray) > 0) {
        normal[0] = -normal[0];
        normal[1] = -normal[1];
        normal[2] = -normal[2];
      }
    }


    __device__ inline float PerturbDepth(const float perturbation,
      const float depth,
      curandState* rand_state) {
      const float depth_min = (1.0f - perturbation) * depth;
      const float depth_max = (1.0f + perturbation) * depth;
      return GenerateRandomDepth(depth_min, depth_max, rand_state);
    }

    __device__ inline void PerturbNormal(const int row, const int col,
      const float perturbation,
      const float normal[3],
      curandState* rand_state,
      float perturbed_normal[3],
      const int num_trials = 0) {
      // Perturbation rotation angles.
      const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
      const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
      const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

      const float sin_a1 = sin(a1);
      const float sin_a2 = sin(a2);
      const float sin_a3 = sin(a3);
      const float cos_a1 = cos(a1);
      const float cos_a2 = cos(a2);
      const float cos_a3 = cos(a3);

      // R = Rx * Ry * Rz
      float R[9];
      R[0] = cos_a2 * cos_a3;
      R[1] = -cos_a2 * sin_a3;
      R[2] = sin_a2;
      R[3] = cos_a1 * sin_a3 + cos_a3 * sin_a1 * sin_a2;
      R[4] = cos_a1 * cos_a3 - sin_a1 * sin_a2 * sin_a3;
      R[5] = -cos_a2 * sin_a1;
      R[6] = sin_a1 * sin_a3 - cos_a1 * cos_a3 * sin_a2;
      R[7] = cos_a3 * sin_a1 + cos_a1 * sin_a2 * sin_a3;
      R[8] = cos_a1 * cos_a2;

      // Perturb the normal vector.
      Mat33DotVec3(R, normal, perturbed_normal);

      // Make sure the perturbed normal is still looking in the same direction as
      // the viewing direction, otherwise try again but with smaller perturbation.
      const float view_ray[3] = { ref_inv_K[0] * col + ref_inv_K[1],
        ref_inv_K[2] * row + ref_inv_K[3], 1.0f };
      if (DotProduct3(perturbed_normal, view_ray) >= 0.0f) {
        const int kMaxNumTrials = 3;
        if (num_trials < kMaxNumTrials) {
          PerturbNormal(row, col, 0.5f * perturbation, normal, rand_state,
            perturbed_normal, num_trials + 1);
          return;
        }
        else {
          perturbed_normal[0] = normal[0];
          perturbed_normal[1] = normal[1];
          perturbed_normal[2] = normal[2];
          return;
        }
      }

      // Make sure normal has unit norm.
      const float inv_norm = rsqrt(DotProduct3(perturbed_normal, perturbed_normal));
      perturbed_normal[0] *= inv_norm;
      perturbed_normal[1] *= inv_norm;
      perturbed_normal[2] *= inv_norm;
    }

    __device__ inline void ComputePointAtDepth(const float row, const float col,
      const float depth, float point[3]) {
      point[0] = depth * (ref_inv_K[0] * col + ref_inv_K[1]);
      point[1] = depth * (ref_inv_K[2] * row + ref_inv_K[3]);
      point[2] = depth;
    }

    // Transfer depth on plane from viewing ray at row1 to row2. The returned
    // depth is the intersection of the viewing ray through row2 with the plane
    // at row1 defined by the given depth and normal.
    __device__ inline float PropagateDepth(const float depth1,
      const float normal1[3], const float row1,
      const float row2) {
      // Point along first viewing ray.
      const float x1 = depth1 * (ref_inv_K[2] * row1 + ref_inv_K[3]);
      const float y1 = depth1;
      // Point on plane defined by point along first viewing ray and plane normal1.
      const float x2 = x1 + normal1[2];
      const float y2 = y1 - normal1[1];

      // Origin of second viewing ray.
      // const float x3 = 0.0f;
      // const float y3 = 0.0f;
      // Point on second viewing ray.
      const float x4 = ref_inv_K[2] * row2 + ref_inv_K[3];
      // const float y4 = 1.0f;

      // Intersection of the lines ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4)).
      const float denom = x2 - x1 + x4 * (y1 - y2);
      const float kEps = 1e-5f;
      if (abs(denom) < kEps) {
        return depth1;
      }
      const float nom = y1 * x2 - x1 * y2;
      return nom / denom;
    }

    // First, compute triangulation angle between reference and source image for 3D
    // point. Second, compute incident angle between viewing direction of source
    // image and normal direction of 3D point. Both angles are cosine distances.
    __device__ inline void ComputeViewingAngles(const float point[3],
      const float normal[3],
      const int image_idx,
      float* cos_triangulation_angle,
      float* cos_incident_angle) {
      *cos_triangulation_angle = 0.0f;
      *cos_incident_angle = 0.0f;

      // Projection center of source image.
      float C[3];
      for (int i = 0; i < 3; ++i) {
        C[i] = tex2D(poses_texture, i + 16, image_idx);
      }

      // Ray from point to camera.
      const float SX[3] = { C[0] - point[0], C[1] - point[1], C[2] - point[2] };

      // Length of ray from reference image to point.
      const float RX_inv_norm = rsqrt(DotProduct3(point, point));

      // Length of ray from source image to point.
      const float SX_inv_norm = rsqrt(DotProduct3(SX, SX));

      *cos_incident_angle = DotProduct3(SX, normal) * SX_inv_norm;
      *cos_triangulation_angle = DotProduct3(SX, point) * RX_inv_norm * SX_inv_norm;
    }

    __device__ inline void ComposeHomography(const int image_idx, const int row,
      const int col, const float depth,
      const float normal[3], float H[9]) {
      // Calibration of source image.
      float K[4];
      for (int i = 0; i < 4; ++i) {
        K[i] = tex2D(poses_texture, i, image_idx);
      }

      // Relative rotation between reference and source image.
      float R[9];
      for (int i = 0; i < 9; ++i) {
        R[i] = tex2D(poses_texture, i + 4, image_idx);
      }

      // Relative translation between reference and source image.
      float T[3];
      for (int i = 0; i < 3; ++i) {
        T[i] = tex2D(poses_texture, i + 13, image_idx);
      }

      // Distance to the plane.
      const float dist =
        depth * (normal[0] * (ref_inv_K[0] * col + ref_inv_K[1]) +
          normal[1] * (ref_inv_K[2] * row + ref_inv_K[3]) + normal[2]);
      const float inv_dist = 1.0f / dist;

      const float inv_dist_N0 = inv_dist * normal[0];
      const float inv_dist_N1 = inv_dist * normal[1];
      const float inv_dist_N2 = inv_dist * normal[2];

      // Homography as H = K * (R - T * n' / d) * Kref^-1.
      H[0] = ref_inv_K[0] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
        K[1] * (R[6] + inv_dist_N0 * T[2]));
      H[1] = ref_inv_K[2] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
        K[1] * (R[7] + inv_dist_N1 * T[2]));
      H[2] = K[0] * (R[2] + inv_dist_N2 * T[0]) +
        K[1] * (R[8] + inv_dist_N2 * T[2]) +
        ref_inv_K[1] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
          K[1] * (R[6] + inv_dist_N0 * T[2])) +
        ref_inv_K[3] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
          K[1] * (R[7] + inv_dist_N1 * T[2]));
      H[3] = ref_inv_K[0] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
        K[3] * (R[6] + inv_dist_N0 * T[2]));
      H[4] = ref_inv_K[2] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
        K[3] * (R[7] + inv_dist_N1 * T[2]));
      H[5] = K[2] * (R[5] + inv_dist_N2 * T[1]) +
        K[3] * (R[8] + inv_dist_N2 * T[2]) +
        ref_inv_K[1] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
          K[3] * (R[6] + inv_dist_N0 * T[2])) +
        ref_inv_K[3] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
          K[3] * (R[7] + inv_dist_N1 * T[2]));
      H[6] = ref_inv_K[0] * (R[6] + inv_dist_N0 * T[2]);
      H[7] = ref_inv_K[2] * (R[7] + inv_dist_N1 * T[2]);
      H[8] = R[8] + ref_inv_K[1] * (R[6] + inv_dist_N0 * T[2]) +
        ref_inv_K[3] * (R[7] + inv_dist_N1 * T[2]) + inv_dist_N2 * T[2];
    }

    // The return values is 1 - NCC, so the range is [0, 2], the smaller the
    // value, the better the color consistency.
    struct PhotoConsistencyCostComputer {
      const int kWindowSize;
      const int kWindowStep;
      const int kWindowRadius;

      __device__ PhotoConsistencyCostComputer(
        const int k_window_size,
        const int k_window_step,
        const float sigma_spatial,
        const float sigma_color)
        : bilateral_weight_computer_(sigma_spatial, sigma_color),
        kWindowSize(k_window_size),
        kWindowStep(k_window_step),
        kWindowRadius(k_window_size / 2)
        {}

      // Maximum photo consistency cost as 1 - min(NCC).
      const float kMaxCost = 2.0f;

      // Image data in local window around patch.
      const float* local_ref_image = nullptr;

      // Precomputed sum of raw and squared image intensities.
      float local_ref_sum = 0.0f;
      float local_ref_squared_sum = 0.0f;

      // Index of source image.
      int src_image_idx = -1;

      // Center position of patch in reference image.
      int row = -1;
      int col = -1;

      // Center position of patch in local reference image.
      int local_row_center = -1;
      int local_col_center = -1;

      // Depth and normal for which to warp patch.
      float depth = 0.0f;
      const float* normal = nullptr;

      __device__ inline float Compute() const {
        float tform[9];
        ComposeHomography(src_image_idx, row, col, depth, normal, tform);

        float tform_step[9];
        for (int i = 0; i < 9; ++i) {
          tform_step[i] = kWindowStep * tform[i];
        }

        //const int thread_id = threadIdx.x;
        const int row_start = row - kWindowRadius;
        const int col_start = col - kWindowRadius;

        float col_src = tform[0] * col_start + tform[1] * row_start + tform[2];
        float row_src = tform[3] * col_start + tform[4] * row_start + tform[5];
        float z = tform[6] * col_start + tform[7] * row_start + tform[8];
        float base_col_src = col_src;
        float base_row_src = row_src;
        float base_z = z;

        const int local_row_start = local_row_center - kWindowRadius;
        const int local_col_start = local_col_center - kWindowRadius;
        int local_r = local_row_start;
        int local_c = local_col_start;
        //int ref_image_idx = THREADS_PER_BLOCK - kWindowRadius + thread_id;
        //int ref_image_base_idx = ref_image_idx;

        const float ref_center_color =
          local_ref_image[local_row_center * SHARED_W + local_col_center];
        const float ref_color_sum = local_ref_sum;
        const float ref_color_squared_sum = local_ref_squared_sum;
        float src_color_sum = 0.0f;
        float src_color_squared_sum = 0.0f;
        float src_ref_color_sum = 0.0f;
        float bilateral_weight_sum = 0.0f;

        for (int row = -kWindowRadius; row <= kWindowRadius; row += kWindowStep) {
          for (int col = -kWindowRadius; col <= kWindowRadius; col += kWindowStep) {
            const float inv_z = 1.0f / z;
            const float norm_col_src = inv_z * col_src + 0.5f;
            const float norm_row_src = inv_z * row_src + 0.5f;
            const float ref_color = local_ref_image[local_r * SHARED_W + local_c];
            const float src_color = tex2DLayered(src_images_texture, norm_col_src,
              norm_row_src, src_image_idx);

            const float bilateral_weight = bilateral_weight_computer_.Compute(
              row, col, ref_center_color, ref_color);

            const float bilateral_weight_src = bilateral_weight * src_color;

            src_color_sum += bilateral_weight_src;
            src_color_squared_sum += bilateral_weight_src * src_color;
            src_ref_color_sum += bilateral_weight_src * ref_color;
            bilateral_weight_sum += bilateral_weight;

            local_c += kWindowStep;

            // Accumulate warped source coordinates per row to reduce numerical
            // errors. Note that this is necessary since coordinates usually are in
            // the order of 1000s as opposed to the color values which are
            // normalized to the range [0, 1].
            col_src += tform_step[0];
            row_src += tform_step[3];
            z += tform_step[6];
          }

          local_r += kWindowStep;

          base_col_src += tform_step[1];
          base_row_src += tform_step[4];
          base_z += tform_step[7];

          col_src = base_col_src;
          row_src = base_row_src;
          z = base_z;
        }

        const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
        src_color_sum *= inv_bilateral_weight_sum;
        src_color_squared_sum *= inv_bilateral_weight_sum;
        src_ref_color_sum *= inv_bilateral_weight_sum;

        const float ref_color_var =
          ref_color_squared_sum - ref_color_sum * ref_color_sum;
        const float src_color_var =
          src_color_squared_sum - src_color_sum * src_color_sum;

        // Based on Jensen's Inequality for convex functions, the variance
        // should always be larger than 0. Do not make this threshold smaller.
        const float kMinVar = 1e-5f;
        if (ref_color_var < kMinVar || src_color_var < kMinVar) {
          return kMaxCost;
        }
        else {
          const float src_ref_color_covar =
            src_ref_color_sum - ref_color_sum * src_color_sum;
          const float src_ref_color_var = sqrt(ref_color_var * src_color_var);
          return max(0.0f,
            min(kMaxCost, 1.0f - src_ref_color_covar / src_ref_color_var));
        }
      }

    private:
      const BilateralWeightComputer bilateral_weight_computer_;
    };

    // Find index of minimum in given values.
    template <int kNumCosts>
    __device__ inline int FindMinCost(const float costs[kNumCosts]) {
      float min_cost = costs[0];
      int min_cost_idx = 0;
      for (int idx = 1; idx < kNumCosts; ++idx) {
        if (costs[idx] <= min_cost) {
          min_cost = costs[idx];
          min_cost_idx = idx;
        }
      }
      return min_cost_idx;
    }

    __device__ inline void ReadRefImageIntoSharedMemory(float* local_image) {
      // Read SHARED_W * SHARED_H data into shared memory.
      // SHARED_W = 3 * blockDim.x
      // SHARED_H = 4 * blockDim.y

      const int global_r_start = (blockIdx.y - 1)*blockDim.y + threadIdx.y;
      const int global_c_start = (blockIdx.x - 1)*blockDim.x + threadIdx.x;

      int global_r = global_r_start;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int global_c = global_c_start;
#pragma unroll
        for (int j = 0; j < 3; j++)
        {
          const int local_r = j * blockDim.y + threadIdx.y;
          const int local_c = i * blockDim.x + threadIdx.x;
          local_image[local_r * 3 * blockDim.x + local_c] = tex2D(ref_image_texture, global_c, global_r);
          global_c += blockDim.x;
        }
        global_r += blockDim.y;
      }
      __syncthreads();

    }

    // Initilize normal map.
    __global__ void InitNormalMap(GpuMat<float> normal_map,
      GpuMat<curandState> rand_state_map) {
      const int row = blockDim.y * blockIdx.y + threadIdx.y;
      const int col = blockDim.x * blockIdx.x + threadIdx.x;
      if (col < normal_map.GetWidth() && row < normal_map.GetHeight()) {
        curandState rand_state = rand_state_map.Get(row, col);
        float normal[3];
        GenerateRandomNormal(row, col, &rand_state, normal);
        normal_map.SetSlice(row, col, normal);
        rand_state_map.Set(row, col, rand_state);
      }
    }

    __global__ void ComputeInitialCost(const int kWindowSize,
      const int kWindowStep,
      GpuMat<float> cost_map,
      const GpuMat<float> depth_map,
      const GpuMat<float> normal_map,
      const GpuMat<float> ref_sum_image,
      const GpuMat<float> ref_squared_sum_image,
      const float sigma_spatial,
      const float sigma_color) {

      // Shared memory for current thread.
      __shared__ float local_ref_image[SHARED_W*SHARED_H];
      // Read 3x4 pixels' data from texture memory for each thread.
      ReadRefImageIntoSharedMemory(local_ref_image);

      PhotoConsistencyCostComputer pcc_computer(kWindowSize, kWindowStep,
        sigma_spatial, sigma_color);
      pcc_computer.local_ref_image = local_ref_image;

      float normal[3];
      pcc_computer.normal = normal;

      // Calculate costs for 2 pixels.
      const int row_start = blockIdx.y * blockDim.y + threadIdx.y;
      const int col = blockIdx.x * blockDim.x + threadIdx.x;
      if (col < cost_map.GetWidth())
      {
        int row = row_start;
        pcc_computer.col = col;
        pcc_computer.local_col_center = 1 * blockDim.x + threadIdx.x;
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
          if (row < cost_map.GetHeight()) {
            pcc_computer.row = row;
            pcc_computer.local_row_center = (1 + i) * blockDim.y + threadIdx.y;
            pcc_computer.depth = depth_map.Get(row, col);
            normal_map.GetSlice(row, col, normal);

            pcc_computer.local_ref_sum = ref_sum_image.Get(row, col);
            pcc_computer.local_ref_squared_sum = ref_squared_sum_image.Get(row, col);

            for (int image_idx = 0; image_idx < cost_map.GetDepth(); ++image_idx) {
              pcc_computer.src_image_idx = image_idx;
              cost_map.Set(row, col, image_idx, pcc_computer.Compute());
            }
          }
          else break;
          row += blockDim.y;
        }
      }
    }

    AcmhPatchMatchCuda::AcmhPatchMatchCuda(const AcmhPatchMatchOptions& options, const PatchMatch::Problem& problem)
      : options_(options),
        problem_(problem),
        ref_width_(0),
        ref_height_(0)
    {
      SetBestCudaDevice(std::stoi(options_.gpu_index));
      InitRefImage();
      InitSourceImages();
      InitTransforms();
      InitWorkspaceMemory();
    }

    AcmhPatchMatchCuda::~AcmhPatchMatchCuda()
    {
    }
    void AcmhPatchMatchCuda::Run()
    {
      #define CASE_WINDOW_RADIUS(window_radius, window_step)              \
        case window_radius:                                               \
          RunWithWindowSizeAndStep<2 * window_radius + 1, window_step>(); \
          break;

      #define CASE_WINDOW_STEP(window_step)                                 \
        case window_step:                                                   \
          switch (options_.window_radius) {                                 \
            CASE_WINDOW_RADIUS(1, window_step)                              \
            CASE_WINDOW_RADIUS(2, window_step)                              \
            CASE_WINDOW_RADIUS(3, window_step)                              \
            CASE_WINDOW_RADIUS(4, window_step)                              \
            CASE_WINDOW_RADIUS(5, window_step)                              \
            CASE_WINDOW_RADIUS(6, window_step)                              \
            CASE_WINDOW_RADIUS(7, window_step)                              \
            CASE_WINDOW_RADIUS(8, window_step)                              \
            CASE_WINDOW_RADIUS(9, window_step)                              \
            CASE_WINDOW_RADIUS(10, window_step)                             \
            CASE_WINDOW_RADIUS(11, window_step)                             \
            CASE_WINDOW_RADIUS(12, window_step)                             \
            CASE_WINDOW_RADIUS(13, window_step)                             \
            CASE_WINDOW_RADIUS(14, window_step)                             \
            CASE_WINDOW_RADIUS(15, window_step)                             \
            CASE_WINDOW_RADIUS(16, window_step)                             \
            CASE_WINDOW_RADIUS(17, window_step)                             \
            CASE_WINDOW_RADIUS(18, window_step)                             \
            CASE_WINDOW_RADIUS(19, window_step)                             \
            CASE_WINDOW_RADIUS(20, window_step)                             \
            default: {                                                      \
              std::cerr << "Error: Window size not supported" << std::endl; \
              break;                                                        \
            }                                                               \
          }                                                                 \
          break;

            switch (options_.window_step) {
              CASE_WINDOW_STEP(1)
                CASE_WINDOW_STEP(2)
            default: {
                std::cerr << "Error: Window step not supported" << std::endl;
                break;
              }
            }

      #undef SWITCH_WINDOW_RADIUS
      #undef CALL_RUN_FUNC
    }
    DepthMap AcmhPatchMatchCuda::GetDepthMap() const
    {
      return DepthMap();
    }
    NormalMap AcmhPatchMatchCuda::GetNormalMap() const
    {
      return NormalMap();
    }
    Mat<float> AcmhPatchMatchCuda::GetSelProbMap() const
    {
      return Mat<float>();
    }
    std::vector<int> AcmhPatchMatchCuda::GetConsistentImageIdxs() const
    {
      return std::vector<int>();
    }

    template <int kWindowSize, int kWindowStep>
    void AcmhPatchMatchCuda::RunWithWindowSizeAndStep() {
      // Wait for all initializations to finish.
      CUDA_SYNC_AND_CHECK();

      CudaTimer total_timer;
      CudaTimer init_timer;

      ComputeCudaConfig();
      //ComputeInitialCost<kWindowSize, kWindowStep>
      //  <<<elem_wise_grid_size_, elem_wise_block_size_ >>>(
      //    *cost_map_, *depth_map_, *normal_map_, *ref_image_->sum_image,
      //    *ref_image_->squared_sum_image, options_.sigma_spatial,
      //    options_.sigma_color);
      CUDA_SYNC_AND_CHECK();

      init_timer.Print("Initialization");

    }
    void AcmhPatchMatchCuda::ComputeCudaConfig()
    {
      // red iteration     black iteration
      // - * - * - *       * - * - * -
      // * - * - * -       - * - * - *
      // - * - * - *       * - * - * -
      // * - * - * -       - * - * - *
      //
      // * : changing   - : unchanged
      checker_block_size_.x = BLOCK_W;
      checker_block_size_.y = BLOCK_H;
      checker_block_size_.z = 1;
      checker_grid_size_.x = (depth_map_->GetWidth() - 1) / TILE_W + 1;
      checker_grid_size_.y = (depth_map_->GetHeight() - 1) / TILE_H + 1;
      checker_grid_size_.z = 1;

      //elem_wise_block_size_.x = BLOCK_W;
      //elem_wise_block_size_.y = BLOCK_H;
      //elem_wise_block_size_.z = 1;
      //elem_wise_grid_size_.x = (depth_map_->GetWidth() - 1) / BLOCK_W + 1;
      //elem_wise_grid_size_.y = (depth_map_->GetHeight() - 1) / BLOCK_H + 1;
      //elem_wise_grid_size_.z = 1;
    }
    void AcmhPatchMatchCuda::InitRefImage()
    {
      const Image& ref_image = problem_.images->at(problem_.ref_image_idx);

      ref_width_ = ref_image.GetWidth();
      ref_height_ = ref_image.GetHeight();

      // Upload to device.
      ref_image_.reset(new GpuMatRefImage(ref_width_, ref_height_));
      const std::vector<uint8_t> ref_image_array =
        ref_image.GetBitmap().ConvertToRowMajorArray();
      ref_image_->Filter(ref_image_array.data(), options_.window_radius,
        options_.window_step, options_.sigma_spatial,
        options_.sigma_color);

      ref_image_device_.reset(
        new CudaArrayWrapper<uint8_t>(ref_width_, ref_height_, 1));
      ref_image_device_->CopyFromGpuMat(*ref_image_->image);

      // Create texture.
      ref_image_texture.addressMode[0] = cudaAddressModeBorder;
      ref_image_texture.addressMode[1] = cudaAddressModeBorder;
      ref_image_texture.addressMode[2] = cudaAddressModeBorder;
      ref_image_texture.filterMode = cudaFilterModePoint;
      ref_image_texture.normalized = false;
      CUDA_SAFE_CALL(
        cudaBindTextureToArray(ref_image_texture, ref_image_device_->GetPtr()));
    }
    void AcmhPatchMatchCuda::InitSourceImages()
    {
      // Determine maximum image size.
      size_t max_width = 0;
      size_t max_height = 0;
      for (const auto image_idx : problem_.src_image_idxs) {
        const Image& image = problem_.images->at(image_idx);
        if (image.GetWidth() > max_width) {
          max_width = image.GetWidth();
        }
        if (image.GetHeight() > max_height) {
          max_height = image.GetHeight();
        }
      }

      // Upload source images to device.
      {
        // Copy source images to contiguous memory block.
        const uint8_t kDefaultValue = 0;
        std::vector<uint8_t> src_images_host_data(
          static_cast<size_t>(max_width * max_height *
            problem_.src_image_idxs.size()),
          kDefaultValue);
        for (size_t i = 0; i < problem_.src_image_idxs.size(); ++i) {
          const Image& image = problem_.images->at(problem_.src_image_idxs[i]);
          const Bitmap& bitmap = image.GetBitmap();
          uint8_t* dest = src_images_host_data.data() + max_width * max_height * i;
          for (size_t r = 0; r < image.GetHeight(); ++r) {
            memcpy(dest, bitmap.GetScanline(r), image.GetWidth() * sizeof(uint8_t));
            dest += max_width;
          }
        }

        // Upload to device.
        src_images_device_.reset(new CudaArrayWrapper<uint8_t>(
          max_width, max_height, problem_.src_image_idxs.size()));
        src_images_device_->CopyToDevice(src_images_host_data.data());

        // Create source images texture.
        src_images_texture.addressMode[0] = cudaAddressModeBorder;
        src_images_texture.addressMode[1] = cudaAddressModeBorder;
        src_images_texture.addressMode[2] = cudaAddressModeBorder;
        src_images_texture.filterMode = cudaFilterModeLinear;
        src_images_texture.normalized = false;
        CUDA_SAFE_CALL(cudaBindTextureToArray(src_images_texture,
          src_images_device_->GetPtr()));
      }
    }
    void AcmhPatchMatchCuda::InitTransforms()
    {
      const Image& ref_image = problem_.images->at(problem_.ref_image_idx);

      //////////////////////////////////////////////////////////////////////////////
      // Generate rotated versions (counter-clockwise) of calibration matrix.
      //////////////////////////////////////////////////////////////////////////////

      for (size_t i = 0; i < 4; ++i) {
        ref_K_host_[i][0] = ref_image.GetK()[0];
        ref_K_host_[i][1] = ref_image.GetK()[2];
        ref_K_host_[i][2] = ref_image.GetK()[4];
        ref_K_host_[i][3] = ref_image.GetK()[5];
      }

      // Rotated by 90 degrees.
      std::swap(ref_K_host_[1][0], ref_K_host_[1][2]);
      std::swap(ref_K_host_[1][1], ref_K_host_[1][3]);
      ref_K_host_[1][3] = ref_width_ - 1 - ref_K_host_[1][3];

      // Rotated by 180 degrees.
      ref_K_host_[2][1] = ref_width_ - 1 - ref_K_host_[2][1];
      ref_K_host_[2][3] = ref_height_ - 1 - ref_K_host_[2][3];

      // Rotated by 270 degrees.
      std::swap(ref_K_host_[3][0], ref_K_host_[3][2]);
      std::swap(ref_K_host_[3][1], ref_K_host_[3][3]);
      ref_K_host_[3][1] = ref_height_ - 1 - ref_K_host_[3][1];

      // Extract 1/fx, -cx/fx, fy, -cy/fy.
      for (size_t i = 0; i < 4; ++i) {
        ref_inv_K_host_[i][0] = 1.0f / ref_K_host_[i][0];
        ref_inv_K_host_[i][1] = -ref_K_host_[i][1] / ref_K_host_[i][0];
        ref_inv_K_host_[i][2] = 1.0f / ref_K_host_[i][2];
        ref_inv_K_host_[i][3] = -ref_K_host_[i][3] / ref_K_host_[i][2];
      }

      // Bind 0 degrees version to constant global memory.
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(ref_K, ref_K_host_[0], sizeof(float) * 4, 0,
        cudaMemcpyHostToDevice));
      CUDA_SAFE_CALL(cudaMemcpyToSymbol(ref_inv_K, ref_inv_K_host_[0],
        sizeof(float) * 4, 0,
        cudaMemcpyHostToDevice));

      //////////////////////////////////////////////////////////////////////////////
      // Generate rotated versions of camera poses.
      //////////////////////////////////////////////////////////////////////////////

      float rotated_R[9];
      memcpy(rotated_R, ref_image.GetR(), 9 * sizeof(float));

      float rotated_T[3];
      memcpy(rotated_T, ref_image.GetT(), 3 * sizeof(float));

      // Matrix for 90deg rotation around Z-axis in counter-clockwise direction.
      const float R_z90[9] = { 0, 1, 0, -1, 0, 0, 0, 0, 1 };

      for (size_t i = 0; i < 4; ++i) {
        const size_t kNumTformParams = 4 + 9 + 3 + 3 + 12 + 12;
        std::vector<float> poses_host_data(kNumTformParams *
          problem_.src_image_idxs.size());
        int offset = 0;
        for (const auto image_idx : problem_.src_image_idxs) {
          const Image& image = problem_.images->at(image_idx);

          const float K[4] = { image.GetK()[0], image.GetK()[2], image.GetK()[4],
            image.GetK()[5] };
          memcpy(poses_host_data.data() + offset, K, 4 * sizeof(float));
          offset += 4;

          float rel_R[9];
          float rel_T[3];
          ComputeRelativePose(rotated_R, rotated_T, image.GetR(), image.GetT(),
            rel_R, rel_T);
          memcpy(poses_host_data.data() + offset, rel_R, 9 * sizeof(float));
          offset += 9;
          memcpy(poses_host_data.data() + offset, rel_T, 3 * sizeof(float));
          offset += 3;

          float C[3];
          ComputeProjectionCenter(rel_R, rel_T, C);
          memcpy(poses_host_data.data() + offset, C, 3 * sizeof(float));
          offset += 3;

          float P[12];
          ComposeProjectionMatrix(image.GetK(), rel_R, rel_T, P);
          memcpy(poses_host_data.data() + offset, P, 12 * sizeof(float));
          offset += 12;

          float inv_P[12];
          ComposeInverseProjectionMatrix(image.GetK(), rel_R, rel_T, inv_P);
          memcpy(poses_host_data.data() + offset, inv_P, 12 * sizeof(float));
          offset += 12;
        }

        poses_device_[i].reset(new CudaArrayWrapper<float>(
          kNumTformParams, problem_.src_image_idxs.size(), 1));
        poses_device_[i]->CopyToDevice(poses_host_data.data());

        RotatePose(R_z90, rotated_R, rotated_T);
      }

      poses_texture.addressMode[0] = cudaAddressModeBorder;
      poses_texture.addressMode[1] = cudaAddressModeBorder;
      poses_texture.addressMode[2] = cudaAddressModeBorder;
      poses_texture.filterMode = cudaFilterModePoint;
      poses_texture.normalized = false;
      CUDA_SAFE_CALL(
        cudaBindTextureToArray(poses_texture, poses_device_[0]->GetPtr()));
    }
    void AcmhPatchMatchCuda::InitWorkspaceMemory()
    {
      rand_state_map_.reset(new GpuMatPRNG(ref_width_, ref_height_));

      depth_map_.reset(new GpuMat<float>(ref_width_, ref_height_));
      depth_map_->FillWithRandomNumbers(options_.depth_min, options_.depth_max,
        *rand_state_map_);

      normal_map_.reset(new GpuMat<float>(ref_width_, ref_height_, 3));
      InitNormalMap <<<elem_wise_grid_size_, elem_wise_block_size_ >>> (
        *normal_map_, *rand_state_map_);

      cost_map_.reset(new GpuMat<float>(ref_width_, ref_height_,
        problem_.src_image_idxs.size()));

      best_view_map_.reset(new GpuMat<uint8_t>(ref_width_, ref_height_,
        problem_.src_image_idxs.size()));
    }
  }
}

