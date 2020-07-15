// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef RMD_SEED_UPDATE_CU
#define RMD_SEED_UPDATE_CU

#include <rmd/mvs_device_data.cuh>
#include <rmd/seed_matrix.cuh>
#include <rmd/texture_memory.cuh>

#include "triangulation.cu"

namespace rmd
{

__device__ __forceinline__
float normpdf(
    const float &x,
    const float &mu,
    const float & sigma_sq)
{
  return (expf(-(x-mu)*(x-mu) / (2.0f*sigma_sq))) * rsqrtf(2.0f*M_PI*sigma_sq);
}

/**
 * @brief 跟新深度滤波器，GPU实现函数
 * @param dev_ptr     参数指针
 * @param T_ref_curr  参考帧到当前帧坐标变换关系
 * 
 * */
__global__
void seedUpdateKernel(
    mvs::DeviceData *dev_ptr,
    SE3<float> T_ref_curr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= dev_ptr->width || y >= dev_ptr->height)
    return;

  const float xx = x+0.5f;
  const float yy = y+0.5f;

  // Check convergence status of seed
  // 是否当前点已经收敛或者发散了
  if( (ConvergenceStates::CONVERGED == tex2D(convergence_tex, xx, yy)) ||
      (ConvergenceStates::DIVERGED  == tex2D(convergence_tex, xx, yy)) )
    return;
  // 如果当前点依旧需要更新
  if( ConvergenceStates::UPDATE == tex2D(convergence_tex, xx, yy) )
  {
    // Bayesian update
    // Retrieve current estimations of parameters
    // 检索当前估计参数
    const float mu = tex2D(mu_tex, xx, yy);
    const float sigma_sq = tex2D(sigma_tex, xx, yy);
    const float a = tex2D(a_tex, xx, yy);
    const float b = tex2D(b_tex, xx, yy);

    // The pixel in reference frame
    // 参考帧像素点
    const float2 px_ref = make_float2((float)x, (float)y);
    // 参考帧相机平面点
    const float3 f_ref = normalize(dev_ptr->cam.cam2world(px_ref));

    // and the epipolar match
    // 极线匹配结果
    const float2 epipolar_match = tex2D(epipolar_matches_tex, xx, yy);
    const float3 f_epi_match  = normalize(dev_ptr->cam.cam2world(epipolar_match));
    // 三角化得到3d点位置
    const float3 pt_xyz_ref = triangulatenNonLin(
          f_ref,
          f_epi_match,
          T_ref_curr);
    // 去除掉在相机后面的点      
    if(pt_xyz_ref.z < 0.0f)
    {
      return;
    }
    // 当前测量得到的深度
    const float depth = norm(pt_xyz_ref);
    //float z = pt_xyz_ref.z;
    // 三角化带来的不确定度
    float tau = triangulationUncertainty(
          depth,
          f_ref,
          T_ref_curr.getTranslation(),
          dev_ptr->cam.getOnePixAngle());
    // 深度滤波器   
    // 计算mu
    const float tau_sq = tau * tau;
    const float s_sq = (tau_sq * sigma_sq) / (tau_sq + sigma_sq);
    const float m    = s_sq * (mu / sigma_sq + depth / tau_sq);
    // 
    float c1   = (a / (a+b)) * normpdf(depth, mu, sigma_sq+tau_sq);
    float c2   = (b / (a+b)) * (1.0f / dev_ptr->scene.depth_range);
    const float norm_const = c1 + c2;
    c1 = c1 / norm_const;
    c2 = c2 / norm_const;
    const float f = c1 * ((a + 1.0f) / (a + b + 1.0f)) + c2 *(a / (a + b + 1.0f));
    const float e = c1 * (( (a + 1.0f)*(a + 2.0f)) / ((a + b + 1.0f) * (a + b + 2.0f))) +
        c2 *(a*(a + 1.0f) / ((a + b + 1.0f) * (a + b + 2.0f)));

    if(isnan(c1*m))
    {
      return;
    }

    // 更新参数
    // 当前估计值 mu c1:上一次估计权重 c2:新一帧测量估计权重
    const float mu_prime = c1 * m + c2 * mu;
    // 更新方差
    dev_ptr->sigma->atXY(x, y) = c1 *(s_sq + m*m) + c2 * (sigma_sq + mu*mu) - mu_prime*mu_prime;
    dev_ptr->mu->atXY(x, y) = mu_prime;
    // 更新ab
    const float a_prime = ( e - f ) / ( f - e/f );
    dev_ptr->a->atXY(x, y) = a_prime;
    dev_ptr->b->atXY(x, y) = a_prime * ( 1.0f-f ) / f;
  }
  else if(ConvergenceStates::NO_MATCH == tex2D(convergence_tex, xx, yy))
  { // no match but projection inside the image: penalize the seed
    // 惩罚种子
    const float b = tex2D(b_tex, xx, yy) + 1.0f;
    dev_ptr->b->atXY(x, y) = b;
  }
  else if (ConvergenceStates::NOT_VISIBLE == tex2D(convergence_tex, xx, yy))
  { // no match, projection out of the image
  }
}

} // rmd namespace

#endif
