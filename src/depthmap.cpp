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

#include <rmd/depthmap.h>


/**
 * @brief 构造函数
 * @param width  宽度
 * @param height 高度
 * @param fx     针孔相机内参
 * @param cx
 * @param fy
 * @param cy
 * */
rmd::Depthmap::Depthmap(size_t width,
                        size_t height,
                        float fx,
                        float cx,
                        float fy,
                        float cy)
  : width_(width)
  , height_(height)
  , is_distorted_(false)
  , seeds_(width, height, rmd::PinholeCamera(fx, fy, cx, cy))
  , fx_(fx)
  , fy_(fy)
  , cx_(cx)
  , cy_(cy)
{
  // opencv格式的内参矩阵
  cv_K_ = (cv::Mat_<float>(3, 3) << fx, 0.0f, cx, 0.0f, fy, cy, 0.0f, 0.0f, 1.0f);
  // 构造 DepthmapDenoiser 类，去除深度地图的噪声类 
  denoiser_.reset(new rmd::DepthmapDenoiser(width_, height_));
  // 新建Mat文件
  output_depth_32fc1_ = cv::Mat_<float>(height_, width_);
  output_convergence_int_ = cv::Mat_<int>(height_, width_);
  img_undistorted_32fc1_.create(height_, width_, CV_32FC1);
  img_undistorted_8uc1_.create(height_, width_, CV_8UC1);
  ref_img_undistorted_8uc1_.create(height_, width_, CV_8UC1);
}

/**
 * @brief 初始化无畸变的地图
 * @param k1 畸变参数
 * @param k2
 * @param r1
 * @param r2
 * */
void rmd::Depthmap::initUndistortionMap(
    float k1,
    float k2,
    float r1,
    float r2)
{
  // 畸变参数矩阵
  cv_D_ = (cv::Mat_<float>(1, 4) << k1, k2, r1, r2);
  // 去除畸变
  // cv_K_ cv格式内参矩阵
  // cv_D_ cv格式畸变参数
  cv::initUndistortRectifyMap(
        cv_K_,
        cv_D_,
        cv::Mat_<double>::eye(3,3),
        cv_K_,
        cv::Size(width_, height_),
        CV_16SC2,
        undist_map1_, undist_map2_);
  is_distorted_ = true;
}

/**
 * @brief 初始化时调用，设置当前帧为参考帧
 * @param img_curr      当前帧图像数据
 * @param T_curr_world  当前帧到世界的坐标变换关系
 * @param min_depth     最小深度值
 * @param max_depth     最大深度值
 * 
 * */
bool rmd::Depthmap::setReferenceImage(
    const cv::Mat &img_curr,
    const rmd::SE3<float> &T_curr_world,
    const float &min_depth,
    const float &max_depth)
{
  // 根据min、max的 sigma 
  denoiser_->setLargeSigmaSq(max_depth-min_depth);
  // 输入当前的图片
  inputImage(img_curr);
  // 设置当前帧为参考帧？？目的应该是只初始化一次
  const bool ret = seeds_.setReferenceImage(reinterpret_cast<float*>(img_undistorted_32fc1_.data),
                                            T_curr_world,
                                            min_depth,
                                            max_depth);

  {
    // 在括号内线程锁起作用
    std::lock_guard<std::mutex> lock(ref_img_mutex_);
    // 复制参考帧图片
    img_undistorted_8uc1_.copyTo(ref_img_undistorted_8uc1_);
    // 设置参考帧位姿
    T_world_ref_ = T_curr_world.inv();
  }

  return ret;
}

/**
 * @brief 正常处理流程下更新深度滤波器
 * @param img_curr      当前的图像数据
 * @param T_curr_world  当前帧和世界坐标系的关系
 * 
 * */
void rmd::Depthmap::update(
    const cv::Mat &img_curr,
    const rmd::SE3<float> &T_curr_world)
{
  // 读取当前图像数据/去除畸变
  inputImage(img_curr);
  // 种子更新
  seeds_.update(
        reinterpret_cast<float*>(img_undistorted_32fc1_.data),
        T_curr_world);
}

/**
 * @brief 输入一张新的图片
 * @param img_8uyc1 图片数据
 * 
 * */
void rmd::Depthmap::inputImage(const cv::Mat &img_8uc1)
{
  if(is_distorted_)
  {// 如果有畸变，则去除畸变
    cv::remap(img_8uc1, img_undistorted_8uc1_, undist_map1_, undist_map2_, CV_INTER_LINEAR);
  }
  else
  {// 没有畸变的话直接复制
    img_undistorted_8uc1_ = img_8uc1;
  }
  img_undistorted_8uc1_.convertTo(img_undistorted_32fc1_, CV_32F, 1.0f/255.0f);
}

void rmd::Depthmap::downloadDepthmap()
{
  seeds_.downloadDepthmap(reinterpret_cast<float*>(output_depth_32fc1_.data));
}

void rmd::Depthmap::downloadDenoisedDepthmap(float lambda, int iterations)
{
  denoiser_->denoise(
        seeds_.getMu(),
        seeds_.getSigmaSq(),
        seeds_.getA(),
        seeds_.getB(),
        reinterpret_cast<float*>(output_depth_32fc1_.data),
        lambda,
        iterations);
}

const cv::Mat_<float> rmd::Depthmap::getDepthmap() const
{
  return output_depth_32fc1_;
}

void rmd::Depthmap::downloadConvergenceMap()
{
  seeds_.downloadConvergence(reinterpret_cast<int*>(output_convergence_int_.data));
}

const cv::Mat_<int> rmd::Depthmap::getConvergenceMap() const
{
  return output_convergence_int_;
}

const cv::Mat rmd::Depthmap::getReferenceImage() const
{
  return ref_img_undistorted_8uc1_;
}

size_t rmd::Depthmap::getConvergedCount() const
{
  return seeds_.getConvergedCount();
}

float rmd::Depthmap::getConvergedPercentage() const
{
  const size_t count = rmd::Depthmap::getConvergedCount();
  return static_cast<float>(count) / static_cast<float>(width_*height_) * 100.0f;
}

// Scale depth in [0,1] and cvt to color
// only for test and debug
cv::Mat rmd::Depthmap::scaleMat(const cv::Mat &depthmap)
{
  cv::Mat scaled_depthmap = depthmap.clone();
  double min_val, max_val;
  cv::minMaxLoc(scaled_depthmap, &min_val, &max_val);
  cv::Mat converted;
  scaled_depthmap = (scaled_depthmap - min_val) * 1.0 / (max_val - min_val);
  scaled_depthmap.convertTo(converted, CV_8UC1, 255);
  cv::Mat colored(converted.rows, converted.cols, CV_8UC3);
  cv::cvtColor(converted, colored, CV_GRAY2BGR);
  return colored;
}

float rmd::Depthmap::getDistFromRef() const
{
  return seeds_.getDistFromRef();
}
