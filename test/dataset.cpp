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

#include "dataset.h"

// 获取数据集信息类

/**
 * @brief  获取图片文件名称
 * @return 图片文件名称
 * */
const std::string & rmd::test::DatasetEntry::getImageFileName() const
{
  return image_file_name_;
}
std::string & rmd::test::DatasetEntry::getImageFileName()
{
  return image_file_name_;
}

/**
 * @brief  获取深度图文件名
 * @return 深度地图文件名
 * */
const std::string & rmd::test::DatasetEntry::getDepthmapFileName() const
{
  return depthmap_file_name_;
}
std::string & rmd::test::DatasetEntry::getDepthmapFileName()
{
  return depthmap_file_name_;
}

/**
 * @brief  获得平移向量
 * @return 平移向量
 * */
const Eigen::Vector3f & rmd::test::DatasetEntry::getTranslation() const
{
  return translation_;
}
Eigen::Vector3f & rmd::test::DatasetEntry::getTranslation()
{
  return translation_;
}

/**
 * @brief  获得旋转四元数
 * @return 旋转四元数
 * */
const Eigen::Quaternionf & rmd::test::DatasetEntry::getQuaternion() const
{
  return quaternion_;
}
Eigen::Quaternionf & rmd::test::DatasetEntry::getQuaternion()
{
  return quaternion_;
}

/**
 * @brief 构造函数
 * @param dataset_path  数据集路径
 * @param sequence_file 数据集序列文件名称
 * */
rmd::test::Dataset::Dataset(
    const std::string &dataset_path,
    const std::string &sequence_file)
  : dataset_path_(dataset_path)
  , sequence_file_(sequence_file)
{
}
rmd::test::Dataset::Dataset(
    const std::string &sequence_file)
  : dataset_path_(std::string())
  , sequence_file_(sequence_file)
{
}
rmd::test::Dataset::Dataset()
  : dataset_path_(std::string())
  , sequence_file_(std::string())
{
}

/**
 * @brief  读取数据集序列
 * @param start  开始位置
 * @param end    结束位置，0表示末尾
 * */
bool rmd::test::Dataset::readDataSequence(size_t start, size_t end)
{
  // 路径非空
  if(dataset_path_.empty() || sequence_file_.empty())
  {
    return false;
  }
  // 清空之前的数据
  dataset_.clear();
  std::string line;
  // 序列文件路径
  const auto sequence_file_path = boost::filesystem::path(dataset_path_) / sequence_file_;
  // 打开文件
  std::ifstream sequence_file_str(sequence_file_path.string());
  if (sequence_file_str.is_open())
  {
    size_t line_cnt = 0;
    while (getline(sequence_file_str, line))
    {// 读取一行数据
      if(line_cnt >= start)
      {// 从开始
        if(line_cnt < end || 0 == end)
        {// 到结尾，0表示所有
          std::stringstream line_str(line);
          DatasetEntry data;
          std::string imgFileName;
          line_str >> imgFileName;
          // 得到当前对应图片名称
          data.getImageFileName() = imgFileName;
          // 得到当前对应深度图名称
          const std::string depthmapFileName = imgFileName.substr(0, imgFileName.find('.')+1) + "depth";
          data.getDepthmapFileName() = depthmapFileName;
          // 得到当前位姿信息
          line_str >> data.getTranslation().x();
          line_str >> data.getTranslation().y();
          line_str >> data.getTranslation().z();
          line_str >> data.getQuaternion().x();
          line_str >> data.getQuaternion().y();
          line_str >> data.getQuaternion().z();
          line_str >> data.getQuaternion().w();
          // 存储在 dataset_ 数据中
          dataset_.push_back(data);
        }
      }
      line_cnt += 1;
    }
    sequence_file_str.close();
    return true;
  }
  else
    return false;
}
/**
 * @brief 读取数据集中所有信息
 * */
bool rmd::test::Dataset::readDataSequence()
{
  return readDataSequence(0, 0);
}

/**
 * @brief 获取图片一张
 * @param img         返回图片
 * @param file_name   文件名称
 * @return            是否读取成功
 * */
bool rmd::test::Dataset::readImage(cv::Mat &img, const char *file_name) const
{
  const boost::filesystem::path dataset_path(dataset_path_);
  const auto img_file_path = dataset_path / "images" / file_name;
  img = cv::imread(img_file_path.string(), CV_LOAD_IMAGE_GRAYSCALE);
  if(img.data == NULL)
    return false;
  else
    return true;
}

/**
 * @brief 获取图片一张
 * @param img         返回图片
 * @param entry   文件名称
 * @return            是否读取成功
 * */
bool rmd::test::Dataset::readImage(cv::Mat &img, const DatasetEntry &entry) const
{
  return readImage(img, entry.getImageFileName().c_str());
}

/**
 * @brief 获取相机位姿
 * @param pose    SE3位姿
 * @param entry   文件名称
 * */
void rmd::test::Dataset::readCameraPose(rmd::SE3<float> &pose, const DatasetEntry &entry) const
{
  pose = rmd::SE3<float>(
        entry.getQuaternion().w(),
        entry.getQuaternion().x(),
        entry.getQuaternion().y(),
        entry.getQuaternion().z(),
        entry.getTranslation().x(),
        entry.getTranslation().y(),
        entry.getTranslation().z()
        );
}

/**
 * @brief 读取一张深度地图
 * @param depthmap 返回的深度地图
 * @param entry    文件名称
 * @param width    地图宽
 * @param height   地图长
 * @return  是否读取成功
 * */
bool rmd::test::Dataset::readDepthmap(
    cv::Mat &depthmap,
    const DatasetEntry &entry,
    const size_t &width,
    const size_t &height) const
{
  // 获得地图文件名称
  const boost::filesystem::path dataset_path(dataset_path_);
  const auto depthmap_file_path = dataset_path / "depthmaps" / entry.getDepthmapFileName();
  std::ifstream depthmap_file_str(depthmap_file_path.string());

  if (depthmap_file_str.is_open())
  {
    // 创建一张同样大小的图片
    depthmap.create(height, width, CV_32FC1);
    float z;
    for(size_t r=0; r<height; ++r)
    {
      for(size_t c=0; c<width; ++c)
      {
        // 读取深度值
        depthmap_file_str >> z;
        // 给指定位置赋值
        depthmap.at<float>(r, c) = z / 100.0f;
      }
    }
    depthmap_file_str.close();
    return true;
  }
  else
    return false;
}

// 获取序列文件中的数据
std::vector<rmd::test::DatasetEntry>::const_iterator rmd::test::Dataset::begin() const
{ return dataset_.begin(); }

std::vector<rmd::test::DatasetEntry>::const_iterator rmd::test::Dataset::end() const
{ return dataset_.end(); }

const rmd::test::DatasetEntry & rmd::test::Dataset::operator()(size_t index) const
{
  return dataset_.at(index);
}

// 从环境变量中得到文件路径
bool rmd::test::Dataset::loadPathFromEnv()
{
  const char *env_path = std::getenv(data_path_env_var);
  if(nullptr != env_path)
  {
    dataset_path_ = std::string(env_path);
    return true;
  }
  return false;
}

const char * rmd::test::Dataset::getDataPathEnvVar()
{
  return data_path_env_var;
}
