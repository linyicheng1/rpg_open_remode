#include "dataset.h"

const std::string & rmd::test::DatasetEntry::getImageFileName() const
{
  return image_file_name_;
}

std::string & rmd::test::DatasetEntry::getImageFileName()
{
  return image_file_name_;
}

const std::string & rmd::test::DatasetEntry::getDepthmapFileName() const
{
  return depthmap_file_name_;
}

std::string & rmd::test::DatasetEntry::getDepthmapFileName()
{
  return depthmap_file_name_;
}

const Eigen::Vector3f & rmd::test::DatasetEntry::getTranslation() const
{
  return translation_;
}

Eigen::Vector3f & rmd::test::DatasetEntry::getTranslation()
{
  return translation_;
}

const Eigen::Quaternionf & rmd::test::DatasetEntry::getQuaternion() const
{
  return quaternion_;
}

Eigen::Quaternionf & rmd::test::DatasetEntry::getQuaternion()
{
  return quaternion_;
}

rmd::test::Dataset::Dataset(const std::string &dataset_path, const std::string &sequence_file_path)
  : dataset_path_(dataset_path)
  , sequence_file_path_(sequence_file_path)
{
}

bool rmd::test::Dataset::readDataSequence(){
  dataset_.clear();
  std::string line;
  std::ifstream sequence_file_str(sequence_file_path_);
  if (sequence_file_str.is_open())
  {
    while (getline(sequence_file_str, line))
    {
      std::stringstream line_str(line);
      DatasetEntry data;
      std::string imgFileName;
      line_str >> imgFileName;
      data.getImageFileName() = imgFileName;
      const std::string depthmapFileName = imgFileName.substr(0, imgFileName.find('.')+1) + "depth";
      data.getDepthmapFileName() = depthmapFileName;
      line_str >> data.getTranslation().x();
      line_str >> data.getTranslation().y();
      line_str >> data.getTranslation().z();
      line_str >> data.getQuaternion().x();
      line_str >> data.getQuaternion().y();
      line_str >> data.getQuaternion().z();
      line_str >> data.getQuaternion().w();
      dataset_.push_back(data);
    }
    sequence_file_str.close();
    return true;
  }
  else
    return false;
}

bool rmd::test::Dataset::readImage(cv::Mat &img, const DatasetEntry &entry) const
{
  const boost::filesystem::path dataset_path(dataset_path_);
  const auto img_file_path = dataset_path / "images" / entry.getImageFileName();
  img = cv::imread(img_file_path.string(), CV_LOAD_IMAGE_GRAYSCALE);
  if(img.data == NULL)
    return false;
  else
    return true;
}

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

bool rmd::test::Dataset::readDepthmap(
    cv::Mat &depthmap,
    const DatasetEntry &entry,
    const size_t &width,
    const size_t &height) const
{
  const boost::filesystem::path dataset_path(dataset_path_);
  const auto depthmap_file_path = dataset_path / "depthmaps" / entry.getDepthmapFileName();
  std::ifstream depthmap_file_str(depthmap_file_path.string());
  if (depthmap_file_str.is_open())
  {
    depthmap.create(height, width, CV_32FC1);
    float f;
    for(size_t r=0; r<height; ++r)
    {
      for(size_t c=0; c<width; ++c)
      {
        depthmap_file_str >> f;
        depthmap.at<float>(r, c) = f;
      }
    }
    depthmap_file_str.close();
    return true;
  }
  else
    return false;
}

std::vector<rmd::test::DatasetEntry>::const_iterator rmd::test::Dataset::begin() const
{ return dataset_.begin(); }

std::vector<rmd::test::DatasetEntry>::const_iterator rmd::test::Dataset::end() const
{ return dataset_.end(); }