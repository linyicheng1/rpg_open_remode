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

#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include <rmd/check_cuda_device.cuh>

// 判断当前系统是否存在CUDA设备
bool rmd::checkCudaDevice(int argc, char **argv)
{
  printf("Running executable: %s\nChecking available CUDA-capable devices...\n", argv[0]);

  int dev_cnt;
  // 获取CUDA设备的数量
  cudaError err = cudaGetDeviceCount(&dev_cnt);
  if(cudaSuccess != err)
  {// 获取设备数量信息失败
    printf("ERROR: cudaGetDeviceCount %s\n", cudaGetErrorString(err));
    return false;
  }
  
  if(0 == dev_cnt)
  {// 得到的CUDA设备数量为零，输出错误信息
    printf("ERROR: no CUDA-capable device found.\n");
    return false;
  }

  cudaDeviceProp device_prop;
  device_prop.major = 0;
  device_prop.minor = 0;

  printf("%d CUDA-capable GPU detected:\n", dev_cnt);
  int dev_id;
  // 遍历所有的CUDA设备
  for(dev_id=0; dev_id<dev_cnt; ++dev_id)
  {
    // 读取CUDA设备的参数
    err = cudaGetDeviceProperties(&device_prop, dev_id);
    if(cudaSuccess != err)
    {// 读取失败，报错
      printf("ERROR: cudaGetDeviceProperties could not get properties for device %d. %s\n", dev_id, cudaGetErrorString(err));
    }
    else
    {// 读取成功则数据设备信息
      printf("Device %d - %s\n", dev_id, device_prop.name);
    }
  }

  dev_id = 0;
  const std::string device_arg("--device=");
  for(int i=1; i<argc; ++i)
  {// 遍历输入参数 "--device=" ,找到使用设备 dev_id
    const std::string arg(argv[i]);
    if(device_arg == arg.substr(0, device_arg.size()))
    {
      dev_id = atoi(arg.substr(device_arg.size(), arg.size()).c_str());
      printf("User-specified device: %d\n", dev_id);
      break;
    }
  }

  if( (dev_id<0) || (dev_id>dev_cnt-1) )
  {// 没有指定设备，报错
    printf("ERROR: invalid device ID specified. Please specify a value in [0, %d].\n", dev_cnt-1);
    return false;
  }
  // 获取指定设备的属性
  err = cudaGetDeviceProperties(&device_prop, dev_id);
  if(cudaSuccess != err)
  {// 读取属性失败
    printf("ERROR: cudaGetDeviceProperties %s\n", cudaGetErrorString(err));
    return false;
  }
  printf("Using GPU device %d: \"%s\" with compute capability %d.%d\n", dev_id, device_prop.name, device_prop.major, device_prop.minor);
  printf("GPU device %d has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
         dev_id, device_prop.multiProcessorCount, device_prop.major, device_prop.minor);
  // 版本信息
  const int version = (device_prop.major * 0x10 + device_prop.minor);

  if (version < 0x20)
  {// 版本太低了也不行
    printf("ERROR: a minimum CUDA compute 2.0 capability is required.\n");
    return false;
  }

  if (cudaComputeModeProhibited == device_prop.computeMode)
  {// 当前状态为禁止运行
    printf("ERROR: device is running in 'Compute Mode Prohibited'\n");
    return false;
  }

  if (device_prop.major < 1)
  {// 当前设备不兼容CUDA
    printf("ERROR: device %d is not a CUDA-capable GPU.\n", dev_id);
    return false;
  }
  // 指定使用设备
  err = cudaSetDevice(dev_id);
  if(cudaSuccess != err)
  {
    printf("ERROR: cudaSetDevice %s\n", cudaGetErrorString(err));
    return false;
  }

  return true;
}
