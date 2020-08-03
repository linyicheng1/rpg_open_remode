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

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <rmd/check_cuda_device.cuh>

/// Run all the tests that were declared with TEST()
// main 函数入口
int main(int argc, char **argv)
{
  // 确保cuda驱动、设备存在
  if(rmd::checkCudaDevice(argc, argv))
  {
    // 初始化
    testing::InitGoogleTest(&argc, argv);
    // 
    int ret = RUN_ALL_TESTS();
    // 重置cuda设备
    cudaDeviceReset();

    return ret;
  }
  else return EXIT_FAILURE;
}
