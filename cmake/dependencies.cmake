# Include Cmake
include(${CMAKE_CURRENT_LIST_DIR}/CPM.cmake)

# spdlog
CPMAddPackage(
  NAME spdlog
  GITHUB_REPOSITORY gabime/spdlog
  VERSION 1.15.3
)
# need to add the following cuz of the following compiler error
# error: identifier "_BitInt" is undefined in fmt/base.h
target_compile_definitions(spdlog INTERFACE FMT_USE_BITINT=0)

# nvbench
# https://github.com/NVIDIA/nvbench_demo/blob/main/CMakeLists.txt
CPMAddPackage("gh:NVIDIA/nvbench#main")

# gtest
CPMAddPackage(
  NAME gtest
  GITHUB_REPOSITORY google/googletest
  VERSION 1.17.0
  OPTIONS
    "INSTALL_GTEST OFF"
    "gtest_force_shared_crt ON"
)
