# Include Cmake
include(${CMAKE_CURRENT_LIST_DIR}/CPM.cmake)

# spdlog
CPMAddPackage(
  NAME spdlog
  GITHUB_REPOSITORY gabime/spdlog
  VERSION 1.15.3
)

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

# cutlass
CPMAddPackage(
    NAME cutlass
    GITHUB_REPOSITORY nvidia/cutlass
    GIT_TAG v4.5.1
    OPTIONS
        "CUTLASS_ENABLE_HEADERS_ONLY ON"
)
