set(SOURCES
  "src/sm90_pipelined_tma_mma_gemm/gemm.cu"
  "src/simple_gemm.cu"
  "src/tiled_gemm.cu"
  "src/gemm_ops.cpp"
)

python_add_library(gemm_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})
add_library(gemm_ops_static STATIC ${SOURCES})

foreach(target gemm_ops gemm_ops_static)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDABOX_CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${CUDABOX_CXX_FLAGS}>
  )

  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/include"
  )

  # need to add the following cuz of the following compiler error
  # error: identifier "_BitInt" is undefined in fmt/base.h
  target_compile_definitions(${target} PRIVATE FMT_USE_BITINT=0)

  target_link_libraries(
    ${target} PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cuda_driver CUDA::cudart
    spdlog::spdlog_header_only
    nvidia::cutlass::cutlass
  )

  set_target_properties(${target} PROPERTIES
    CUDA_ARCHITECTURES "${CUDABOX_CUDA_ARCH}"
  )
endforeach()

install(TARGETS gemm_ops LIBRARY DESTINATION ${PROJECT_NAME})

add_benchmark(
  simple_gemm_bench
  benchmarks/simple_gemm_bench.cu
  LIBRARIES gemm_ops_static ${TORCH_LIBRARIES}
)

add_benchmark(
  tiled_gemm_bench
  benchmarks/tiled_gemm_bench.cu
  LIBRARIES gemm_ops_static ${TORCH_LIBRARIES}
)

add_benchmark(
  sm90_pipelined_tma_mma_gemm_bench
  benchmarks/sm90_pipelined_tma_mma_gemm_bench.cu
  LIBRARIES gemm_ops_static ${TORCH_LIBRARIES} nvidia::cutlass::cutlass
)
