set(SOURCES
  "src/sgemm.cu"
  "src/tgemm.cu"
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

  target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES} c10 cuda spdlog::spdlog)
endforeach()

install(TARGETS gemm_ops LIBRARY DESTINATION ${PROJECT_NAME})

add_benchmark(
  sgemm_bench
  benchmarks/sgemm_bench.cu
  LIBRARIES gemm_ops_static
)
