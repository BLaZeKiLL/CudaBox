set(SOURCES
  "src/sgemm.cu"
  "src/tgemm.cu"
  "src/gemm_ops.cpp"
)

python_add_library(gemm_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

target_compile_options(gemm_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDABOX_CUDA_FLAGS}>)
target_compile_options(gemm_ops PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CUDABOX_CXX_FLAGS}>)

target_include_directories(gemm_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/include"
)

target_link_libraries(gemm_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda spdlog::spdlog)

install(TARGETS gemm_ops LIBRARY DESTINATION ${PROJECT_NAME})
