# Adapt from: https://github.com/neuralmagic/vllm-flash-attention/blob/main/cmake/utils.cmake
#
# Clear all `-gencode` flags from `CMAKE_CUDA_FLAGS` and store them in
# `CUDA_ARCH_FLAGS`.
#
# Example:
#   CMAKE_CUDA_FLAGS="-Wall -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75"
#   clear_cuda_arches(CUDA_ARCH_FLAGS)
#   CUDA_ARCH_FLAGS="-gencode arch=compute_70,code=sm_70;-gencode arch=compute_75,code=sm_75"
#   CMAKE_CUDA_FLAGS="-Wall"
#
macro(clear_cuda_arches CUDA_ARCH_FLAGS)
  # Extract all `-gencode` flags from `CMAKE_CUDA_FLAGS`
  string(REGEX MATCHALL "-gencode arch=[^ ]+" CUDA_ARCH_FLAGS
    ${CMAKE_CUDA_FLAGS})

  # Remove all `-gencode` flags from `CMAKE_CUDA_FLAGS` since they will be modified
  # and passed back via the `CUDA_ARCHITECTURES` property.
  string(REGEX REPLACE "-gencode arch=[^ ]+ *" "" CMAKE_CUDA_FLAGS
    ${CMAKE_CUDA_FLAGS})
endmacro()

function(add_benchmark TARGET_NAME SOURCES)
  if(NOT CUDABOX_ENABLE_BENCHMARKS)
    return()
  endif()

  cmake_parse_arguments(ARG "" "" "LIBRARIES" ${ARGN})

  add_executable(${TARGET_NAME} ${SOURCES})
  target_link_libraries(${TARGET_NAME} PRIVATE nvbench::main gtest ${ARG_LIBRARIES})

  set_target_properties(${TARGET_NAME} PROPERTIES CUDA_ARCHITECTURES "${CUDABOX_CUDA_ARCH}")

  # Automatically register as CTest test
  add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME} --color)
endfunction()

# Probe the local GPU's compute capability via `nvidia-smi` and choose the
# right CUDA architecture string for CMAKE_CUDA_ARCHITECTURES, opting into the
# architecture-specific `a` variant where it exists (Hopper+).
#
# `a` variants (sm_90a, sm_100a, sm_103a, sm_110a, sm_120a, sm_121a) unlock
# arch-specific instructions like WGMMA / TMA that CUTLASS Hopper+ kernels
# require. They are non-portable (forward-incompatible), so this function is
# intended for *local-dev* builds, not for shipping wheels.
#
# Result is written to the given output variable (PARENT_SCOPE).
# Falls back to ${FALLBACK} (default: 90a) if probing fails.
#
# Example:
#   detect_cuda_arch(MY_ARCH FALLBACK 90a)
#   set(CMAKE_CUDA_ARCHITECTURES ${MY_ARCH})
function(detect_cuda_arch OUTVAR)
  cmake_parse_arguments(ARG "" "FALLBACK" "" ${ARGN})
  if(NOT ARG_FALLBACK)
    set(ARG_FALLBACK "90a")
  endif()

  # Archs that have an `a` (architecture-specific) variant available in the
  # CUDA toolkit. Update as new archs land. Anything not in this list will be
  # used verbatim (e.g. compute capability 8.0 -> "80").
  set(_archs_with_a_variant 90 100 101 103 110 120 121)

  find_program(_nvidia_smi nvidia-smi)
  if(NOT _nvidia_smi)
    message(STATUS "detect_cuda_arch: nvidia-smi not found; using fallback "
                   "CUDA_ARCH=${ARG_FALLBACK}")
    set(${OUTVAR} ${ARG_FALLBACK} PARENT_SCOPE)
    return()
  endif()

  execute_process(
    COMMAND ${_nvidia_smi} --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _raw_cc
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _rc
  )
  if(NOT _rc EQUAL 0 OR _raw_cc STREQUAL "")
    message(STATUS "detect_cuda_arch: nvidia-smi query failed; using fallback "
                   "CUDA_ARCH=${ARG_FALLBACK}")
    set(${OUTVAR} ${ARG_FALLBACK} PARENT_SCOPE)
    return()
  endif()

  # Multi-GPU: take the first reported compute capability.
  string(REGEX REPLACE "\r?\n.*" "" _raw_cc "${_raw_cc}")
  # "9.0" -> "90"
  string(REPLACE "." "" _arch_num "${_raw_cc}")

  if(_arch_num IN_LIST _archs_with_a_variant)
    set(_chosen "${_arch_num}a")
  else()
    set(_chosen "${_arch_num}")
  endif()

  message(STATUS "detect_cuda_arch: GPU compute_cap=${_raw_cc} -> "
                 "CUDA_ARCH=${_chosen}")
  set(${OUTVAR} ${_chosen} PARENT_SCOPE)
endfunction()
