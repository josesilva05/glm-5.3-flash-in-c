# GLM-5.3-Flash inference engine.
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release -j
#   ctest --test-dir build -C Release --output-on-failure

cmake_minimum_required(VERSION 3.16)

project(glm-5.3-flash-in-c
        VERSION 2.0.0
        DESCRIPTION "GLM-5.3-Flash inference engine in portable C"
        LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(GLM53F_NATIVE_ARCH "Optimise for the building CPU (-march/-mcpu=native)" OFF)
option(GLM53F_CUDA "Build the CUDA backend: trunk on NVIDIA GPUs (--gpu)" OFF)

if(GLM53F_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)   # Ampere (RTX 30xx); override with -DCMAKE_CUDA_ARCHITECTURES=...
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

find_package(OpenMP)
find_package(Threads REQUIRED)

add_library(glm53f_flags INTERFACE)
if(MSVC)
  target_compile_options(glm53f_flags INTERFACE
    "$<$<COMPILE_LANGUAGE:C>:/W3;/O2;/arch:AVX2;/fp:precise;/D_CRT_SECURE_NO_WARNINGS;/D_CRT_NONSTDC_NO_DEPRECATE>")
else()
  target_compile_options(glm53f_flags INTERFACE
    "$<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpointer-arith;-Wshadow;-Wvla;-Wno-unused-parameter;-Wno-unused-function;-ffp-contract=off>")
  if(GLM53F_NATIVE_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
      target_compile_options(glm53f_flags INTERFACE $<$<COMPILE_LANGUAGE:C>:-mcpu=native>)
    else()
      target_compile_options(glm53f_flags INTERFACE $<$<COMPILE_LANGUAGE:C>:-march=native>)
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    target_compile_options(glm53f_flags INTERFACE "$<$<COMPILE_LANGUAGE:C>:-mavx2;-mfma>")
  endif()
endif()

# --------------------------------------------------------------------------- engine --
add_library(glm53f STATIC
  src/core/glm53f_ops.c
  src/core/glm53f_dsa.c
  src/io/glm53f_st.c
  src/io/glm53f_load.c
  src/cache/glm53f_cache.c
  src/model/glm53f_bind.c
  src/model/glm53f_model.c
  src/model/glm53f_mtp.c)

target_include_directories(glm53f PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/glm53f
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/src/core
  ${CMAKE_CURRENT_SOURCE_DIR}/src/io
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cache
  ${CMAKE_CURRENT_SOURCE_DIR}/src/model
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer)

target_link_libraries(glm53f PUBLIC glm53f_flags Threads::Threads)
if(NOT MSVC)
  target_link_libraries(glm53f PUBLIC m)
endif()
if(GLM53F_CUDA)
  target_sources(glm53f PRIVATE src/gpu/glm53f_gpu.c src/gpu/glm53f_gpu_kernels.cu)
  target_include_directories(glm53f PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu)
  target_compile_definitions(glm53f PUBLIC GLM53F_CUDA=1)
  target_link_libraries(glm53f PUBLIC CUDA::cudart CUDA::cublas)
  # No automatic FMA contraction in device code: kernels repeat the CPU arithmetic exactly.
  set_source_files_properties(src/gpu/glm53f_gpu_kernels.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")
endif()
if(OpenMP_C_FOUND)
  target_link_libraries(glm53f PUBLIC OpenMP::OpenMP_C)
endif()

# ------------------------------------------------------------------------------ cli --
add_executable(glm53f_cli src/cli/glm53f_run.c)
set_target_properties(glm53f_cli PROPERTIES OUTPUT_NAME glm53f)
target_link_libraries(glm53f_cli PRIVATE glm53f)
if(WIN32)
  target_link_libraries(glm53f_cli PRIVATE psapi)
endif()

# ---------------------------------------------------------------------------- tests --
enable_testing()

function(glm53f_add_test name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE glm53f)
endfunction()

set(FIX ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)

glm53f_add_test(test_st  tests/unit/test_st.c)
glm53f_add_test(test_tok tests/unit/test_tok.c)
glm53f_add_test(test_glm_tiny tests/unit/test_glm_tiny.c)
glm53f_add_test(test_cfg tests/unit/test_cfg.c)
glm53f_add_test(test_fp8 tests/unit/test_fp8.c)

add_test(NAME tiny_oracle COMMAND test_glm_tiny ${FIX}/glm_tiny)
if(GLM53F_CUDA)
  # Two layers per device, so the GPU gates also cross device boundaries.
  set_tests_properties(tiny_oracle PROPERTIES ENVIRONMENT "GLM53F_GPU_LAYERS_PER_DEV=2")
endif()
add_test(NAME fp8_kernel COMMAND test_fp8)
add_test(NAME config_tiny COMMAND test_cfg ${FIX}/glm_tiny/config.json)
add_test(NAME config_released COMMAND test_cfg ${CMAKE_CURRENT_SOURCE_DIR}/config.json)

add_test(NAME safetensors COMMAND test_st ${FIX}/st ${CMAKE_CURRENT_BINARY_DIR}/idx.json
         plain.f32.2d plain.bf16.1d tricky.f16.1d packed.u8.2d scalar.f32 second.shard.f32)

add_test(NAME tokenizer
         COMMAND ${CMAKE_COMMAND}
                 -DTEST_TOK=$<TARGET_FILE:test_tok>
                 -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/README.md
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_tok_test.cmake)
set_tests_properties(tokenizer PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED:")

# Measurement tool, needs the real checkpoint: expert read throughput versus concurrency.
add_executable(bench_expert_io tools/bench_expert_io.c)
target_link_libraries(bench_expert_io PRIVATE glm53f)

install(TARGETS glm53f_cli RUNTIME DESTINATION bin)
