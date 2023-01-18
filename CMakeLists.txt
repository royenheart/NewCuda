cmake_minimum_required(VERSION 3.16)

# deprecated

project(tCUDA VERSION 0.1.0 LANGUAGES CUDA CXX)

# 编译参数、设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/builds/out)

# set include path
include_directories(${CMAKE_SOURCE_DIR}/inc)

# find_package(NVHPC REQUIRED COMPONENTS CUDA NVSHMEM HOSTUTILS)
# link_libraries(NVHPC::CUDART NVHPC::NVSHMEM NVHPC::HOSTUTILS NVHPC::NVRTC)

# OpenCV required
find_package(OpenCV 4.4.0 REQUIRED core highgui imgproc)
if (OpenCV_core_FOUND AND OpenCV_highgui_FOUND AND OpenCV_imgproc_FOUND)
    message(STATUS "OpenCV FOUND!")
    include_directories(${OpenCV_INCLUDE_DIRS})
else ()
    message(FATAL_ERROR "OpenCV NOT FOUND, Compile STOP")
endif ()

add_executable(atomOptCPU 
    ${CMAKE_SOURCE_DIR}/src/atomOptCPU.cu
    ${CMAKE_SOURCE_DIR}/src/data.cu
)

add_executable(atomOptGPU 
    ${CMAKE_SOURCE_DIR}/src/atomOptGPU.cu
    ${CMAKE_SOURCE_DIR}/src/data.cu
)

add_executable(bitmap_sharedMem 
    ${CMAKE_SOURCE_DIR}/src/bitmap_sharedMem.cu
    ${CMAKE_SOURCE_DIR}/src/image.cu
)
target_link_libraries(bitmap_sharedMem PUBLIC ${OpenCV_LIBS})

add_executable(cudaMultiStream ${CMAKE_SOURCE_DIR}/src/cudaMultiStream.cu)

add_executable(cudaMultiStreamOverlap ${CMAKE_SOURCE_DIR}/src/cudaMultiStreamOverlap.cu)

add_executable(cudaStream ${CMAKE_SOURCE_DIR}/src/cudaStream.cu)

add_executable(dotMul ${CMAKE_SOURCE_DIR}/src/dotMul.cu)

add_executable(dotMulAtomLock ${CMAKE_SOURCE_DIR}/src/dotMulAtomLock.cu)

add_executable(GameLife 
    ${CMAKE_SOURCE_DIR}/src/GameLife.cu
    ${CMAKE_SOURCE_DIR}/src/image.cu
)
target_link_libraries(GameLife PUBLIC ${OpenCV_LIBS})

add_executable(heatConductionNoTexture 
    ${CMAKE_SOURCE_DIR}/src/heatConductionNoTexture.cu
    ${CMAKE_SOURCE_DIR}/src/image.cu
)
target_link_libraries(heatConductionNoTexture PUBLIC ${OpenCV_LIBS})

add_executable(heatConductionTexture 
    ${CMAKE_SOURCE_DIR}/src/heatConductionTexture.cu
    ${CMAKE_SOURCE_DIR}/src/image.cu
)
target_link_libraries(heatConductionTexture PUBLIC ${OpenCV_LIBS})

add_executable(ioNoPagelockMem ${CMAKE_SOURCE_DIR}/src/ioNoPagelockMem.cu)

add_executable(ioPagelockMem ${CMAKE_SOURCE_DIR}/src/ioPagelockMem.cu)

add_executable(juila 
    ${CMAKE_SOURCE_DIR}/src/juila.cu
    ${CMAKE_SOURCE_DIR}/src/image.cu
)
target_link_libraries(juila PUBLIC ${OpenCV_LIBS})

add_executable(rayTracing 
    ${CMAKE_SOURCE_DIR}/src/rayTracing.cu
    ${CMAKE_SOURCE_DIR}/src/image.cu
)
target_link_libraries(rayTracing PUBLIC ${OpenCV_LIBS})

add_executable(zeroCopyMem ${CMAKE_SOURCE_DIR}/src/zeroCopyMem.cu)