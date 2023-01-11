cmake_minimum_required(VERSION 3.14)

project(tCUDA VERSION 0.1.0 LANGUAGES CUDA CXX)

# 编译参数、设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

# set include path
include_directories(${CMAKE_SOURCE_DIR}/src)
# include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# OpenCV required
find_package(OpenCV 4.4.0 REQUIRED core highgui imgproc)
if (OpenCV_core_FOUND AND OpenCV_highgui_FOUND AND OpenCV_imgproc_FOUND)
    message(STATUS "OpenCV FOUND!")
    include_directories(${OpenCV_INCLUDE_DIRS})
else ()
    message(FATAL_ERROR "OpenCV NOT FOUND, Compile STOP")
endif ()

# link_libraries(${CUDA_LIBRARIES})

add_library(data SHARED ${CMAKE_SOURCE_DIR}/src/data.cpp)
add_library(image SHARED ${CMAKE_SOURCE_DIR}/src/image.cpp)

target_link_libraries(image PUBLIC ${OpenCV_LIBS})

add_executable(atomOptCPU ${CMAKE_SOURCE_DIR}/src/atomOptCPU.cu)
target_link_libraries(atomOptCPU PUBLIC data)

add_executable(atomOptGPU ${CMAKE_SOURCE_DIR}/src/atomOptGPU.cu)
target_link_libraries(atomOptGPU PUBLIC data)

add_executable(bitmap_sharedMem ${CMAKE_SOURCE_DIR}/src/bitmap_sharedMem.cu)
target_link_libraries(bitmap_sharedMem PUBLIC image)

add_executable(cudaMultiStream ${CMAKE_SOURCE_DIR}/src/cudaMultiStream.cu)

add_executable(cudaMultiStreamOverlap ${CMAKE_SOURCE_DIR}/src/cudaMultiStreamOverlap.cu)

add_executable(cudaStream ${CMAKE_SOURCE_DIR}/src/cudaStream.cu)

add_executable(dotMul ${CMAKE_SOURCE_DIR}/src/dotMul.cu)

add_executable(dotMulAtomLock ${CMAKE_SOURCE_DIR}/src/dotMulAtomLock.cu)

add_executable(GameLife ${CMAKE_SOURCE_DIR}/src/GameLife.cu)
target_link_libraries(GameLife PUBLIC image)

add_executable(heatConductionNoTexture ${CMAKE_SOURCE_DIR}/src/heatConductionNoTexture.cu)
target_link_libraries(heatConductionNoTexture PUBLIC image)

add_executable(heatConductionTexture ${CMAKE_SOURCE_DIR}/src/heatConductionTexture.cu)
target_link_libraries(heatConductionTexture PUBLIC image)

add_executable(ioNoPagelockMem ${CMAKE_SOURCE_DIR}/src/ioNoPagelockMem.cu)

add_executable(ioPagelockMem ${CMAKE_SOURCE_DIR}/src/ioPagelockMem.cu)

add_executable(juila ${CMAKE_SOURCE_DIR}/src/juila.cu)
target_link_libraries(juila PUBLIC image)

add_executable(rayTracing ${CMAKE_SOURCE_DIR}/src/rayTracing.cu)
target_link_libraries(rayTracing PUBLIC image)

add_executable(zeroCopyMem ${CMAKE_SOURCE_DIR}/src/zeroCopyMem.cu)