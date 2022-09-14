cmake_minimum_required(VERSION 3.16.3)

project(NewCUDA_unittest LANGUAGES CXX CUDA)

include_directories(${CMAKE_SOURCE_DIR}/headers)

# OpenCV required
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

link_directories(${CMAKE_SOURCE_DIR}/lib)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

add_executable(atomOptCPU
    ${CMAKE_SOURCE_DIR}/src/atomOptCPU.cu
)

add_executable(atomOptGPU
    ${CMAKE_SOURCE_DIR}/src/atomOptGPU.cu
)

add_executable(bitmap_sharedMem
    ${CMAKE_SOURCE_DIR}/src/bitmap_sharedMem.cu
)

add_executable(bookTest
    ${CMAKE_SOURCE_DIR}/src/bookTest.cu
)

add_executable(cudaMultiStream
    ${CMAKE_SOURCE_DIR}/src/cudaMultiStream.cu
)

add_executable(cudaMultiStreamOverlap
    ${CMAKE_SOURCE_DIR}/src/cudaMultiStreamOverlap.cu
)

add_executable(cudaStream
    ${CMAKE_SOURCE_DIR}/src/cudaStream.cu
)

add_executable(dotMul
    ${CMAKE_SOURCE_DIR}/src/dotMul.cu
)

add_executable(dotMulAtomLock
    ${CMAKE_SOURCE_DIR}/src/dotMulAtomLock.cu
)

add_executable(GameLife
    ${CMAKE_SOURCE_DIR}/src/GameLife.cu
)

add_executable(heatConductionNoTexture
    ${CMAKE_SOURCE_DIR}/src/heatConductionNoTexture.cu
)

add_executable(heatConductionTexture
    ${CMAKE_SOURCE_DIR}/src/heatConductionTexture.cu
)

add_executable(ioNoPagelockMem
    ${CMAKE_SOURCE_DIR}/src/ioNoPagelockMem.cu
)

add_executable(ioPagelockMem
    ${CMAKE_SOURCE_DIR}/src/ioPagelockMem.cu
)

add_executable(juila
    ${CMAKE_SOURCE_DIR}/src/juila.cu
)

add_executable(rayTracing
    ${CMAKE_SOURCE_DIR}/src/rayTracing.cu
)

add_executable(zeroCopyMem
    ${CMAKE_SOURCE_DIR}/src/zeroCopyMem.cu
)