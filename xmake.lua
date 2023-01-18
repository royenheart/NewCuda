-- configure: xmake f -c -p linux -a x86_64 -m debug -o builds/build [--cuda=PATH TO CUDA SDK]
-- configure example using NVHPC: xmake f -c -p linux -a x86_64 -m debug -o builds/build --cuda=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda/11.8
-- configure toolchain-select: xmake f --toolchain=xxx [--sdk=/aaa/xxx(find the location of toolchain)]
-- build: xmake b -r -v -D
-- install: xmake install -o builds/install
-- run: xmake run GameLife
-- clean&uninstall: xmake clean && xmake uninstall

add_rules("mode.debug", "mode.release") 
set_languages("c17", "c++17")
set_warnings("all")

add_requires("cuda")
add_requires("cmake::OpenCV::4.6.0", {alias = "opencv", system = true, 
                                     configs = {components = {"core", "imgproc", "highgui"}}})
add_includedirs("inc")

target("data")
    set_kind("shared")
    add_files("src/data.cu")
    add_cugencodes("native")

target("image")
    set_kind("shared")
    add_files("src/image.cu")
    add_cugencodes("native")
    add_packages("cuda", "opencv")

target("atomOptCPU")
    set_kind("binary")
    add_files("src/atomOptCPU.cu")
    add_cugencodes("native")
    add_deps("data")
    add_packages("cuda")

target("atomOptGPU")
    set_kind("binary")
    add_files("src/atomOptGPU.cu")
    add_cugencodes("native")
    add_deps("data")
    add_packages("cuda")

target("bitmap_sharedMem")
    set_kind("binary")
    add_files("src/bitmap_sharedMem.cu")
    add_cugencodes("native")
    add_deps("image")
    add_packages("cuda", "opencv")

target("caustics")
    set_kind("binary")
    add_files("src/caustics.cu")
    add_cugencodes("native")
    add_packages("cuda")

target("cudaMultiStream")
    set_kind("binary")
    add_files("src/cudaMultiStream.cu")
    add_cugencodes("native")
    add_packages("cuda")
    
target("cudaMultiStreamOverlap")
    set_kind("binary")
    add_files("src/cudaMultiStream.cu")
    add_cugencodes("native")
    add_packages("cuda")

target("cudaStream")
    set_kind("binary")
    add_files("src/cudaStream.cu")
    add_cugencodes("native")
    add_packages("cuda")

target("dotMul")
    set_kind("binary")
    add_files("src/dotMul.cu")
    add_cugencodes("native")
    add_packages("cuda")

target("dotMulAtomLock")
    set_kind("binary")
    add_files("src/dotMulAtomLock.cu")
    add_cugencodes("native")
    add_packages("cuda")

target("GameLife")
    set_kind("binary")
    add_files("src/GameLife.cu")
    add_cugencodes("native")
    add_deps("image")
    add_packages("cuda", "opencv")

target("getCudaDevProp")
    set_kind("binary")
    add_files("src/getCudaDevProp.cu")
    add_cugencodes("native")
    add_packages("cuda")

target("heatConductionNoTexture")
    set_kind("binary")
    add_files("src/heatConductionNoTexture.cu")
    add_cugencodes("native")
    add_deps("image")
    add_packages("cuda", "opencv")

target("heatConductionTexture")
    set_kind("binary")
    add_files("src/heatConductionTexture.cu")
    add_cugencodes("native")
    add_deps("image")
    add_packages("cuda", "opencv")

target("ioNoPageLockMem")
    set_kind("binary")
    add_files("src/ioNoPagelockMem.cu")
    add_cugencodes("native")
    add_packages("cuda")

target("ioPageLockMem")
    set_kind("binary")
    add_files("src/ioPagelockMem.cu")
    add_cugencodes("native")
    add_packages("cuda")

target("juila")
    set_kind("binary")
    add_files("src/juila.cu")
    add_cugencodes("native")
    add_deps("image")
    add_packages("cuda", "opencv")

target("rayTracing")
    set_kind("binary")
    add_files("src/rayTracing.cu")
    add_cugencodes("native")
    add_deps("image")
    add_packages("cuda", "opencv")

target("zeroCopyMem")
    set_kind("binary")
    add_files("src/zeroCopyMem.cu")
    add_cugencodes("native")
    add_packages("cuda")

-- target("tCUDA")
--     set_kind("shared")
--     add_files("src/**.cu")
--     add_includedirs("inc")

--     -- generate relocatable device code for device linker of dependents.
--     -- if __device__ or __global__ functions will be called cross file,
--     -- or dynamic parallelism will be used,
--     -- this instruction should be opted in.
--     -- add_cuflags("-rdc=true")

--     -- generate SASS code for SM architecture of current host
--     add_cugencodes("native")

--     -- generate PTX code for the virtual architecture to guarantee compatibility
--     add_cugencodes("compute_30")

--     -- -- generate SASS code for each SM architecture
--     -- add_cugencodes("sm_30", "sm_35", "sm_37", "sm_50", "sm_52", "sm_60", "sm_61", "sm_70", "sm_75")

--     -- -- generate PTX code from the highest SM architecture to guarantee forward-compatibility
--     -- add_cugencodes("compute_75")

--
-- If you want to known more usage about xmake, please see https://xmake.io
--
-- ## FAQ
--
-- You can enter the project directory firstly before building project.
--
--   $ cd projectdir
--
-- 1. How to build project?
--
--   $ xmake
--
-- 2. How to configure project?
--
--   $ xmake f -p [macosx|linux|iphoneos ..] -a [x86_64|i386|arm64 ..] -m [debug|release]
--
-- 3. Where is the build output directory?
--
--   The default output directory is `./build` and you can configure the output directory.
--
--   $ xmake f -o outputdir
--   $ xmake
--
-- 4. How to run and debug target after building project?
--
--   $ xmake run [targetname]
--   $ xmake run -d [targetname]
--
-- 5. How to install target to the system directory or other output directory?
--
--   $ xmake install
--   $ xmake install -o installdir
--
-- 6. Add some frequently-used compilation flags in xmake.lua
--
-- @code
--    -- add debug and release modes
--    add_rules("mode.debug", "mode.release")
--
--    -- add macro defination
--    add_defines("NDEBUG", "_GNU_SOURCE=1")
--
--    -- set warning all as error
--    set_warnings("all", "error")
--
--    -- set language: c99, c++11
--    set_languages("c99", "c++11")
--
--    -- set optimization: none, faster, fastest, smallest
--    set_optimize("fastest")
--
--    -- add include search directories
--    add_includedirs("/usr/include", "/usr/local/include")
--
--    -- add link libraries and search directories
--    add_links("tbox")
--    add_linkdirs("/usr/local/lib", "/usr/lib")
--
--    -- add system link libraries
--    add_syslinks("z", "pthread")
--
--    -- add compilation and link flags
--    add_cxflags("-stdnolib", "-fno-strict-aliasing")
--    add_ldflags("-L/usr/local/lib", "-lpthread", {force = true})
--
-- @endcode
--

