# NewCuda

CUDA Learning

## How to build

### Windows

1. Replace the settings in the vscode configuration file that involve the location of the software with the location corresponding to the compilation platform.
2. Correctly install nvidia gpu driver and nvidia cuda toolkit.
3. Correctly install opencv(4.4.0 and above is okay).
4. Add necessary runtime path of opencv and nvidia cuda toolkit to the environment.
5. Using cmake or vscode C/C++ plugins to auto configure the build the project.

### Linux

1. Same as the step 1-4 of Windows.
2. Using cmake to configure and build the project
    ```shell
    # for unitTests
    cd unitTests
    mkdir build && cd build
    cmake ..
    make -j
    ```