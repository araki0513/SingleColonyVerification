cmake_minimum_required(VERSION 3.15)
project(App4 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    set(OpenCV_DIR "T:\\DevloperEnvs\\opencv\\build\\x64\\vc16\\lib")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
else()
    set(OpenCV_DIR "D:\\env\\openCV-Built-by-minGW-x64-4.9.0")
endif()

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

file(GLOB_RECURSE CPP_SOURCES src/*.cpp)
file(GLOB_RECURSE CU_SOURCES src/*.cu)
file(GLOB_RECURSE HEADERS include/*.h include/*.hpp include/*.cuh)

if(MSVC)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler /MD --expt-relaxed-constexpr --fmad=false")
endif()

cuda_add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CU_SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE include ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES)

# 解决方案：将 OpenCV 的 bin 目录自动添加到调试工作目录环境变量中
# 这在 VS 开发环境或通过终端启动构建时非常有用
if(MSVC)
    # 获取 OpenCV bin 目录 (假设结构为 build/x64/vc16/lib 的兄弟目录)
    get_filename_component(OPENCV_BIN_DIR "${OpenCV_DIR}/../bin" ABSOLUTE)
    
    # 设置 VS 调试器的环境变量 (仅在 VS 内部启动时有效)
    set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=${OPENCV_BIN_DIR};%PATH%")
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/NODEFAULTLIB:LIBCMT")
    
    # 或者：在生成后自动执行 DLL 拷贝（虽然你不想手动考，但这是工业界最常用的“自动发布”方式）
    # add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    #    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    #    "${OPENCV_BIN_DIR}/opencv_world4100.dll"
    #    $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()