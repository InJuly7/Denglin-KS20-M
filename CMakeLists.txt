cmake_minimum_required(VERSION 3.5)
project(PROGRAM)

# 设置 SDK_DIR 目录
# DLICC_PATH: /dl/sdk/bin  SDK_DIR /dl/sdk/
# set(SDK_DIR $ENV{DLICC_PATH}/../)
set(SDK_DIR "/dl/sdk/")
set(OPENCV_DIR "/usr/include/opencv4/")
set(EIGEN_DIR "/usr/local/include/eigen3/")
set(ORT_DIR "/usr/local/include/onnxruntime-linux-aarch64-1.23.2/")
set(CMAKE_C_COMPILER "/dl/sdk/bin/dlcc")
set(CMAKE_CXX_COMPILER "/dl/sdk/bin/dlcc")

set(ORT_LIB "${ORT_DIR}/lib/libonnxruntime.so")

message(STATUS CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR})

# 默认的 CPP 文件列表
set(DEFAULT_CPPS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/op_gpu.cc
)
message(STATUS "DEFAULT_CPPS = ${DEFAULT_CPPS}")

# /dl/sdk/include /dl/include/dlnne
# 包含目录
include_directories(${SDK_DIR}/include)
include_directories(${SDK_DIR}/include/dlnne)
include_directories(${SDK_DIR}/include/curt)
include_directories(${OPENCV_DIR})
include_directories(${EIGEN_DIR}) # 纯头文件库
include_directories(${ORT_DIR}/include)


# 编译选项
set(CXX_FLAGS -std=c++14 -fPIC -fpermissive -Wno-attributes)
set(CUDA_CXX_FLAGS -x cuda --offload-arch=dlgput64,dlgpux64)
add_compile_options(${CXX_FLAGS})

# 选项：控制是否编译特定可执行文件（默认为ON，您可以在cmake命令中设置）
option(BUILD_YOLOV11_SEG "Build the yolov11_seg executable" ON)
option(BUILD_YOLOV11 "Build the yolov11 executable" ON)
option(BUILD_YOLOV11_SAM2 "Build the yolov11+sam2 pipeline executable" ON)
option(BUILD_SAM2 "Build the sam2 executable" ON)
option(BUILD_TEST_RTSP "Build Test RTSP" OFF)
option(BUILD_TEST_ONNXRUNTIME "Build Test ORT" OFF)


# 编译可执行文件：yolov11_seg
if(BUILD_YOLOV11_SEG)
	find_package(OpenCV REQUIRED)
	set(YOLOV11_SEG_CPPS
		${CMAKE_CURRENT_SOURCE_DIR}/src/op_gpu.cc
  	)
	add_executable(yolov11_seg
		${CMAKE_CURRENT_SOURCE_DIR}/test_yolov11_seg.cc
		${YOLOV11_SEG_CPPS}
	)
	target_compile_options(yolov11_seg PRIVATE -x cuda --offload-arch=dlgput64,dlgpux64 ${CXX_FLAGS})
	target_link_libraries(yolov11_seg PRIVATE curt dlnne ${OpenCV_LIBS})

  # 设置链接器语言
  set_target_properties(yolov11_seg PROPERTIES LINKER_LANGUAGE CXX)
endif()

# 编译可执行文件：yolov11
if(BUILD_YOLOV11)
	find_package(OpenCV REQUIRED)
	set(YOLOV11_CPPS
		${CMAKE_CURRENT_SOURCE_DIR}/src/op_gpu.cc
  	)
	add_executable(yolov11
		${CMAKE_CURRENT_SOURCE_DIR}/test_yolov11.cc
		${YOLOV11_CPPS}
	)
	target_compile_options(yolov11 PRIVATE -x cuda --offload-arch=dlgput64,dlgpux64 ${CXX_FLAGS})
	target_link_libraries(yolov11 PRIVATE curt dlnne ${OpenCV_LIBS})

  # 设置链接器语言
  set_target_properties(yolov11 PROPERTIES LINKER_LANGUAGE CXX)
endif()

# 编译可执行文件：yolov11_sam2_pipeline
if(BUILD_YOLOV11_SAM2)
	find_package(OpenCV REQUIRED)
	set(YOLOV11_SAM2_CPPS
		${CMAKE_CURRENT_SOURCE_DIR}/src/op_gpu.cc
  	)
	add_executable(yolov11_sam2
		${CMAKE_CURRENT_SOURCE_DIR}/test_yolov11_sam2.cc
		${YOLOV11_SAM2_CPPS}
	)
	target_compile_options(yolov11_sam2 PRIVATE -x cuda --offload-arch=dlgput64,dlgpux64 ${CXX_FLAGS})
	target_link_libraries(yolov11_sam2 PRIVATE curt dlnne ${OpenCV_LIBS} ${ORT_LIB})

  set_target_properties(yolov11_sam2 PROPERTIES LINKER_LANGUAGE CXX)
endif()

# 编译可执行文件：sam2
if(BUILD_SAM2)
	find_package(OpenCV REQUIRED)
	set(BUILD_SAM2_CPPS
		${CMAKE_CURRENT_SOURCE_DIR}/src/op_gpu.cc
  	)
	add_executable(sam2
		${CMAKE_CURRENT_SOURCE_DIR}/test_sam2.cc
		${BUILD_SAM2_CPPS}
	)
	target_compile_options(sam2 PRIVATE -x cuda --offload-arch=dlgput64,dlgpux64 ${CXX_FLAGS})
	target_link_libraries(sam2 PRIVATE curt dlnne ${OpenCV_LIBS} ${ORT_LIB})

  # 设置链接器语言
  set_target_properties(sam2 PROPERTIES LINKER_LANGUAGE CXX)
endif()

if(BUILD_TEST_ONNXRUNTIME)
	add_executable(test_ort
		${CMAKE_CURRENT_SOURCE_DIR}/test_onnrxunrime.cc
	)
	target_compile_options(test_ort PRIVATE -x cuda --offload-arch=dlgput64,dlgpux64 ${CXX_FLAGS})
	target_link_libraries(test_ort PRIVATE curt dlnne ${ORT_LIB})

	# 设置链接器语言
	set_target_properties(test_ort PROPERTIES LINKER_LANGUAGE CXX)
endif()


# 编译可执行文件：test_rtsp
# if(BUILD_TEST_RTSP)
# 	find_package(OpenCV 4.0)
# 	add_executable(test_rtsp.cc)
# 	target_compile_options(test_rtsp PRIVATE ${CXX_FLAGS})
# 	target_link_libraries(test_rtsp PRIVATE curt dlnne ${OpenCV_LIBS})
# 	set_target_properties(test_rtsp PROPERTIES LINKER_LANGUAGE CXX)
# endif()



