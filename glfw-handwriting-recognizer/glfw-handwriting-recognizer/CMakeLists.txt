cmake_minimum_required(VERSION 3.10)
project(onnx_example)

set(CMAKE_CXX_STANDARD 17)

# Update with the correct NuGet path
set(ONNX_RUNTIME_DIR "${CMAKE_SOURCE_DIR}/external/Microsoft.ML.OnnxRuntime.1.17.0")
set(GLFW_ROOT "${CMAKE_SOURCE_DIR}/external/glfw.3.3.8")

# Include GLFW headers
include_directories(${GLFW_ROOT}/build/native/include)

# Link GLFW static library (adjust for your configuration/platform)
if (WIN32)
    set(GLFW_LIB "${GLFW_ROOT}/build/native/lib/static/v143/x64/glfw3.lib")
    # Use lib-v142/win-x86 if you're using Visual Studio 2019 32-bit
endif()

# -------------------------------------
# FetchContent: GLEW
# -------------------------------------
#include(FetchContent)
#FetchContent_Declare(
#  glew
#  GIT_REPOSITORY https://github.com/nigels-com/glew.git
#  GIT_TAG glew-2.2.0
#)
#FetchContent_MakeAvailable(glew)

include_directories("${ONNX_RUNTIME_DIR}/build/native/include")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # 64-bit
    link_directories("${ONNX_RUNTIME_DIR}/runtimes/win-x64/native")
else()
    # 32-bit
    link_directories("${ONNX_RUNTIME_DIR}/runtimes/win-x86/native")
endif()

#include_directories(${glew_SOURCE_DIR}/include)
add_subdirectory(lib/glew EXCLUDE_FROM_ALL) 

# Use standard main() entry point (not WinMain)
add_executable(glfw-handwriting-recognizer main.cpp)
target_link_libraries(glfw-handwriting-recognizer onnxruntime ${GLFW_LIB} libglew_static)
# target_link_options(glfw-handwriting-recognizer PRIVATE "/SUBSYSTEM:WINDOWS") # Removed to use console subsystem

# Windows might have an onnxruntime.dll in the system directory so it's more robust to manually copy the dlls to
# the output dir. Define a function to do so. This is called from the cmake file in the subdirectories.
function(copy_ort_dlls target_name)
  if (MSVC)
      file(GLOB ORT_DLLS ${ONNX_RUNTIME_DIR}/runtimes/win-x64/native/*.dll)
      foreach(ORT_DLL ${ORT_DLLS})
          add_custom_command(TARGET ${target_name} POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E copy ${ORT_DLL}  $<TARGET_FILE_DIR:${target_name}>)
      endforeach()
  endif()
endfunction()

copy_ort_dlls(glfw-handwriting-recognizer)

# change working directory for VS debugger to same directory the binary is created in, and copy the model there.
set_property(TARGET glfw-handwriting-recognizer PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(OutDir)")
add_custom_command(TARGET glfw-handwriting-recognizer POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/mnist.onnx $<TARGET_FILE_DIR:glfw-handwriting-recognizer>)
