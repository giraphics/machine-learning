cmake_minimum_required(VERSION 3.10)
project(onnx_example)

set(CMAKE_CXX_STANDARD 17)

# Update with the correct NuGet path
set(ONNX_RUNTIME_DIR "${CMAKE_SOURCE_DIR}/external/Microsoft.ML.OnnxRuntime.1.17.0")

include_directories("${ONNX_RUNTIME_DIR}/build/native/include")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # 64-bit
    link_directories("${ONNX_RUNTIME_DIR}/runtimes/win-x64/native")
else()
    # 32-bit
    link_directories("${ONNX_RUNTIME_DIR}/runtimes/win-x86/native")
endif()

add_executable(onnx_example main.cpp)
target_link_libraries(onnx_example onnxruntime)
target_link_options(onnx_example PRIVATE "/SUBSYSTEM:WINDOWS")

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

copy_ort_dlls(onnx_example)

# change working directory for VS debugger to same directory the binary is created in, and copy the model there.
set_property(TARGET onnx_example PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$(OutDir)")
add_custom_command(TARGET onnx_example POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/mnist.onnx $<TARGET_FILE_DIR:onnx_example>)
