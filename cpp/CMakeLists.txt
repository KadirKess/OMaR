cmake_minimum_required(VERSION 3.20)
project(omr_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-sign-compare)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(nlohmann_json REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")

file(GLOB_RECURSE OMR_CORE_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(omr_core STATIC ${OMR_CORE_SOURCES})
target_include_directories(omr_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(omr_core PUBLIC
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/apps/pipeline_eval.cpp")
  add_library(omr_pipeline_common STATIC apps/pipeline_common.cpp)
  target_include_directories(omr_pipeline_common PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/apps"
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(omr_pipeline_common PUBLIC omr_core)

  add_executable(omr_pipeline_eval apps/pipeline_eval.cpp)
  target_link_libraries(omr_pipeline_eval PRIVATE omr_pipeline_common)

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/apps/midi_export.cpp")
    add_executable(omr_midi_export apps/midi_export.cpp)
    target_link_libraries(omr_midi_export PRIVATE omr_pipeline_common)
  endif()
endif()

option(OMR_BUILD_TESTS "Build the GoogleTest suite" ON)
if(OMR_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_subdirectory(tests)
endif()
