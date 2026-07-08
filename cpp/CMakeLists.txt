cmake_minimum_required(VERSION 3.20)
project(omr_cpp LANGUAGES CXX)

# ---- Standard ---------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra)

# ---- Dependencies (brew: opencv nlohmann-json googletest) -------------------
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# ---- Core library -----------------------------------------------------------
# Every module under src/ compiles into one static library that the CLI and the
# tests link against. CONFIGURE_DEPENDS re-globs on rebuild, so adding a new
# .cpp does not require re-running cmake by hand.
file(GLOB_RECURSE OMR_CORE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(omr_core STATIC ${OMR_CORE_SOURCES})
target_include_directories(omr_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(omr_core PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
target_include_directories(omr_core PUBLIC ${OpenCV_INCLUDE_DIRS})

# ---- CLI executable ---------------------------------------------------------
# Wired once the pipeline entry point exists (plan task K5).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/apps/pipeline_eval.cpp")
  add_executable(omr_pipeline_eval apps/pipeline_eval.cpp)
  target_link_libraries(omr_pipeline_eval PRIVATE omr_core)
endif()

# ---- Tests ------------------------------------------------------------------
option(OMR_BUILD_TESTS "Build the GoogleTest suite" ON)
if(OMR_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  enable_testing()
  add_subdirectory(tests)
endif()
