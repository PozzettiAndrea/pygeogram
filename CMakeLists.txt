cmake_minimum_required(VERSION 3.15...3.26)
project(pygeogram LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# Configure geogram as a static library (no GUI, no apps)
# These MUST be set before add_subdirectory so geogram's CMakeLists.txt sees them.
# ---------------------------------------------------------------------------
# Force static library so everything is linked into the nanobind .so module.
# geogram's platform detection defaults to -dynamic variants, override to static.
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(VORPALINE_PLATFORM "Linux64-clang" CACHE STRING "" FORCE)
    else()
      set(VORPALINE_PLATFORM "Linux64-gcc" CACHE STRING "" FORCE)
    endif()
  else()
    set(VORPALINE_PLATFORM "Linux64-gcc-aarch64" CACHE STRING "" FORCE)
  endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(VORPALINE_PLATFORM "Darwin-aarch64-clang" CACHE STRING "" FORCE)
  else()
    set(VORPALINE_PLATFORM "Darwin-clang" CACHE STRING "" FORCE)
  endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(VORPALINE_PLATFORM "Win-vs-generic" CACHE STRING "" FORCE)
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Need PIC for static lib objects that will be linked into a shared .so
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GEOGRAM_LIB_ONLY ON CACHE BOOL "" FORCE)
set(GEOGRAM_WITH_GRAPHICS OFF CACHE BOOL "" FORCE)
set(GEOGRAM_WITH_LUA OFF CACHE BOOL "" FORCE)
set(GEOGRAM_WITH_TRIANGLE ON CACHE BOOL "" FORCE)
set(GEOGRAM_WITH_TETGEN ON CACHE BOOL "" FORCE)
set(GEOGRAM_WITH_HLBFGS ON CACHE BOOL "" FORCE)
set(GEOGRAM_WITH_FPG OFF CACHE BOOL "" FORCE)

# Build geogram — EXCLUDE_FROM_ALL suppresses geogram's own install rules
# so only our nanobind module gets packaged into the wheel.
add_subdirectory(geogram EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Python + nanobind
# ---------------------------------------------------------------------------
find_package(Python 3.10
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

find_package(nanobind CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Build the nanobind extension module
# ---------------------------------------------------------------------------
nanobind_add_module(
  _pygeogram
  STABLE_ABI
  src/_pygeogram.cpp
)

# Link geogram + OpenMP (geogram uses OpenMP internally)
find_package(OpenMP)
target_link_libraries(_pygeogram PRIVATE geogram)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_pygeogram PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(_pygeogram PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/geogram/src/lib
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(_pygeogram PRIVATE cxx_std_17)

# Compiler-specific optimization
if(MSVC)
  target_compile_options(_pygeogram PRIVATE /O2)
else()
  target_compile_options(_pygeogram PRIVATE -O3)
endif()

# Install into the pygeogram Python package directory
install(TARGETS _pygeogram LIBRARY DESTINATION pygeogram)
