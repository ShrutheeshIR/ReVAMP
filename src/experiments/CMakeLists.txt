cmake_minimum_required(VERSION 3.16...3.22)
project(vamp_experiments LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(VAMP_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../vamp")

if(NOT EXISTS "${VAMP_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "Could not find the vamp submodule at ${VAMP_SOURCE_DIR}. "
    "Run `git submodule update --init` from the PVAMP repo root.")
endif()

# CPMAddPackage is defined by vamp/cmake/FetchInitCPM.cmake. We pull it in
# ourselves (rather than relying on vamp's own add_subdirectory() to define
# it) because CMake function/macro scope follows directory nesting: anything
# vamp defines inside its own add_subdirectory() call is invisible up here in
# the parent, but anything we define here IS visible to vamp's CMakeLists
# when we add_subdirectory() it below.
list(APPEND CMAKE_MODULE_PATH "${VAMP_SOURCE_DIR}/cmake")
include(FetchInitCPM)
CPMAddPackage("gh:nlohmann/json@3.11.3")

# vamp/cmake/CompilerSettings.cmake sets CMAKE_CXX_FLAGS (-march=native
# -mavx2 / -mcpu=native / etc., matching whatever SIMD ISA vamp_cpp was
# compiled against). It normally runs as part of vamp's own CMakeLists.txt,
# but since we add_subdirectory() that below, those flags would only apply
# in that child scope and never propagate back up to the executables we
# define here -- so pull it in ourselves first, or these binaries silently
# compile without the right target attributes and fail with confusing
# "target specific option mismatch" inlining errors.
include(CompilerSettings)

# We only need the vamp_cpp interface library, built as a plain subdirectory
# (no python/wasm bindings, no install rules, no demo executables of its own).
set(VAMP_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(VAMP_BUILD_CPP_DEMO OFF CACHE BOOL "" FORCE)
set(VAMP_BUILD_OMPL_DEMO OFF CACHE BOOL "" FORCE)
set(VAMP_BUILD_WASM OFF CACHE BOOL "" FORCE)
set(VAMP_INSTALL_CPP_LIBRARY OFF CACHE BOOL "" FORCE)

if(NOT TARGET vamp_cpp)
  add_subdirectory("${VAMP_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/vamp")
endif()

# Registers an experiment executable, linking vamp_cpp plus any extra
# libraries (e.g. nlohmann_json::nlohmann_json) passed as trailing args, and
# silences the narrowing/sign-compare warnings these demo scripts trip under
# Clang (matches vamp's own former demo build settings). `name` is the
# target/binary name (e.g. vamp_iiwa_branch_selector); the source file is the
# same name with the "vamp_" prefix stripped (iiwa_branch_selector.cc).
function(vamp_experiment_executable name)
  string(REGEX REPLACE "^vamp_" "" source_stem "${name}")
  add_executable(${name} "${CMAKE_CURRENT_SOURCE_DIR}/${source_stem}.cc")
  target_link_libraries(${name} PRIVATE vamp_cpp ${ARGN})
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${name} PRIVATE -Wno-c++11-narrowing -Wno-sign-compare)
  endif()
endfunction()

if(EXISTS "${VAMP_SOURCE_DIR}/src/impl/vamp/robots/rby1.hh")
  vamp_experiment_executable(vamp_rby1_task_space_planner nlohmann_json::nlohmann_json)
  vamp_experiment_executable(vamp_rby1_gcp_branch_selector)
  vamp_experiment_executable(vamp_rby1_mcvamp_planner nlohmann_json::nlohmann_json)
endif()

if(EXISTS "${VAMP_SOURCE_DIR}/src/impl/vamp/robots/iiwa_marker.hh")
  vamp_experiment_executable(vamp_iiwa_maze_solver nlohmann_json::nlohmann_json)
  vamp_experiment_executable(vamp_iiwa_maze_solver_benchmark nlohmann_json::nlohmann_json)
  vamp_experiment_executable(vamp_iiwa_maze_tsr_benchmark nlohmann_json::nlohmann_json)
  vamp_experiment_executable(vamp_iiwa_branch_selector nlohmann_json::nlohmann_json)
  vamp_experiment_executable(vamp_iiwa_maze_problem_generator nlohmann_json::nlohmann_json)
endif()

if(EXISTS "${VAMP_SOURCE_DIR}/src/impl/vamp/robots/fr3_marker.hh")
  vamp_experiment_executable(vamp_fr3_maze_problem_generator nlohmann_json::nlohmann_json)
  vamp_experiment_executable(vamp_fr3_maze_solver nlohmann_json::nlohmann_json)
  vamp_experiment_executable(vamp_fr3_maze_solver_benchmark nlohmann_json::nlohmann_json)
  vamp_experiment_executable(vamp_fr3_maze_tsr_benchmark nlohmann_json::nlohmann_json)
  vamp_experiment_executable(vamp_fr3_mcvamp_problem_generator nlohmann_json::nlohmann_json)
endif()

if(EXISTS "${VAMP_SOURCE_DIR}/src/impl/vamp/robots/bimanual_iiwa.hh")
  vamp_experiment_executable(vamp_bimanual_iiwa_leader_follower_shelf nlohmann_json::nlohmann_json)
  vamp_experiment_executable(vamp_bimanual_iiwa_projection_shelf nlohmann_json::nlohmann_json)
  vamp_experiment_executable(vamp_bimanual_iiwa_parameterized_shelf nlohmann_json::nlohmann_json)
endif()
