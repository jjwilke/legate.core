macro(legate_include_rapids)
  if (NOT _LEGATE_HAS_RAPIDS)
    if(NOT EXISTS ${CMAKE_BINARY_DIR}/LEGATE_RAPIDS.cmake)
      file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-23.02/RAPIDS.cmake
           ${CMAKE_BINARY_DIR}/LEGATE_RAPIDS.cmake)
    endif()
    include(${CMAKE_BINARY_DIR}/LEGATE_RAPIDS.cmake)
    include(rapids-cmake)
    include(rapids-cpm)
    include(rapids-cuda)
    include(rapids-export)
    include(rapids-find)
    set(_LEGATE_HAS_RAPIDS ON)
  endif()
endmacro()

function(legate_default_cpp_install target)
  set(options)
  set(one_value_args EXPORT)
  set(multi_value_args)
  cmake_parse_arguments(
    LEGATE_OPT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if (NOT LEGATE_OPT_EXPORT)
    message(FATAL_ERROR "Need EXPORT name for legate_default_install")
  endif()

  legate_include_rapids()

  rapids_cmake_install_lib_dir(lib_dir)

  install(TARGETS ${target}
          DESTINATION ${lib_dir}
	  EXPORT ${LEGATE_OPT_EXPORT})

  set(final_code_block
    "set(${target}_BUILD_LIBDIR ${CMAKE_BINARY_DIR}/legate_${target})"
  )

  rapids_export(
    INSTALL ${target}
    EXPORT_SET ${LEGATE_OPT_EXPORT}
    GLOBAL_TARGETS ${target}
    NAMESPACE legate::
    LANGUAGES ${ENABLED_LANGUAGES}
  )

  # build export targets
  rapids_export(
    BUILD ${target}
    EXPORT_SET ${LEGATE_OPT_EXPORT}
    GLOBAL_TARGETS ${target}
    NAMESPACE legate::
    FINAL_CODE_BLOCK final_code_block
    LANGUAGES ${ENABLED_LANGUAGES}
  )
endfunction()

function(legate_add_cffi header)
  if (NOT DEFINED CMAKE_C_COMPILER)
    message(FATAL_ERROR "Must enable C language to build Legate projects")
  endif()

  set(options)
  set(one_value_args TARGET)
  set(multi_value_args)
  cmake_parse_arguments(
    LEGATE_OPT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  # abbreviate for the function below
  set(target ${LEGATE_OPT_TARGET})
  set(install_info_in
[=[
from pathlib import Path

def get_libpath():
    import os, sys, platform
    join = os.path.join
    exists = os.path.exists
    dirname = os.path.dirname
    cn_path = dirname(dirname(__file__))
    so_ext = {
        "": "",
        "Java": ".jar",
        "Linux": ".so",
        "Darwin": ".dylib",
        "Windows": ".dll"
    }[platform.system()]

    def find_lib(libdir):
        target = f"lib@target@{so_ext}*"
        search_path = Path(libdir)
        matches = [m for m in search_path.rglob(target)]
        if matches:
          return matches[0].parent
        return None

    return (
        find_lib("@libdir@") or
        find_lib(join(dirname(dirname(dirname(cn_path))), "lib")) or
        find_lib(join(dirname(dirname(sys.executable)), "lib")) or
        ""
    )

libpath: str = get_libpath()

header: str = """
  @header@
  void @target@_perform_registration();
"""
]=])
  set(install_info_py_in ${CMAKE_BINARY_DIR}/legate_${target}/install_info.py.in)
  set(install_info_py ${CMAKE_SOURCE_DIR}/${target}/install_info.py)
  file(WRITE ${install_info_py_in} "${install_info_in}")

  set(generate_script_content
  [=[
    execute_process(
      COMMAND ${CMAKE_C_COMPILER}
        -E
        -P @header@
      ECHO_ERROR_VARIABLE
      OUTPUT_VARIABLE header
      COMMAND_ERROR_IS_FATAL ANY
    )
    configure_file(
        @install_info_py_in@
        @install_info_py@
        @ONLY)
  ]=])

  set(generate_script ${CMAKE_BINARY_DIR}/gen_install_info.cmake)
  file(CONFIGURE
       OUTPUT ${generate_script}
       CONTENT "${generate_script_content}"
       @ONLY
  )

  if (DEFINED ${target}_BUILD_LIBDIR)
    # this must have been imported from an existing editable build
    set(libdir ${${target}_BUILD_LIBDIR})
  else()
    # libraries are built in a common spot
    set(libdir ${CMAKE_BINARY_DIR}/legate_${target})
    message("libdir to binary dir")
  endif()
  add_custom_target("${target}_generate_install_info_py" ALL
    COMMAND ${CMAKE_COMMAND}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -Dtarget=${target}
      -Dlibdir=${libdir}
      -P ${generate_script}
    OUTPUT ${install_info_py}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating install_info.py"
    DEPENDS ${header}
  )
endfunction()

function(legate_default_python_install target)
  set(options)
  set(one_value_args EXPORT)
  set(multi_value_args)
  cmake_parse_arguments(
    LEGATE_OPT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if (NOT LEGATE_OPT_EXPORT)
    message(FATAL_ERROR "Need EXPORT name for legate_default_python_install")
  endif()

  if (SKBUILD)
    add_library(${target}_python INTERFACE)
    add_library(legate::${target}_python ALIAS ${target}_python)
    target_link_libraries(${target}_python INTERFACE legate::core legate::${target})

    install(TARGETS ${target}_python
            DESTINATION ${lib_dir}
            EXPORT ${LEGATE_OPT_EXPORT})

    legate_include_rapids()
    rapids_export(
      INSTALL ${target}_python
      EXPORT_SET ${LEGATE_OPT_EXPORT}
      GLOBAL_TARGETS ${target}_python
      NAMESPACE legate::
    )
  endif()
endfunction()

function(legate_add_cpp_subdirectory dir)
  set(options)
  set(one_value_args EXPORT TARGET)
  set(multi_value_args)
  cmake_parse_arguments(
    LEGATE_OPT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if (NOT LEGATE_OPT_EXPORT)
    message(FATAL_ERROR "Need EXPORT name for legate_default_install")
  endif()

  if (NOT LEGATE_OPT_TARGET)
    message(FATAL_ERROR "Need TARGET name for Legate package")
  endif()
  # abbreviate for the function
  set(target ${LEGATE_OPT_TARGET})

  legate_include_rapids()

  rapids_find_package(legate_core CONFIG
          GLOBAL_TARGETS legate::core
          BUILD_EXPORT_SET ${LEGATE_OPT_EXPORT}
          INSTALL_EXPORT_SET ${LEGATE_OPT_EXPORT})

  if (SKBUILD)
    if (NOT DEFINED ${target}_ROOT)
      set(${target}_ROOT ${CMAKE_SOURCE_DIR}/build)
    endif()
    rapids_find_package(${target} CONFIG
      GLOBAL_TARGETS legate::${target}
      BUILD_EXPORT_SET ${LEGATE_OPT_EXPORT}
      INSTALL_EXPORT_SET ${LEGATE_OPT_EXPORT})
    if (NOT ${target}_FOUND)
      add_subdirectory(${dir} ${CMAKE_BINARY_DIR}/legate_${target})
      legate_default_cpp_install(${target} EXPORT ${LEGATE_OPT_EXPORT})
    else()
      # Make sure the libdir is visible to other functions
      set(${target}_BUILD_LIBDIR "${${target}_BUILD_LIBDIR}" PARENT_SCOPE)
    endif()
  else()
    add_subdirectory(${dir} ${CMAKE_BINARY_DIR}/legate_${target})
    legate_default_cpp_install(${target} EXPORT ${LEGATE_OPT_EXPORT})
  endif()

endfunction()

function(legate_cpp_library_template target output_sources_variable)
  set(file_template
[=[
#pragma once

#include "legate.h"

namespace @target@ {

struct Registry {
 public:
  template <typename... Args>
  static void record_variant(Args&&... args)
  {
    get_registrar().record_variant(std::forward<Args>(args)...);
  }
  static legate::TaskRegistrar& get_registrar();
};

template <typename T, int ID>
struct Task : public legate::LegateTask<T> {
  using Registrar = Registry;
  static constexpr int TASK_ID = ID;
};

}
]=])
  string(CONFIGURE "${file_template}" file_content @ONLY)
  file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/legate_library.h "${file_content}")

  set(file_template
[=[
#include "legate_library.h"
#include "core/mapping/mapping.h"

namespace @target@ {

class Mapper : public legate::mapping::LegateMapper {
 public:
  Mapper(){}

 private:
  Mapper(const Mapper& rhs)            = delete;
  Mapper& operator=(const Mapper& rhs) = delete;

  // Legate mapping functions
 public:
  void set_machine(const legate::mapping::MachineQueryInterface* machine) override {
    machine_ = machine;
  }

  legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::TaskTarget>& options) override {
    return *options.begin();
  }

  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override {
    using legate::mapping::StoreMapping;
    std::vector<StoreMapping> mappings;
    auto& inputs  = task.inputs();
    auto& outputs = task.outputs();
    for (auto& input : inputs) {
      mappings.push_back(StoreMapping::default_mapping(input, options.front()));
      mappings.back().policy.exact = true;
    }
    for (auto& output : outputs) {
      mappings.push_back(StoreMapping::default_mapping(output, options.front()));
      mappings.back().policy.exact = true;
    }
    return std::move(mappings);
  }

  legate::Scalar tunable_value(legate::TunableID tunable_id) override {
    return 0;
  }

 private:
  const legate::mapping::MachineQueryInterface* machine_;
};

static const char* const library_name = "@target@";

Legion::Logger log_@target@(library_name);

/*static*/ legate::TaskRegistrar& Registry::get_registrar()
{
  static legate::TaskRegistrar registrar;
  return registrar;
}

void registration_callback()
{
  legate::ResourceConfig config;
  config.max_mappers       = 1;
  config.max_tasks         = 1024;
  config.max_reduction_ops = 8;
  legate::LibraryContext context(library_name, config);

  Registry::get_registrar().register_all_tasks(context);

  // Now we can register our mapper with the runtime
  context.register_mapper(std::make_unique<Mapper>(), 0);
}

}  // namespace @target@

extern "C" {

void @target@_perform_registration(void)
{
  // Tell the runtime about our registration callback so we hook it
  // in before the runtime starts and make it global so that we know
  // that this call back is invoked everywhere across all nodes
  legate::Core::perform_registration<@target@::registration_callback>();
}

}
]=])
  string(CONFIGURE "${file_template}" file_content @ONLY)
  file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/legate_library.cc "${file_content}")

  set(${output_sources_variable}
    legate_library.h
    legate_library.cc
    PARENT_SCOPE
  )
endfunction()

function(legate_python_library_template target)
set(file_template
[=[
from legate.core import (
    Library,
    ResourceConfig,
    get_legate_runtime,
)
import os
from typing import Any

class UserLibrary(Library):
    def __init__(self, name: str) -> None:
        self.name = name
        self.shared_object: Any = None

    @property
    def cffi(self) -> Any:
        return self.shared_object

    def get_name(self) -> str:
        return self.name

    def get_shared_library(self) -> str:
        from @target@.install_info import libpath
        return os.path.join(libpath, f"lib@target@{self.get_library_extension()}")

    def get_c_header(self) -> str:
        from @target@.install_info import header

        return header

    def get_registration_callback(self) -> str:
        return "@target@_perform_registration"

    def get_resource_configuration(self) -> ResourceConfig:
        assert self.shared_object is not None
        config = ResourceConfig()
        config.max_tasks = 1024
        config.max_mappers = 1
        config.max_reduction_ops = 8
        config.max_projections = 0
        config.max_shardings = 0
        return config

    def initialize(self, shared_object: Any) -> None:
        self.shared_object = shared_object

    def destroy(self) -> None:
        pass

user_lib = UserLibrary("@target@")
user_context = get_legate_runtime().register_library(user_lib)
]=])
  string(CONFIGURE "${file_template}" file_content @ONLY)
  file(WRITE ${CMAKE_SOURCE_DIR}/${target}/library.py "${file_content}")
endfunction()
