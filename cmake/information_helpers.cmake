# TODO: we may use file(GENERATE ... TARGET ...) to generate config file based on the target property
#       when we bump the CMake minimum version to 3.15/3.19
function(filter_generator_expressions INPUT OUTPUT)
    set(TMP "${INPUT}")
    # We need to explicitly evaluate and remove a few generator expressions,
    # since they are only available in file or target contexts
    string(REGEX REPLACE "\\$<COMPILE_LANG_AND_ID:CUDA[^>]*>" "0" TMP "${TMP}")
    string(REGEX REPLACE "\\$<BUILD_INTERFACE:[^<>]*>" "" TMP "${TMP}")
    string(REGEX REPLACE "\\$<INSTALL_INTERFACE:[^<>]*>" "" TMP "${TMP}")
    string(REGEX REPLACE "\\$<HOST_LINK:" "$<1:" TMP "${TMP}")
    string(REGEX REPLACE "SHELL:" "" TMP "${TMP}")
    # Ignore hwloc include if it is the internal one
    string(REGEX REPLACE "${PROJECT_BINARY_DIR}.*hwloc/src/include.*" "" TMP "${TMP}")
    set(${OUTPUT} "${TMP}" PARENT_SCOPE)
endfunction()

macro(ginkgo_interface_libraries_recursively INTERFACE_LIBS)
    foreach(_lib ${INTERFACE_LIBS})
        if (TARGET ${_lib})
            if("${_lib}" MATCHES "ginkgo.*")
                list(APPEND GINKGO_INTERFACE_LIBS
                    "-l${_lib}$<$<CONFIG:Debug>:${CMAKE_DEBUG_POSTFIX}>")
            endif()
            # Get the link flags and treat them
            get_target_property(_link_flags "${_lib}" INTERFACE_LINK_OPTIONS)
            if (_link_flags)
                filter_generator_expressions("${_link_flags}" _link_flags_filtered)
                list(APPEND GINKGO_INTERFACE_LIBS "${_link_flags_filtered}")
            endif()
            # Populate the include directories
            get_target_property(_incs "${_lib}" INTERFACE_INCLUDE_DIRECTORIES)
            if (_incs)
                filter_generator_expressions("${_incs}" _incs_filtered)
                foreach(_inc ${_incs_filtered})
                    if(_inc MATCHES ".* .*")
                        list(APPEND GINKGO_INTERFACE_CFLAGS "-I\"${_inc}\"")
                    elseif(_inc)
                        list(APPEND GINKGO_INTERFACE_CFLAGS "-I${_inc}")
                    endif()
                endforeach()
            endif()

            # Populate the compiler options and definitions if needed
            get_target_property(_defs "${_lib}" INTERFACE_COMPILE_DEFINITIONS)
            if (_defs)
                filter_generator_expressions("${_defs}" _defs_filtered)
                foreach(_def ${_defs_filtered})
                    if(_def MATCHES "^-D.*")
                        list(APPEND GINKGO_INTERFACE_CFLAGS "${_def}")
                    else()
                        list(APPEND GINKGO_INTERFACE_CFLAGS "-D${_def}")
                    endif()
                endforeach()
            endif()
            get_target_property(_opts "${_lib}" INTERFACE_COMPILE_OPTIONS)
            if (_opts)
                filter_generator_expressions("${_opts}" _opts_filtered)
                list(APPEND GINKGO_INTERFACE_CFLAGS "${_opts_filtered}")
            endif()

            # Keep recursing through the libraries
            get_target_property(_libs "${_lib}" INTERFACE_LINK_LIBRARIES)
            ginkgo_interface_libraries_recursively("${_libs}")
        elseif(EXISTS ${_lib})
            get_filename_component(_dir "${_lib}" DIRECTORY)
            get_filename_component(_libname "${_lib}" NAME_WE)
            if("${_lib}" MATCHES "${PROJECT_BINARY_DIR}.*hwloc.so")
                set(_dir ${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_LIBRARY_DIR})
            endif()
            if(NOT ("${_libname}" MATCHES "^lib.*"))
                message(FATAL_ERROR "Library is missing lib prefix: ${_lib}")
            endif()
            string(REGEX REPLACE "^lib" "" _libname_pure "${_libname}")
            list(APPEND GINKGO_INTERFACE_LIBS "-l${_libname_pure}")
            if(_dir MATCHES ".* .*")
                list(APPEND GINKGO_INTERFACE_LIB_DIRS "-L\"${_dir}\"")
            else()
                list(APPEND GINKGO_INTERFACE_LIB_DIRS "-L${_dir}")
            endif()
        endif()
    endforeach()
endmacro()

macro(ginkgo_interface_information)
    unset(GINKGO_INTERFACE_LIBS)
    unset(GINKGO_INTERFACE_LIB_DIRS)
    unset(GINKGO_INTERFACE_CFLAGS)
    # Prepare recursively populated library list
    list(APPEND GINKGO_INTERFACE_LIBS "-lginkgo$<$<CONFIG:Debug>:${CMAKE_DEBUG_POSTFIX}>")
    list(APPEND GINKGO_INTERFACE_LIB_DIRS "-L${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_LIBRARY_DIR}")
    # Prepare recursively populated include directory list
    list(APPEND GINKGO_INTERFACE_CFLAGS
        "-I${CMAKE_INSTALL_PREFIX}/${GINKGO_INSTALL_INCLUDE_DIR}")

    # Call the recursive interface libraries macro
    get_target_property(GINKGO_INTERFACE_LINK_LIBRARIES ginkgo INTERFACE_LINK_LIBRARIES)
    ginkgo_interface_libraries_recursively("${GINKGO_INTERFACE_LINK_LIBRARIES}")

    # Format and store the interface libraries found
    # remove duplicates on the reversed list to keep the dependecy in the end of list.
    list(REVERSE GINKGO_INTERFACE_LIBS)
    list(REMOVE_DUPLICATES GINKGO_INTERFACE_LIBS)
    list(REVERSE GINKGO_INTERFACE_LIBS)
    list(REMOVE_ITEM GINKGO_INTERFACE_LIBS "")
    list(REMOVE_DUPLICATES GINKGO_INTERFACE_LIB_DIRS)
    # keep it as list 
    set(GINKGO_INTERFACE_LDFLAGS ${GINKGO_INTERFACE_LIB_DIRS} ${GINKGO_INTERFACE_LIBS})
    # Format and store the interface cflags found
    list(REMOVE_ITEM GINKGO_INTERFACE_CFLAGS "")
    # Keep it as list
endmacro(ginkgo_interface_information)

macro(ginkgo_git_information)
    if(EXISTS "${Ginkgo_SOURCE_DIR}/.git")
        find_package(Git QUIET)
        if(GIT_FOUND)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} describe --contains --all HEAD
                WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR}
                OUTPUT_VARIABLE GINKGO_GIT_BRANCH
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} log -1 --format=%H ${Ginkgo_SOURCE_DIR}
                WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR}
                OUTPUT_VARIABLE GINKGO_GIT_REVISION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} log -1 --format=%h ${Ginkgo_SOURCE_DIR}
                WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR}
                OUTPUT_VARIABLE GINKGO_GIT_SHORTREV
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        endif()
    endif()
endmacro(ginkgo_git_information)
