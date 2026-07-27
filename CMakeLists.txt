cmake_minimum_required(VERSION 3.24)
project(nanogpt_for_nspire_runtime LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

include(CTest)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

add_library(
    nanogpt_runtime
    STATIC
    runtime/src/ng_crc32.c
    runtime/src/ng_model.c
    runtime/src/ng_ops.c
    runtime/src/ng_runtime.c
)
target_include_directories(
    nanogpt_runtime
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/src"
)
target_include_directories(
    nanogpt_runtime
    PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/include"
)
if(MSVC)
    target_compile_options(nanogpt_runtime PRIVATE /W4 /WX)
else()
    target_compile_options(
        nanogpt_runtime
        PRIVATE
        -Wall
        -Wextra
        -Werror
        -Wpedantic
        -Wshadow
        -Wconversion
    )
endif()

set(TEST_FIXTURE_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-fixtures")
set(TEST_MODEL_FIXTURE "${TEST_FIXTURE_DIR}/tiny-fp32.ngm")
add_custom_command(
    OUTPUT "${TEST_MODEL_FIXTURE}"
    COMMAND
        "${CMAKE_COMMAND}" -E make_directory "${TEST_FIXTURE_DIR}"
    COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_test_model_fixture.py"
        --output "${TEST_MODEL_FIXTURE}"
    DEPENDS
        tools/generate_test_model_fixture.py
        training/nanogpt_nspire/export_format.py
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VERBATIM
)
add_custom_target(test_model_fixture DEPENDS "${TEST_MODEL_FIXTURE}")

add_executable(test_model_loader tests/c/test_model_loader.c)
target_link_libraries(test_model_loader PRIVATE nanogpt_runtime)
add_dependencies(test_model_loader test_model_fixture)
if(MSVC)
    target_compile_options(test_model_loader PRIVATE /W4 /WX)
else()
    target_compile_options(
        test_model_loader
        PRIVATE
        -Wall
        -Wextra
        -Werror
        -Wpedantic
        -Wshadow
        -Wconversion
    )
endif()
add_test(
    NAME model_loader
    COMMAND test_model_loader "${TEST_MODEL_FIXTURE}"
)

add_executable(
    test_ops
    tests/c/test_ops.c
)
target_include_directories(
    test_ops
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/runtime/src"
)
if(MSVC)
    target_compile_options(test_ops PRIVATE /W4 /WX)
else()
    target_compile_options(
        test_ops
        PRIVATE
        -Wall
        -Wextra
        -Werror
        -Wpedantic
        -Wshadow
        -Wconversion
    )
    target_link_libraries(test_ops PRIVATE m)
endif()
target_link_libraries(test_ops PRIVATE nanogpt_runtime)
add_test(NAME scalar_ops COMMAND test_ops)

add_executable(test_runtime tests/c/test_runtime.c)
target_link_libraries(test_runtime PRIVATE nanogpt_runtime)
add_dependencies(test_runtime test_model_fixture)
if(MSVC)
    target_compile_options(test_runtime PRIVATE /W4 /WX)
else()
    target_compile_options(
        test_runtime
        PRIVATE
        -Wall
        -Wextra
        -Werror
        -Wpedantic
        -Wshadow
        -Wconversion
    )
endif()
add_test(
    NAME incremental_runtime
    COMMAND test_runtime "${TEST_MODEL_FIXTURE}"
)

add_executable(model_inspect runtime/platform/host/model_inspect.c)
target_link_libraries(model_inspect PRIVATE nanogpt_runtime)
if(MSVC)
    target_compile_options(model_inspect PRIVATE /W4 /WX)
else()
    target_compile_options(
        model_inspect
        PRIVATE
        -Wall
        -Wextra
        -Werror
        -Wpedantic
        -Wshadow
        -Wconversion
    )
endif()

add_executable(run_model runtime/platform/host/run_model.c)
target_link_libraries(run_model PRIVATE nanogpt_runtime)
if(MSVC)
    target_compile_options(run_model PRIVATE /W4 /WX)
else()
    target_compile_options(
        run_model
        PRIVATE
        -Wall
        -Wextra
        -Werror
        -Wpedantic
        -Wshadow
        -Wconversion
    )
endif()
