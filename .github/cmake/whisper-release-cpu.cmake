# whisper-rs-sys forwards CMAKE_* variables to CMake. Restrict this hook to
# whisper.cpp so the other native dependencies keep their own settings.
if(NOT PROJECT_NAME STREQUAL "whisper.cpp")
    return()
endif()

set(GGML_NATIVE OFF CACHE BOOL "Disable build-host CPU detection" FORCE)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
    # Raspberry Pi 4 supports ARMv8-A with NEON.
    set(GGML_CPU_ARM_ARCH armv8-a CACHE STRING "Raspberry Pi 4 baseline" FORCE)
else()
    # Keep ggml's x86 instructions within the documented x86-64-v3 floor.
    foreach(feature IN ITEMS SSE42 AVX AVX2 BMI2 FMA F16C)
        set(GGML_${feature} ON CACHE BOOL "Enable x86-64-v3 instructions" FORCE)
    endforeach()
    foreach(feature IN ITEMS AVX_VNNI AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 AMX_TILE AMX_INT8 AMX_BF16)
        set(GGML_${feature} OFF CACHE BOOL "Disable post-x86-64-v3 instructions" FORCE)
    endforeach()
endif()
