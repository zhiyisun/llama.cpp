set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR ARM64)

set(CMAKE_C_COMPILER   "cl.exe")
set(CMAKE_CXX_COMPILER "cl.exe")

set(CMAKE_C_COMPILER_TARGET   ${target})
set(CMAKE_CXX_COMPILER_TARGET ${target})

set(arch_c_flags "/arch:ARMV8.4-A /fp:fast /favor:AMD64")
set(warn_c_flags "/wd4101 /wd4102 /wd4505")

set(CMAKE_C_FLAGS_INIT   "${arch_c_flags} ${warn_c_flags}")
set(CMAKE_CXX_FLAGS_INIT "${arch_c_flags} ${warn_c_flags}")
