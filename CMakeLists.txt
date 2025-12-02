# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("tsetlin.c")

# Include sub-projects.
add_subdirectory ("tsetlin")
add_subdirectory ("libraries")

add_executable(main "main.c" "tsetlin_config.h")
add_executable(main_mnist "main_mnist.c" "tsetlin_config.h")

# List of libraries
set(MAIN_LIBS 
    log
    tsetlin
)

# Cross-platform math library linking
if (UNIX)
    # On Linux/macOS, link libm for math functions
    list(APPEND MAIN_LIBS m)
endif()

# On Windows (MSVC), math functions are already in the runtime, no extra linking needed

# Link to the subfolder libraries
target_link_libraries(main 
	PRIVATE ${MAIN_LIBS}
)

target_link_libraries(main_mnist 
	PRIVATE ${MAIN_LIBS} 
    PRIVATE tqdm
)

configure_file(${CMAKE_SOURCE_DIR}/iris.csv
               ${CMAKE_BINARY_DIR}/iris.csv
               COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/mnist/train-images-idx3-ubyte
               ${CMAKE_BINARY_DIR}/mnist/train-images-idx3-ubyte
               COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/mnist/train-labels-idx1-ubyte
               ${CMAKE_BINARY_DIR}/mnist/train-labels-idx1-ubyte
               COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/mnist/t10k-images-idx3-ubyte
               ${CMAKE_BINARY_DIR}/mnist/t10k-images-idx3-ubyte
               COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/mnist/t10k-labels-idx1-ubyte
               ${CMAKE_BINARY_DIR}/mnist/t10k-labels-idx1-ubyte
               COPYONLY)

# Enable Unit testing
enable_testing()
include(CTest)

# Add tests folder
add_subdirectory(tests)
