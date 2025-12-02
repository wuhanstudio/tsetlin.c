add_library(log STATIC
    log.c/src/log.c
)

target_compile_definitions(log PUBLIC LOG_USE_COLOR)
target_include_directories(log PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/log.c/src)

add_library(unity STATIC
    Unity/src/unity.c
)

target_include_directories(unity PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Unity/src
)

add_library(tqdm STATIC
    tqdm/tqdm.c
)

target_include_directories(tqdm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/tqdm
)

# Cross-platform math library linking
if (UNIX)
    # On Linux/macOS, link libm for math functions
    target_link_libraries(tqdm PUBLIC m)
endif()