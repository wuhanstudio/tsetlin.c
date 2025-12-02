# CMakeList.txt : CMake project for tsetlin.c, include source and define
# project specific logic here.
#


add_library(tsetlin STATIC
 "tsetlin.c" "tsetlin.h"
 "automaton.h" "automaton.c"  
 "clause.h" "clause.c"
)

target_include_directories(tsetlin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
