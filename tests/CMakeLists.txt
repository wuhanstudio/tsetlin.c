# Create test executable
add_executable(unit_test_automaton test_automaton/test_automaton.c)

# Link your library + GTest
target_link_libraries(unit_test_automaton
    PRIVATE tsetlin
    PRIVATE unity
    PRIVATE log
)

# Register test with CTest
add_test(NAME unit_test_automaton COMMAND unit_test_automaton)
