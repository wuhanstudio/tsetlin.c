/* =========================================================================
    Unity - A Test Framework for C
    ThrowTheSwitch.org
    Copyright (c) 2007-25 Mike Karlesky, Mark VanderVoord, & Greg Williams
    SPDX-License-Identifier: MIT
========================================================================= */

#include <unity.h>
#include <log.h>

#include <automaton.h>

void setUp(void) {
    // set stuff up here
}

void tearDown(void) {
    // clean stuff up here
}

static void test_action(void) {
    automaton_t* a = automaton_new(10, 5);
    TEST_ASSERT_NOT_NULL(a);
    
    TEST_ASSERT_EQUAL_INT(0, automaton_action(a));
    automaton_free(a);

    a = automaton_new(10, 6);
    TEST_ASSERT_NOT_NULL(a);

    TEST_ASSERT_EQUAL_INT(1, automaton_action(a));
    automaton_free(a);
}

static void test_automaton_reward(void) {
    automaton_t* a = automaton_new(10, 5);
    TEST_ASSERT_NOT_NULL(a);

    automaton_reward(a);
    TEST_ASSERT_EQUAL_INT(6, a->state);

    automaton_reward(a);
    TEST_ASSERT_EQUAL_INT(7, a->state);

    automaton_free(a);
}

static void test_automaton_penalty(void) {
    automaton_t* a = automaton_new(10, 6);
    TEST_ASSERT_NOT_NULL(a);

    automaton_penalty(a);
    TEST_ASSERT_EQUAL_INT(5, a->state);

    automaton_penalty(a);
    TEST_ASSERT_EQUAL_INT(4, a->state);

    automaton_free(a);
}

void test_function_should_doBlahAndBlah(void) {
    //test stuff
}

// not needed when using generate_test_runner.rb
int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_automaton_reward);
    RUN_TEST(test_automaton_penalty);

    return UNITY_END();
}
