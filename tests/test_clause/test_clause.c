/* =========================================================================
    Unity - A Test Framework for C
    ThrowTheSwitch.org
    Copyright (c) 2007-25 Mike Karlesky, Mark VanderVoord, & Greg Williams
    SPDX-License-Identifier: MIT
========================================================================= */

#include <unity.h>
#include <log.h>

#include <clause.h>

void setUp(void) {
}

void tearDown(void) {
}

/* Helper to update automata actions after direct state manipulation */
static void update_actions_and_compress(clause_t* c) {
    for (int i = 0; i < c->N_feature; ++i) {
        automaton_update(c->p_automata[i]);
        automaton_update(c->n_automata[i]);
    }
    clause_compress(c, -1);
}

static void test_clause_evaluate(void) {
    clause_t* clause = clause_new(3, 10);
    TEST_ASSERT_NOT_NULL(clause);

    /* Manually set automata states to control actions */
    clause->p_automata[0]->state = 6; /* Include feature 0 */
    clause->n_automata[0]->state = 5; /* Exclude (NOT feature 0) */

    clause->p_automata[1]->state = 4; /* Exclude feature 1 */
    clause->n_automata[1]->state = 7; /* Include (NOT feature 1) */

    clause->p_automata[2]->state = 6; /* Include feature 2 */
    clause->n_automata[2]->state = 5; /* Exclude (NOT feature 2) */

    /* Recompute actions and rebuild compressed lists */
    update_actions_and_compress(clause);

    /* Test case where clause should evaluate to 1 */
    int X1[3] = { 1, 0, 1 };
    int output = clause_evaluate(clause, X1);
    TEST_ASSERT_EQUAL_INT(1, output);

    /* Test case where clause should evaluate to 0 */
    int X2[3] = { 1, 1, 1 };
    output = clause_evaluate(clause, X2);
    TEST_ASSERT_EQUAL_INT(0, output);

    clause_free(clause);
}

static void test_clause_update(void) {
    clause_t* clause = clause_new(2, 10);
    TEST_ASSERT_NOT_NULL(clause);

    /* Manually set automata states to control actions */
    clause->p_automata[0]->state = 6; /* Include feature 0 */
    clause->n_automata[0]->state = 5; /* Exclude (NOT feature 0) */

    clause->p_automata[1]->state = 4; /* Exclude feature 1 */
    clause->n_automata[1]->state = 7; /* Include (NOT feature 1) */

    update_actions_and_compress(clause);

    int X[2] = { 1, 0 };
    int clause_output = clause_evaluate(clause, X);
    TEST_ASSERT_EQUAL_INT(1, clause_output);

    /* Perform update: match_target=1 (Type I), s=3.0, threshold=-1 */
    int feedback = clause_update(clause, X, 1, clause_output, 3.0, -1);

    /* At least some feedback count may be zero depending on RNG; still check state property */
    TEST_ASSERT_TRUE(clause->p_automata[0]->state >= 6);

    (void)feedback; /* silence unused-variable if needed */

    clause_free(clause);
}

/* not needed when using generate_test_runner.rb */
int main(void) {
    UNITY_BEGIN();

    RUN_TEST(test_clause_evaluate);
    RUN_TEST(test_clause_update);

    return UNITY_END();
}