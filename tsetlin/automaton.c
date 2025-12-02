#include <assert.h>
#include <stdlib.h>

#include "automaton.h"

static int _compute_action(int state, int middle_state) {
    return (state > middle_state) ? 1 : 0;
}

void automaton_init(automaton_t* a, int N_state, int state) {
    assert(a != NULL);
    assert((N_state % 2) == 0); /* N_state must be even */
    a->N_state = N_state;
    a->middle_state = N_state / 2;
    a->state = state;
    a->action = _compute_action(a->state, a->middle_state);
}

automaton_t* automaton_new(int N_state, int state) {
    automaton_t* a = (automaton_t*)malloc(sizeof(automaton_t));
    if (!a) return NULL;
    automaton_init(a, N_state, state);
    return a;
}

void automaton_free(automaton_t* a) {
    free(a);
}

int automaton_action(const automaton_t* a) {
    return a ? a->action : 0;
}

bool automaton_reward(automaton_t* a) {
    int previous_action = a->action;
    a->state += 1;
    a->action = _compute_action(a->state, a->middle_state);
    return previous_action != a->action;
}

bool automaton_penalty(automaton_t* a) {
    int previous_action = a->action;
    a->state -= 1;
    a->action = _compute_action(a->state, a->middle_state);
    return previous_action != a->action;
}

void automaton_update(automaton_t* a) {
    if (a) a->action = _compute_action(a->state, a->middle_state);
}
