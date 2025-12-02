#ifndef TSETLIN_AUTOMATON_H
#define TSETLIN_AUTOMATON_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct {
        int N_state;
        int middle_state;
        int state;
        int action; /* 0 or 1 */
    } automaton_t;

    /* Initialize an existing automaton_t. N_state must be even. */
    void automaton_init(automaton_t* a, int N_state, int state);

    /* Allocate, initialize and return a new automaton_t (caller must free). */
    automaton_t* automaton_new(int N_state, int state);

    /* Free a heap-allocated automaton created by automaton_new. */
    void automaton_free(automaton_t* a);

    /* Return current action (0 or 1). */
    int automaton_action(const automaton_t* a);

    /* Apply reward: increment state, update action, return true if action changed. */
    bool automaton_reward(automaton_t* a);

    /* Apply penalty: decrement state, update action, return true if action changed. */
    bool automaton_penalty(automaton_t* a);

    /* Recompute action from current state (no state change). */
    void automaton_update(automaton_t* a);

#ifdef __cplusplus
}
#endif

#endif /* TSETLIN_AUTOMATON_H */
