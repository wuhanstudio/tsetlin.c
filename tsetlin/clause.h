#ifndef TSETLIN_CLAUSE_H
#define TSETLIN_CLAUSE_H

#include <stdbool.h>
#include <stddef.h>

#include "automaton.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct {
        int N_feature;
        int N_states;
        int N_literals;

        automaton_t** p_automata; /* array length N_feature */
        automaton_t** n_automata; /* array length N_feature */

        /* Included literal index lists (built by compress). */
        int* p_included_idxs;
        int p_included_count;
        int* n_included_idxs;
        int n_included_count;

        /* Trainable literal index lists (optional, when threshold >= 0). */
        int* p_trainable_idxs;
        int p_trainable_count;
        int* n_trainable_idxs;
        int n_trainable_count;
    } clause_t;

    /* Allocate and initialize a clause. Caller must free with clause_free(). */
    clause_t* clause_new(int N_feature, int N_states);

    /* Free clause and owned automata and internal arrays. */
    void clause_free(clause_t* c);

    /* Rebuild included and trainable index arrays. If threshold < 0, trainable lists are cleared. */
    void clause_compress(clause_t* c, int threshold);

    /* Evaluate clause on input X (array length N_feature). Returns 1 or 0. */
    int clause_evaluate(const clause_t* c, const int* X);

    /*
     * Update clause according to algorithm.
     * X: input feature array length N_feature (values 0 or 1)
     * match_target: 1 for Type I feedback, 0 for Type II
     * clause_output: clause evaluation result (0 or 1)
     * s: parameter s (>1) controlling probabilities
     * threshold: optional threshold for trainable automata (-1 means disabled)
     * Returns number of automata feedback events applied.
     */
    int clause_update(clause_t* c, const int* X, int match_target, int clause_output, double s, int threshold);

    /* Set automata states from states array length 2 * N_feature.
     * states[0..N_feature-1] -> p_automata states
     * states[N_feature..2*N_feature-1] -> n_automata states
     * threshold passed to compress(). */
    void clause_set_state(clause_t* c, const int* states, int threshold);

    /* Get current states. Returns a newly allocated array of length 2 * N_feature.
     * Caller must free the returned pointer. */
    int* clause_get_state(const clause_t* c);

#ifdef __cplusplus
}
#endif

#endif /* TSETLIN_CLAUSE_H */
