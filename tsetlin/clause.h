#ifndef TSETLIN_CLAUSE_INT_H
#define TSETLIN_CLAUSE_INT_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct {
        int N_feature;
        int N_states;
        int N_literals;
        int middle;

        /* integer states for positive and negative literals */
        int* p_states; /* length N_feature */
        int* n_states; /* length N_feature */

        /* compressed index lists */
        int* p_included_idxs;
        int p_included_count;
        int* n_included_idxs;
        int n_included_count;

        int* p_trainable_idxs;
        int p_trainable_count;
        int* n_trainable_idxs;
        int n_trainable_count;
    } clause_t;

    /* Allocate and initialize; caller must free with clause_int_free(). */
    clause_t* clause_new(int N_feature, int N_states);

    /* Free clause created by clause_int_new(). */
    void clause_free(clause_t* c);

    /* Rebuild included/trainable index lists. threshold < 0 disables trainable lists. */
    void clause_compress(clause_t* c, int threshold);

    /* Evaluate clause on X (array length N_feature). Returns 1 or 0. */
    int clause_evaluate(const clause_t* c, const int* X);

    /*
     * Update clause using integer-state feedback rules.
     * match_target: 1 => Type I feedback, 0 => Type II feedback.
     * clause_output: current clause output (0/1).
     * s: specificity parameter (>1 normally).
     * threshold: optional threshold for trainable literals (-1 disables).
     * Returns number of feedback operations applied.
     */
    int clause_update(clause_t* c, const int* X, int match_target, int clause_output, double s, int threshold);

    /* Set states from array length 2*N_feature (p_states then n_states). */
    void clause_set_state(clause_t* c, const int* states, int threshold);

    /* Get pointer to newly allocated states array length 2*N_feature (caller frees). */
    int* clause_get_state(const clause_t* c);

#ifdef __cplusplus
}
#endif

#endif /* TSETLIN_CLAUSE_INT_H */