#ifndef TSETLIN_TSETLIN_H
#define TSETLIN_TSETLIN_H

#include <stdbool.h>

#include "clause.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct {
        int target_type1;
        int target_type2;
        int non_target_type1;
        int non_target_type2;
    } tsetlin_feedback_t;

    typedef struct {
        int n_features;
        int n_classes;
        int n_clauses;
        int n_states;

        /* Arrays: pos_clauses[c] is clause_t** of length n_clauses/2 */
        clause_t*** pos_clauses;
        clause_t*** neg_clauses;
    } tsetlin_t;

    /* Allocate and initialize a Tsetlin object. Caller must free with tsetlin_free. */
    tsetlin_t* tsetlin_new(int N_feature, int N_class, int N_clause, int N_state);

    /* Free a tsetlin instance and all allocated clauses. */
    void tsetlin_free(tsetlin_t* ts);

    /* Predict class for single sample X (array length n_features).
     * If votes_out is non-NULL it must point to an int array of length n_classes and it will be filled. */
    int tsetlin_predict(const tsetlin_t* ts, const int* X, int* votes_out);

    /* Single training step. If out_feedback is non-NULL it will be filled. threshold <= -1 disables thresholding. */
    tsetlin_feedback_t* tsetlin_step(tsetlin_t* ts, const int* X, int y_target, int T, double s, tsetlin_feedback_t* out_feedback, int threshold);

    /* Fit over dataset X (array of n_samples pointers to int arrays) and labels y (length n_samples). */
    void tsetlin_fit(tsetlin_t* ts, const int** X, const int* y, int n_samples, int T, double s, int epochs);

#ifdef __cplusplus
}
#endif

#endif /* TSETLIN_TSETLIN_H */
