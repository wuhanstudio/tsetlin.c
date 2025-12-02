#include "tsetlin.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Helpers */
static int argmax_int(const int* arr, int len) {
    int best = 0;
    for (int i = 1; i < len; ++i) {
        if (arr[i] > arr[best]) best = i;
    }
    return best;
}

static int clip_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* Create a new tsetlin instance */
tsetlin_t* tsetlin_new(int N_feature, int N_class, int N_clause, int N_state) {
    assert((N_state % 2) == 0);
    assert((N_clause % 2) == 0);

    tsetlin_t* ts = (tsetlin_t*)calloc(1, sizeof(tsetlin_t));
    if (!ts) return NULL;

    ts->n_features = N_feature;
    ts->n_classes = N_class;
    ts->n_clauses = N_clause;
    ts->n_states = N_state;

    ts->pos_clauses = (clause_t***)calloc(N_class, sizeof(clause_t**));
    ts->neg_clauses = (clause_t***)calloc(N_class, sizeof(clause_t**));
    if (!ts->pos_clauses || !ts->neg_clauses) {
        tsetlin_free(ts);
        return NULL;
    }

    int half = N_clause / 2;
    for (int c = 0; c < N_class; ++c) {
        ts->pos_clauses[c] = (clause_t**)calloc(half, sizeof(clause_t*));
        ts->neg_clauses[c] = (clause_t**)calloc(half, sizeof(clause_t*));
        if (!ts->pos_clauses[c] || !ts->neg_clauses[c]) {
            tsetlin_free(ts);
            return NULL;
        }
        for (int j = 0; j < half; ++j) {
            ts->pos_clauses[c][j] = clause_new(N_feature, N_state);
            ts->neg_clauses[c][j] = clause_new(N_feature, N_state);
            if (!ts->pos_clauses[c][j] || !ts->neg_clauses[c][j]) {
                tsetlin_free(ts);
                return NULL;
            }
        }
    }

    /* Seed RNG once for library usage (caller may override with srand). */
    srand((unsigned)time(NULL));

    return ts;
}

void tsetlin_free(tsetlin_t* ts) {
    if (!ts) return;
    if (ts->pos_clauses) {
        int half = ts->n_clauses / 2;
        for (int c = 0; c < ts->n_classes; ++c) {
            if (ts->pos_clauses[c]) {
                for (int j = 0; j < half; ++j) {
                    clause_free(ts->pos_clauses[c][j]);
                }
                free(ts->pos_clauses[c]);
            }
        }
        free(ts->pos_clauses);
    }
    if (ts->neg_clauses) {
        int half = ts->n_clauses / 2;
        for (int c = 0; c < ts->n_classes; ++c) {
            if (ts->neg_clauses[c]) {
                for (int j = 0; j < half; ++j) {
                    clause_free(ts->neg_clauses[c][j]);
                }
                free(ts->neg_clauses[c]);
            }
        }
        free(ts->neg_clauses);
    }
    free(ts);
}

/* Predict single sample */
int tsetlin_predict(const tsetlin_t* ts, const int* X, int* votes_out) {
    assert(ts != NULL);
    assert(X != NULL);

    int* votes = NULL;
    int local_votes[64]; /* small fast path; if classes > 64 we allocate */
    if (ts->n_classes <= 64) {
        votes = local_votes;
        memset(votes, 0, sizeof(int) * ts->n_classes);
    }
    else {
        votes = (int*)calloc(ts->n_classes, sizeof(int));
        if (!votes) return 0;
    }

    int half = ts->n_clauses / 2;
    for (int c = 0; c < ts->n_classes; ++c) {
        int sum = 0;
        for (int j = 0; j < half; ++j) {
            sum += clause_evaluate(ts->pos_clauses[c][j], X);
            sum -= clause_evaluate(ts->neg_clauses[c][j], X);
        }
        votes[c] = sum;
    }

    int pred = argmax_int(votes, ts->n_classes);

    if (votes_out) {
        memcpy(votes_out, votes, sizeof(int) * ts->n_classes);
    }

    if (votes != local_votes) free(votes);
    return pred;
}

/* Single training step following the Python logic (pair-wise learning). */
tsetlin_feedback_t* tsetlin_step(tsetlin_t* ts, const int* X, int y_target, int T, double s, tsetlin_feedback_t* out_feedback, int threshold) {
    assert(ts != NULL);
    assert(X != NULL);
    assert(y_target >= 0 && y_target < ts->n_classes);
    if (out_feedback) {
        out_feedback->target_type1 = 0;
        out_feedback->target_type2 = 0;
        out_feedback->non_target_type1 = 0;
        out_feedback->non_target_type2 = 0;
    }

    int half = ts->n_clauses / 2;

    /* Pair 1: Target class */
    int class_sum = 0;
    int* pos_vals = (int*)malloc(sizeof(int) * half);
    int* neg_vals = (int*)malloc(sizeof(int) * half);
    if (!pos_vals || !neg_vals) {
        free(pos_vals); free(neg_vals);
        return;
    }

    for (int i = 0; i < half; ++i) {
        pos_vals[i] = clause_evaluate(ts->pos_clauses[y_target][i], X);
        neg_vals[i] = clause_evaluate(ts->neg_clauses[y_target][i], X);
        class_sum += pos_vals[i];
        class_sum -= neg_vals[i];
    }

    class_sum = clip_int(class_sum, -T, T);
    double c1 = (double)(T - class_sum) / (2.0 * (double)T);

    for (int i = 0; i < half; ++i) {
        if (((double)rand() / RAND_MAX) <= c1) {
            int feedback_count = clause_update(ts->pos_clauses[y_target][i], X, 1, pos_vals[i], s, threshold);
            if (out_feedback) out_feedback->target_type1 += feedback_count;
        }
        if (((double)rand() / RAND_MAX) <= c1) {
            int feedback_count = clause_update(ts->neg_clauses[y_target][i], X, 0, neg_vals[i], s, threshold);
            if (out_feedback) out_feedback->target_type2 += feedback_count;
        }
    }

    /* Pair 2: Non-target class (random) */
    int other_class = 0;
    if (ts->n_classes == 1) other_class = 0;
    else {
        int r = rand() % (ts->n_classes - 1);
        /* map r in [0..n_classes-2] to class != y_target */
        if (r >= y_target) other_class = r + 1;
        else other_class = r;
    }

    class_sum = 0;
    for (int i = 0; i < half; ++i) {
        pos_vals[i] = clause_evaluate(ts->pos_clauses[other_class][i], X);
        neg_vals[i] = clause_evaluate(ts->neg_clauses[other_class][i], X);
        class_sum += pos_vals[i];
        class_sum -= neg_vals[i];
    }

    class_sum = clip_int(class_sum, -T, T);
    double c2 = (double)(T + class_sum) / (2.0 * (double)T);

    for (int i = 0; i < half; ++i) {
        if (((double)rand() / RAND_MAX) <= c2) {
            int feedback_count = clause_update(ts->pos_clauses[other_class][i], X, 0, pos_vals[i], s, threshold);
            if (out_feedback) out_feedback->non_target_type2 += feedback_count;
        }
        if (((double)rand() / RAND_MAX) <= c2) {
            int feedback_count = clause_update(ts->neg_clauses[other_class][i], X, 1, neg_vals[i], s, threshold);
            if (out_feedback) out_feedback->non_target_type1 += feedback_count;
        }
    }

    free(pos_vals);
    free(neg_vals);
}

/* Fit across dataset. X is array of sample pointers (each sample is int array length n_features). */
void tsetlin_fit(tsetlin_t* ts, const int** X, const int* y, int n_samples, int T, double s, int epochs) {
    assert(ts != NULL);
    assert(X != NULL);
    assert(y != NULL);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < n_samples; ++i) {
            tsetlin_step(ts, X[i], y[i], T, s, NULL, -1);
        }
    }
}
