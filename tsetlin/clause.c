#include "clause.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Helpers: dynamic int list append/remove/search */
static void append_idx(int** arr, int* count, int idx) {
    int new_count = (*count) + 1;
    int* tmp = (int*)realloc(*arr, sizeof(int) * new_count);
    if (!tmp) return;
    tmp[new_count - 1] = idx;
    *arr = tmp;
    *count = new_count;
}

static void remove_idx(int** arr, int* count, int idx) {
    if (!*arr || *count == 0) return;
    int found = -1;
    for (int i = 0; i < *count; ++i) {
        if ((*arr)[i] == idx) { found = i; break; }
    }
    if (found < 0) return;
    for (int i = found; i < (*count) - 1; ++i) (*arr)[i] = (*arr)[i + 1];
    --(*count);
    if (*count == 0) { free(*arr); *arr = NULL; }
    else {
        int* tmp = (int*)realloc(*arr, sizeof(int) * (*count));
        if (tmp) *arr = tmp;
    }
}

static bool is_included(const int* arr, int count, int idx) {
    for (int i = 0; i < count; ++i) if (arr[i] == idx) return true;
    return false;
}

clause_t* clause_new(int N_feature, int N_states) {
    assert((N_states % 2) == 0);
    clause_t* c = (clause_t*)calloc(1, sizeof(clause_t));
    if (!c) return NULL;

    c->N_feature = N_feature;
    c->N_states = N_states;
    c->N_literals = 2 * N_feature;
    c->middle = N_states / 2;

    c->p_states = (int*)malloc(sizeof(int) * N_feature);
    c->n_states = (int*)malloc(sizeof(int) * N_feature);
    if (!c->p_states || !c->n_states) {
        clause_free(c);
        return NULL;
    }

    /* initialize to middle and randomize middle + {0,1} complementary */
    for (int i = 0; i < N_feature; ++i) {
        int choice = rand() % 2;
        c->p_states[i] = c->middle + choice;
        c->n_states[i] = c->middle + (1 - choice);
    }

    clause_compress(c, -1);
    return c;
}

void clause_free(clause_t* c) {
    if (!c) return;
    free(c->p_states);
    free(c->n_states);
    free(c->p_included_idxs);
    free(c->n_included_idxs);
    free(c->p_trainable_idxs);
    free(c->n_trainable_idxs);
    free(c);
}

void clause_compress(clause_t* c, int threshold) {
    if (!c) return;

    free(c->p_included_idxs); c->p_included_idxs = NULL; c->p_included_count = 0;
    free(c->n_included_idxs); c->n_included_idxs = NULL; c->n_included_count = 0;
    free(c->p_trainable_idxs); c->p_trainable_idxs = NULL; c->p_trainable_count = 0;
    free(c->n_trainable_idxs); c->n_trainable_idxs = NULL; c->n_trainable_count = 0;

    for (int i = 0; i < c->N_feature; ++i) {
        if (c->p_states[i] > c->middle) append_idx(&c->p_included_idxs, &c->p_included_count, i);
        if (c->n_states[i] > c->middle) append_idx(&c->n_included_idxs, &c->n_included_count, i);

        if (threshold > 0) {
            if (abs(c->p_states[i] - c->middle) <= threshold)
                append_idx(&c->p_trainable_idxs, &c->p_trainable_count, i);
            if (abs(c->n_states[i] - c->middle) <= threshold)
                append_idx(&c->n_trainable_idxs, &c->n_trainable_count, i);
        }
    }
}

int clause_evaluate(const clause_t* c, const int* X) {
    if (!c || !X) return 0;
    for (int i = 0; i < c->p_included_count; ++i) {
        int idx = c->p_included_idxs[i];
        if (X[idx] == 0) return 0;
    }
    for (int i = 0; i < c->n_included_count; ++i) {
        int idx = c->n_included_idxs[i];
        if (X[idx] == 1) return 0;
    }
    return 1;
}

int clause_update(clause_t* c, const int* X, int match_target, int clause_output, double s, int threshold) {
    assert(c != NULL);
    assert(X != NULL);
    int feedback_count = 0;
    double s1 = 0.0, s2 = 0.0;
    if (s > 0.0) { s1 = 1.0 / s; s2 = (s - 1.0) / s; }

    /* Type I feedback */
    if (match_target == 1) {
        if (clause_output == 0) {
            /* Erase pattern: penalize to reduce included literals */
            if (threshold < 0) {
                for (int i = 0; i < c->N_feature; ++i) {
                    /* positive literal penalty */
                    if (c->p_states[i] > 1 && ((double)rand() / (double)RAND_MAX) <= s1) {
                        c->p_states[i] -= 1;
                        feedback_count++;
                        if (c->p_states[i] == c->middle) {
                            remove_idx(&c->p_included_idxs, &c->p_included_count, i);
                        }
                    }
                    /* negative literal penalty */
                    if (c->n_states[i] > 1 && ((double)rand() / (double)RAND_MAX) <= s1) {
                        c->n_states[i] -= 1;
                        feedback_count++;
                        if (c->n_states[i] == c->middle) {
                            remove_idx(&c->n_included_idxs, &c->n_included_count, i);
                        }
                    }
                }
            }
            else {
                for (int ii = 0; ii < c->p_trainable_count; ++ii) {
                    int i = c->p_trainable_idxs[ii];
                    if (c->p_states[i] > 1 && ((double)rand() / (double)RAND_MAX) <= s1) {
                        c->p_states[i] -= 1;
                        feedback_count++;
                        if (c->p_states[i] == c->middle) remove_idx(&c->p_included_idxs, &c->p_included_count, i);
                    }
                }
                for (int ii = 0; ii < c->n_trainable_count; ++ii) {
                    int i = c->n_trainable_idxs[ii];
                    if (c->n_states[i] > 1 && ((double)rand() / (double)RAND_MAX) <= s1) {
                        c->n_states[i] -= 1;
                        feedback_count++;
                        if (c->n_states[i] == c->middle) remove_idx(&c->n_included_idxs, &c->n_included_count, i);
                    }
                }
            }
        }
        else {
            /* Recognize pattern: increase included literals */
            if (threshold < 0) {
                for (int i = 0; i < c->N_feature; ++i) {
                    if (X[i] == 1) {
                        /* reward positive literal */
                        if (c->p_states[i] < c->N_states && ((double)rand() / (double)RAND_MAX) <= s2) {
                            c->p_states[i] += 1;
                            feedback_count++;
                            if (c->p_states[i] == c->middle + 1 && !is_included(c->p_included_idxs, c->p_included_count, i)) {
                                append_idx(&c->p_included_idxs, &c->p_included_count, i);
                            }
                        }
                        /* penalty negative literal */
                        if (c->n_states[i] > 1 && ((double)rand() / (double)RAND_MAX) <= s1) {
                            c->n_states[i] -= 1;
                            feedback_count++;
                            if (c->n_states[i] == c->middle) remove_idx(&c->n_included_idxs, &c->n_included_count, i);
                        }
                    }
                    else {
                        /* X == 0: reward negative literal */
                        if (c->n_states[i] < c->N_states && ((double)rand() / (double)RAND_MAX) <= s2) {
                            c->n_states[i] += 1;
                            feedback_count++;
                            if (c->n_states[i] == c->middle + 1 && !is_included(c->n_included_idxs, c->n_included_count, i)) {
                                append_idx(&c->n_included_idxs, &c->n_included_count, i);
                            }
                        }
                        /* penalty positive literal */
                        if (c->p_states[i] > 1 && ((double)rand() / (double)RAND_MAX) <= s1) {
                            c->p_states[i] -= 1;
                            feedback_count++;
                            if (c->p_states[i] == c->middle) remove_idx(&c->p_included_idxs, &c->p_included_count, i);
                        }
                    }
                }
            }
            else {
                for (int ii = 0; ii < c->p_trainable_count; ++ii) {
                    int i = c->p_trainable_idxs[ii];
                    if (X[i] == 1) {
                        if (c->p_states[i] < c->N_states && ((double)rand() / (double)RAND_MAX) <= s2) {
                            c->p_states[i] += 1;
                            feedback_count++;
                            if (c->p_states[i] == c->middle + 1 && !is_included(c->p_included_idxs, c->p_included_count, i))
                                append_idx(&c->p_included_idxs, &c->p_included_count, i);
                        }
                    }
                    else {
                        if (c->p_states[i] > 1 && ((double)rand() / (double)RAND_MAX) <= s1) {
                            c->p_states[i] -= 1;
                            feedback_count++;
                            if (c->p_states[i] == c->middle) remove_idx(&c->p_included_idxs, &c->p_included_count, i);
                        }
                    }
                }
                for (int ii = 0; ii < c->n_trainable_count; ++ii) {
                    int i = c->n_trainable_idxs[ii];
                    if (X[i] == 1) {
                        if (c->n_states[i] > 1 && ((double)rand() / (double)RAND_MAX) <= s1) {
                            c->n_states[i] -= 1;
                            feedback_count++;
                            if (c->n_states[i] == c->middle) remove_idx(&c->n_included_idxs, &c->n_included_count, i);
                        }
                    }
                    else {
                        if (c->n_states[i] < c->N_states && ((double)rand() / (double)RAND_MAX) <= s2) {
                            c->n_states[i] += 1;
                            feedback_count++;
                            if (c->n_states[i] == c->middle + 1 && !is_included(c->n_included_idxs, c->n_included_count, i))
                                append_idx(&c->n_included_idxs, &c->n_included_count, i);
                        }
                    }
                }
            }
        }
    }
    /* Type II feedback */
    else {
        if (clause_output == 1) {
            if (threshold < 0) {
                for (int i = 0; i < c->N_feature; ++i) {
                    if (X[i] == 0 && c->p_states[i] <= c->middle) {
                        c->p_states[i] += 1;
                        feedback_count++;
                        if (c->p_states[i] == c->middle + 1 && !is_included(c->p_included_idxs, c->p_included_count, i))
                            append_idx(&c->p_included_idxs, &c->p_included_count, i);
                    }
                    else if (X[i] == 1 && c->n_states[i] <= c->middle) {
                        c->n_states[i] += 1;
                        feedback_count++;
                        if (c->n_states[i] == c->middle + 1 && !is_included(c->n_included_idxs, c->n_included_count, i))
                            append_idx(&c->n_included_idxs, &c->n_included_count, i);
                    }
                }
            }
            else {
                for (int ii = 0; ii < c->p_trainable_count; ++ii) {
                    int i = c->p_trainable_idxs[ii];
                    if (X[i] == 0 && c->p_states[i] <= c->middle) {
                        c->p_states[i] += 1;
                        feedback_count++;
                        if (c->p_states[i] == c->middle + 1 && !is_included(c->p_included_idxs, c->p_included_count, i))
                            append_idx(&c->p_included_idxs, &c->p_included_count, i);
                    }
                }
                for (int ii = 0; ii < c->n_trainable_count; ++ii) {
                    int i = c->n_trainable_idxs[ii];
                    if (X[i] == 1 && c->n_states[i] <= c->middle) {
                        c->n_states[i] += 1;
                        feedback_count++;
                        if (c->n_states[i] == c->middle + 1 && !is_included(c->n_included_idxs, c->n_included_count, i))
                            append_idx(&c->n_included_idxs, &c->n_included_count, i);
                    }
                }
            }
        }
    }

    /* Rebuild trainable lists / included lists to keep consistency */
    clause_compress(c, threshold);
    return feedback_count;
}

void clause_set_state(clause_t* c, const int* states, int threshold) {
    assert(c != NULL);
    assert(states != NULL);
    for (int i = 0; i < c->N_feature; ++i) {
        c->p_states[i] = states[i];
        c->n_states[i] = states[i + c->N_feature];
    }
    clause_compress(c, threshold);
}

int* clause_get_state(const clause_t* c) {
    if (!c) return NULL;
    int* states = (int*)malloc(sizeof(int) * 2 * c->N_feature);
    if (!states) return NULL;
    for (int i = 0; i < c->N_feature; ++i) {
        states[i] = c->p_states[i];
        states[i + c->N_feature] = c->n_states[i];
    }
    return states;
}