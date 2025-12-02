#include "clause.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Helper: append index to dynamic int array, resizing as needed. */
static void append_idx(int** arr, int* count, int idx) {
    int new_count = (*count) + 1;
    int* tmp = (int*)realloc(*arr, sizeof(int) * new_count);
    if (!tmp) return; /* allocation failure: silently ignore (best-effort) */
    tmp[new_count - 1] = idx;
    *arr = tmp;
    *count = new_count;
}

/* Helper: remove index from list if present. Order of remaining elements preserved. */
static void remove_idx(int** arr, int* count, int idx) {
    if (!*arr || *count == 0) return;
    int found = -1;
    for (int i = 0; i < *count; ++i) {
        if ((*arr)[i] == idx) { found = i; break; }
    }
    if (found < 0) return;
    for (int i = found; i < (*count) - 1; ++i) {
        (*arr)[i] = (*arr)[i + 1];
    }
    --(*count);
    if (*count == 0) {
        free(*arr);
        *arr = NULL;
    }
    else {
        int* tmp = (int*)realloc(*arr, sizeof(int) * (*count));
        if (tmp) *arr = tmp;
    }
}

clause_t* clause_new(int N_feature, int N_states) {
    assert((N_states % 2) == 0);

    clause_t* c = (clause_t*)calloc(1, sizeof(clause_t));
    if (!c) return NULL;

    c->N_feature = N_feature;
    c->N_states = N_states;
    c->N_literals = 2 * N_feature;

    c->p_automata = (automaton_t**)calloc(N_feature, sizeof(automaton_t*));
    c->n_automata = (automaton_t**)calloc(N_feature, sizeof(automaton_t*));
    if (!c->p_automata || !c->n_automata) {
        clause_free(c);
        return NULL;
    }

    /* Create automata; initialize with state = -1 as in Python. */
    for (int i = 0; i < N_feature; ++i) {
        c->p_automata[i] = automaton_new(N_states, -1);
        c->n_automata[i] = automaton_new(N_states, -1);
        if (!c->p_automata[i] || !c->n_automata[i]) {
            clause_free(c);
            return NULL;
        }
    }

    /* Randomly initialise automata states: middle_state + {0,1} and complementary. */
    /* Do not reseed global RNG here; use rand() as-is. */
    for (int i = 0; i < N_feature; ++i) {
        int choice = rand() % 2; /* 0 or 1 */
        c->p_automata[i]->state = (N_states / 2) + choice;
        c->n_automata[i]->state = (N_states / 2) + (1 - choice);
        automaton_update(c->p_automata[i]);
        automaton_update(c->n_automata[i]);
    }

    /* initial compress (no threshold) */
    clause_compress(c, -1);

    return c;
}

void clause_free(clause_t* c) {
    if (!c) return;
    if (c->p_automata) {
        for (int i = 0; i < c->N_feature; ++i) {
            automaton_free(c->p_automata[i]);
        }
        free(c->p_automata);
    }
    if (c->n_automata) {
        for (int i = 0; i < c->N_feature; ++i) {
            automaton_free(c->n_automata[i]);
        }
        free(c->n_automata);
    }

    free(c->p_included_idxs);
    free(c->n_included_idxs);
    free(c->p_trainable_idxs);
    free(c->n_trainable_idxs);
    free(c);
}

void clause_compress(clause_t* c, int threshold) {
    if (!c) return;

    /* Clear current lists */
    free(c->p_included_idxs);
    c->p_included_idxs = NULL;
    c->p_included_count = 0;

    free(c->n_included_idxs);
    c->n_included_idxs = NULL;
    c->n_included_count = 0;

    free(c->p_trainable_idxs);
    c->p_trainable_idxs = NULL;
    c->p_trainable_count = 0;

    free(c->n_trainable_idxs);
    c->n_trainable_idxs = NULL;
    c->n_trainable_count = 0;

    for (int i = 0; i < c->N_feature; ++i) {
        if (automaton_action(c->p_automata[i]) == 1) {
            append_idx(&c->p_included_idxs, &c->p_included_count, i);
        }
        if (automaton_action(c->n_automata[i]) == 1) {
            append_idx(&c->n_included_idxs, &c->n_included_count, i);
        }

        if (threshold > 0) {
            if (abs(c->p_automata[i]->state - (c->N_states / 2)) <= threshold) {
                append_idx(&c->p_trainable_idxs, &c->p_trainable_count, i);
            }
            if (abs(c->n_automata[i]->state - (c->N_states / 2)) <= threshold) {
                append_idx(&c->n_trainable_idxs, &c->n_trainable_count, i);
            }
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

/* Helper to test membership quickly (search included idx lists). */
static bool is_included(int* idxs, int count, int idx) {
    for (int i = 0; i < count; ++i) if (idxs[i] == idx) return true;
    return false;
}

int clause_update(clause_t* c, const int* X, int match_target, int clause_output, double s, int threshold) {
    assert(c != NULL);
    assert(X != NULL);
    int feedback_count = 0;

    double s1 = 0.0, s2 = 0.0;
    if (s > 0.0) {
        s1 = 1.0 / s;
        s2 = (s - 1.0) / s;
    }

    /* Type I feedback (match_target == 1) */
    if (match_target == 1) {
        /* Erase Pattern: reduce included literals when clause_output == 0 */
        if (clause_output == 0) {
            if (threshold < 0) {
                for (int i = 0; i < c->N_feature; ++i) {
                    /* Positive automaton */
                    if (c->p_automata[i]->state > 1 && ((double)rand() / RAND_MAX) <= s1) {
                        feedback_count++;
                        if (automaton_penalty(c->p_automata[i]) && is_included(c->p_included_idxs, c->p_included_count, i)) {
                            remove_idx(&c->p_included_idxs, &c->p_included_count, i);
                        }
                    }

                    /* Negative automaton */
                    if (c->n_automata[i]->state > 1 && ((double)rand() / RAND_MAX) <= s1) {
                        feedback_count++;
                        if (automaton_penalty(c->n_automata[i]) && is_included(c->n_included_idxs, c->n_included_count, i)) {
                            remove_idx(&c->n_included_idxs, &c->n_included_count, i);
                        }
                    }
                }
            }
            else {
                /* thresholded: only trainable lists */
                for (int ii = 0; ii < c->p_trainable_count; ++ii) {
                    int i = c->p_trainable_idxs[ii];
                    if (c->p_automata[i]->state > 1 && ((double)rand() / RAND_MAX) <= s1) {
                        feedback_count++;
                        if (automaton_penalty(c->p_automata[i]) && is_included(c->p_included_idxs, c->p_included_count, i)) {
                            remove_idx(&c->p_included_idxs, &c->p_included_count, i);
                        }
                    }
                }
                for (int ii = 0; ii < c->n_trainable_count; ++ii) {
                    int i = c->n_trainable_idxs[ii];
                    if (c->n_automata[i]->state > 1 && ((double)rand() / RAND_MAX) <= s1) {
                        feedback_count++;
                        if (automaton_penalty(c->n_automata[i]) && is_included(c->n_included_idxs, c->n_included_count, i)) {
                            remove_idx(&c->n_included_idxs, &c->n_included_count, i);
                        }
                    }
                }
            }
        }

        /* Recognize Pattern: increase included literals when clause_output == 1 */
        if (clause_output == 1) {
            if (threshold < 0) {
                for (int i = 0; i < c->N_feature; ++i) {
                    if (X[i] == 1) {
                        /* Positive literal X */
                        if (c->p_automata[i]->state < c->N_states && ((double)rand() / RAND_MAX) <= s2) {
                            feedback_count++;
                            if (automaton_reward(c->p_automata[i]) && !is_included(c->p_included_idxs, c->p_included_count, i)) {
                                append_idx(&c->p_included_idxs, &c->p_included_count, i);
                            }
                        }
                        /* Negative automaton: penalize to remove NOT X */
                        if (c->n_automata[i]->state > 1 && ((double)rand() / RAND_MAX) <= s1) {
                            feedback_count++;
                            if (automaton_penalty(c->n_automata[i]) && is_included(c->n_included_idxs, c->n_included_count, i)) {
                                remove_idx(&c->n_included_idxs, &c->n_included_count, i);
                            }
                        }
                    }
                    else { /* X[i] == 0 */
                        /* Negative literal NOT X */
                        if (c->n_automata[i]->state < c->N_states && ((double)rand() / RAND_MAX) <= s2) {
                            feedback_count++;
                            if (automaton_reward(c->n_automata[i]) && !is_included(c->n_included_idxs, c->n_included_count, i)) {
                                append_idx(&c->n_included_idxs, &c->n_included_count, i);
                            }
                        }
                        /* Positive automaton: penalize to remove X */
                        if (c->p_automata[i]->state > 1 && ((double)rand() / RAND_MAX) <= s1) {
                            feedback_count++;
                            if (automaton_penalty(c->p_automata[i]) && is_included(c->p_included_idxs, c->p_included_count, i)) {
                                remove_idx(&c->p_included_idxs, &c->p_included_count, i);
                            }
                        }
                    }
                }
            }
            else {
                /* thresholded operations */
                for (int ii = 0; ii < c->p_trainable_count; ++ii) {
                    int i = c->p_trainable_idxs[ii];
                    if (X[i] == 1) {
                        if (c->p_automata[i]->state < c->N_states && ((double)rand() / RAND_MAX) <= s2) {
                            feedback_count++;
                            if (automaton_reward(c->p_automata[i]) && !is_included(c->p_included_idxs, c->p_included_count, i)) {
                                append_idx(&c->p_included_idxs, &c->p_included_count, i);
                            }
                        }
                    }
                    else {
                        if (c->p_automata[i]->state > 1 && ((double)rand() / RAND_MAX) <= s1) {
                            feedback_count++;
                            if (automaton_penalty(c->p_automata[i]) && is_included(c->p_included_idxs, c->p_included_count, i)) {
                                remove_idx(&c->p_included_idxs, &c->p_included_count, i);
                            }
                        }
                    }
                }

                for (int ii = 0; ii < c->n_trainable_count; ++ii) {
                    int i = c->n_trainable_idxs[ii];
                    if (X[i] == 1) {
                        if (c->n_automata[i]->state > 1 && ((double)rand() / RAND_MAX) <= s1) {
                            feedback_count++;
                            if (automaton_penalty(c->n_automata[i]) && is_included(c->n_included_idxs, c->n_included_count, i)) {
                                remove_idx(&c->n_included_idxs, &c->n_included_count, i);
                            }
                        }
                    }
                    else {
                        if (c->n_automata[i]->state < c->N_states && ((double)rand() / RAND_MAX) <= s2) {
                            feedback_count++;
                            if (automaton_reward(c->n_automata[i]) && !is_included(c->n_included_idxs, c->n_included_count, i)) {
                                append_idx(&c->n_included_idxs, &c->n_included_count, i);
                            }
                        }
                    }
                }
            }
        }
    }
    /* Type II feedback (match_target == 0) */
    else {
        if (clause_output == 1) {
            if (threshold < 0) {
                for (int i = 0; i < c->N_feature; ++i) {
                    if ((X[i] == 0) && (automaton_action(c->p_automata[i]) == 0)) {
                        feedback_count++;
                        if (automaton_reward(c->p_automata[i]) && !is_included(c->p_included_idxs, c->p_included_count, i)) {
                            append_idx(&c->p_included_idxs, &c->p_included_count, i);
                        }
                    }
                    else if ((X[i] == 1) && (automaton_action(c->n_automata[i]) == 0)) {
                        feedback_count++;
                        if (automaton_reward(c->n_automata[i]) && !is_included(c->n_included_idxs, c->n_included_count, i)) {
                            append_idx(&c->n_included_idxs, &c->n_included_count, i);
                        }
                    }
                }
            }
            else {
                for (int ii = 0; ii < c->p_trainable_count; ++ii) {
                    int i = c->p_trainable_idxs[ii];
                    if ((X[i] == 0) && (automaton_action(c->p_automata[i]) == 0)) {
                        feedback_count++;
                        if (automaton_reward(c->p_automata[i]) && !is_included(c->p_included_idxs, c->p_included_count, i)) {
                            append_idx(&c->p_included_idxs, &c->p_included_count, i);
                        }
                    }
                }
                for (int ii = 0; ii < c->n_trainable_count; ++ii) {
                    int i = c->n_trainable_idxs[ii];
                    if ((X[i] == 1) && (automaton_action(c->n_automata[i]) == 0)) {
                        feedback_count++;
                        if (automaton_reward(c->n_automata[i]) && !is_included(c->n_included_idxs, c->n_included_count, i)) {
                            append_idx(&c->n_included_idxs, &c->n_included_count, i);
                        }
                    }
                }
            }
        }
    }

    /* After updates, automata action fields are maintained by automaton functions.
     * Rebuild compress lists if threshold parameter used or to keep lists current.
     */
    clause_compress(c, threshold);
    return feedback_count;
}

void clause_set_state(clause_t* c, const int* states, int threshold) {
    assert(c != NULL);
    assert(states != NULL);

    for (int i = 0; i < c->N_feature; ++i) {
        c->p_automata[i]->state = states[i];
        automaton_update(c->p_automata[i]);
        c->n_automata[i]->state = states[i + c->N_feature];
        automaton_update(c->n_automata[i]);
    }
    clause_compress(c, threshold);
}

int* clause_get_state(const clause_t* c) {
    if (!c) return NULL;
    int* states = (int*)malloc(sizeof(int) * 2 * c->N_feature);
    if (!states) return NULL;
    for (int i = 0; i < c->N_feature; ++i) {
        states[i] = c->p_automata[i]->state;
        states[i + c->N_feature] = c->n_automata[i]->state;
    }
    return states;
}
