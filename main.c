#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "tsetlin/tsetlin.h"
#include "log.h"

/*
  Simplified standalone implementation of the Python script in C.

  - Loads "iris.csv" (expects 5 columns: 4 numeric features + species string).
  - Normalizes features, booleanizes into bits per feature.
  - Splits into train/test.
  - Trains a Tsetlin machine using the C API implemented in this project.
  - Evaluates and prints accuracy.

  Notes:
  - This is intended as a runnable example. It omits model (de)serialization
    and Optuna/profiling support and uses a simple booleanization strategy.
  - Adjust paths and compile via your CMake configuration.
*/

/* ---------- CSV loader for Iris (simple) ---------- */
/* Maps species strings to ints: "setosa"->0, "versicolor"->1, "virginica"->2 */
static int species_to_label(const char* s) {
    if (strstr(s, "setosa")) return 0;
    if (strstr(s, "versicolor")) return 1;
    if (strstr(s, "virginica")) return 2;
    return -1;
}

#define N_FEATUES 4

/* Loads CSV into X (allocated double[n_samples][n_features]) and y (allocated int[n_samples])
   Returns 0 on success, non-zero on error. */
static int load_iris_csv(const char* path, double*** out_X, int* out_n_samples, int* out_n_features, int** out_y) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;

    /* read all lines into buffer (Iris is small) */
    char line[256];
    double** X = NULL;
    int* y = NULL;
    int n = 0;

    while (fgets(line, sizeof(line), f)) {
        /* trim newline */
        char* nl = strchr(line, '\n');
        if (nl) *nl = '\0';
        if (strlen(line) == 0) continue;

        /* parse 4 doubles and a string */
        double vals[N_FEATUES];
        char species[64];
        int scanned = sscanf(line, "%lf,%lf,%lf,%lf,%63s", &vals[0], &vals[1], &vals[2], &vals[3], species);
        if (scanned < 5) {
            /* Try species with spaces (some CSVs may have quotes) */
            char* last_comma = strrchr(line, ',');
            if (!last_comma) continue;
            strncpy(species, last_comma + 1, sizeof(species) - 1);
            species[sizeof(species) - 1] = '\0';
            /* parse numeric part */
            char numeric[256];
            size_t nlen = last_comma - line;
            if (nlen >= sizeof(numeric)) continue;
            memcpy(numeric, line, nlen);
            numeric[nlen] = '\0';
            if (sscanf(numeric, "%lf,%lf,%lf,%lf", &vals[0], &vals[1], &vals[2], &vals[3]) != 4) continue;
        }

        double* row = (double*)malloc(sizeof(double) * N_FEATUES);
        if (!row) { fclose(f); return -1; }
        for (int i = 0; i < N_FEATUES; ++i) row[i] = vals[i];

        double** tmpX = (double**)realloc(X, sizeof(double*) * (n + 1));
        int* tmpy = (int*)realloc(y, sizeof(int) * (n + 1));
        if (!tmpX || !tmpy) { free(row); fclose(f); return -1; }
        X = tmpX; y = tmpy;
        X[n] = row;
        y[n] = species_to_label(species);
        ++n;
    }

    fclose(f);

    *out_X = X;
    *out_y = y;
    *out_n_samples = n;
    *out_n_features = N_FEATUES;
    return 0;
}

/* ---------- Mean / Std helpers ---------- */
static void compute_mean_std(double** X, int n_samples, int n_features, double* mean_out, double* std_out) {
    for (int j = 0; j < n_features; ++j) {
        double sum = 0.0;
        for (int i = 0; i < n_samples; ++i) sum += X[i][j];
        double mean = sum / (double)n_samples;
        mean_out[j] = mean;

        double var = 0.0;
        for (int i = 0; i < n_samples; ++i) {
            double d = X[i][j] - mean;
            var += d * d;
        }
        var /= (double)n_samples;
        std_out[j] = (var > 0.0) ? sqrt(var) : 1.0;
    }
}

/* ---------- Booleanize features ---------- */
/* Converts each real-valued feature into `num_bits` boolean features.
   Strategy: generate `num_bits` thresholds linearly spaced around mean ± std,
   and set bit = 1 when value > threshold. Output is int matrix of size
   n_samples x (n_features * num_bits).
   Caller must free returned array (array of int*). */
static int** booleanize_features(double** X, int n_samples, int n_features, const double* mean, const double* std, int num_bits) {
    int out_features = n_features * num_bits;
    int** Xb = (int**)malloc(sizeof(int*) * n_samples);
    if (!Xb) return NULL;

    /* compute thresholds per feature */
    double* thresholds = (double*)malloc(sizeof(double) * n_features * num_bits);
    if (!thresholds) { free(Xb); return NULL; }
    for (int j = 0; j < n_features; ++j) {
        double left = mean[j] - std[j];
        double right = mean[j] + std[j];
        for (int b = 0; b < num_bits; ++b) {
            double t = left + ((double)b + 0.5) * (right - left) / (double)num_bits;
            thresholds[j * num_bits + b] = t;
        }
    }

    for (int i = 0; i < n_samples; ++i) {
        int* row = (int*)malloc(sizeof(int) * out_features);
        if (!row) { /* free allocated and return */
            for (int k = 0; k < i; ++k) free(Xb[k]);
            free(Xb); free(thresholds); return NULL;
        }
        for (int j = 0; j < n_features; ++j) {
            double v = X[i][j];
            for (int b = 0; b < num_bits; ++b) {
                int idx = j * num_bits + b;
                row[idx] = (v > thresholds[idx]) ? 1 : 0;
            }
        }
        Xb[i] = row;
    }

    free(thresholds);
    return Xb;
}

/* ---------- Train/Test split ---------- */
/* Simple deterministic split with random_state seed. Returns train arrays via pointers.
   Allocates new arrays for train/test sample pointers; does not copy feature rows. */
static void train_test_split(int** Xb, int* y, int n_samples, double test_size, int random_state,
    int*** X_train_out, int** y_train_out, int* n_train_out,
    int*** X_test_out, int** y_test_out, int* n_test_out) {
    int* idx = (int*)malloc(sizeof(int) * n_samples);
    for (int i = 0; i < n_samples; ++i) idx[i] = i;

    srand((unsigned)random_state);
    /* Fisher-Yates shuffle */
    for (int i = n_samples - 1; i > 0; --i) {
        int r = rand() % (i + 1);
        int tmp = idx[i]; idx[i] = idx[r]; idx[r] = tmp;
    }

    int n_test = (int)round(test_size * n_samples);
    int n_train = n_samples - n_test;

    int** X_train = (int**)malloc(sizeof(int*) * n_train);
    int* y_train = (int*)malloc(sizeof(int) * n_train);
    int** X_test = (int**)malloc(sizeof(int*) * n_test);
    int* y_test = (int*)malloc(sizeof(int) * n_test);

    int ti = 0, vi = 0;
    for (int i = 0; i < n_samples; ++i) {
        if (i < n_train) {
            X_train[ti] = Xb[idx[i]];
            y_train[ti] = y[idx[i]];
            ++ti;
        }
        else {
            X_test[vi] = Xb[idx[i]];
            y_test[vi] = y[idx[i]];
            ++vi;
        }
    }

    free(idx);
    *X_train_out = X_train; *y_train_out = y_train; *n_train_out = n_train;
    *X_test_out = X_test; *y_test_out = y_test; *n_test_out = n_test;
}

/* ---------- Accuracy helper ---------- */
static double compute_accuracy(tsetlin_t* ts, int** X_samples, int* y, int n_samples) {
    int correct = 0;
    for (int i = 0; i < n_samples; ++i) {
        int pred = tsetlin_predict(ts, X_samples[i], NULL);
        if (pred == y[i]) ++correct;
    }
    return (double)correct / (double)n_samples;
}

/* ---------- Simple log wrapper using log.h library if available ---------- */
static void my_log(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
}

/* ---------- main (command-line) ---------- */
int main(int argc, char** argv) {
    const char* csv_path = "iris.csv";
    int epochs = 10;
    int N_CLAUSE = 20;
    int N_STATE = 10;
    int N_BIT = 4;
    int T = 30;
    double s = 6.0;
    int optuna = 0;

    /* Basic argument parsing (minimal) */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n_clause") == 0 && i + 1 < argc) N_CLAUSE = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n_state") == 0 && i + 1 < argc) N_STATE = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n_bit") == 0 && i + 1 < argc) N_BIT = atoi(argv[++i]);
        else if (strcmp(argv[i], "--T") == 0 && i + 1 < argc) T = atoi(argv[++i]);
        else if (strcmp(argv[i], "--s") == 0 && i + 1 < argc) s = atof(argv[++i]);
        else if (strcmp(argv[i], "--optuna") == 0) optuna = 1;
    }

    if (!(N_BIT == 1 || N_BIT == 2 || N_BIT == 4 || N_BIT == 8)) {
        fprintf(stderr, "n_bit must be one of [1,2,4,8]\n");
        return 1;
    }

    /* Load iris dataset */
    double** X_real = NULL;
    int* y_labels = NULL;
    int n_samples = 0;
    int n_features = 0;
    if (load_iris_csv(csv_path, &X_real, &n_samples, &n_features, &y_labels) != 0) {
        fprintf(stderr, "Failed to load %s\n", csv_path);
        return 1;
    }
    my_log("Loaded %d samples, %d features", n_samples, n_features);

    /* Normalization stats */
    double* mean = (double*)malloc(sizeof(double) * n_features);
    double* std = (double*)malloc(sizeof(double) * n_features);
    compute_mean_std(X_real, n_samples, n_features, mean, std);

    /* Booleanize features */
    int** Xb = booleanize_features(X_real, n_samples, n_features, mean, std, N_BIT);
    int bool_features = n_features * N_BIT;
    if (!Xb) {
        fprintf(stderr, "Booleanization failed\n");
        return 1;
    }

    /* Prepare train/test split (test_size=0.2, random_state=0 as in Python) */
    int** X_train = NULL, ** X_test = NULL;
    int* y_train = NULL, * y_test = NULL;
    int n_train = 0, n_test = 0;
    train_test_split(Xb, y_labels, n_samples, 0.2, 0, &X_train, &y_train, &n_train, &X_test, &y_test, &n_test);
    my_log("Train samples: %d, Test samples: %d. Boolean features: %d", n_train, n_test, bool_features);

    if (optuna) {
        my_log("Optuna-style optimization is not implemented in this C example; run without --optuna.");
    }

    /* Create tsetlin */
    tsetlin_t* ts = tsetlin_new(bool_features, 3, N_CLAUSE, N_STATE);
    if (!ts) {
        fprintf(stderr, "Failed to allocate Tsetlin instance\n");
        return 1;
    }

    /* Initial evaluation */
    double test_acc = compute_accuracy(ts, X_test, y_test, n_test);
    my_log("Initial test accuracy: %.2f%%", test_acc * 100.0);

    /* Training loops */
    for (int epoch = 0; epoch < epochs; ++epoch) {
        my_log("[Epoch %d/%d] Starting", epoch + 1, epochs);
        for (int i = 0; i < n_train; ++i) {
            tsetlin_step(ts, X_train[i], y_train[i], T, s, NULL, -1);
        }
        double train_acc = compute_accuracy(ts, X_train, y_train, n_train);
        my_log("[Epoch %d/%d] Train Accuracy: %.2f%%", epoch + 1, epochs, train_acc * 100.0);
    }

    /* Final evaluation */
    test_acc = compute_accuracy(ts, X_test, y_test, n_test);
    my_log("Final test accuracy: %.2f%%", test_acc * 100.0);

    /* Clean up */
    tsetlin_free(ts);
    for (int i = 0; i < n_samples; ++i) free(X_real[i]);
    free(X_real);
    free(y_labels);
    for (int i = 0; i < n_samples; ++i) free(Xb[i]);
    free(Xb);
    free(mean);
    free(std);
    free(X_train);
    free(y_train);
    free(X_test);
    free(y_test);

    return 0;
}
