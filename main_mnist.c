#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#include <tsetlin.h>
#include <log.h>

#include <tqdm.h>

/* Simple IDX file readers for MNIST (train/test files must be present in CWD):
   - train-images-idx3-ubyte
   - train-labels-idx1-ubyte
   - t10k-images-idx3-ubyte
   - t10k-labels-idx1-ubyte
*/

/* Read big-endian 32-bit */
static uint32_t read_be_u32(FILE* f) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return (uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | (uint32_t)b[3];
}

static uint8_t* load_idx_images(const char* path, int* out_count, int* out_rows, int* out_cols) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic = read_be_u32(f);
    uint32_t count = read_be_u32(f);
    uint32_t rows = read_be_u32(f);
    uint32_t cols = read_be_u32(f);
    if (magic != 0x00000803) { fclose(f); return NULL; }
    size_t total = (size_t)count * rows * cols;
    uint8_t* buf = (uint8_t*)malloc(total);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, total, f) != total) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *out_count = (int)count;
    *out_rows = (int)rows;
    *out_cols = (int)cols;
    return buf;
}

static uint8_t* load_idx_labels(const char* path, int* out_count) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic = read_be_u32(f);
    uint32_t count = read_be_u32(f);
    if (magic != 0x00000801) { fclose(f); return NULL; }
    uint8_t* buf = (uint8_t*)malloc(count);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, count, f) != count) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *out_count = (int)count;
    return buf;
}

/* Convert raw uint8 images to binary int arrays (0/1) using threshold */
static int** binarize_images(uint8_t* images, int n, int rows, int cols, int threshold) {
    int features = rows * cols;
    int** out = (int**)malloc(sizeof(int*) * n);
    if (!out) return NULL;
    for (int i = 0; i < n; ++i) {
        int* r = (int*)malloc(sizeof(int) * features);
        if (!r) {
            for (int j = 0; j < i; ++j) free(out[j]);
            free(out);
            return NULL;
        }
        uint8_t* src = images + (size_t)i * features;
        for (int k = 0; k < features; ++k) r[k] = (src[k] > threshold) ? 1 : 0;
        out[i] = r;
    }
    return out;
}

/* Compute accuracy by predicting each sample */
static double compute_accuracy(tsetlin_t* ts, int** X_samples, uint8_t* y, int n_samples) {
    int correct = 0;
    for (int i = 0; i < n_samples; ++i) {
        int pred = tsetlin_predict(ts, X_samples[i], NULL);
        if (pred == (int)y[i]) ++correct;
    }
    return (double)correct / (double)n_samples;
}

int main(int argc, char** argv) {
    int epochs = 5;
    int N_CLAUSE = 200;
    int N_STATE = 100;
    int T = 100;
    double s = 5.0;
    bool flag_feedback = false;
    bool flag_compression = false;
    int threshold = -1;

    /* parse minimal arguments */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n_clause") == 0 && i + 1 < argc) N_CLAUSE = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n_state") == 0 && i + 1 < argc) N_STATE = atoi(argv[++i]);
        else if (strcmp(argv[i], "--T") == 0 && i + 1 < argc) T = atoi(argv[++i]);
        else if (strcmp(argv[i], "--s") == 0 && i + 1 < argc) s = atof(argv[++i]);
        else if (strcmp(argv[i], "--feedback") == 0) flag_feedback = true;
        else if (strcmp(argv[i], "--compression") == 0) flag_compression = true;
        else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) threshold = atoi(argv[++i]);
    }

    /* deterministic RNG same as Python seed(0) */
    srand(0);

    log_info("Number of clauses: %d, Number of states: %d", N_CLAUSE, N_STATE);
    log_info("Threshold T: %d, Specificity s: %.2f", T, s);

    /* Load MNIST (expects files in current directory) */
    int train_count, rows, cols;
    uint8_t* train_images = load_idx_images("mnist/train-images-idx3-ubyte", &train_count, &rows, &cols);
    if (!train_images) { log_error("Failed to load train images"); return 1; }
    int train_labels_count;
    uint8_t* train_labels = load_idx_labels("mnist/train-labels-idx1-ubyte", &train_labels_count);
    if (!train_labels || train_labels_count != train_count) { log_error("Failed to load train labels"); return 1; }

    int test_count;
    uint8_t* test_images = load_idx_images("mnist/t10k-images-idx3-ubyte", &test_count, &rows, &cols);
    if (!test_images) { log_error("Failed to load test images"); return 1; }
    int test_labels_count;
    uint8_t* test_labels = load_idx_labels("mnist/t10k-labels-idx1-ubyte", &test_labels_count);
    if (!test_labels || test_labels_count != test_count) { log_error("Failed to load test labels"); return 1; }

    log_info("Train images: %d, Test images: %d, Image shape: %dx%d", train_count, test_count, rows, cols);

    /* Binarize with threshold 75 */
    int** X_train = binarize_images(train_images, train_count, rows, cols, 75);
    int** X_test = binarize_images(test_images, test_count, rows, cols, 75);
    if (!X_train || !X_test) { log_error("Failed to binarize images"); return 1; }

    int n_features = rows * cols;
    tsetlin_t* ts = tsetlin_new(n_features, 10, N_CLAUSE, N_STATE);
    if (!ts) { log_error("Failed to allocate Tsetlin"); return 1; }

    double accuracy = compute_accuracy(ts, X_train, train_labels, train_count);
    log_info("Initial train accuracy: %.2f%%", accuracy * 100.0);

    /* feedback accumulators per epoch if requested */
    for (int epoch = 0; epoch < epochs; ++epoch) {
        log_info("[Epoch %d/%d] Train Accuracy: %.2f%%", epoch + 1, epochs, accuracy * 100.0);
        long target_type_1_count = 0;
        long target_type_2_count = 0;
        long non_target_type_1_count = 0;
        long non_target_type_2_count = 0;

        tqdm_t bar;
        tqdm_init(&bar, (size_t)train_count, "Training", 50);

        for (int i = 0; i < train_count; ++i) {
            /* call training step; header signature may return feedback pointer or accept out param.
               We ignore returned feedback pointer here to avoid depending on a particular
               implementation. If your tsetlin_step implementation fills an out parameter,
               adjust this call accordingly. */
            (void)tsetlin_step(ts, X_train[i], (int)train_labels[i], T, s, NULL, threshold);
            /* If your implementation returns a pointer with feedback counts, you can accumulate them here. */
            if ((i & 0x3) == 0) { /* update occasionally for performance */
                tqdm_update(&bar, (size_t)(i + 1));
            }
        }

        accuracy = compute_accuracy(ts, X_train, train_labels, train_count);

        if (flag_feedback) {
            log_info("Epoch feedback (not collected in this C port): Target Type I: %ld, Type II: %ld, NonTarget Type I: %ld, Type II: %ld",
                target_type_1_count, target_type_2_count, non_target_type_1_count, non_target_type_2_count);
        }

        if (flag_compression) {
            /* The Python script asks user whether to compress and exit — here we skip interactive prompt for non-interactive runs. */
            log_info("Compression enabled but interactive prompt is skipped in C port.");
        }
    }

    /* Final evaluation */
    double test_acc = compute_accuracy(ts, X_test, test_labels, test_count);
    log_info("Test Accuracy: %.2f%%", test_acc * 100.0);

    log_info("Model save/load not implemented in this C port.");

    /* Cleanup */
    tsetlin_free(ts);
    for (int i = 0; i < train_count; ++i) free(X_train[i]);
    for (int i = 0; i < test_count; ++i) free(X_test[i]);
    free(X_train);
    free(X_test);
    free(train_images);
    free(test_images);
    free(train_labels);
    free(test_labels);

    return 0;
}
