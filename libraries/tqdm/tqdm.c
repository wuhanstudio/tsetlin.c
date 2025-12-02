#include "tqdm.h"

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double _now_seconds(void) {
#if defined(_WIN32)
    return (double)clock() / (double)CLOCKS_PER_SEC;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return ts.tv_sec + ts.tv_nsec * 1e-9;
    }
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

void tqdm_init(tqdm_t* t, size_t total, const char* desc, size_t width) {
    if (!t) return;
    t->total = total;
    t->width = (width > 0) ? width : 40;
    t->start_time = _now_seconds();
    t->last_time = t->start_time;
    t->last_count = 0;
    t->enabled = 1;
    if (desc) {
        strncpy(t->desc, desc, sizeof(t->desc) - 1);
        t->desc[sizeof(t->desc) - 1] = '\0';
    }
    else {
        t->desc[0] = '\0';
    }
    /* print initial 0% line */
    fprintf(stderr, "%s ", t->desc);
    fflush(stderr);
}

void tqdm_update(tqdm_t* t, size_t count) {
    if (!t || !t->enabled) return;
    double now = _now_seconds();
    /* rate-limit updates to avoid excessive output */
    if (count == t->last_count && (now - t->last_time) < 0.05) return;
    if ((now - t->last_time) < 0.05 && count < t->total) return;

    t->last_time = now;
    t->last_count = count;

    if (t->total == 0) {
        fprintf(stderr, "\r%s %zu", t->desc, count);
        fflush(stderr);
        return;
    }

    double frac = (double)count / (double)t->total;
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;

    size_t filled = (size_t)floor(frac * (double)t->width);
    size_t empty = (t->width > filled) ? (t->width - filled) : 0;

    /* ETA calculation */
    double elapsed = now - t->start_time;
    double rate = (count > 0 && elapsed > 0.0) ? ((double)count / elapsed) : 0.0;
    double eta = (rate > 0.0) ? ((t->total - count) / rate) : 0.0;

    /* Build bar in-place and print */
    fprintf(stderr, "\r%s [", t->desc);
    for (size_t i = 0; i < filled; ++i) fputc('=', stderr);
    if (filled < t->width) fputc('>', stderr);
    for (size_t i = 0; i + filled + 1 <= empty; ++i) fputc(' ', stderr);
    fprintf(stderr, "] %3.0f%% (%zu/%zu) ETA: %4.0fs", frac * 100.0, count, t->total, eta);
    fflush(stderr);

    if (count >= t->total) {
        /* finish immediately */
        fputc('\n', stderr);
        fflush(stderr);
        t->enabled = 0;
    }
}

void tqdm_finish(tqdm_t* t) {
    if (!t) return;
    if (t->enabled) {
        /* force final print */
        tqdm_update(t, t->total);
    }
}