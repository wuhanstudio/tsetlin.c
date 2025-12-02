#ifndef TQDM_H
#define TQDM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef struct {
        size_t total;
        size_t width;
        double start_time;
        double last_time;
        size_t last_count;
        char desc[64];
        int enabled;
    } tqdm_t;

    void tqdm_init(tqdm_t* t, size_t total, const char* desc, size_t width);
    void tqdm_update(tqdm_t* t, size_t count);
    void tqdm_finish(tqdm_t* t);

#ifdef __cplusplus
}
#endif

#endif /* TQDM_H */