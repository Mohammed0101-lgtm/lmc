#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "value.h"

struct Value*** get_batch(
    const int* __restrict const data, const int data_size, const int context_length,
    const int batch_size);

char* read_file(FILE* _file);

int*  encode(const char* str, int* result_size, int *voca_size);

#endif // DATA_LOADER_H