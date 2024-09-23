#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <stdbool.h>

struct Embedding {
    struct Value** __table;
    int            _size;
    int            _column_size;
};

bool _embd_init_(struct Embedding** table, const int dims[2]);

struct Value**
table_forward(struct Embedding** table, struct Value** input, int input_size, int input_colSize);

void _table_backward(
    struct Embedding* table, struct Value** input, int input_size, int input_colsize,
    struct Value** grad_output, int grad_output_size);

void free_table(struct Embedding* t);

#endif // EMBEDDING_H