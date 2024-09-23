#include "embedding.h"
#include "value.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


bool _embd_init_(struct Embedding** table, const int dims[2]) {
    printf("Initializing the embedding table with dimensions [%i, %i] ...\n", dims[0], dims[1]);

    if (dims[0] <= 0 || dims[1] <= 0) {
        fprintf(stderr, "Cannot init table with fewer than two dims\n");
        exit(EXIT_FAILURE);
    }

    *table = (struct Embedding*)malloc(sizeof(struct Embedding));

    if (!table) {
        fprintf(stderr, "Malloc Failed!\n");
        exit(EXIT_FAILURE);
    }

    (*table)->_size        = dims[0];
    (*table)->_column_size = dims[1];

    (*table)->__table      = (struct Value**)malloc(dims[0] * sizeof(struct Value*));

    if (!(*table)->__table) {
        fprintf(stderr, "Malloc Failed!\n");
        free_table(*table);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < dims[0]; i++) {
        (*table)->__table[i] = (struct Value*)malloc(dims[1] * sizeof(struct Value));

        if (!(*table)->__table[i]) {
            fprintf(stderr, "Malloc Failed!\n");
            for (int j = 0; j < i; j++)
                free((*table)->__table[j]);

            free((*table)->__table);
            free(*table);
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < dims[0]; i++) {
        srand(time(NULL));
        for (int j = 0; j < dims[1]; j++) {
            struct Value* temp = _init_value((float)(rand() / RAND_MAX));
            if (!temp)
                break;

            (*table)->__table[i][j] = *temp;
            free(temp);
        }
    }

    return true;
}

struct Value**
table_forward(struct Embedding** table, struct Value** input, int input_size, int input_colSize) {
    assert(*table != NULL && *input != NULL);
    assert((*table)->__table != NULL && (*table)->_size > 0 && (*table)->_column_size > 0);
    assert(input_size > 0 && input_colSize > 0);

    int num_indices = input_size * input_colSize;
    int indices[num_indices];
    int k = 0;

    for (int i = 0; i < (*table)->_size; i++) 
        for (int j = 0; j < (*table)->_column_size; j++) 
            indices[k++] = (*table)->__table[i][j].data;

    int            dims[2] = {num_indices, (*table)->_column_size};

    struct Value** output  = (struct Value**)malloc(dims[0] * sizeof(struct Value*));
    if (!output) {
        fprintf(stderr, "Malloc Failed!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < dims[0]; i++) {
        output[i] = (struct Value*)malloc(dims[1] * sizeof(struct Value));
        if (!output[i]) {
            fprintf(stderr, "Malloc Failed!\n");
            for (int j = 0; j < i; j++)
                free(output[j]);

            free(output);
            return NULL;
        }
    }

    for (int i = 0; i < num_indices; i++) {
        int            idx = indices[i];
        struct Value** row = &(*table)->__table[idx];

        if (!*row)
            break;

        output[i] = *row;
    }

    return output;
}

void _table_backward(
    struct Embedding* table, struct Value** input, int input_size, int input_colsize,
    struct Value** grad_output, int grad_output_size) {

    assert(table->__table && input && grad_output);

    int num_indices = input_size * input_colsize;
    int indices[num_indices];
    int k = 0;

    for (int i = 0; i < input_size; i++)
        for (int j = 0; j < input_colsize; j++)
            indices[k++] = input[i][j].data;

    for (int i = 0; i < num_indices; i++) {
        for (int j = 0; j < grad_output_size; j++) {
            int table_index = indices[i] * table->_column_size + j;
            table->__table[i][table_index].grad += grad_output[i][j].grad;
        }
    }
}

void free_table(struct Embedding* t) {
    if (!t)
        return;

    if (t->__table) {
        for (int i = 0; i < t->_size; i++)
            free(t->__table[i]);

        free(t->__table);
    }

    free(t);
}
