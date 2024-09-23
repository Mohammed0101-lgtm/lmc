#include "linear.h"
#include "operations.h"
#include "value.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


bool _linear_init_(struct Linear** layer_ptr, size_t __size, int vocab_size) {
    printf("Initializing linear layer (dimensions : [%zu, %i]) ...\n", __size, vocab_size);

    assert(__size > 0 && vocab_size > 0);

    if (*layer_ptr)
        free_linear(*layer_ptr);

    *layer_ptr = (struct Linear*)malloc(sizeof(struct Linear));
    if (!(*layer_ptr)) {
        fprintf(stderr, "Malloc Failed!\n");
        exit(EXIT_FAILURE);
    }

    (*layer_ptr)->_size   = __size;

    (*layer_ptr)->weights = (struct Value**)malloc(sizeof(struct Value*) * __size);
    if (!(*layer_ptr)->weights) {
        fprintf(stderr, "Malloc Failed!\n");
        free_linear(*layer_ptr);
        exit(EXIT_FAILURE);
    }

    srand(time(NULL));

    for (size_t i = 0; i < __size; i++) {
        (*layer_ptr)->weights[i] = (struct Value*)malloc(sizeof(struct Value) * vocab_size);
        if (!(*layer_ptr)->weights[i]) {
            fprintf(stderr, "Malloc Failed!\n");
            free_linear(*layer_ptr);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < vocab_size; j++) {
            float         rand_val      = ((float)rand() / (float)RAND_MAX * 2) - 1;
            struct Value* temp          = _init_value(rand_val);
            (*layer_ptr)->weights[i][j] = *temp;
            free(temp);
        }
    }

    (*layer_ptr)->bias = _init_value(0.0f);

    return true;
}

struct Value**
_linear_forward(struct Linear* layer, struct Value** data, size_t dims[2], int vocab_size) {

    assert(dims[0] == layer->_size);

    int            output_size, output_colSize;
    struct Value** output = matmul(
        data, layer->weights, dims[0], layer->_size, dims[1], vocab_size, &output_size,
        &output_colSize);

    if (!output) {
        fprintf(stderr, "Matmul Failed!\n");
        return NULL;
    }

    __relu(&output, output_size, output_colSize);

    if (layer->bias) {
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < output_colSize; j++) {
                struct Value* sum = add(&output[i][j], layer->bias);

                if (!sum) {
                    for (int k = 0; k < output_size; k++)
                        free(output[k]);

                    free(output);
                    return NULL;
                }

                output[i][j] = *sum;
                free(sum);
            }
        }
    }

    return output;
}

void free_linear(struct Linear* layer) {
    if (!layer)
        return;

    if (layer->weights) {
        for (size_t i = 0; i < layer->_size; i++)
            if (layer->weights[i])
                free(layer->weights[i]);

        free(layer->weights);
    }

    if (layer->bias)
        free(layer->bias);

    free(layer);
}
