#include "nn.h"
#include "embedding.h"
#include "linear.h"
#include "operations.h"
#include "util.h"
#include "value.h"

#include <assert.h>

bool _nn_init_(struct NeuralNetwork** nn, int input_size, int vocab_size, int layer_size) {
    printf("Initializing neural net ...\n");

    *nn = (struct NeuralNetwork*)malloc(sizeof(struct NeuralNetwork));
    if (!*nn) {
        fprintf(stderr, "Malloc Failed!\n");
        exit(EXIT_FAILURE);
    }

    (*nn)->parameters = (struct Value**)malloc(sizeof(struct Value*) * (input_size * 2 + 1));
    if (!(*nn)->parameters) {
        fprintf(stderr, "Malloc Failed!\n");
        exit(EXIT_FAILURE);
    }

    (*nn)->lookup_table = (struct Embedding*)malloc(sizeof(struct Embedding));
    if (!(*nn)->lookup_table) {
        fprintf(stderr, "Malloc Failed for Lookup Table!\n");
        free_nn(*nn);
        exit(EXIT_FAILURE);
    }

    int dims[2] = {vocab_size, 2};
    if (!_embd_init_(&(*nn)->lookup_table, dims)) {
        free_nn(*nn);
        exit(EXIT_FAILURE);
    }

    (*nn)->hidden_layer = (struct Linear*)malloc(sizeof(struct Linear));
    if (!(*nn)->hidden_layer) {
        fprintf(stderr, "Malloc Failed for Hidden Layer!\n");
        free_nn(*nn);
        exit(EXIT_FAILURE);
    }

    if (!_linear_init_(&(*nn)->hidden_layer, layer_size, vocab_size)) {
        fprintf(stderr, "Linear Hidden Layer Init Failed!\n");
        free_nn(*nn);
        exit(EXIT_FAILURE);
    }

    (*nn)->output_layer = (struct Linear*)malloc(sizeof(struct Linear));
    if (!(*nn)->output_layer) {
        fprintf(stderr, "Malloc Failed for Output Layer!\n");
        free_nn(*nn);
        exit(EXIT_FAILURE);
    }

    if (!_linear_init_(&(*nn)->output_layer, vocab_size, layer_size)) {
        fprintf(stderr, "Linear Output Layer Init Failed!\n");
        free_nn(*nn);
        exit(EXIT_FAILURE);
    }

    return true;
}

struct Value* feed_forward(
    struct NeuralNetwork* nn, struct Value** input_data, int data_size, int input_colSize,
    int layer_size, int vocab_size) {

    assert(nn != NULL);
    assert(input_data != NULL);
    assert(input_colSize > 0 && data_size > 0);
    assert(layer_size != 0 && vocab_size > 0);

    struct Value** logits = table_forward(&nn->lookup_table, input_data, data_size, input_colSize);

    if (!logits) {
        fprintf(stderr, "'table_forward' failed!\n");
        exit(EXIT_FAILURE);
    }

    size_t hidden_dims[2] = { (size_t)layer_size, 1 };
    logits                = _linear_forward(nn->hidden_layer, logits, hidden_dims, vocab_size);

    if (!logits) {
        fprintf(stderr, "'_linear_forward_' failed for Hidden Layer!\n");
        exit(EXIT_FAILURE);
    }

    __tanh(&logits, hidden_dims[0], hidden_dims[1]);

    logits = _linear_forward(nn->output_layer, logits, hidden_dims, vocab_size);

    if (!logits) {
        fprintf(stderr, "'_linear_forward' failed for Output Layer!\n");
        exit(EXIT_FAILURE);
    }

    struct Value* ans =
        (struct Value*)malloc((hidden_dims[0] * hidden_dims[1]) * sizeof(struct Value));

    if (!ans) {
        fprintf(stderr, "Malloc Failed for Output!\n");
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0, k = 0; i < hidden_dims[0]; i++)
        for (size_t j = 0; j < hidden_dims[1]; j++)
            ans[k++] = logits[i][j];

    for (size_t i = 0; i < hidden_dims[0]; i++)
        free(logits[i]);

    free(logits);
    return ans;
}

void free_nn(struct NeuralNetwork* nn) {
    if (nn->hidden_layer)
        free_linear(nn->hidden_layer);

    if (nn->output_layer)
        free_linear(nn->output_layer);

    if (nn->lookup_table)
        free_table(nn->lookup_table);

    if (nn->parameters)
        free(nn->parameters);

    free(nn);
}

struct Value** array_to_matrix(const struct Value* arr, int arr_size, int rows, int cols) {
    if (rows * cols != arr_size) {
        fprintf(
            stderr, "Invalid arguments : The number of elements in the vector does not match the "
                    "specified matrix dimensions.");
        exit(EXIT_FAILURE);
    }

    struct Value** mat = (struct Value**)malloc(rows * sizeof(struct Value*));
    if (!mat)
        return NULL;

    for (int i = 0; i < rows; i++) {
        mat[i] = (struct Value*)malloc(cols * sizeof(struct Value));
        if (!mat[i]) {
            for (int j = 0; j < i; j++)
                free(mat[j]);

            free(mat);
            fprintf(stderr, "Malloc Failed!\n");
            exit(EXIT_FAILURE);
        }
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i][j] = arr[i * cols + j];
        }
    }

    return mat;
}

void back_propagate(
    struct NeuralNetwork* nn, struct Value* output, struct Value* targets, int output_size,
    int targets_size, float learning_rate, int batch_size, int vocab_size, int context_length) {

    assert(output_size == targets_size);
    float         loss      = mse_loss(output, output_size, targets);
    struct Value* loss_grad = (struct Value*)malloc(sizeof(struct Value) * output_size);
    if (!loss_grad) {
        fprintf(stderr, "Malloc Failed!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < output_size; i++)
        loss_grad[i] = *sub(&output[i], &targets[i]);

    for (int i = 0; i < output_size; i++)
        loss_grad[i].data *= (1.0f / output_size);

    struct Value* d_logits = (struct Value*)malloc(sizeof(struct Value) * output_size);
    if (!d_logits) {
        fprintf(stderr, "Malloc Failed!\n");
        free(loss_grad);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < output_size; i++)
        d_logits[i] = output[i];

    softmax(d_logits, output_size);

    for (size_t i = 0; i < nn->output_layer->_size; i++) {
        for (int j = 0; j < vocab_size; j++) {
            nn->output_layer->weights[i][j].data -=
                learning_rate * nn->output_layer->weights[i][j].grad;
            nn->output_layer->weights[i][j].data = 0.0f;
        }
    }

    size_t         dims[2] = {(size_t)batch_size, (size_t)context_length};
    struct Value** mat     = array_to_matrix(output, output_size, batch_size, context_length);
    if (!mat) {
        fprintf(stderr, "Failed to create the matrix for an array!\n");
        free(d_logits);
        free(loss_grad);
        exit(EXIT_FAILURE);
    }


    struct Value** hidden_grad = _linear_forward(nn->hidden_layer, mat, dims, vocab_size);

    for (size_t i = 0; i < nn->hidden_layer->_size; i++) {
        for (int j = 0; j < vocab_size; j++) {
            nn->hidden_layer->weights[i][j].data -=
                (learning_rate * nn->hidden_layer->weights[i][j].grad);
            nn->hidden_layer->weights[i][j].grad = 0.0f;
        }
    }

    struct Value** lookup_grad =
        table_forward(&nn->lookup_table, hidden_grad, batch_size, context_length);

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < context_length; j++) {
            int index = hidden_grad[i][j].index;
            nn->lookup_table->__table[index][j].data -= (learning_rate * lookup_grad[i][j].grad);
        }
    }
}

int* generate(
    struct NeuralNetwork* nn, struct Value** idx, int idx_size, int max_tokens, int batch_size,
    int context_length, int layer_size, int vocab_size) {
    int* cat = (int*)malloc(max_tokens * idx_size * sizeof(int));
    int  k   = 0;

    for (int i = 0; i < max_tokens; i++) {
        struct Value** mat = array_to_matrix(*idx, idx_size, batch_size, context_length);
        struct Value*  logits =
            feed_forward(nn, mat, batch_size, context_length, layer_size, vocab_size);
        softmax(logits, batch_size);
        struct Value* next = multinomial(logits, idx_size, idx_size);

        for (int j = 0; j < idx_size; j++)
            cat[k++] = (int)(idx[j]->data);

        for (int j = 0; j < idx_size; j++)
            cat[k++] = (int)(next[j].data);
    }

    return cat;
}