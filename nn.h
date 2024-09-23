#ifndef NN_H
#define NN_H

#include <stdbool.h>

struct NeuralNetwork {
    struct Value**    parameters;
    struct Embedding* lookup_table;
    struct Linear*    hidden_layer;
    struct Linear*    output_layer;
};

bool          _nn_init_(struct NeuralNetwork** nn, int input_size, int vocab_size, int layer_size);

struct Value* feed_forward(
    struct NeuralNetwork* nn, struct Value** input_data, int data_size, int input_colSize,
    int layer_size, int vocab_size);

void           free_nn(struct NeuralNetwork* nn);

struct Value** array_to_matrix(const struct Value* arr, int arr_size, int rows, int cols);

void           back_propagate(
              struct NeuralNetwork* nn, struct Value* output, struct Value* targets, int output_size,
              int targets_size, float learning_rate, int batch_size, int vocab_size, int context_length);

int* generate(
    struct NeuralNetwork* nn, struct Value** idx, int idx_size, int max_tokens, int batch_size,
    int context_length, int layer_size, int vocab_size);

#endif // NN_H