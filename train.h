#ifndef TRAIN_H
#define TRAIN_H

#include "nn.h"

float estimate_loss(
    struct NeuralNetwork* nn, int* data, int data_size, int batch_size, int eval_iters,
    int context_length, int vocab_size, int layer_size);
void train_model(
    struct NeuralNetwork* nn, int* data, int data_size, int batch_size, int max_iterations,
    int eval_interval, int context_length, int vocab_size, int eval_iters, int layer_size);

#endif // TRAIN_He