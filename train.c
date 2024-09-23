#include "train.h"
#include "dataloader.h"
#include "embedding.h"
#include "linear.h"
#include "operations.h"
#include "util.h"
#include "value.h"

float estimate_loss(
    struct NeuralNetwork* nn, int* data, int data_size, int batch_size, int eval_iters,
    int context_length, int vocab_size, int layer_size) {
    float  output;

    float* losses = (float*)malloc(eval_iters * sizeof(float));
    if (!losses) {
        fprintf(stderr, "Malloc Failed!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < eval_iters; i++) {
        struct Value*** batches = get_batch(data, data_size, context_length, batch_size);
        struct Value**  inputs  = batches[0];
        struct Value**  targets = batches[0];

        struct Value*   logits =
            feed_forward(nn, inputs, batch_size, context_length, layer_size, vocab_size);
        float loss = mse_loss(logits, vocab_size, *targets);
        losses[i]  = loss;
    }

    float sum = 0;

    for (int i = 0; i < eval_iters; i++)
        sum += losses[i];

    output = sum / eval_iters;

    return output;
}

void train_model(
    struct NeuralNetwork* nn, int* data, int data_size, int batch_size, int max_iterations,
    int eval_interval, int context_length, int vocab_size, int eval_iters, int layer_size) {
    
    printf("Training model..");

    int  train_size = data_size * 0.9;
    int* train_data = (int*)malloc(train_size * sizeof(int));
    
    if (!train_data) {
        fprintf(stderr, "Malloc Failed!\n");
        exit(EXIT_FAILURE);
    }

    int  val_size = data_size - train_size;
    int* val_data = (int*)malloc(val_size * sizeof(int));
    
    if (!val_data) {
        fprintf(stderr, "Malloc Failed!\n");
        free(train_data);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < train_size; i++)
        train_data[i] = data[i];

    for (int i = 0, k = train_size + 1; i < val_size; i++)
        val_data[i++] = data[k++];

    for (int iter = 0; iter < max_iterations; iter++) {
        if (iter % eval_interval == 0) {
            float train_loss = estimate_loss(
                nn, train_data, train_size, batch_size, eval_iters, context_length, vocab_size,
                layer_size);
            float val_loss = estimate_loss(
                nn, val_data, val_size, batch_size, eval_iters, context_length, vocab_size,
                layer_size);
            printf("step %i: train_loss %f, val loss %f\n", iter, train_loss, val_loss);
        }

        struct Value*** batch   = get_batch(train_data, train_size, context_length, batch_size);
        struct Value**  inputs  = batch[0];
        struct Value**  targets = batch[0];

        struct Value*   logits =
            feed_forward(nn, inputs, batch_size, context_length, layer_size, vocab_size);
        back_propagate(
            nn, logits, *targets, vocab_size, batch_size, 0.01f, batch_size, vocab_size,
            context_length);
    }
}
