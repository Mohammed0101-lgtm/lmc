#include "dataloader.h"
#include "nn.h"
#include "train.h"

#include <stdio.h>

int context_length = 8;
int batch_size     = 16;
int max_iterations = 10000;
int eval_interval  = 200;
int vocab_size     = 0;
int eval_iters     = 100;
int layer_size     = 100;

int main(int argc, char** argv) {
    printf("start ...\n");

    if (argc <= 0) {
        fprintf(stderr, "Too few arguments!\n");
        return -1;
    } else if (argc > 2) {
        fprintf(stderr, "Too many arguments!\n");
        return -1;
    }

    FILE* fp = fopen(argv[1], "r");
    if (!fp) {
        fprintf(stderr, "Failed to open file : %s\n", argv[1]);
        return -1;
    }

    char* input_text = read_file(fp);
    fclose(fp);

    if (!input_text) {
        fprintf(stderr, "read_file Failed!\n");
        return -1;
    }

    int  encoded_size;
    int* encoding = encode(input_text, &encoded_size, &vocab_size);
    if (!encoding) {
        fprintf(stderr, "Failed to encode text!\n");
        free(input_text);
        return -1;
    }

    free(input_text);

    struct NeuralNetwork* nn = NULL;
    if (!_nn_init_(&nn, context_length, vocab_size, layer_size)) {
        fprintf(stderr, "Failed to initialize the neural network!\n");
        free(input_text);
        free(encoding);
        return -1;
    }

    train_model(
        nn, encoding, encoded_size, batch_size, max_iterations, eval_interval, context_length,
        vocab_size, eval_iters, layer_size);

    free(encoding);
    free_nn(nn);

    return 0;
}