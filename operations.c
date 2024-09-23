#include "operations.h"
#include "value.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


float float_max(int _count, ...) {
    va_list arg_ptr;
    va_start(arg_ptr, _count);

    float maximum = __FLT_MIN__;
    float next;

    for (int i = 0; i < _count; i++) {
        next = va_arg(arg_ptr, double);
        if (next >= maximum)
            maximum = next;
    }

    return maximum;
}

void __relu(struct Value*** __restrict data, int data_size, int column_size) {
    if (!data || data_size <= 0 || column_size <= 0)
        return;

    for (int i = 0; i < data_size; i++)
        for (int j = 0; j < column_size; j++)
            data[i][j]->data = float_max(2, 0.0f, data[i][j]->data);
}

void __tanh(struct Value*** __restrict data, int data_size, int column_size) {
    if (!data || data_size <= 0 || column_size <= 0)
        return;

    for (int i = 0; i < data_size; i++)
        for (int j = 0; j < column_size; j++)
            data[i][j]->data = tanh(data[i][j]->data);
}

struct Value* multinomial(struct Value* data, int data_size, int num_samples) {
    if (!data || data_size <= 0 || num_samples <= 0)
        return NULL;

    float* cumulative_probs = (float*)malloc(data_size * sizeof(float));

    if (!cumulative_probs) {
        fprintf(stderr, "Malloc Failed!\n");
        return NULL;
    }

    cumulative_probs[0] = data[0].data;

    for (int i = 1; i < data_size; i++)
        cumulative_probs[i] = cumulative_probs[i - 1] + data[i].data;

    float total_sum = cumulative_probs[data_size - 1];

    for (int i = 0; i < data_size; i++)
        cumulative_probs[i] /= total_sum;

    struct Value* result = (struct Value*)malloc(num_samples * sizeof(struct Value));

    if (!result) {
        fprintf(stderr, "Malloc Failed!\n");
        free(cumulative_probs);
        return NULL;
    }

    srand((unsigned int)time(NULL));

    for (int sample = 0; sample < num_samples; sample++) {
        float rand_prob = (float)(rand() / (float)RAND_MAX);

        int   index     = 0;
        while (index < data_size - 1 && rand_prob > cumulative_probs[index])
            index++;

        result[sample] = data[index];
    }

    free(cumulative_probs);

    return result;
}

float max_element(struct Value* arr, int array_size) {
    if (!arr || array_size <= 0)
        return __FLT_MIN__;

    float max = __FLT_MIN__;

    for (int i = 0; i < array_size; i++)
        if (arr[i].data > max)
            max = arr[i].data;

    return max;
}

void softmax(struct Value* data, int data_size) {
    float max_val = max_element(data, data_size);
    if (max_val == __FLT_MIN__)
        return;

    float sumExp = 0.0f;

    for (int i = 0; i < data_size; i++)
        sumExp += exp(data[i].data - max_val);

    for (int i = 0; i < data_size; i++)
        data[i].data = exp(data[i].data - max_val) / sumExp;
}

struct Value** matmul(
    struct Value** mat_1, struct Value** mat_2, int mat1_size, int mat2_size, int mat1_colsize,
    int mat2_colsize, int* return_size, int* return_colSize) {

    if (mat1_colsize != mat2_size) {
        fprintf(stderr, "Incompatible matrix dimensions!\n");
        fprintf(
            stderr, "first mat : [%i, %i], second mat : [%i, %i]\n", mat1_size, mat1_colsize,
            mat2_size, mat2_colsize);
        return NULL;
    }

    struct Value** result = (struct Value**)malloc(mat1_size * sizeof(struct Value*));

    if (!result) {
        fprintf(stderr, "Malloc Failed!\n");
        return NULL;
    }

    *return_size    = mat1_size;
    *return_colSize = mat2_colsize;

    for (int i = 0; i < mat1_size; i++) {
        result[i] = (struct Value*)malloc(mat2_colsize * sizeof(struct Value));
        if (!result[i]) {
            fprintf(stderr, "Malloc Failed!\n");
            for (int j = 0; j < i; j++)
                free(result[j]);

            free(result);
            return NULL;
        }
    }

    for (int i = 0; i < mat1_size; i++) {
        for (int j = 0; j < mat2_colsize; j++) {
            result[i][j].data = 0.0f;
            result[i][j].grad = 0.0f;

            for (int k = 0; k < mat1_colsize; k++) {
                struct Value* product = mul(&mat_1[i][k], &mat_2[k][j]);
                struct Value* sum     = add(&result[i][j], product);

                result[i][j].data     = sum->data;
                result[i][j].grad     = sum->grad;
                result[i][j]._op      = sum->_op;
                result[i][j].index    = sum->index;
                result[i][j]._prev[0] = product->_prev[0];
                result[i][j]._prev[1] = product->_prev[1];

                free(product);
                free(sum);
            }
        }
    }

    return result;
}

float mse_loss(struct Value* logits, int logits_size, struct Value* targets) {
    float sumSqDif = 0.0f;

    for (int i = 0; i < logits_size; i++) {
        float diff = logits[i].data - targets[i].data;
        sumSqDif += diff * diff;
    }

    return sumSqDif / logits_size;
}