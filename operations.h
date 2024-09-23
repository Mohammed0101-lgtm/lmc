#ifndef _OPERATIONS_H_
#define _OPERATIONS_H_

#include "value.h"

float          float_max(int _count, ...);

void           __relu(struct Value*** __restrict data, int data_size, int column_size);

void           __tanh(struct Value*** __restrict data, int data_size, int column_size);

struct Value*  multinomial(struct Value* data, int data_size, int num_samples);

float          max_element(struct Value* arr, int array_size);

void           softmax(struct Value* data, int data_size);

float          mse_loss(struct Value* logits, int logits_size, struct Value* targets);

struct Value** matmul(
    struct Value** mat_1, struct Value** mat_2, int mat1_size, int mat2_size, int mat1_colsize,
    int mat2_colsize, int* return_size, int* return_colSize);


#endif // operations_h