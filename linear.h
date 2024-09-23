#ifndef LINEAR_H
#define LINEAR_H

#include <stdlib.h>
#include <stdbool.h>


struct Linear {
    struct Value** weights;
    size_t         _size;
    struct Value*  bias;
};


bool _linear_init_(struct Linear** layer_ptr, size_t __size, int vocab_size);

struct Value**
     _linear_forward(struct Linear* layer, struct Value** data, size_t dims[2], int vocab_size);

void free_linear(struct Linear* layer);

#endif // LINEAR_H