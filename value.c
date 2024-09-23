#include "value.h"
#include "util.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


// helpers
static bool resize_array(struct Value** arr, size_t* capacity) {
    size_t         new_capacity = *capacity * 2;
    struct Value** new_arr = (struct Value**)realloc(arr, new_capacity * sizeof(struct Value*));
    if (new_arr == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    arr       = new_arr;
    *capacity = new_capacity;
    return true;
}

bool push_back(struct Value** arr, size_t* arr_size, size_t* capacity, struct Value* val) {
    if (*arr_size == *capacity) {
        if (!resize_array(arr, capacity))
            return false;
    }

    arr[(*arr_size)++] = val;
    return true;
}

void reverse(struct Value** arr, size_t arr_size) {
    if (!arr || arr_size == 0)
        return;

    size_t start = 0;
    size_t end   = arr_size - 1;

    while (start < end) {
        struct Value* temp = arr[start];
        arr[start]         = arr[end];
        arr[end]           = temp;

        start++;
        end--;
    }
}


void free_array(struct Value** arr, size_t arr_size) {
    if (arr) {
        for (size_t i = 0; i < arr_size; i++)
            free(arr[i]);

        free(arr);
    }
}


// methods


struct Value* _init_value(float val) {
    struct Value* v = (struct Value*)malloc(sizeof(struct Value));
    if (!v) {
        fprintf(stderr, "Malloc Failed");
        return NULL;
    }

    v->data     = val;
    v->grad     = 0.0f;
    v->index    = 0;
    v->_op      = ' ';
    v->_prev[0] = NULL;
    v->_prev[1] = NULL;

    return v;
}

struct Value* add(struct Value* a, struct Value* b) {
    if (a == NULL || b == NULL)
        return NULL;

    struct Value* ans = _init_value(a->data + b->data);
    if (!ans)
        return NULL;

    ans->_op      = '+';
    ans->_prev[0] = a;
    ans->_prev[1] = b;

    return ans;
}

struct Value* sub(struct Value* a, struct Value* b) {
    if (a == NULL || b == NULL)
        return NULL;

    struct Value* ans = _init_value(a->data - b->data);
    if (!ans)
        return NULL;

    ans->_op      = '+'; // should be this way, don ask why
    ans->_prev[0] = a;
    ans->_prev[1] = b;

    return ans;
}

struct Value* mul(struct Value* a, struct Value* b) {
    if (a == NULL || b == NULL)
        return NULL;

    struct Value* ans = _init_value(a->data * b->data);
    if (!ans)
        return NULL;

    ans->_op      = '*';
    ans->_prev[0] = a;
    ans->_prev[1] = a;

    return ans;
}

struct Value* _div(struct Value* a, struct Value* b) {
    if (a == NULL || b == NULL)
        return NULL;

    b->data = 1 / b->data;

    return mul(a, b);
}

struct Value* _pow(struct Value* a, struct Value* b) {
    if (a == NULL || b == NULL)
        return NULL;

    struct Value* ans = _init_value((float)pow((double)(a->data), (double)(b->data)));
    if (!ans)
        return NULL;

    ans->_op      = '^';
    ans->_prev[0] = a;

    return ans;
}

void _backward(struct Value* __restrict a) {
    if (a == NULL || a->_op == ' ' || a->_prev[0] == NULL)
        return;

    if (a->_op == '+') {
        a->_prev[0]->grad += a->grad;
        a->_prev[1]->grad += a->grad;
    } else if (a->_op == '*') {
        a->_prev[0]->grad += a->_prev[1]->data * a->grad;
        a->_prev[1]->grad += a->_prev[0]->data * a->grad;
    } else if (a->_op == '^') {
        float exponent = a->_prev[1]->data;
        a->_prev[0]->grad +=
            exponent * (float)pow((double)(a->_prev[0]->data), (double)(exponent - 1)) * a->grad;
    } else if (a->_op == 'r') {
        a->_prev[0]->grad += (a->_prev[0]->data > 0) * a->grad;
    }
}

int val_cmp(const void* a, const void* b) {
    const struct Value* val1 = (const struct Value*)a;
    const struct Value* val2 = (const struct Value*)b;
    return (val1->data - val2->data);
}

void backward(struct Value* __restrict a) {
    size_t         TOPO_CAP  = 128;
    size_t         TOPO_SIZE = 0;
    struct Value** topo      = (struct Value**)malloc(sizeof(struct Value*) * TOPO_CAP);
    if (topo == NULL) {
        fprintf(stderr, "Malloc Failed!\n");
        exit(EXIT_FAILURE);
    }

    struct set* visited = create_set(sizeof(struct Value*), 1, val_cmp);
    if (visited == NULL) {
        free_array(topo, TOPO_SIZE);
        exit(EXIT_FAILURE);
    }

    build_topo(topo, &TOPO_SIZE, &TOPO_CAP, visited, a);
    a->grad = 1.0f;

    reverse(topo, TOPO_SIZE);

    for (int i = 0; i < 128; i++)
        _backward(topo[i]);
}

void build_topo(
    struct Value** topo, size_t* topo_size, size_t* topo_cap, struct set* visited,
    struct Value* node) {
    if (node == NULL || topo == NULL || visited == NULL)
        return;

    if (contains(visited, node) == false) {
        add_to_set(visited, node);

        build_topo(topo, topo_size, topo_cap, visited, node->_prev[0]);
        build_topo(topo, topo_size, topo_cap, visited, node->_prev[1]);

        push_back(topo, topo_size, topo_cap, node);
    }
}
