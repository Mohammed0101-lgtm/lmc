#ifndef VALUE_H
#define VALUE_H

#include "util.h"

struct Value {
    float         data;
    float         grad;
    struct Value* _prev[2];
    char          _op;
    int           index;
};

struct Value* _init_value(float val);

struct Value* add(struct Value* a, struct Value* b);
struct Value* sub(struct Value* a, struct Value* b);
struct Value* mul(struct Value* a, struct Value* b);

struct Value* _div(struct Value* a, struct Value* b);
struct Value* _pow(struct Value* a, struct Value* b);

void          backward(struct Value* __restrict a);
void          _backward(struct Value* __restrict a);
void          build_topo(
             struct Value** topo, size_t* topo_size, size_t* topo_cap, struct set* visited,
             struct Value* node);

#endif // VALUE_H