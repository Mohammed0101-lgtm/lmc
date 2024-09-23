#ifndef _UTIL_H_
#define _UTIL_H_

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct entry {
    char*         key;
    int           value;
    struct entry* next;
};

struct map {
    struct entry** buckets;
    int            capacity;
};

struct set {
    void** data;
    size_t size;
    size_t capacity;
    size_t element_size;
    int (*cmp)(const void*, const void*);
};

struct array {
    void** data;
    size_t size;
    size_t capacity;
    size_t element_size;
};

unsigned long long hash(const char* str);

struct map*        create_map(int capacity);

int                get(struct map* m, char* key);

void               put(struct map* m, const char* key, int value);

void               free_map(struct map* m);

struct set*
     create_set(size_t element_size, size_t initial_capacity, int (*cmp)(const void*, const void*));

bool contains(struct set* set, void* value);

void add_to_set(struct set* set, void* value);

void free_set(struct set* set);

struct array* create_array(size_t element_size, size_t initial_capacity);

#endif // _UTIL_H_