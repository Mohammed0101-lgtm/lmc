#include "util.h"

unsigned long long hash(const char* str) {
    unsigned long long h = 5381;
    int                c;

    while ((c = *str++)) {
        h = ((h << 5) + h) + c;
    }

    return h;
}

struct map* create_map(int capacity) {
    struct map* m = (struct map*)malloc(sizeof(struct map));
    if (m == NULL)
        return NULL;

    m->capacity = capacity;
    m->buckets  = (struct entry**)calloc(capacity, sizeof(struct entry));
    if (m->buckets == NULL) {
        free(m);
        return NULL;
    }

    return m;
}

int get(struct map* m, char* key) {
    int           index = hash(key) % m->capacity;
    struct entry* e     = m->buckets[index];

    while (e != NULL) {
        if (strcmp(key, e->key) == 0)
            return e->value;

        e = e->next;
    }
    return 0;
}

void put(struct map* m, const char* key, int value) {
    int           index = hash(key) % m->capacity;
    struct entry* e     = m->buckets[index];

    while (e != NULL) {
        if (strcmp(e->key, key) == 0) {
            e->value = value;
            return;
        }

        e = e->next;
    }

    struct entry* new_entry = (struct entry*)malloc(sizeof(struct entry));
    if (new_entry == NULL)
        return;

    new_entry->value = value;
    new_entry->key   = strdup(key);
    if (new_entry->key == NULL) {
        free(new_entry);
        return;
    }

    new_entry->next   = m->buckets[index];
    m->buckets[index] = new_entry;
}

void free_map(struct map* m) {
    for (int i = 0; i < m->capacity; i++) {
        struct entry* e = m->buckets[i];
        while (e != NULL) {
            struct entry* next = e->next;
            free(e->key);
            free(e);
            e = next;
        }
    }

    free(m->buckets);
    free(m);
}

struct set*
create_set(size_t element_size, size_t initial_capacity, int (*cmp)(const void*, const void*)) {
    struct set* set = (struct set*)malloc(sizeof(struct set));
    if (set == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return NULL;
    }

    set->data = (void**)malloc(initial_capacity * sizeof(void*));
    if (set->data == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        free(set);
        return NULL;
    }

    set->size         = 0;
    set->capacity     = initial_capacity;
    set->element_size = element_size;
    set->cmp          = cmp;

    return set;
}

bool contains(struct set* set, void* value) {
    for (size_t i = 0; i < set->size; i++)
        if (set->cmp(set->data[i], value) == 0)
            return true;

    return false;
}

void add_to_set(struct set* set, void* value) {
    if (contains(set, value) == true)
        return;

    if (set->size == set->capacity) {
        size_t new_capacity = set->capacity * 2;
        void** new_data     = (void**)realloc(set->data, new_capacity * sizeof(void*));
        if (new_data == NULL) {
            fprintf(stderr, "Malloc failed!\n");
            exit(EXIT_FAILURE);
        }

        set->data     = new_data;
        set->capacity = new_capacity;
    }

    void* element = malloc(set->element_size);
    if (element == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        exit(EXIT_FAILURE);
    }

    memcpy(element, value, set->element_size);
    set->data[set->size++] = element;
}

void free_set(struct set* set) {
    if (set != NULL) {
        for (size_t i = 0; i < set->size; i++)
            free(set->data[i]);

        free(set->data);
        free(set);
    }
}

struct array* create_array(size_t element_size, size_t initial_capacity) {
    struct array* arr = (struct array*)malloc(sizeof(struct array));
    if (arr == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        exit(EXIT_FAILURE);
    }

    arr->data = (void**)malloc(initial_capacity * sizeof(void*));
    if (arr->data == NULL) {
        fprintf(stderr, "Malloc failed!\n");
        free(arr);
        exit(EXIT_FAILURE);
    }

    arr->size         = 0;
    arr->capacity     = initial_capacity;
    arr->element_size = element_size;

    return arr;
}