#include "dataloader.h"
#include "value.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


struct Value*** get_batch(
    const int* __restrict const data, const int data_size, const int context_length,
    const int batch_size) {

    srand(time(NULL));
    int            random_offset = rand() % (data_size - context_length * batch_size);
    struct Value** inputs        = (struct Value**)malloc(batch_size * sizeof(struct Value*));

    if (!inputs) {
        fprintf(stderr, "Malloc Failed!\n");
        return NULL;
    }

    for (int i = 0; i < batch_size; i++) {
        inputs[i] = (struct Value*)malloc(context_length * sizeof(struct Value));

        if (!inputs[i]) {
            for (int j = 0; j < i; j++)
                free(inputs[j]);

            free(inputs);
            fprintf(stderr, "Malloc Failed!\n");
            return NULL;
        }
    }

    struct Value* targets = (struct Value*)malloc(batch_size * sizeof(struct Value));

    if (!targets) {
        fprintf(stderr, "Malloc Failed\n");

        for (int i = 0; i < batch_size; i++)
            free(inputs[i]);

        free(inputs);
        return NULL;
    }

    int chunk_begin = random_offset;
    int chunk_end   = random_offset + context_length;

    for (int i = 0; i < batch_size; i++) {
        if (chunk_end >= data_size)
            break;

        for (int j = 0; j < context_length; j++)
            inputs[i][j] = *_init_value((float)(data[chunk_begin + j]));

        targets[i] = *_init_value((float)(data[chunk_end]));

        chunk_begin += context_length;
        chunk_end += context_length;
    }

    struct Value*** output = (struct Value***)malloc(2 * sizeof(struct Value**));

    if (!output) {
        fprintf(stderr, "Malloc Failed\n");

        for (int i = 0; i < batch_size; i++)
            free(inputs[i]);

        free(inputs);
        free(targets);
        return NULL;
    }

    output[0] = inputs;
    output[1] = (struct Value**)targets;

    return output;
}

char* read_file(FILE* _file) {
    if (_file == NULL)
        return NULL;

    if (fseek(_file, 0, SEEK_END) != 0)
        return NULL;

    long f_size = ftell(_file);

    if (f_size == -1L)
        return NULL;

    if (fseek(_file, 0, SEEK_SET) != 0)
        return NULL;

    if (f_size < 0)
        return NULL;

    char* buf = (char*)malloc((f_size + 1) * sizeof(char));
    if (buf == NULL) {
        fprintf(stderr, "Malloc failed\n");
        return NULL;
    }

    size_t read_size = fread(buf, sizeof(char), (size_t)f_size, _file);

    if (read_size != (size_t)f_size) {
        free(buf);

        if (feof(_file))
            fprintf(stderr, "Unexpected end of file\n");
        else if (ferror(_file))
            fprintf(stderr, "fread failed\n");

        return NULL;
    }

    buf[f_size] = '\0';
    return buf;
}

#define MAX_CHAR 256

typedef struct {
    char character;
    int  index;
} CharMap;

CharMap char_to_index[MAX_CHAR];
char    global_idx[MAX_CHAR];

int     find_char_index(char c) {
    for (int i = 0; i < MAX_CHAR; ++i) {
        if (char_to_index[i].character == c) {
            return char_to_index[i].index;
        }
    }
    return -1;
}

int* encode(const char* str, int* result_size, int* vocab_size) {
    for (int i = 0; i < MAX_CHAR; ++i) {
        char_to_index[i].character = 0;
        char_to_index[i].index     = -1;
        global_idx[i]              = 0;
    }

    *vocab_size  = 0;
    int  index   = 0;
    int  str_len = strlen(str);

    int* result  = (int*)malloc(str_len * sizeof(int));

    if (result == NULL) {
        perror("Failed to allocate memory for result");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < str_len; ++i) {
        char c          = str[i];
        int  char_index = find_char_index(c);

        if (char_index == -1) {
            (*vocab_size)++;
            char_to_index[index].character = c;
            char_to_index[index].index     = index;
            char_index                     = index++;
        }

        result[i] = char_index;
    }

    for (int i = 0; i < index; ++i)
        global_idx[char_to_index[i].index] = char_to_index[i].character;

    *result_size = str_len;
    return result;
}