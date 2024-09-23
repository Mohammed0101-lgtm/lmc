CC = clang
CFLAGS = -Wall -Wextra -O2
TARGET = network
OBJECTS = main.o train.o nn.o linear.o embedding.o dataloader.o operations.o value.o util.o
DEPS = $(OBJECTS:.o=.d)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET)

-include $(DEPS)

%.o: %.c
	$(CC) $(CFLAGS) -MMD -c $< -o $@

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJECTS) $(DEPS)
