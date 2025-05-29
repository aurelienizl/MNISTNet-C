# Makefile for MNISTNet-C

CC       := gcc
CFLAGS   := -std=c99 -O2 -fopenmp -march=native 
LDFLAGS  := -lm

SRC_DIR  := src
SRCS     := $(SRC_DIR)/main.c \
		    $(SRC_DIR)/nn_optimized.c \
		    $(SRC_DIR)/loaders/mnist.c \
		    $(SRC_DIR)/loaders/cifar10.c \

TARGET   := mnist_simple


all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDFLAGS)

debug: CFLAGS += -g -fsanitize=address
debug: $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean debug
