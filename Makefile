# Makefile for MNISTNet-C

CC       := gcc
CFLAGS   := -std=c99 -O2 -fopenmp -march=native
LDFLAGS  := -lm

SRC_DIR  := src
SRCS     := $(wildcard $(SRC_DIR)/*.c)
TARGET   := mnist_simple

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
