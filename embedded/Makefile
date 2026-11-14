# Build the C log-mel front-end as a shared library (for the Python parity test)
# and a static object (for linking into firmware).
CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -fPIC
UNAME := $(shell uname)
SO := liblogmel.$(if $(filter Darwin,$(UNAME)),dylib,so)

all: $(SO) logmel.o

mel_tables.h: gen_tables.py
	cd .. && .venv/bin/python embedded/gen_tables.py

logmel.o: logmel.c logmel.h mel_tables.h
	$(CC) $(CFLAGS) -c logmel.c -o $@

$(SO): logmel.o
	$(CC) -shared -o $@ logmel.o -lm

size: logmel.o
	size logmel.o

clean:
	rm -f *.o *.so *.dylib

.PHONY: all clean size
