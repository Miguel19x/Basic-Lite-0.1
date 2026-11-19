CC ?= gcc
CFLAGS ?= -O3 -march=native -fopenmp -shared -fPIC
TARGET = libpaged_gemm.so
PYTHON ?= $(shell if [ -f venv/bin/python ]; then echo venv/bin/python; else echo python3; fi)

.PHONY: all clean test

all: $(TARGET)

$(TARGET): paged_gemm_kernel.c
	$(CC) $(CFLAGS) $< -o $@

test: $(TARGET)
	PYTHONPATH=. $(PYTHON) tests/test_smoke.py

clean:
	rm -f $(TARGET)
