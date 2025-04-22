# Name of the output binary
TARGET = main

# Source files
SRC = main.c
UTILS = $(wildcard utils/*.c)
ALL_SRC = $(SRC) $(UTILS)





# Compiler and flags
CXX = gcc

WARNINGS = -Wall -Werror -Wpedantic -Wextra -Wunused -Wuninitialized -Wshadow \
           -Wformat -Wconversion -Wfloat-equal -Wcast-qual -Wcast-align \
           -Wstrict-aliasing -Wswitch-default -Werror=return-type \
           -Werror=uninitialized -Werror=sign-compare -Wunused-function

OPTIMIZE = -O3 -funroll-loops -finline-functions -march=native -fpeel-loops #-mavx2

CXXFLAGS = $(WARNINGS) $(OPTIMIZE)

# Default build target
all: $(TARGET)

$(TARGET): $(ALL_SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(ALL_SRC) -lX11 -lm -fopenmp

run: $(TARGET)
	./$(TARGET)

bug: $(TARGET)
	gdb ./$(TARGET)

clean:
	rm -f $(TARGET)
