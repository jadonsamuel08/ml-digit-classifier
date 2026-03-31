CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic -Iinclude
TARGET := mnist_viewer
SRCS := src/main.cpp src/mnist_loader.cpp

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
