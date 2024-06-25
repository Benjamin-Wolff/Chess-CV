CC = clang++
CCX = $(CC)

# Directories
SRCDIR := src
INCDIR := include
BINDIR := bin

# Files
SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(SRCS:$(SRCDIR)/%.cpp=$(BINDIR)/%.o)
# EXEC := project1

# OSX include paths (for homebrew, probably)
CFLAGS := -Wc++11-extensions -std=c++17 -I$(INCDIR) $(shell pkg-config --cflags opencv4 nlohmann_json) -I/usr/local/Cellar/cpr/1.10.5/include

CXXFLAGS = $(CFLAGS)

# Linker flags
LDFLAGS := $(shell pkg-config --libs opencv4 nlohmann_json) -lcpr

# Compilation rule
$(BINDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Build rule

chessCV: $(BINDIR)/chessCV.o $(BINDIR)/csv_util.o $(BINDIR)/processingOps.o $(BINDIR)/pieceDetectionOps.o $(BINDIR)/chessAnalysis.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $(BINDIR)/$@

.PHONY: clean

clean:
	rm -f $(BINDIR)/*.o $(BINDIR)/$(EXEC)