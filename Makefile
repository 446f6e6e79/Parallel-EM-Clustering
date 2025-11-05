# MPI compiler and flags
MPICC   ?= mpicc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -D_POSIX_C_SOURCE=200809L -I./src
LDFLAGS ?= -lm

# Target binary executable
TARGET  := bin/EM_Clustering

# Source files
SRC     := src/main.c src/file_io.c 

# Object and dependency directories
OBJDIR  := src/obj
DEPDIR  := src/dep

# Derived object and dependency file paths
OBJ     := $(SRC:src/%.c=$(OBJDIR)/%.o)
DEPS    := $(SRC:src/%.c=$(DEPDIR)/%.d)

.PHONY: all clean

all: $(TARGET)

# Link the target binary
$(TARGET): $(OBJ)
	@mkdir -p $(dir $@)
	$(MPICC) $(LDFLAGS) -o $@ $(OBJ)

# Compile .c -> .o and generate dependency file (.d)
$(OBJDIR)/%.o: src/%.c
	@mkdir -p $(dir $@) $(DEPDIR)/$(dir $(<:src/%=%))
	$(MPICC) $(CFLAGS) -MMD -MP -c $< -o $@
	@mv $(OBJDIR)/$(basename $(notdir $(@F))).d $(DEPDIR)/ || true

# Include dependency files if they exist
-include $(DEPS)

clean:
	rm -rf $(OBJDIR) $(DEPDIR) $(TARGET)
