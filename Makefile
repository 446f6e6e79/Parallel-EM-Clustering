# MPI compiler and flags
MPICC    ?= mpicc
CFLAGS   ?= -std=gnu11 -O2 -Wall -Wextra -I./src -D_GNU_SOURCE
LDFLAGS  ?= -lm

# Commons source codes
COMMON_SRC := src/file_io.c src/utils.c src/debug.c src/em_algorithm.c

# Directory for objects and dependencies
OBJDIR := src/obj
DEPDIR := src/dep

# parallel main version
MPI_MAIN   := src/main.c
TARGET_MPI := bin/EM_Clustering

# sequential main version
SEQ_MAIN   := src/sequential/multiFeature.c
TARGET_SEQ := bin/EM_Sequential

# Libreria statica
LIB_NAME   := libem.a
LIB_PATH   := bin/$(LIB_NAME)



# Objects files
OBJ_COMMON := $(COMMON_SRC:src/%.c=$(OBJDIR)/%.o)
OBJ_MPI    := $(MPI_MAIN:src/%.c=$(OBJDIR)/%.o)
OBJ_SEQ    := $(SEQ_MAIN:src/%.c=$(OBJDIR)/%.o)

# Dependency files
DEPS_COMMON := $(COMMON_SRC:src/%.c=$(DEPDIR)/%.d)
DEPS_MPI    := $(MPI_MAIN:src/%.c=$(DEPDIR)/%.d)
DEPS_SEQ    := $(SEQ_MAIN:src/%.c=$(DEPDIR)/%.d)

.PHONY: all mpi sequential clean

# Default: build MPI parallel version
all: mpi

mpi: $(TARGET_MPI)

sequential: $(LIB_PATH) $(TARGET_SEQ)

# Static library
$(LIB_PATH): $(OBJ_COMMON)
	@mkdir -p $(dir $@)
	@rm -f $@
	ar rcs $@ $(OBJ_COMMON)

# MPI executable
$(TARGET_MPI): $(OBJ_MPI) $(OBJ_COMMON)
	@mkdir -p $(dir $@)
	$(MPICC) $(LDFLAGS) -o $@ $(OBJ_MPI) $(OBJ_COMMON)

# Sequential executable
$(TARGET_SEQ): $(OBJ_SEQ) $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(MPICC) $(LDFLAGS) -o $@ $(OBJ_SEQ) $(LIB_PATH)

# Generic compilation rule
$(OBJDIR)/%.o: src/%.c
	@mkdir -p $(dir $@) $(dir $(DEPDIR)/$*.d)
	$(MPICC) $(CFLAGS) -MMD -MP -MF $(DEPDIR)/$*.d -c $< -o $@

# Include dependency files
-include $(DEPS_COMMON) $(DEPS_MPI) $(DEPS_SEQ)

clean:
	rm -rf $(OBJDIR) $(DEPDIR) $(TARGET_MPI) $(TARGET_SEQ) $(LIB_PATH)
