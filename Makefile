# MPI compiler and flags
MPICC    ?= mpicc
CFLAGS   ?= -std=gnu11 -O2 -Wall -Wextra -I./src -I./src/headers -D_GNU_SOURCE
LDFLAGS  ?= -lm

# Enable debug mode if DEBUG=1 is passed
DEBUG ?= 0
ifeq ($(DEBUG),1)
  CFLAGS += -DDEBUG
endif

# Binary suffix based on debug mode
BIN_SUFFIX := $(if $(filter 1,$(DEBUG)),_debug,)

# Base names
BASE_TARGET_MPI := EM_Clustering
BASE_TARGET_SEQ := EM_Sequential
BASE_LIB_NAME   := libem

# Final paths
TARGET_MPI := bin/$(BASE_TARGET_MPI)$(BIN_SUFFIX)
TARGET_SEQ := bin/$(BASE_TARGET_SEQ)$(BIN_SUFFIX)
LIB_NAME   := $(BASE_LIB_NAME)$(BIN_SUFFIX).a
LIB_PATH   := bin/$(LIB_NAME)

# Commons source codes
COMMON_SRC := src/io_utils.c src/utils.c src/em_algorithm.c src/mpi_utils.c src/debug.c

# Directory for objects and dependencies
OBJDIR := src/obj
DEPDIR := src/dep

# parallel main version
MPI_MAIN := src/main.c
# sequential main version
SEQ_MAIN := src/sequential/multiFeature.c

# Objects files
OBJ_COMMON := $(COMMON_SRC:src/%.c=$(OBJDIR)/%.o)
OBJ_MPI    := $(MPI_MAIN:src/%.c=$(OBJDIR)/%.o)
OBJ_SEQ    := $(SEQ_MAIN:src/%.c=$(OBJDIR)/%.o)

# Dependency files
DEPS_COMMON := $(COMMON_SRC:src/%.c=$(DEPDIR)/%.d)
DEPS_MPI    := $(MPI_MAIN:src/%.c=$(DEPDIR)/%.d)
DEPS_SEQ    := $(SEQ_MAIN:src/%.c=$(DEPDIR)/%.d)

.PHONY: all mpi sequential clean debug sequential-debug

# Default: build MPI
all: mpi

mpi: $(TARGET_MPI)

sequential: $(LIB_PATH) $(TARGET_SEQ)

debug:
	@$(MAKE) DEBUG=1 mpi

sequential-debug:
	@$(MAKE) DEBUG=1 sequential

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
	rm -rf $(OBJDIR) $(DEPDIR) \
	       bin/$(BASE_TARGET_MPI) bin/$(BASE_TARGET_MPI)_debug \
	       bin/$(BASE_TARGET_SEQ) bin/$(BASE_TARGET_SEQ)_debug \
	       bin/$(BASE_LIB_NAME).a bin/$(BASE_LIB_NAME)_debug.a