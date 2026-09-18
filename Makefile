# GLM-5.3-Flash inference engine.
#
#   make                build the engine (bin/glm53f)
#   make test           run every test that needs no model weights
#   make portable       build without -march/-mcpu=native
#   make clean
#
# CMake is the reference build on Windows/MSVC (see CMakeLists.txt); this Makefile serves
# Linux, macOS (Homebrew libomp) and MSYS2 MinGW64. Both build the same sources.

CC     ?= cc
BUILD  ?= build-make
BIN    ?= bin

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
  ARCH ?= $(if $(filter arm64,$(UNAME_M)),-mcpu=native,-march=native)
  OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
  OMP_CFLAGS ?= -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  OMP_LDFLAGS ?= -L$(OMP_PREFIX)/lib -lomp
else ifneq ($(findstring MINGW,$(UNAME_S)),)
  ARCH ?= -march=native
  OMP_CFLAGS ?= -fopenmp
  OMP_LDFLAGS ?= -fopenmp -static -lpsapi
else
  ARCH ?= $(if $(filter aarch64,$(UNAME_M)),-mcpu=native,-march=native)
  OMP_CFLAGS ?= -fopenmp
  OMP_LDFLAGS ?= -fopenmp
endif

# -Wpointer-arith: weight pointers are void*, and arithmetic on them is a silent GNU
# extension striding by one byte. -ffp-contract=off keeps floating point reproducible.
WARN    := -Wall -Wextra -Wpointer-arith -Wshadow -Wvla -Wno-unused-parameter -Wno-unused-function
CFLAGS  ?= -O3 -std=gnu99 $(WARN) $(ARCH) $(OMP_CFLAGS) -pthread -ffp-contract=off
LDFLAGS ?= -lm $(OMP_LDFLAGS) -pthread

INCLUDES := -Iinclude -Iinclude/glm53f -Ithird_party \
            -Isrc/core -Isrc/io -Isrc/cache -Isrc/model -Isrc/tokenizer

ENGINE_SRC := src/core/glm53f_ops.c src/core/glm53f_dsa.c src/io/glm53f_st.c \
              src/io/glm53f_load.c src/io/glm53f_i4file.c src/cache/glm53f_cache.c src/model/glm53f_bind.c \
              src/model/glm53f_model.c src/model/glm53f_mtp.c
ENGINE_OBJ := $(patsubst %.c,$(BUILD)/%.o,$(ENGINE_SRC))

TESTS := test_glm_tiny test_cfg test_st test_tok
FIX   ?= tests/fixtures

.PHONY: all test portable clean

all: $(BIN)/glm53f $(BIN)/localcode

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BIN)/glm53f: src/cli/glm53f_run.c src/cli/glm53f_chat.c src/cli/glm53f_tui.c $(ENGINE_OBJ)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/localcode: src/cli/glm53f_run.c src/cli/glm53f_chat.c src/cli/glm53f_tui.c $(ENGINE_OBJ)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) -DGLM53F_LOCALCODE=1 $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_%: tests/unit/test_%.c $(ENGINE_OBJ)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

test: $(BIN)/glm53f $(addprefix $(BIN)/,$(TESTS))
	@echo "== tiny oracle (engine vs transformers glm5_next) =="; ./$(BIN)/test_glm_tiny $(FIX)/glm_tiny
	@echo "== config reader =="; ./$(BIN)/test_cfg $(FIX)/glm_tiny/config.json && ./$(BIN)/test_cfg config.json
	@echo "== safetensors ==";   ./$(BIN)/test_st $(FIX)/st $(BUILD)/st_index.json \
	    plain.f32.2d plain.bf16.1d tricky.f16.1d packed.u8.2d scalar.f32 second.shard.f32
	@echo "ALL WEIGHTLESS TESTS PASSED"

portable:
	$(MAKE) ARCH="$(if $(filter arm64 aarch64,$(UNAME_M)),,-mavx2 -mfma)" all

clean:
	rm -rf $(BUILD) $(BIN)
