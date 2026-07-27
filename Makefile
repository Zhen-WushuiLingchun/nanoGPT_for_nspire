# Ndless ARM compile/link/package smoke for the portable runtime.
#
#   eval "$(tools/ndless-env.sh)"
#   make ndless-smoke

GCC := nspire-gcc
GENZEHN := genzehn
MAKE_PRG := make-prg

BUILD_DIR := build/arm-runtime-smoke
DIST_DIR := dist
ELF := $(DIST_DIR)/nanogpt-runtime-smoke.elf
TNS := $(DIST_DIR)/nanogpt-runtime-smoke.tns

CFLAGS := -std=c11 -marm -Os -Wall -Wextra -Werror -Wshadow \
	-Wpointer-arith -ffunction-sections -fdata-sections -Iruntime/include
LDFLAGS := -Wl,--gc-sections -Wl,--no-warn-rwx-segments -lm

SOURCES := \
	runtime/src/ng_crc32.c \
	runtime/src/ng_model.c \
	runtime/src/ng_ops.c \
	runtime/src/ng_runtime.c \
	runtime/platform/ndless/crt_compat.c \
	runtime/platform/ndless/main_ndless.c
OBJECTS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(SOURCES))

.PHONY: ndless-smoke check-ndless clean

ndless-smoke: $(TNS)
	@bytes=$$(stat -c %s "$(TNS)"); \
	echo "Ndless smoke package: $(TNS) ($$bytes bytes)"

check-ndless:
	@command -v $(GCC) >/dev/null 2>&1 || { \
		echo "error: nspire-gcc not found; run eval \"\$$(tools/ndless-env.sh)\"" >&2; \
		exit 1; \
	}
	@command -v $(GENZEHN) >/dev/null 2>&1 || { \
		echo "error: genzehn not found" >&2; \
		exit 1; \
	}
	@command -v $(MAKE_PRG) >/dev/null 2>&1 || { \
		echo "error: make-prg not found" >&2; \
		exit 1; \
	}

$(BUILD_DIR)/%.o: %.c | check-ndless
	@mkdir -p "$(dir $@)"
	$(GCC) $(CFLAGS) -c "$<" -o "$@"

$(ELF): $(OBJECTS)
	@mkdir -p "$(DIST_DIR)"
	$(GCC) $(OBJECTS) -o "$@" $(LDFLAGS)

$(TNS): $(ELF)
	$(GENZEHN) \
		--input "$<" \
		--output "$@.zehn" \
		--name "nanoGPT Runtime Smoke" \
		--author "nanoGPT for Nspire" \
		--version 1 \
		--ndless-min 31 \
		--ndless-rev-min 2022 \
		--clickpad-support true \
		--color-support true \
		--240x320-support false \
		--compress
	$(MAKE_PRG) "$@.zehn" "$@"
	@rm -f "$@.zehn"

clean:
	rm -rf "$(BUILD_DIR)" "$(ELF)" "$(TNS)"
