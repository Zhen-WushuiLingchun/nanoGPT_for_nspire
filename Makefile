# Ndless ARM compile/link/package smoke for the portable runtime.
#
#   eval "$(tools/ndless-env.sh)"
#   make ndless-smoke

GCC := nspire-gcc
GENZEHN := genzehn
MAKE_PRG := make-prg

SMOKE_BUILD_DIR := build/arm-runtime-smoke
CHAT_BUILD_DIR := build/arm-chat
DIST_DIR := dist
ELF := $(DIST_DIR)/nanogpt-runtime-smoke.elf
TNS := $(DIST_DIR)/nanogpt-runtime-smoke.tns
CHAT_ELF := $(DIST_DIR)/nanogpt-chat.elf
CHAT_TNS := $(DIST_DIR)/nanogpt-chat.tns

CFLAGS := -std=c11 -marm -Os -Wall -Wextra -Werror -Wshadow \
	-Wpointer-arith -ffunction-sections -fdata-sections -Iruntime/include
LDFLAGS := -Wl,--gc-sections -Wl,--no-warn-rwx-segments -lm

SMOKE_SOURCES := \
	runtime/src/ng_crc32.c \
	runtime/src/ng_model.c \
	runtime/src/ng_ops.c \
	runtime/src/ng_runtime.c \
	runtime/platform/ndless/crt_compat.c \
	runtime/platform/ndless/main_ndless.c
SMOKE_OBJECTS := $(patsubst %.c,$(SMOKE_BUILD_DIR)/%.o,$(SMOKE_SOURCES))

CHAT_SOURCES := \
	runtime/src/ng_crc32.c \
	runtime/src/ng_chat.c \
	runtime/src/ng_model.c \
	runtime/src/ng_ops.c \
	runtime/src/ng_runtime.c \
	runtime/src/ng_gfx.c \
	runtime/src/ng_chat_view.c \
	runtime/platform/ndless/crt_compat.c \
	runtime/platform/ndless/chat_platform_ndless.c \
	runtime/platform/ndless/chat_main_ndless.c
CHAT_OBJECTS := $(patsubst %.c,$(CHAT_BUILD_DIR)/%.o,$(CHAT_SOURCES))

.PHONY: ndless-smoke ndless-chat check-ndless clean

ndless-smoke: $(TNS)
	@bytes=$$(stat -c %s "$(TNS)"); \
	echo "Ndless smoke package: $(TNS) ($$bytes bytes)"

ndless-chat: $(CHAT_TNS)
	@bytes=$$(stat -c %s "$(CHAT_TNS)"); \
	echo "Ndless chat package: $(CHAT_TNS) ($$bytes bytes)"

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

$(SMOKE_BUILD_DIR)/%.o: %.c | check-ndless
	@mkdir -p "$(dir $@)"
	$(GCC) $(CFLAGS) -c "$<" -o "$@"

$(CHAT_BUILD_DIR)/%.o: %.c | check-ndless
	@mkdir -p "$(dir $@)"
	$(GCC) $(CFLAGS) -c "$<" -o "$@"

$(ELF): $(SMOKE_OBJECTS)
	@mkdir -p "$(DIST_DIR)"
	$(GCC) $(SMOKE_OBJECTS) -o "$@" $(LDFLAGS)

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

$(CHAT_ELF): $(CHAT_OBJECTS)
	@mkdir -p "$(DIST_DIR)"
	$(GCC) $(CHAT_OBJECTS) -o "$@" $(LDFLAGS)

$(CHAT_TNS): $(CHAT_ELF)
	$(GENZEHN) \
		--input "$<" \
		--output "$@.zehn" \
		--name "nanoGPT Pixel Chat" \
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
	rm -rf \
		"$(SMOKE_BUILD_DIR)" \
		"$(CHAT_BUILD_DIR)" \
		"$(ELF)" \
		"$(TNS)" \
		"$(CHAT_ELF)" \
		"$(CHAT_TNS)"
