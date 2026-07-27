#!/usr/bin/env bash
# Print shell exports for the already-installed Ndless SDK and ARM toolchain.

set -euo pipefail

sdk_root="${NANOGPT_NDLESS_ROOT:-$HOME/.phy-nspire}"
ndless_sdk="${NDLESS_SDK:-$sdk_root/Ndless/ndless-sdk}"

if [ -n "${_NDLESS_TOOLCHAIN_PATH:-}" ]; then
    toolchain_bin="$_NDLESS_TOOLCHAIN_PATH"
else
    toolchain_dir="$(
        find "$sdk_root" -maxdepth 1 -type d \
            -name 'arm-gnu-toolchain-*-x86_64-arm-none-eabi' \
            -print |
            sort -V |
            tail -n 1
    )"
    toolchain_bin="$toolchain_dir/bin"
fi

if [ ! -x "$ndless_sdk/bin/nspire-gcc" ]; then
    echo "error: Ndless SDK not found at $ndless_sdk" >&2
    exit 1
fi
if [ ! -x "$toolchain_bin/arm-none-eabi-gcc" ]; then
    echo "error: ARM toolchain not found below $sdk_root" >&2
    exit 1
fi

printf 'export NDLESS_SDK=%q\n' "$ndless_sdk"
printf 'export _NDLESS_TOOLCHAIN_PATH=%q\n' "$toolchain_bin"
printf 'export PATH=%q:%q:$PATH\n' "$ndless_sdk/bin" "$toolchain_bin"
