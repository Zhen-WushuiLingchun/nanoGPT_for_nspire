/*
 * Compatibility with prebuilt Arm GNU Toolchain newlib.
 *
 * Ndless's crt0 already walks .init_array and .fini_array. Prebuilt newlib
 * nevertheless references _init/_fini, while the Ndless startup objects do
 * not define them. Empty weak hooks satisfy newlib without running
 * constructors or destructors twice.
 */

void _init(void);
void _fini(void);

__attribute__((weak)) void _init(void)
{
}

__attribute__((weak)) void _fini(void)
{
}
