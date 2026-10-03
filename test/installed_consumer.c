#include <jb.h>

#include <stdio.h>

int main(void) {
    jb_abi_info info = {.struct_size = sizeof info};
    if (jb_get_abi_info(&info) != JB_OK || info.api_version != JB_API_VERSION)
        return 1;
    printf("jev-bush %s ABI %u\n", jb_version(), info.api_version);
    return 0;
}
