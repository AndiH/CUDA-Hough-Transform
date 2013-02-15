# Minimal Code Snippet for Bug Report
When compiling my code with the Nvidia C Compiler (NVCC) I get some strange error. Something like
```bash
In file included from /private/cuda-5.0/bin/crt/link.stub:79:0:
/tmp/tmpxft_000034aa_00000000-1_bad_dlink.reg.c:2:1: error: redefinition of âconst unsigned char def_module_id_str__NV_MODULE_ID []â
```

Using the `-dlink` flag solves / circumvents this problem. But that's strange.

## Using this example
The Makefile provides some example build configurations
* `make good`: successfully builds with `-dlink` flag
* `make bad`: fails building (no `-dlink' flag is set)
* `make one`: alternatively build code as a one-liner (and let nvcc worry about everything) -- sucessfull!
