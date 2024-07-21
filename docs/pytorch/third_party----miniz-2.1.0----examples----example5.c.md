# `.\pytorch\third_party\miniz-2.1.0\examples\example5.c`

```
// example5.c - Demonstrates how to use miniz.c's low-level tdefl_compress() and tinfl_inflate() API's for simple file to file compression/decompression.
// The low-level API's are the fastest, make no use of dynamic memory allocation, and are the most flexible functions exposed by miniz.c.
// Public domain, April 11 2012, Rich Geldreich, richgel99@gmail.com. See "unlicense" statement at the end of tinfl.c.
// For simplicity, this example is limited to files smaller than 4GB, but this is not a limitation of miniz.c.

// Purposely disable a whole bunch of stuff this low-level example doesn't use.
#define MINIZ_NO_STDIO
#define MINIZ_NO_ARCHIVE_APIS
#define MINIZ_NO_TIME
#define MINIZ_NO_ZLIB_APIS
#define MINIZ_NO_MALLOC
#include "miniz.h"

// Now include stdio.h because this test uses fopen(), etc. (but we still don't want miniz.c's stdio stuff, for testing).
#include <stdio.h>
#include <limits.h>

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;

#define my_max(a,b) (((a) > (b)) ? (a) : (b))
#define my_min(a,b) (((a) < (b)) ? (a) : (b))

// IN_BUF_SIZE is the size of the file read buffer.
// IN_BUF_SIZE must be >= 1
#define IN_BUF_SIZE (1024*512)
static uint8 s_inbuf[IN_BUF_SIZE];

// COMP_OUT_BUF_SIZE is the size of the output buffer used during compression.
// COMP_OUT_BUF_SIZE must be >= 1 and <= OUT_BUF_SIZE
#define COMP_OUT_BUF_SIZE (1024*512)

// OUT_BUF_SIZE is the size of the output buffer used during decompression.
// OUT_BUF_SIZE must be a power of 2 >= TINFL_LZ_DICT_SIZE (because the low-level decompressor not only writes, but reads from the output buffer as it decompresses)
//#define OUT_BUF_SIZE (TINFL_LZ_DICT_SIZE)
#define OUT_BUF_SIZE (1024*512)
static uint8 s_outbuf[OUT_BUF_SIZE];

// tdefl_compressor contains all the state needed by the low-level compressor so it's a pretty big struct (~300k).
// This example makes it a global vs. putting it on the stack, of course in real-world usage you'll probably malloc() or new it.
tdefl_compressor g_deflator;

// main function, entry point of the program
int main(int argc, char *argv[])
{
    // The following lines of code are intentionally left blank
    // because this comment section only includes the initial part of the main function.
    // The actual implementation of the main function is missing from this comment block.
}
```