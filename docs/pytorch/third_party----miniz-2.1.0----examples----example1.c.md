# `.\pytorch\third_party\miniz-2.1.0\examples\example1.c`

```
// example1.c - 展示了miniz.c中的compress()和uncompress()函数（与zlib的相同）。
// 作者：Rich Geldreich，发布于公共领域，日期：2011年5月15日，联系方式：richgel99@gmail.com。
// tinfl.c文件末尾包含了“无许可证”声明。

#include <stdio.h>
#include "miniz.h"

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint;

// 待压缩的字符串。
static const char *s_pStr = "Good morning Dr. Chandra. This is Hal. I am ready for my first lesson."
                            "Good morning Dr. Chandra. This is Hal. I am ready for my first lesson."
                            "Good morning Dr. Chandra. This is Hal. I am ready for my first lesson."
                            "Good morning Dr. Chandra. This is Hal. I am ready for my first lesson."
                            "Good morning Dr. Chandra. This is Hal. I am ready for my first lesson."
                            "Good morning Dr. Chandra. This is Hal. I am ready for my first lesson."
                            "Good morning Dr. Chandra. This is Hal. I am ready for my first lesson.";

int main(int argc, char *argv[])
{
  uint step = 0;
  int cmp_status;
  uLong src_len = (uLong)strlen(s_pStr);
  uLong cmp_len = compressBound(src_len); // 计算压缩后可能的最大长度
  uLong uncomp_len = src_len;             // 解压后的长度等于原始数据长度
  uint8 *pCmp, *pUncomp;
  uint total_succeeded = 0;
  (void)argc, (void)argv;

  printf("miniz.c version: %s\n", MZ_VERSION);

  do
  {
    // 分配空间用于存放压缩和解压缩后的数据
    pCmp = (mz_uint8 *)malloc((size_t)cmp_len);   // 分配压缩数据的空间
    pUncomp = (mz_uint8 *)malloc((size_t)src_len); // 分配解压缩数据的空间
    if ((!pCmp) || (!pUncomp))
    {
      printf("Out of memory!\n");
      return EXIT_FAILURE;
    }

    // 压缩字符串
    cmp_status = compress(pCmp, &cmp_len, (const unsigned char *)s_pStr, src_len);
    if (cmp_status != Z_OK)
    {
      printf("compress() failed!\n");
      free(pCmp);
      free(pUncomp);
      return EXIT_FAILURE;
    }

    printf("Compressed from %u to %u bytes\n", (mz_uint32)src_len, (mz_uint32)cmp_len);

    if (step)
    {
      // 如果进行模糊测试，则故意损坏压缩数据（这是一个非常简单的模糊测试）
      uint n = 1 + (rand() % 3);
      while (n--)
      {
        uint i = rand() % cmp_len;
        pCmp[i] ^= (rand() & 0xFF);
      }
    }

    // 解压缩
    cmp_status = uncompress(pUncomp, &uncomp_len, pCmp, cmp_len);
    total_succeeded += (cmp_status == Z_OK);

    if (step)
    {
      printf("Simple fuzzy test: step %u total_succeeded: %u\n", step, total_succeeded);
    }
    else
    {
      if (cmp_status != Z_OK)
      {
        printf("uncompress failed!\n");
        free(pCmp);
        free(pUncomp);
        return EXIT_FAILURE;
      }

      printf("Decompressed from %u to %u bytes\n", (mz_uint32)cmp_len, (mz_uint32)uncomp_len);

      // 确保uncompress()返回了预期的数据
      if ((uncomp_len != src_len) || (memcmp(pUncomp, s_pStr, (size_t)src_len)))
      {
        printf("Decompression failed!\n");
        free(pCmp);
        free(pUncomp);
        return EXIT_FAILURE;
      }
    }

    free(pCmp);
    free(pUncomp);

    step++;
    // 只要命令行参数的数量大于等于2，就继续进行模糊测试。
    } while (argc >= 2);
    
    // 打印成功消息
    printf("Success.\n");
    // 返回成功的退出状态码
    return EXIT_SUCCESS;
}


注释：

# 这行代码是一个单独的右大括号 '}'，通常用于结束代码块或数据结构的定义。
```