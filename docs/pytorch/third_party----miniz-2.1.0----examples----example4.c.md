# `.\pytorch\third_party\miniz-2.1.0\examples\example4.c`

```py
// example4.c - 使用 tinfl.c 将内存中的 zlib 流解压到输出文件
// 公有领域，作者 Rich Geldreich，联系邮箱 richgel99@gmail.com
// 请参阅 tinfl.c 结尾的 "unlicense" 声明

#include "miniz_tinfl.h"   // 包含 tinfl 库头文件
#include <stdio.h>         // 标准输入输出库
#include <limits.h>        // 包含整数限制的定义

typedef unsigned char uint8;    // 定义 8 位无符号整数类型 uint8
typedef unsigned short uint16;  // 定义 16 位无符号整数类型 uint16
typedef unsigned int uint;       // 定义 无符号整数类型 uint

#define my_max(a,b) (((a) > (b)) ? (a) : (b))  // 宏定义，返回两个数中的最大值
#define my_min(a,b) (((a) < (b)) ? (a) : (b))  // 宏定义，返回两个数中的最小值

// 回调函数，将缓冲区内容写入文件
static int tinfl_put_buf_func(const void* pBuf, int len, void *pUser)
{
  return len == (int)fwrite(pBuf, 1, len, (FILE*)pUser);  // 写入缓冲区内容到文件并返回写入的字节数
}

// 主函数，程序入口
int main(int argc, char *argv[])
{
  int status;               // 状态变量
  FILE *pInfile, *pOutfile; // 输入文件指针，输出文件指针
  uint infile_size, outfile_size;  // 输入文件大小，输出文件大小
  size_t in_buf_size;       // 输入缓冲区大小
  uint8 *pCmp_data;         // 压缩数据指针
  long file_loc;            // 文件位置变量

  // 检查命令行参数数量
  if (argc != 3)
  {
    printf("Usage: example4 infile outfile\n");  // 打印用法信息
    printf("Decompresses zlib stream in file infile to file outfile.\n");  // 打印功能描述
    printf("Input file must be able to fit entirely in memory.\n");  // 提示输入文件必须完全放入内存
    printf("example3 can be used to create compressed zlib streams.\n");  // 提示使用 example3 创建压缩的 zlib 流
    return EXIT_FAILURE;  // 返回失败状态码
  }

  // 打开输入文件
  pInfile = fopen(argv[1], "rb");  // 以只读二进制方式打开文件
  if (!pInfile)
  {
    printf("Failed opening input file!\n");  // 打开输入文件失败
    return EXIT_FAILURE;  // 返回失败状态码
  }

  // 确定输入文件的大小
  fseek(pInfile, 0, SEEK_END);  // 将文件指针移动到文件末尾
  file_loc = ftell(pInfile);   // 获取文件指针当前位置
  fseek(pInfile, 0, SEEK_SET);  // 将文件指针移动到文件开头

  // 检查文件大小是否在可处理范围内
  if ((file_loc < 0) || (file_loc > INT_MAX))
  {
     // 这不是 miniz 或 tinfl 的限制，而是本示例的限制
     printf("File is too large to be processed by this example.\n");  // 文件过大无法处理
     return EXIT_FAILURE;  // 返回失败状态码
  }

  infile_size = (uint)file_loc;  // 将文件大小转换为无符号整数

  pCmp_data = (uint8 *)malloc(infile_size);  // 分配足够大小的内存存储压缩数据
  if (!pCmp_data)
  {
    printf("Out of memory!\n");  // 内存分配失败
    return EXIT_FAILURE;  // 返回失败状态码
  }
  if (fread(pCmp_data, 1, infile_size, pInfile) != infile_size)
  {
    printf("Failed reading input file!\n");  // 读取输入文件失败
    return EXIT_FAILURE;  // 返回失败状态码
  }

  // 打开输出文件
  pOutfile = fopen(argv[2], "wb");  // 以只写二进制方式打开文件
  if (!pOutfile)
  {
    printf("Failed opening output file!\n");  // 打开输出文件失败
    return EXIT_FAILURE;  // 返回失败状态码
  }

  printf("Input file size: %u\n", infile_size);  // 打印输入文件大小信息

  in_buf_size = infile_size;  // 输入缓冲区大小设置为输入文件大小
  // 解压缩内存中的数据到回调函数写入到输出文件中
  status = tinfl_decompress_mem_to_callback(pCmp_data, &in_buf_size, tinfl_put_buf_func, pOutfile, TINFL_FLAG_PARSE_ZLIB_HEADER);
  if (!status)
  {
    printf("tinfl_decompress_mem_to_callback() failed with status %i!\n", status);  // 解压缩失败
    return EXIT_FAILURE;  // 返回失败状态码
  }

  outfile_size = ftell(pOutfile);  // 获取输出文件大小

  fclose(pInfile);  // 关闭输入文件
  if (EOF == fclose(pOutfile))
  {
    printf("Failed writing to output file!\n");  // 写入输出文件失败
    return EXIT_FAILURE;  // 返回失败状态码
  }

  printf("Total input bytes: %u\n", (uint)in_buf_size);  // 打印总输入字节数
  printf("Total output bytes: %u\n", outfile_size);  // 打印总输出字节数
  printf("Success.\n");  // 打印成功信息
  return EXIT_SUCCESS;  // 返回成功状态码
}
```