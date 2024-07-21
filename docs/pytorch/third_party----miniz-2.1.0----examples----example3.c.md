# `.\pytorch\third_party\miniz-2.1.0\examples\example3.c`

```py
// example3.c - 展示如何使用 miniz.c 的 deflate() 和 inflate() 函数进行简单的文件压缩
// 公共领域，2011年5月15日，Rich Geldreich，richgel99@gmail.com。请参阅tinfl.c末尾的“unlicense”声明。

#include <stdio.h>
#include <limits.h>
#include "miniz.h"  // 包含 miniz 压缩库的头文件

typedef unsigned char uint8;  // 定义无符号8位整数类型 uint8
typedef unsigned short uint16; // 定义无符号16位整数类型 uint16
typedef unsigned int uint;     // 定义无符号32位整数类型 uint

#define my_max(a,b) (((a) > (b)) ? (a) : (b))  // 定义宏函数，返回两个数中较大的一个
#define my_min(a,b) (((a) < (b)) ? (a) : (b))  // 定义宏函数，返回两个数中较小的一个

#define BUF_SIZE (1024 * 1024)  // 定义缓冲区大小为 1MB
static uint8 s_inbuf[BUF_SIZE];  // 定义输入缓冲区
static uint8 s_outbuf[BUF_SIZE]; // 定义输出缓冲区

int main(int argc, char *argv[])
{
  const char *pMode;    // 压缩或解压模式
  FILE *pInfile, *pOutfile;  // 输入和输出文件指针
  uint infile_size;     // 输入文件大小
  int level = Z_BEST_COMPRESSION;  // 压缩级别，默认为 Z_BEST_COMPRESSION
  z_stream stream;      // zlib 压缩流
  int p = 1;            // 命令行参数索引
  const char *pSrc_filename;  // 输入文件名
  const char *pDst_filename;  // 输出文件名
  long file_loc;         // 文件位置

  printf("miniz.c version: %s\n", MZ_VERSION);  // 打印 miniz.c 的版本信息

  // 检查命令行参数数量是否正确
  if (argc < 4)
  {
    printf("Usage: example3 [options] [mode:c or d] infile outfile\n");  // 提示用户使用说明
    printf("\nModes:\n");
    printf("c - Compresses file infile to a zlib stream in file outfile\n");  // 压缩模式说明
    printf("d - Decompress zlib stream in file infile to file outfile\n");   // 解压模式说明
    printf("\nOptions:\n");
    printf("-l[0-10] - Compression level, higher values are slower.\n");  // 压缩级别选项说明
    return EXIT_FAILURE;  // 程序退出，返回失败状态
  }

  // 解析命令行参数中的选项
  while ((p < argc) && (argv[p][0] == '-'))
  {
    switch (argv[p][1])
    {
      case 'l':
      {
        level = atoi(&argv[1][2]);  // 解析压缩级别参数
        if ((level < 0) || (level > 10))  // 检查压缩级别是否有效
        {
          printf("Invalid level!\n");  // 提示用户压缩级别无效
          return EXIT_FAILURE;    // 程序退出，返回失败状态
        }
        break;
      }
      default:
      {
        printf("Invalid option: %s\n", argv[p]);  // 提示用户选项无效
        return EXIT_FAILURE;    // 程序退出，返回失败状态
      }
    }
    p++;
  }

  // 检查命令行参数是否包含必要的输入输出文件名
  if ((argc - p) < 3)
  {
    printf("Must specify mode, input filename, and output filename after options!\n");  // 提示用户必须指定模式、输入和输出文件名
    return EXIT_FAILURE;    // 程序退出，返回失败状态
  }
  else if ((argc - p) > 3)
  {
    printf("Too many filenames!\n");  // 提示用户文件名过多
    return EXIT_FAILURE;    // 程序退出，返回失败状态
  }

  pMode = argv[p++];    // 获取压缩或解压模式
  if (!strchr("cCdD", pMode[0]))
  {
    printf("Invalid mode!\n");  // 提示用户模式无效
    return EXIT_FAILURE;    // 程序退出，返回失败状态
  }

  pSrc_filename = argv[p++];    // 获取输入文件名
  pDst_filename = argv[p++];    // 获取输出文件名

  printf("Mode: %c, Level: %u\nInput File: \"%s\"\nOutput File: \"%s\"\n", pMode[0], level, pSrc_filename, pDst_filename);  // 打印压缩或解压模式、级别以及输入输出文件名信息

  // 打开输入文件
  pInfile = fopen(pSrc_filename, "rb");
  if (!pInfile)
  {
    printf("Failed opening input file!\n");  // 提示用户打开输入文件失败
    return EXIT_FAILURE;    // 程序退出，返回失败状态
  }

  // 确定输入文件的大小
  fseek(pInfile, 0, SEEK_END);  // 定位到文件末尾
  file_loc = ftell(pInfile);    // 获取文件位置
  fseek(pInfile, 0, SEEK_SET);  // 重新定位到文件开头

  if ((file_loc < 0) || (file_loc > INT_MAX))
  {
     // 这不是 miniz 或 tinfl 的限制，而是此示例的限制
     printf("File is too large to be processed by this example.\n");  // 提示用户文件过大，无法处理
     return EXIT_FAILURE;    // 程序退出，返回失败状态
  }

  infile_size = (uint)file_loc;  // 将文件大小转换为无符号整数类型

  // 打开输出文件
  pOutfile = fopen(pDst_filename, "wb");
  if (!pOutfile)
  {
    printf("Failed opening output file!\n");  // 提示用户打开输出文件失败
    return EXIT_FAILURE;    // 程序退出，返回失败状态
  }
    // 如果程序执行到这里，说明发生了错误，返回失败状态码
    return EXIT_FAILURE;
  }

  // 打印输入文件大小
  printf("Input file size: %u\n", infile_size);

  // 初始化压缩流对象 z_stream
  memset(&stream, 0, sizeof(stream));
  stream.next_in = s_inbuf;     // 设置输入缓冲区的起始位置
  stream.avail_in = 0;          // 设置输入缓冲区可用字节数为0
  stream.next_out = s_outbuf;   // 设置输出缓冲区的起始位置
  stream.avail_out = BUF_SIZE;  // 设置输出缓冲区可用字节数为BUF_SIZE

  if ((pMode[0] == 'c') || (pMode[0] == 'C'))
  {
    // 如果是压缩模式
    uint infile_remaining = infile_size;

    // 初始化压缩流
    if (deflateInit(&stream, level) != Z_OK)
    {
      printf("deflateInit() failed!\n");
      return EXIT_FAILURE;
    }

    // 压缩主循环
    for ( ; ; )
    {
      int status;
      if (!stream.avail_in)
      {
        // 输入缓冲区为空，从输入文件读取更多字节
        uint n = my_min(BUF_SIZE, infile_remaining);

        if (fread(s_inbuf, 1, n, pInfile) != n)
        {
          printf("Failed reading from input file!\n");
          return EXIT_FAILURE;
        }

        stream.next_in = s_inbuf;
        stream.avail_in = n;

        infile_remaining -= n;
        //printf("Input bytes remaining: %u\n", infile_remaining);
      }

      // 压缩数据
      status = deflate(&stream, infile_remaining ? Z_NO_FLUSH : Z_FINISH);

      if ((status == Z_STREAM_END) || (!stream.avail_out))
      {
        // 输出缓冲区已满，或者压缩完成，将缓冲区内容写入输出文件
        uint n = BUF_SIZE - stream.avail_out;
        if (fwrite(s_outbuf, 1, n, pOutfile) != n)
        {
          printf("Failed writing to output file!\n");
          return EXIT_FAILURE;
        }
        stream.next_out = s_outbuf;
        stream.avail_out = BUF_SIZE;
      }

      if (status == Z_STREAM_END)
        break;
      else if (status != Z_OK)
      {
        printf("deflate() failed with status %i!\n", status);
        return EXIT_FAILURE;
      }
    }

    // 结束压缩流
    if (deflateEnd(&stream) != Z_OK)
    {
      printf("deflateEnd() failed!\n");
      return EXIT_FAILURE;
    }
  }
  else if ((pMode[0] == 'd') || (pMode[0] == 'D'))
  {
    // 如果是解压缩模式
    uint infile_remaining = infile_size;

    // 初始化解压缩流
    if (inflateInit(&stream))
    {
      printf("inflateInit() failed!\n");
      return EXIT_FAILURE;
    }

    // 解压缩主循环
    for ( ; ; )
    {
      // 定义变量 status 用于存储函数返回状态
      int status;
      // 如果输入缓冲区为空
      if (!stream.avail_in)
      {
        // 输入缓冲区为空，因此从输入文件中读取更多字节。
        uint n = my_min(BUF_SIZE, infile_remaining);

        // 从输入文件中读取 n 字节到 s_inbuf 中
        if (fread(s_inbuf, 1, n, pInfile) != n)
        {
          // 读取失败，输出错误消息并返回失败状态
          printf("Failed reading from input file!\n");
          return EXIT_FAILURE;
        }

        // 设置 zlib 流的下一个输入指针和可用输入字节数
        stream.next_in = s_inbuf;
        stream.avail_in = n;

        // 更新剩余待处理的输入文件字节数
        infile_remaining -= n;
      }

      // 调用 zlib 库的 inflate 函数进行解压缩
      status = inflate(&stream, Z_SYNC_FLUSH);

      // 如果解压缩完成或输出缓冲区已满
      if ((status == Z_STREAM_END) || (!stream.avail_out))
      {
        // 输出缓冲区已满，或解压缩完成，因此将缓冲区写入输出文件。
        uint n = BUF_SIZE - stream.avail_out;
        if (fwrite(s_outbuf, 1, n, pOutfile) != n)
        {
          // 写入输出文件失败，输出错误消息并返回失败状态
          printf("Failed writing to output file!\n");
          return EXIT_FAILURE;
        }
        // 重置 zlib 流的下一个输出指针和可用输出字节数
        stream.next_out = s_outbuf;
        stream.avail_out = BUF_SIZE;
      }

      // 如果解压缩完成
      if (status == Z_STREAM_END)
        break;
      // 如果解压缩返回非成功状态
      else if (status != Z_OK)
      {
        // 输出错误消息和解压缩状态码，并返回失败状态
        printf("inflate() failed with status %i!\n", status);
        return EXIT_FAILURE;
      }
    }

    // 结束 zlib 流的解压缩操作
    if (inflateEnd(&stream) != Z_OK)
    {
      // 如果结束解压缩操作失败，输出错误消息并返回失败状态
      printf("inflateEnd() failed!\n");
      return EXIT_FAILURE;
    }
  }
  else
  {
    // 如果模式无效，输出错误消息并返回失败状态
    printf("Invalid mode!\n");
    return EXIT_FAILURE;
  }

  // 关闭输入文件
  fclose(pInfile);
  // 关闭输出文件，并检查是否失败
  if (EOF == fclose(pOutfile))
  {
    // 如果关闭输出文件失败，输出错误消息并返回失败状态
    printf("Failed writing to output file!\n");
    return EXIT_FAILURE;
  }

  // 输出总的输入字节数
  printf("Total input bytes: %u\n", (mz_uint32)stream.total_in);
  // 输出总的输出字节数
  printf("Total output bytes: %u\n", (mz_uint32)stream.total_out);
  // 输出成功消息并返回成功状态
  printf("Success.\n");
  return EXIT_SUCCESS;
}



# 这行代码表示一个代码块的结束，关闭了之前的函数定义或者控制流结构（例如 if、for、while 等）。
```