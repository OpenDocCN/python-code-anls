# `.\numpy\numpy\_core\src\common\npy_cpuinfo_parser.h`

```py
/*
 * Copyright (C) 2010 The Android Open Source Project
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
#ifndef NUMPY_CORE_SRC_COMMON_NPY_CPUINFO_PARSER_H_
#define NUMPY_CORE_SRC_COMMON_NPY_CPUINFO_PARSER_H_

#include <errno.h>
#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stddef.h>

#define NPY__HWCAP  16
#define NPY__HWCAP2 26

// arch/arm/include/uapi/asm/hwcap.h
#define NPY__HWCAP_HALF   (1 << 1)
#define NPY__HWCAP_NEON   (1 << 12)
#define NPY__HWCAP_VFPv3  (1 << 13)
#define NPY__HWCAP_VFPv4  (1 << 16)
#define NPY__HWCAP2_AES   (1 << 0)
#define NPY__HWCAP2_PMULL (1 << 1)
#define NPY__HWCAP2_SHA1  (1 << 2)
#define NPY__HWCAP2_SHA2  (1 << 3)
#define NPY__HWCAP2_CRC32 (1 << 4)
// arch/arm64/include/uapi/asm/hwcap.h
#define NPY__HWCAP_FP       (1 << 0)
#define NPY__HWCAP_ASIMD    (1 << 1)
#define NPY__HWCAP_FPHP     (1 << 9)
#define NPY__HWCAP_ASIMDHP  (1 << 10)
#define NPY__HWCAP_ASIMDDP  (1 << 20)
#define NPY__HWCAP_SVE      (1 << 22)
#define NPY__HWCAP_ASIMDFHM (1 << 23)

/* 
 * Get the size of a file by reading it until the end. This is needed
 * because files under /proc do not always return a valid size when
 * using fseek(0, SEEK_END) + ftell(). Nor can they be mmap()-ed.
 */
static int
get_file_size(const char* pathname)
{
    int fd, result = 0;
    char buffer[256];

    // 打开指定路径的文件，只读方式
    fd = open(pathname, O_RDONLY);
    if (fd < 0) {
        return -1; // 如果打开失败，返回错误码 -1
    }

    // 循环读取文件内容计算文件大小
    for (;;) {
        int ret = read(fd, buffer, sizeof buffer);
        if (ret < 0) {
            if (errno == EINTR) {
                continue; // 如果是被中断，继续读取
            }
            break; // 出现其他错误则跳出循环
        }
        if (ret == 0) {
            break; // 读取到文件末尾，跳出循环
        }
        result += ret; // 累加已读取的字节数
    }
    close(fd); // 关闭文件描述符
    return result; // 返回文件大小
}
/* 
 * Read the content of /proc/cpuinfo into a user-provided buffer.
 * Return the length of the data, or -1 on error. Does *not*
 * zero-terminate the content. Will not read more
 * than 'buffsize' bytes.
 */
static int
read_file(const char*  pathname, char*  buffer, size_t  buffsize)
{
    int  fd, count;

    fd = open(pathname, O_RDONLY);  // 打开指定路径的文件，只读模式
    if (fd < 0) {  // 如果打开文件失败
        return -1;  // 返回错误状态
    }
    count = 0;  // 初始化计数器

    // 循环读取文件内容，直到达到指定的缓冲区大小或出错
    while (count < (int)buffsize) {
        int ret = read(fd, buffer + count, buffsize - count);  // 从文件中读取数据到缓冲区中
        if (ret < 0) {  // 如果读取操作返回错误
            if (errno == EINTR) {  // 如果是被中断信号中断
                continue;  // 继续读取
            }
            if (count == 0) {  // 如果在开始读取前就出错
                count = -1;  // 返回错误状态
            }
            break;  // 退出循环
        }
        if (ret == 0) {  // 如果读取到文件末尾
            break;  // 退出循环
        }
        count += ret;  // 更新已读取数据的字节数
    }
    close(fd);  // 关闭文件
    return count;  // 返回读取到的数据长度或错误状态
}

/* 
 * Extract the content of a the first occurrence of a given field in
 * the content of /proc/cpuinfo and return it as a heap-allocated
 * string that must be freed by the caller.
 *
 * Return NULL if not found
 */
static char*
extract_cpuinfo_field(const char* buffer, int buflen, const char* field)
{
    int fieldlen = strlen(field);  // 计算字段的长度
    const char* bufend = buffer + buflen;  // 缓冲区结束位置
    char* result = NULL;  // 初始化结果指针为NULL
    int len;
    const char *p, *q;

    /* Look for first field occurrence, and ensures it starts the line. */
    p = buffer;  // 从缓冲区开始查找
    for (;;) {
        p = memmem(p, bufend-p, field, fieldlen);  // 在缓冲区中查找字段的第一次出现
        if (p == NULL) {  // 如果未找到字段
            goto EXIT;  // 跳转到退出处理
        }

        if (p == buffer || p[-1] == '\n') {  // 确保字段在行的开头
            break;  // 找到符合条件的字段，退出循环
        }

        p += fieldlen;  // 继续向后查找
    }

    /* Skip to the first column followed by a space */
    p += fieldlen;  // 跳过字段本身
    p = memchr(p, ':', bufend-p);  // 查找字段后的冒号
    if (p == NULL || p[1] != ' ') {  // 如果未找到冒号或冒号后不是空格
        goto EXIT;  // 跳转到退出处理
    }

    /* Find the end of the line */
    p += 2;  // 跳过冒号和空格
    q = memchr(p, '\n', bufend-p);  // 查找行末尾的换行符
    if (q == NULL) {  // 如果未找到换行符
        q = bufend;  // 将结束位置设置为缓冲区末尾
    }

    /* Copy the line into a heap-allocated buffer */
    len = q - p;  // 计算行的长度
    result = malloc(len + 1);  // 分配内存保存行数据，需由调用者释放
    if (result == NULL) {  // 如果内存分配失败
        goto EXIT;  // 跳转到退出处理
    }

    memcpy(result, p, len);  // 复制行数据到结果缓冲区
    result[len] = '\0';  // 添加字符串结尾标志

EXIT:
    return result;  // 返回提取的字段内容或NULL（未找到）
}

/* 
 * Checks that a space-separated list of items contains one given 'item'.
 * Returns 1 if found, 0 otherwise.
 */
static int
has_list_item(const char* list, const char* item)
{
    const char* p = list;  // 指向列表起始位置
    int itemlen = strlen(item);  // 计算待查找项的长度

    if (list == NULL) {  // 如果列表为空
        return 0;  // 直接返回未找到
    }

    while (*p) {  // 循环遍历列表
        const char*  q;

        /* skip spaces */
        while (*p == ' ' || *p == '\t') {  // 跳过空格和制表符
            p++;
        }

        /* find end of current list item */
        q = p;
        while (*q && *q != ' ' && *q != '\t') {  // 查找当前列表项的末尾
            q++;
        }

        if (itemlen == q-p && !memcmp(p, item, itemlen)) {  // 比较当前项与目标项是否相等
            return 1;  // 找到目标项，返回1
        }

        /* skip to next item */
        p = q;  // 移动到下一个列表项
    }
    return 0;  // 未找到目标项，返回0
}

static void setHwcap(char* cpuFeatures, unsigned long* hwcap) {
    *hwcap |= has_list_item(cpuFeatures, "neon") ? NPY__HWCAP_NEON : 0;  // 检查CPU特性中是否包含"neon"，设置对应的标志位
}
    # 如果 cpuFeatures 中包含 "half"，则将 NPY__HWCAP_HALF 的值加到 hwcap 中，否则加 0
    *hwcap |= has_list_item(cpuFeatures, "half") ? NPY__HWCAP_HALF : 0;
    
    # 如果 cpuFeatures 中包含 "vfpv3"，则将 NPY__HWCAP_VFPv3 的值加到 hwcap 中，否则加 0
    *hwcap |= has_list_item(cpuFeatures, "vfpv3") ? NPY__HWCAP_VFPv3 : 0;
    
    # 如果 cpuFeatures 中包含 "vfpv4"，则将 NPY__HWCAP_VFPv4 的值加到 hwcap 中，否则加 0
    *hwcap |= has_list_item(cpuFeatures, "vfpv4") ? NPY__HWCAP_VFPv4 : 0;

    # 如果 cpuFeatures 中包含 "asimd"，则将 NPY__HWCAP_ASIMD 的值加到 hwcap 中，否则加 0
    *hwcap |= has_list_item(cpuFeatures, "asimd") ? NPY__HWCAP_ASIMD : 0;
    
    # 如果 cpuFeatures 中包含 "fp"，则将 NPY__HWCAP_FP 的值加到 hwcap 中，否则加 0
    *hwcap |= has_list_item(cpuFeatures, "fp") ? NPY__HWCAP_FP : 0;
    
    # 如果 cpuFeatures 中包含 "fphp"，则将 NPY__HWCAP_FPHP 的值加到 hwcap 中，否则加 0
    *hwcap |= has_list_item(cpuFeatures, "fphp") ? NPY__HWCAP_FPHP : 0;
    
    # 如果 cpuFeatures 中包含 "asimdhp"，则将 NPY__HWCAP_ASIMDHP 的值加到 hwcap 中，否则加 0
    *hwcap |= has_list_item(cpuFeatures, "asimdhp") ? NPY__HWCAP_ASIMDHP : 0;
    
    # 如果 cpuFeatures 中包含 "asimddp"，则将 NPY__HWCAP_ASIMDDP 的值加到 hwcap 中，否则加 0
    *hwcap |= has_list_item(cpuFeatures, "asimddp") ? NPY__HWCAP_ASIMDDP : 0;
    
    # 如果 cpuFeatures 中包含 "asimdfhm"，则将 NPY__HWCAP_ASIMDFHM 的值加到 hwcap 中，否则加 0
    *hwcap |= has_list_item(cpuFeatures, "asimdfhm") ? NPY__HWCAP_ASIMDFHM : 0;
static int
get_feature_from_proc_cpuinfo(unsigned long *hwcap, unsigned long *hwcap2) {
    // 声明一个指向字符的指针cpuinfo，初始设为NULL
    char* cpuinfo = NULL;
    // 声明一个整型变量cpuinfo_len，用于存储读取的文件大小
    int cpuinfo_len;
    // 调用get_file_size函数获取/proc/cpuinfo文件的大小
    cpuinfo_len = get_file_size("/proc/cpuinfo");
    // 如果获取文件大小失败，则返回0
    if (cpuinfo_len < 0) {
        return 0;
    }
    // 分配cpuinfo_len大小的内存给cpuinfo，存储/proc/cpuinfo文件的内容
    cpuinfo = malloc(cpuinfo_len);
    // 如果内存分配失败，则返回0
    if (cpuinfo == NULL) {
        return 0;
    }
    // 重新调用read_file函数，读取/proc/cpuinfo文件的内容到cpuinfo中
    cpuinfo_len = read_file("/proc/cpuinfo", cpuinfo, cpuinfo_len);
    // 从cpuinfo中提取"Features"字段的值，存储在cpuFeatures中
    char* cpuFeatures = extract_cpuinfo_field(cpuinfo, cpuinfo_len, "Features");
    // 如果未能提取到cpuFeatures，则返回0
    if(cpuFeatures == NULL) {
        return 0;
    }
    // 调用setHwcap函数，解析cpuFeatures并设置hwcap的值
    setHwcap(cpuFeatures, hwcap);
    // 将hwcap的值加到hwcap2
    *hwcap2 |= *hwcap;
    // 如果cpuFeatures中包含"aes"项，则设置NPY__HWCAP2_AES到hwcap2
    *hwcap2 |= has_list_item(cpuFeatures, "aes") ? NPY__HWCAP2_AES : 0;
    // 如果cpuFeatures中包含"pmull"项，则设置NPY__HWCAP2_PMULL到hwcap2
    *hwcap2 |= has_list_item(cpuFeatures, "pmull") ? NPY__HWCAP2_PMULL : 0;
    // 如果cpuFeatures中包含"sha1"项，则设置NPY__HWCAP2_SHA1到hwcap2
    *hwcap2 |= has_list_item(cpuFeatures, "sha1") ? NPY__HWCAP2_SHA1 : 0;
    // 如果cpuFeatures中包含"sha2"项，则设置NPY__HWCAP2_SHA2到hwcap2
    *hwcap2 |= has_list_item(cpuFeatures, "sha2") ? NPY__HWCAP2_SHA2 : 0;
    // 如果cpuFeatures中包含"crc32"项，则设置NPY__HWCAP2_CRC32到hwcap2
    *hwcap2 |= has_list_item(cpuFeatures, "crc32") ? NPY__HWCAP2_CRC32 : 0;
    // 返回1，表示函数执行成功
    return 1;
}
#endif  /* NUMPY_CORE_SRC_COMMON_NPY_CPUINFO_PARSER_H_ */
```