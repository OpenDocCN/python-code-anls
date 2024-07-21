# `.\pytorch\third_party\miniz-2.1.0\miniz.c`

```
/* --------------------------------------------------------------
 * 以下是基于 zlib 风格的 API
 * --------------------------------------------------------------
 */

// 计算 Adler-32 校验和
mz_ulong mz_adler32(mz_ulong adler, const unsigned char *ptr, size_t buf_len)
{
    // 初始化校验和变量 s1 和 s2
    mz_uint32 i, s1 = (mz_uint32)(adler & 0xffff), s2 = (mz_uint32)(adler >> 16);
    // 计算每个数据块的长度，最大为 5552
    size_t block_len = buf_len % 5552;
    // 如果指针为空，返回初始校验和值 MZ_ADLER32_INIT
    if (!ptr)
        return MZ_ADLER32_INIT;
    // 循环处理每个数据块
    while (buf_len)
    {
        // 对当前数据块中的每个 8 字节进行累加校验和计算
        for (i = 0; i + 7 < block_len; i += 8, ptr += 8)
        {
            s1 += ptr[0], s2 += s1;
            s1 += ptr[1], s2 += s1;
            s1 += ptr[2], s2 += s1;
            s1 += ptr[3], s2 += s1;
            s1 += ptr[4], s2 += s1;
            s1 += ptr[5], s2 += s1;
            s1 += ptr[6], s2 += s1;
            s1 += ptr[7], s2 += s1;
        }
        // 处理剩余不足 8 字节的数据
        for (; i < block_len; ++i)
            s1 += *ptr++, s2 += s1;
        // 对 s1 和 s2 进行模运算，保证在 65521 之内
        s1 %= 65521U, s2 %= 65521U;
        // 减去当前处理的数据块长度，并设定下一个数据块长度为 5552
        buf_len -= block_len;
        block_len = 5552;
    }
    // 返回计算得到的 Adler-32 校验和
    return (s2 << 16) + s1;
}
    {
        // 静态常量数组，用于计算 CRC32 校验值的快速查表
        static const mz_uint32 s_crc32[16] = { 0, 0x1db71064, 0x3b6e20c8, 0x26d930ac, 0x76dc4190, 0x6b6b51f4, 0x4db26158, 0x5005713c,
                                               0xedb88320, 0xf00f9344, 0xd6d6a3e8, 0xcb61b38c, 0x9b64c2b0, 0x86d3d2d4, 0xa00ae278, 0xbdbdf21c };
        // 将传入的 CRC 值转换为 mz_uint32 类型
        mz_uint32 crcu32 = (mz_uint32)crc;
        // 如果指针 ptr 为空，则返回初始 CRC32 值
        if (!ptr)
            return MZ_CRC32_INIT;
        // 对 CRC32 的当前值取反
        crcu32 = ~crcu32;
        // 遍历输入的数据缓冲区，计算 CRC32 值
        while (buf_len--)
        {
            // 读取当前指针指向的字节
            mz_uint8 b = *ptr++;
            // 使用快速查表计算新的 CRC32 值
            crcu32 = (crcu32 >> 4) ^ s_crc32[(crcu32 & 0xF) ^ (b & 0xF)];
            crcu32 = (crcu32 >> 4) ^ s_crc32[(crcu32 & 0xF) ^ (b >> 4)];
        }
        // 返回最终计算出的 CRC32 校验值的反码
        return ~crcu32;
    }
#elif defined(USE_EXTERNAL_MZCRC)
/* 如果定义了USE_EXTERNAL_CRC，外部模块将导出mz_crc32()符号供我们使用，例如SSE加速版本。
 * 根据实现的不同，可能需要对输入/输出的CRC值进行反转。
 */
mz_ulong mz_crc32(mz_ulong crc, const mz_uint8 *ptr, size_t buf_len);
#else
/* 更快，但占用更大的CPU缓存。
 */
mz_ulong mz_crc32(mz_ulong crc, const mz_uint8 *ptr, size_t buf_len)
{
    // 初始化CRC32值，应用按位取反
    mz_uint32 crc32 = (mz_uint32)crc ^ 0xFFFFFFFF;
    // 将输入指针转换为字节指针
    const mz_uint8 *pByte_buf = (const mz_uint8 *)ptr;

    // 循环处理缓冲区，每次处理4字节
    while (buf_len >= 4)
    {
        crc32 = (crc32 >> 8) ^ s_crc_table[(crc32 ^ pByte_buf[0]) & 0xFF];
        crc32 = (crc32 >> 8) ^ s_crc_table[(crc32 ^ pByte_buf[1]) & 0xFF];
        crc32 = (crc32 >> 8) ^ s_crc_table[(crc32 ^ pByte_buf[2]) & 0xFF];
        crc32 = (crc32 >> 8) ^ s_crc_table[(crc32 ^ pByte_buf[3]) & 0xFF];
        pByte_buf += 4;
        buf_len -= 4;
    }

    // 处理剩余的字节
    while (buf_len)
    {
        crc32 = (crc32 >> 8) ^ s_crc_table[(crc32 ^ pByte_buf[0]) & 0xFF];
        ++pByte_buf;
        --buf_len;
    }

    // 返回反转后的CRC32值
    return ~crc32;
}
#endif

void mz_free(void *p)
{
    // 调用MZ_FREE释放内存
    MZ_FREE(p);
}

void *miniz_def_alloc_func(void *opaque, size_t items, size_t size)
{
    // 忽略未使用的参数，分配内存并返回
    (void)opaque, (void)items, (void)size;
    return MZ_MALLOC(items * size);
}

void miniz_def_free_func(void *opaque, void *address)
{
    // 忽略未使用的参数，释放内存
    (void)opaque, (void)address;
    MZ_FREE(address);
}

void *miniz_def_realloc_func(void *opaque, void *address, size_t items, size_t size)
{
    // 忽略未使用的参数，重新分配内存并返回
    (void)opaque, (void)address, (void)items, (void)size;
    return MZ_REALLOC(address, items * size);
}

const char *mz_version(void)
{
    // 返回miniz库的版本信息
    return MZ_VERSION;
}

#ifndef MINIZ_NO_ZLIB_APIS

int mz_deflateInit(mz_streamp pStream, int level)
{
    // 调用mz_deflateInit2初始化压缩流，使用默认策略
    return mz_deflateInit2(pStream, level, MZ_DEFLATED, MZ_DEFAULT_WINDOW_BITS, 9, MZ_DEFAULT_STRATEGY);
}

int mz_deflateInit2(mz_streamp pStream, int level, int method, int window_bits, int mem_level, int strategy)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    tdefl_compressor *pComp;
    // 根据参数创建压缩标志
    mz_uint comp_flags = TDEFL_COMPUTE_ADLER32 | tdefl_create_comp_flags_from_zip_params(level, window_bits, strategy);

    // 如果压缩流为空，返回流错误
    if (!pStream)
        return MZ_STREAM_ERROR;
    // 检查方法、内存级别和窗口位是否有效
    if ((method != MZ_DEFLATED) || ((mem_level < 1) || (mem_level > 9)) || ((window_bits != MZ_DEFAULT_WINDOW_BITS) && (-window_bits != MZ_DEFAULT_WINDOW_BITS)))
        return MZ_PARAM_ERROR;

    // 初始化压缩流的各个字段
    pStream->data_type = 0;
    pStream->adler = MZ_ADLER32_INIT;
    pStream->msg = NULL;
    pStream->reserved = 0;
    pStream->total_in = 0;
    pStream->total_out = 0;
    // 如果分配函数为空，设置为默认分配函数
    if (!pStream->zalloc)
        pStream->zalloc = miniz_def_alloc_func;
    // 如果释放函数为空，设置为默认释放函数
    if (!pStream->zfree)
        pStream->zfree = miniz_def_free_func;

    // 分配压缩器结构内存
    pComp = (tdefl_compressor *)pStream->zalloc(pStream->opaque, 1, sizeof(tdefl_compressor));
    // 如果分配失败，返回内存错误
    if (!pComp)
        return MZ_MEM_ERROR;

    // 设置压缩流的状态为分配的压缩器结构
    pStream->state = (struct mz_internal_state *)pComp;
    # 如果压缩初始化函数返回非 TDEFL_STATUS_OKAY，则表示初始化失败
    if (tdefl_init(pComp, NULL, NULL, comp_flags) != TDEFL_STATUS_OKAY)
    {
        # 如果初始化失败，结束当前的压缩流操作
        mz_deflateEnd(pStream);
        # 返回参数错误代码
        return MZ_PARAM_ERROR;
    }

    # 如果初始化成功，则返回操作成功的状态代码
    return MZ_OK;
}

// 重置 zlib 压缩流状态
int mz_deflateReset(mz_streamp pStream)
{
    // 如果流对象为空，或者其状态为空，或者未分配内存函数或释放内存函数，则返回流错误
    if ((!pStream) || (!pStream->state) || (!pStream->zalloc) || (!pStream->zfree))
        return MZ_STREAM_ERROR;
    
    // 将输入和输出的总字节数归零
    pStream->total_in = pStream->total_out = 0;
    // 初始化压缩器状态
    tdefl_init((tdefl_compressor *)pStream->state, NULL, NULL, ((tdefl_compressor *)pStream->state)->m_flags);
    return MZ_OK;
}

// 执行 zlib 压缩操作
int mz_deflate(mz_streamp pStream, int flush)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t in_bytes, out_bytes;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_ulong orig_total_in, orig_total_out;
    int mz_status = MZ_OK;

    // 如果流对象为空，或者其状态为空，或者flush参数不在有效范围内，或者输出缓冲区指针为空，则返回流错误
    if ((!pStream) || (!pStream->state) || (flush < 0) || (flush > MZ_FINISH) || (!pStream->next_out))
        return MZ_STREAM_ERROR;
    
    // 如果输出缓冲区没有可用空间，则返回缓冲区错误
    if (!pStream->avail_out)
        return MZ_BUF_ERROR;

    // 如果flush参数为MZ_PARTIAL_FLUSH，则转换为MZ_SYNC_FLUSH
    if (flush == MZ_PARTIAL_FLUSH)
        flush = MZ_SYNC_FLUSH;

    // 如果上一次压缩操作已经完成，则根据flush参数返回相应的状态
    if (((tdefl_compressor *)pStream->state)->m_prev_return_status == TDEFL_STATUS_DONE)
        return (flush == MZ_FINISH) ? MZ_STREAM_END : MZ_BUF_ERROR;

    // 保存压缩前的输入和输出总字节数
    orig_total_in = pStream->total_in;
    orig_total_out = pStream->total_out;

    // 循环执行压缩操作
    for (;;)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        tdefl_status defl_status;
        in_bytes = pStream->avail_in;
        out_bytes = pStream->avail_out;

        // 调用tdefl_compress函数执行压缩操作
        defl_status = tdefl_compress((tdefl_compressor *)pStream->state, pStream->next_in, &in_bytes, pStream->next_out, &out_bytes, (tdefl_flush)flush);
        
        // 更新输入和输出缓冲区指针及可用字节数
        pStream->next_in += (mz_uint)in_bytes;
        pStream->avail_in -= (mz_uint)in_bytes;
        pStream->total_in += (mz_uint)in_bytes;
        pStream->adler = tdefl_get_adler32((tdefl_compressor *)pStream->state);

        pStream->next_out += (mz_uint)out_bytes;
        pStream->avail_out -= (mz_uint)out_bytes;
        pStream->total_out += (mz_uint)out_bytes;

        // 根据压缩状态判断操作结果
        if (defl_status < 0)
        {
            mz_status = MZ_STREAM_ERROR;
            break;
        }
        else if (defl_status == TDEFL_STATUS_DONE)
        {
            mz_status = MZ_STREAM_END;
            break;
        }
        else if (!pStream->avail_out)
            break;
        else if ((!pStream->avail_in) && (flush != MZ_FINISH))
        {
            // 如果没有输入数据可用，并且不是在完成压缩的情况下，则判断是否可以进行前进，否则返回缓冲区错误
            if ((flush) || (pStream->total_in != orig_total_in) || (pStream->total_out != orig_total_out))
                break;
            return MZ_BUF_ERROR; /* Can't make forward progress without some input.
 */
        }
    }
    return mz_status;
}

// 结束 zlib 压缩流
int mz_deflateEnd(mz_streamp pStream)
{
    // 如果流对象为空，则返回流错误
    if (!pStream)
        return MZ_STREAM_ERROR;
    
    // 如果流对象状态不为空，则释放其状态并置空
    if (pStream->state)
    {
        pStream->zfree(pStream->opaque, pStream->state);
        pStream->state = NULL;
    }
    return MZ_OK;
}

// 计算 zlib 压缩后数据的上限长度
mz_ulong mz_deflateBound(mz_streamp pStream, mz_ulong source_len)
{
    (void)pStream;
    /* This is really over conservative. (And lame, but it's actually pretty tricky to compute a true upper bound given the way tdefl's blocking works.) */
    // 这个值是非常保守的估计（并且有些愚蠢），实际上，根据tdefl的阻塞方式计算真正的上限是相当棘手的。
    # 计算并返回最大值，根据以下两个表达式中的结果选择较大的那个：
    # 1. 128 + (source_len * 110) / 100
    # 2. 128 + source_len + ((source_len / (31 * 1024)) + 1) * 5
    return MZ_MAX(128 + (source_len * 110) / 100, 128 + source_len + ((source_len / (31 * 1024)) + 1) * 5);
}

// 压缩数据到指定目标，使用给定的压缩级别
int mz_compress2(unsigned char *pDest, mz_ulong *pDest_len, const unsigned char *pSource, mz_ulong source_len, int level)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int status;
    mz_stream stream;
    memset(&stream, 0, sizeof(stream));

    /* In case mz_ulong is 64-bits (argh I hate longs). */
    // 检查源数据和目标数据长度，如果超过32位无法处理，返回参数错误
    if ((source_len | *pDest_len) > 0xFFFFFFFFU)
        return MZ_PARAM_ERROR;

    // 设置流对象的输入和输出参数
    stream.next_in = pSource;
    stream.avail_in = (mz_uint32)source_len;
    stream.next_out = pDest;
    stream.avail_out = (mz_uint32)*pDest_len;

    // 初始化压缩流对象
    status = mz_deflateInit(&stream, level);
    if (status != MZ_OK)
        return status;

    // 执行压缩操作，直到结束
    status = mz_deflate(&stream, MZ_FINISH);
    if (status != MZ_STREAM_END)
    {
        mz_deflateEnd(&stream);
        // 如果压缩未正常结束，根据状态返回相应错误或成功
        return (status == MZ_OK) ? MZ_BUF_ERROR : status;
    }

    // 更新输出数据的长度
    *pDest_len = stream.total_out;
    // 结束压缩流对象
    return mz_deflateEnd(&stream);
}

// 使用默认压缩级别进行压缩
int mz_compress(unsigned char *pDest, mz_ulong *pDest_len, const unsigned char *pSource, mz_ulong source_len)
{
    return mz_compress2(pDest, pDest_len, pSource, source_len, MZ_DEFAULT_COMPRESSION);
}

// 计算给定数据的最大可能压缩长度
mz_ulong mz_compressBound(mz_ulong source_len)
{
    return mz_deflateBound(NULL, source_len);
}

// 解压状态结构体
typedef struct
{
    tinfl_decompressor m_decomp;
    mz_uint m_dict_ofs, m_dict_avail, m_first_call, m_has_flushed;
    int m_window_bits;
    mz_uint8 m_dict[TINFL_LZ_DICT_SIZE];
    tinfl_status m_last_status;
} inflate_state;

// 初始化解压流对象
int mz_inflateInit2(mz_streamp pStream, int window_bits)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    inflate_state *pDecomp;
    if (!pStream)
        return MZ_STREAM_ERROR;
    if ((window_bits != MZ_DEFAULT_WINDOW_BITS) && (-window_bits != MZ_DEFAULT_WINDOW_BITS))
        return MZ_PARAM_ERROR;

    // 初始化流对象的基本参数
    pStream->data_type = 0;
    pStream->adler = 0;
    pStream->msg = NULL;
    pStream->total_in = 0;
    pStream->total_out = 0;
    pStream->reserved = 0;

    // 设置内存分配和释放函数，如果未设置，则使用默认函数
    if (!pStream->zalloc)
        pStream->zalloc = miniz_def_alloc_func;
    if (!pStream->zfree)
        pStream->zfree = miniz_def_free_func;

    // 分配并初始化解压状态结构体
    pDecomp = (inflate_state *)pStream->zalloc(pStream->opaque, 1, sizeof(inflate_state));
    if (!pDecomp)
        return MZ_MEM_ERROR;

    // 将解压状态绑定到流对象
    pStream->state = (struct mz_internal_state *)pDecomp;

    // 初始化解压器
    tinfl_init(&pDecomp->m_decomp);
    pDecomp->m_dict_ofs = 0;
    pDecomp->m_dict_avail = 0;
    pDecomp->m_last_status = TINFL_STATUS_NEEDS_MORE_INPUT;
    pDecomp->m_first_call = 1;
    pDecomp->m_has_flushed = 0;
    pDecomp->m_window_bits = window_bits;

    return MZ_OK;
}

// 使用默认窗口位数初始化解压流对象
int mz_inflateInit(mz_streamp pStream)
{
    return mz_inflateInit2(pStream, MZ_DEFAULT_WINDOW_BITS);
}

// 重置解压流对象的状态
int mz_inflateReset(mz_streamp pStream)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    inflate_state *pDecomp;
    if (!pStream)
        return MZ_STREAM_ERROR;

    // 重置流对象的基本参数
    pStream->data_type = 0;
    pStream->adler = 0;
    pStream->msg = NULL;
    pStream->total_in = 0;
    pStream->total_out = 0;
    pStream->reserved = 0;
    # 将指针 pStream 强制类型转换为 inflate_state 结构体指针，并赋给 pDecomp
    pDecomp = (inflate_state *)pStream->state;

    # 使用 tinfl_init 函数初始化解压状态机 pDecomp 中的解压器
    tinfl_init(&pDecomp->m_decomp);

    # 将解压状态机中的字典偏移设置为 0
    pDecomp->m_dict_ofs = 0;

    # 将解压状态机中的字典可用数据长度设置为 0
    pDecomp->m_dict_avail = 0;

    # 将解压状态机中的最后状态设置为需要更多输入数据
    pDecomp->m_last_status = TINFL_STATUS_NEEDS_MORE_INPUT;

    # 将解压状态机的第一次调用标志设置为真
    pDecomp->m_first_call = 1;

    # 将解压状态机的刷新标志设置为未刷新
    pDecomp->m_has_flushed = 0;

    # 设置解压状态机中的窗口位数，但此行代码被注释掉，没有实际作用
    /* pDecomp->m_window_bits = window_bits */;

    # 返回解压状态机初始化成功的状态码 MZ_OK
    return MZ_OK;
}

// 函数 mz_inflate 的实现，用于解压缩数据流
int mz_inflate(mz_streamp pStream, int flush)
{
    // 指向解压缩状态结构体的指针
    inflate_state *pState;
    // 用于标识调用次数的变量
    mz_uint n, first_call, decomp_flags = TINFL_FLAG_COMPUTE_ADLER32;
    // 输入字节数、输出字节数、原始可用输入字节数
    size_t in_bytes, out_bytes, orig_avail_in;
    // 解压缩状态
    tinfl_status status;

    // 检查解压缩流和状态是否有效
    if ((!pStream) || (!pStream->state))
        return MZ_STREAM_ERROR;
    // 如果 flush 类型为 MZ_PARTIAL_FLUSH，则转换为 MZ_SYNC_FLUSH
    if (flush == MZ_PARTIAL_FLUSH)
        flush = MZ_SYNC_FLUSH;
    // 如果 flush 存在且不是 MZ_SYNC_FLUSH 或 MZ_FINISH，则返回错误
    if ((flush) && (flush != MZ_SYNC_FLUSH) && (flush != MZ_FINISH))
        return MZ_STREAM_ERROR;

    // 获取解压缩状态
    pState = (inflate_state *)pStream->state;
    // 如果指定了 m_window_bits，则设置解压缩标志
    if (pState->m_window_bits > 0)
        decomp_flags |= TINFL_FLAG_PARSE_ZLIB_HEADER;
    // 记录原始可用输入字节数
    orig_avail_in = pStream->avail_in;

    // 获取首次调用标志
    first_call = pState->m_first_call;
    pState->m_first_call = 0;
    // 如果上一个状态为负，则返回数据错误
    if (pState->m_last_status < 0)
        return MZ_DATA_ERROR;

    // 如果已经 flush 过，并且不是 MZ_FINISH 操作，则返回流错误
    if (pState->m_has_flushed && (flush != MZ_FINISH))
        return MZ_STREAM_ERROR;
    // 标记已经进行了 flush 操作
    pState->m_has_flushed |= (flush == MZ_FINISH);

    // 如果 flush 类型为 MZ_FINISH，并且是首次调用
    if ((flush == MZ_FINISH) && (first_call))
    {
        /* MZ_FINISH 在第一次调用时意味着输入和输出缓冲区足够大，可以容纳整个压缩/解压缩文件。 */
        // 设置非环绕输出缓冲区标志
        decomp_flags |= TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
        // 记录输入和输出字节数
        in_bytes = pStream->avail_in;
        out_bytes = pStream->avail_out;
        // 调用解压缩函数，解压数据
        status = tinfl_decompress(&pState->m_decomp, pStream->next_in, &in_bytes, pStream->next_out, pStream->next_out, &out_bytes, decomp_flags);
        // 记录最后的解压缩状态
        pState->m_last_status = status;
        // 更新输入指针和剩余输入字节数
        pStream->next_in += (mz_uint)in_bytes;
        pStream->avail_in -= (mz_uint)in_bytes;
        pStream->total_in += (mz_uint)in_bytes;
        // 更新 Adler-32 校验和
        pStream->adler = tinfl_get_adler32(&pState->m_decomp);
        // 更新输出指针和剩余输出字节数
        pStream->next_out += (mz_uint)out_bytes;
        pStream->avail_out -= (mz_uint)out_bytes;
        pStream->total_out += (mz_uint)out_bytes;

        // 如果解压缩状态小于 0，则返回数据错误
        if (status < 0)
            return MZ_DATA_ERROR;
        // 如果解压缩未完成，则标记最后状态为失败，并返回缓冲区错误
        else if (status != TINFL_STATUS_DONE)
        {
            pState->m_last_status = TINFL_STATUS_FAILED;
            return MZ_BUF_ERROR;
        }
        // 返回流结束
        return MZ_STREAM_END;
    }
    /* 如果 flush 不是 MZ_FINISH，则必须假设还有更多的输入。 */
    // 如果 flush 不是 MZ_FINISH，则设置有更多输入标志
    if (flush != MZ_FINISH)
        decomp_flags |= TINFL_FLAG_HAS_MORE_INPUT;

    // 如果字典可用
    if (pState->m_dict_avail)
    {
        // 计算需要复制的字节数
        n = MZ_MIN(pState->m_dict_avail, pStream->avail_out);
        // 复制字典数据到输出缓冲区
        memcpy(pStream->next_out, pState->m_dict + pState->m_dict_ofs, n);
        // 更新输出指针和剩余输出字节数
        pStream->next_out += n;
        pStream->avail_out -= n;
        pStream->total_out += n;
        // 更新字典可用字节数
        pState->m_dict_avail -= n;
        pState->m_dict_ofs = (pState->m_dict_ofs + n) & (TINFL_LZ_DICT_SIZE - 1);
        // 如果最后状态为完成且字典可用为 0，则返回流结束，否则返回 OK
        return ((pState->m_last_status == TINFL_STATUS_DONE) && (!pState->m_dict_avail)) ? MZ_STREAM_END : MZ_OK;
    }

    // 进入无限循环，等待输入继续解压缩
    for (;;)
    {
        // 记录输入流中的可用字节数
        in_bytes = pStream->avail_in;
        // 计算输出字节数，即输出字典中剩余的空间大小
        out_bytes = TINFL_LZ_DICT_SIZE - pState->m_dict_ofs;

        // 调用解压函数进行解压操作，更新输入和输出字节数，以及状态
        status = tinfl_decompress(&pState->m_decomp, pStream->next_in, &in_bytes, pState->m_dict, pState->m_dict + pState->m_dict_ofs, &out_bytes, decomp_flags);
        // 更新最后的状态
        pState->m_last_status = status;

        // 更新输入流的指针和可用字节数
        pStream->next_in += (mz_uint)in_bytes;
        pStream->avail_in -= (mz_uint)in_bytes;
        pStream->total_in += (mz_uint)in_bytes;
        // 计算并更新 Adler-32 校验和
        pStream->adler = tinfl_get_adler32(&pState->m_decomp);

        // 更新输出字典中可用的字节数
        pState->m_dict_avail = (mz_uint)out_bytes;

        // 计算实际写入输出流的字节数
        n = MZ_MIN(pState->m_dict_avail, pStream->avail_out);
        // 将输出字典中的数据复制到输出流中
        memcpy(pStream->next_out, pState->m_dict + pState->m_dict_ofs, n);
        pStream->next_out += n;
        pStream->avail_out -= n;
        pStream->total_out += n;
        pState->m_dict_avail -= n;
        // 更新字典偏移量，确保在字典大小范围内循环
        pState->m_dict_ofs = (pState->m_dict_ofs + n) & (TINFL_LZ_DICT_SIZE - 1);

        // 检查解压状态，若出现错误则返回数据错误
        if (status < 0)
            return MZ_DATA_ERROR; /* Stream is corrupted (there could be some uncompressed data left in the output dictionary - oh well). */
        // 若需要更多输入数据但无输入可用，则返回缓冲区错误
        else if ((status == TINFL_STATUS_NEEDS_MORE_INPUT) && (!orig_avail_in))
            return MZ_BUF_ERROR; /* Signal caller that we can't make forward progress without supplying more input or by setting flush to MZ_FINISH. */
        // 若为 MZ_FINISH 模式，则根据状态返回相应结果
        else if (flush == MZ_FINISH)
        {
            /* The output buffer MUST be large to hold the remaining uncompressed data when flush==MZ_FINISH. */
            // 若解压完成且输出字典仍有数据，则返回缓冲区错误；否则返回流结束
            if (status == TINFL_STATUS_DONE)
                return pState->m_dict_avail ? MZ_BUF_ERROR : MZ_STREAM_END;
            // 若状态为 TINFL_STATUS_HAS_MORE_OUTPUT，但输出流空间不足，则返回缓冲区错误
            /* status here must be TINFL_STATUS_HAS_MORE_OUTPUT, which means there's at least 1 more byte on the way. If there's no more room left in the output buffer then something is wrong. */
            else if (!pStream->avail_out)
                return MZ_BUF_ERROR;
        }
        // 若解压完成或无输入输出可用，或者字典中仍有数据，则退出循环
        else if ((status == TINFL_STATUS_DONE) || (!pStream->avail_in) || (!pStream->avail_out) || (pState->m_dict_avail))
            break;
    }

    // 返回解压的最终状态码
    return ((status == TINFL_STATUS_DONE) && (!pState->m_dict_avail)) ? MZ_STREAM_END : MZ_OK;
}

// 函数 mz_inflateEnd 的定义，用于结束解压缩流操作
int mz_inflateEnd(mz_streamp pStream)
{
    // 如果传入的流对象为空指针，返回流错误代码
    if (!pStream)
        return MZ_STREAM_ERROR;
    // 如果流对象的状态存在
    if (pStream->state)
    {
        // 调用流对象的释放函数释放状态数据，并将状态指针置空
        pStream->zfree(pStream->opaque, pStream->state);
        pStream->state = NULL;
    }
    // 返回操作成功代码
    return MZ_OK;
}

// 函数 mz_uncompress 的定义，用于解压缩数据
int mz_uncompress(unsigned char *pDest, mz_ulong *pDest_len, const unsigned char *pSource, mz_ulong source_len)
{
    // 创建解压缩流对象
    mz_stream stream;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int status;
    // 将解压缩流对象的内存清零
    memset(&stream, 0, sizeof(stream));

    /* In case mz_ulong is 64-bits (argh I hate longs). */
    // 检查输入数据长度是否超过限制，如果超过返回参数错误代码
    if ((source_len | *pDest_len) > 0xFFFFFFFFU)
        return MZ_PARAM_ERROR;

    // 设置解压缩流对象的输入源和输入数据长度
    stream.next_in = pSource;
    stream.avail_in = (mz_uint32)source_len;
    // 设置解压缩流对象的输出目标和输出数据长度
    stream.next_out = pDest;
    stream.avail_out = (mz_uint32)*pDest_len;

    // 初始化解压缩流对象，返回初始化状态
    status = mz_inflateInit(&stream);
    if (status != MZ_OK)
        return status;

    // 执行解压缩操作，直到数据流结束
    status = mz_inflate(&stream, MZ_FINISH);
    // 如果解压缩未完成，执行结束解压缩操作并返回相应状态
    if (status != MZ_STREAM_END)
    {
        mz_inflateEnd(&stream);
        // 如果是缓冲区错误且没有剩余输入数据，则返回数据错误，否则返回当前状态
        return ((status == MZ_BUF_ERROR) && (!stream.avail_in)) ? MZ_DATA_ERROR : status;
    }
    // 将解压缩后的输出数据长度赋值给传入的目标数据长度
    *pDest_len = stream.total_out;

    // 结束解压缩操作并返回操作状态
    return mz_inflateEnd(&stream);
}

// 函数 mz_error 的定义，根据错误码返回相应的错误描述字符串
const char *mz_error(int err)
{
    // 静态数组，存储错误码及对应的错误描述字符串
    static struct
    {
        int m_err;
        const char *m_pDesc;
    } s_error_descs[] =
        {
          { MZ_OK, "" }, { MZ_STREAM_END, "stream end" }, { MZ_NEED_DICT, "need dictionary" }, { MZ_ERRNO, "file error" }, { MZ_STREAM_ERROR, "stream error" }, { MZ_DATA_ERROR, "data error" }, { MZ_MEM_ERROR, "out of memory" }, { MZ_BUF_ERROR, "buf error" }, { MZ_VERSION_ERROR, "version error" }, { MZ_PARAM_ERROR, "parameter error" }
        };
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint i;
    // 遍历静态数组，根据传入的错误码返回相应的错误描述字符串
    for (i = 0; i < sizeof(s_error_descs) / sizeof(s_error_descs[0]); ++i)
        if (s_error_descs[i].m_err == err)
            return s_error_descs[i].m_pDesc;
    // 若未找到匹配的错误码，则返回空指针
    return NULL;
}

#endif /*MINIZ_NO_ZLIB_APIS */

#ifdef __cplusplus
}
#endif
/*
  This section contains legal and licensing information about the software.
  It includes declarations about the software being released into the public domain,
  and permissions granted to users for copying, modifying, and distributing the software.
*/

/**************************************************************************
 *
 * Copyright 2013-2014 RAD Game Tools and Valve Software
 * Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC
 * All Rights Reserved.
 *
 * This part identifies the specific copyright holders and the years during which
 * the copyright applies.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * Here the terms of permission for using and distributing the software are outlined,
 * including the freedom to use, modify, and distribute the software for any purpose,
 * whether commercial or non-commercial, under specified conditions.
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * These lines state that the copyright and permission notices must be retained
 * in all copies or significant parts of the software that are distributed.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Here the disclaimer of warranties and liabilities is provided,
 * stating that the software is provided as-is without warranties of any kind.
 *
 **************************************************************************/
 
#ifdef __cplusplus
extern "C" {
#endif

/*
  The following section defines low-level compression tables used independently
  from decompression APIs. The purpose of this section is to declare static tables
  for faster initialization and to ensure thread safety.
*/

static const mz_uint16 s_tdefl_len_sym[256] =
    # 创建一个包含大量整数的集合，这些整数表示Huffman编码中可能的编码长度。
    {
      257, 258, 259, 260, 261, 262, 263, 264, 265, 265, 266, 266, 267, 267, 268, 268, 269, 269, 269, 269, 
      270, 270, 270, 270, 271, 271, 271, 271, 272, 272, 272, 272, 
      273, 273, 273, 273, 273, 273, 273, 273, 274, 274, 274, 274, 
      274, 274, 274, 274, 275, 275, 275, 275, 275, 275, 275, 275, 
      276, 276, 276, 276, 276, 276, 276, 276, 277, 277, 277, 277, 
      277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 
      278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 
      278, 278, 278, 278, 279, 279, 279, 279, 279, 279, 279, 279, 
      279, 279, 279, 279, 279, 279, 279, 279, 280, 280, 280, 280, 
      280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 
      281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 
      281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 
      281, 281, 281, 281, 281, 281, 281, 281, 282, 282, 282, 282, 
      282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 
      282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 
      282, 282, 282, 282, 283, 283, 283, 283, 283, 283, 283, 283, 
      283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 
      283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 
      284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 
      284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 
      284, 284, 284, 284, 284, 284, 284, 285
    }
// 长度值的额外位数表，用于表示长度编码中每个长度码字后面的额外位数
static const mz_uint8 s_tdefl_len_extra[256] =
    {
      0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0
    };

// 小距离符号表，用于表示距离编码中的小距离
static const mz_uint8 s_tdefl_small_dist_sym[512] =
    {
      0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11,
      11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13,
      13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
      14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
      14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
      15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
      16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
    };



# 这行代码结束了一个对象的定义或一个代码块的声明，通常与开始定义或声明的代码相对应。
// 定义静态常量数组，用于表示小距离码的额外位数，索引范围为0到511
static const mz_uint8 s_tdefl_small_dist_extra[512] =
    {
      0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
      6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
      6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7, 7, 7, 7, 7
    };

// 定义静态常量数组，用于表示大距离码的符号，索引范围为0到127
static const mz_uint8 s_tdefl_large_dist_sym[128] =
    {
      0, 0, 18, 19, 20, 20, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
      26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
      28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29
    };

// 定义静态常量数组，用于表示大距离码的额外位数，索引范围为0到127
static const mz_uint8 s_tdefl_large_dist_extra[128] =
    {
      0, 0, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
      12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
      13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13
    };

/* 根据16位键m_key
// 对符号频率进行基数排序以优化编码效率
static tdefl_sym_freq *tdefl_radix_sort_syms(mz_uint num_syms, tdefl_sym_freq *pSyms0, tdefl_sym_freq *pSyms1)
{
    // 定义总共需要的排序轮数和位移量
    mz_uint32 total_passes = 2, pass_shift, pass, i, hist[256 * 2];
    // 当前和新的符号频率数组指针
    tdefl_sym_freq *pCur_syms = pSyms0, *pNew_syms = pSyms1;
    // 清空历史直方图数组
    MZ_CLEAR_OBJ(hist);
    // 统计每个符号频率的出现次数
    for (i = 0; i < num_syms; i++)
    {
        mz_uint freq = pSyms0[i].m_key;
        hist[freq & 0xFF]++;
        hist[256 + ((freq >> 8) & 0xFF)]++;
    }
    // 计算实际需要的排序轮数
    while ((total_passes > 1) && (num_syms == hist[(total_passes - 1) * 256]))
        total_passes--;
    // 根据每个排序轮的位移量进行基数排序
    for (pass_shift = 0, pass = 0; pass < total_passes; pass++, pass_shift += 8)
    {
        const mz_uint32 *pHist = &hist[pass << 8];
        mz_uint offsets[256], cur_ofs = 0;
        // 计算每个桶的起始偏移量
        for (i = 0; i < 256; i++)
        {
            offsets[i] = cur_ofs;
            cur_ofs += pHist[i];
        }
        // 根据当前位进行排序并存储到新的符号频率数组中
        for (i = 0; i < num_syms; i++)
            pNew_syms[offsets[(pCur_syms[i].m_key >> pass_shift) & 0xFF]++] = pCur_syms[i];
        // 交换当前和新的符号频率数组指针
        {
            tdefl_sym_freq *t = pCur_syms;
            pCur_syms = pNew_syms;
            pNew_syms = t;
        }
    }
    // 返回最终排序好的符号频率数组
    return pCur_syms;
}

/* tdefl_calculate_minimum_redundancy() originally written by: Alistair Moffat, alistair@cs.mu.oz.au, Jyrki Katajainen, jyrki@diku.dk, November 1996. */
// 计算最小冗余性的哈夫曼编码
static void tdefl_calculate_minimum_redundancy(tdefl_sym_freq *A, int n)
{
    // 定义变量：根、叶子、下一个符号、可用、已用和深度
    int root, leaf, next, avbl, used, dpth;
    // 如果符号数量为0则直接返回
    if (n == 0)
        return;
    // 如果只有一个符号，直接设定其码长为1
    else if (n == 1)
    {
        A[0].m_key = 1;
        return;
    }
    // 合并频率最小的两个符号直到剩下一个根节点
    A[0].m_key += A[1].m_key;
    root = 0;
    leaf = 2;
    for (next = 1; next < n - 1; next++)
    {
        if (leaf >= n || A[root].m_key < A[leaf].m_key)
        {
            A[next].m_key = A[root].m_key;
            A[root++].m_key = (mz_uint16)next;
        }
        else
            A[next].m_key = A[leaf++].m_key;
        if (leaf >= n || (root < next && A[root].m_key < A[leaf].m_key))
        {
            A[next].m_key = (mz_uint16)(A[next].m_key + A[root].m_key);
            A[root++].m_key = (mz_uint16)next;
        }
        else
            A[next].m_key = (mz_uint16)(A[next].m_key + A[leaf++].m_key);
    }
    // 设定叶子节点的码长
    A[n - 2].m_key = 0;
    for (next = n - 3; next >= 0; next--)
        A[next].m_key = A[A[next].m_key].m_key + 1;
    // 分配编码长度和深度
    avbl = 1;
    used = dpth = 0;
    root = n - 2;
    next = n - 1;
    while (avbl > 0)
    {
        while (root >= 0 && (int)A[root].m_key == dpth)
        {
            used++;
            root--;
        }
        while (avbl > used)
        {
            A[next--].m_key = (mz_uint16)(dpth);
            avbl--;
        }
        avbl = 2 * used;
        dpth++;
        used = 0;
    }
}

/* 限制规范哈夫曼编码表的最大码长。 */
enum
{
    TDEFL_MAX_SUPPORTED_HUFF_CODESIZE = 32
};
static void tdefl_huffman_enforce_max_code_size(int *pNum_codes, int code_list_len, int max_code_size)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int i;
    mz_uint32 total = 0;
    
    // 如果编码列表长度小于等于1，则直接返回，不需要处理
    if (code_list_len <= 1)
        return;

    // 累加大于最大编码长度的编码数量到最大编码长度位置
    for (i = max_code_size + 1; i <= TDEFL_MAX_SUPPORTED_HUFF_CODESIZE; i++)
        pNum_codes[max_code_size] += pNum_codes[i];

    // 计算总数值，用于后续的调整过程
    for (i = max_code_size; i > 0; i--)
        total += (((mz_uint32)pNum_codes[i]) << (max_code_size - i));

    // 调整编码长度，确保总数等于 (1UL << max_code_size)
    while (total != (1UL << max_code_size))
    {
        pNum_codes[max_code_size]--;
        for (i = max_code_size - 1; i > 0; i--)
            if (pNum_codes[i])
            {
                pNum_codes[i]--;
                pNum_codes[i + 1] += 2;
                break;
            }
        total--;
    }
}

static void tdefl_optimize_huffman_table(tdefl_compressor *d, int table_num, int table_len, int code_size_limit, int static_table)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int i, j, l, num_codes[1 + TDEFL_MAX_SUPPORTED_HUFF_CODESIZE];
    mz_uint next_code[TDEFL_MAX_SUPPORTED_HUFF_CODESIZE + 1];
    
    // 清空编码数量数组
    MZ_CLEAR_OBJ(num_codes);

    // 如果是静态表，统计每个编码长度出现的次数
    if (static_table)
    {
        for (i = 0; i < table_len; i++)
            num_codes[d->m_huff_code_sizes[table_num][i]]++;
    }
    else
    {
        // 非静态表情况下，准备用于排序和计算的符号频率数组
        tdefl_sym_freq syms0[TDEFL_MAX_HUFF_SYMBOLS], syms1[TDEFL_MAX_HUFF_SYMBOLS], *pSyms;
        int num_used_syms = 0;
        const mz_uint16 *pSym_count = &d->m_huff_count[table_num][0];

        // 构建符号频率数组
        for (i = 0; i < table_len; i++)
            if (pSym_count[i])
            {
                syms0[num_used_syms].m_key = (mz_uint16)pSym_count[i];
                syms0[num_used_syms++].m_sym_index = (mz_uint16)i;
            }

        // 对符号进行基数排序，并计算最小冗余度
        pSyms = tdefl_radix_sort_syms(num_used_syms, syms0, syms1);
        tdefl_calculate_minimum_redundancy(pSyms, num_used_syms);

        // 统计每个编码长度的符号数量
        for (i = 0; i < num_used_syms; i++)
            num_codes[pSyms[i].m_key]++;

        // 强制确保每个编码长度不超过限制
        tdefl_huffman_enforce_max_code_size(num_codes, num_used_syms, code_size_limit);

        // 清空压缩器对象中的编码长度和编码数组
        MZ_CLEAR_OBJ(d->m_huff_code_sizes[table_num]);
        MZ_CLEAR_OBJ(d->m_huff_codes[table_num]);

        // 根据编码长度为符号赋予 Huffman 编码
        for (i = 1, j = num_used_syms; i <= code_size_limit; i++)
            for (l = num_codes[i]; l > 0; l--)
                d->m_huff_code_sizes[table_num][pSyms[--j].m_sym_index] = (mz_uint8)(i);
    }

    // 计算下一个编码的起始值
    next_code[1] = 0;
    for (j = 0, i = 2; i <= code_size_limit; i++)
        next_code[i] = j = ((j + num_codes[i - 1]) << 1);

    // 为每个符号生成反转的 Huffman 编码
    for (i = 0; i < table_len; i++)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        mz_uint rev_code = 0, code, code_size;

        // 如果编码长度为0，则跳过
        if ((code_size = d->m_huff_code_sizes[table_num][i]) == 0)
            continue;

        // 计算反转的 Huffman 编码
        code = next_code[code_size]++;
        for (l = code_size; l > 0; l--, code >>= 1)
            rev_code = (rev_code << 1) | (code & 1);
        
        // 将反转后的编码存入压缩器对象中的 Huffman 编码数组
        d->m_huff_codes[table_num][i] = (mz_uint16)rev_code;
    }
}
# 定义宏 TDEFL_PUT_BITS(b, l)，用于将指定的比特位写入压缩数据流中
#define TDEFL_PUT_BITS(b, l)                                       \
    do                                                             \
    {                                                              \
        mz_uint bits = b;                                          \
        mz_uint len = l;                                           \
        MZ_ASSERT(bits <= ((1U << len) - 1U));                     \
        d->m_bit_buffer |= (bits << d->m_bits_in);                 \
        d->m_bits_in += len;                                       \
        while (d->m_bits_in >= 8)                                  \
        {                                                          \
            # 将当前的比特缓冲区内容写入输出缓冲区，直到缓冲区满或比特缓冲区内容小于8位
            if (d->m_pOutput_buf < d->m_pOutput_buf_end)           \
                *d->m_pOutput_buf++ = (mz_uint8)(d->m_bit_buffer); \
            # 将比特缓冲区向右移动8位，以便继续处理下一批比特数据
            d->m_bit_buffer >>= 8;                                 \
            # 调整比特缓冲区中的位数计数，以反映移动8位后的新状态
            d->m_bits_in -= 8;                                     \
        }                                                          \
    }                                                              \
    # 宏定义结束
    MZ_MACRO_END

# 定义宏 TDEFL_RLE_PREV_CODE_SIZE()，用于处理 RLE（Run-Length Encoding）压缩中的前一个码大小
#define TDEFL_RLE_PREV_CODE_SIZE() \
    # 此处应有功能实现，但当前版本未提供具体代码，可能在后续版本中实现
    \
    {
        // 如果存在重复计数（RLE），执行以下逻辑
        if (rle_repeat_count)
        {
            // 如果重复计数小于3，则处理
            if (rle_repeat_count < 3)
            {
                // 更新哈夫曼编码长度为prev_code_size的统计信息
                d->m_huff_count[2][prev_code_size] = (mz_uint16)(d->m_huff_count[2][prev_code_size] + rle_repeat_count);
                // 将prev_code_size重复rle_repeat_count次添加到压缩编码长度数组中
                while (rle_repeat_count--)
                    packed_code_sizes[num_packed_code_sizes++] = prev_code_size;
            }
            else
            {
                // 更新哈夫曼编码长度为16的统计信息
                d->m_huff_count[2][16] = (mz_uint16)(d->m_huff_count[2][16] + 1);
                // 将长度为16的编码标记添加到压缩编码长度数组中
                packed_code_sizes[num_packed_code_sizes++] = 16;
                // 将rle_repeat_count - 3作为字节添加到压缩编码长度数组中
                packed_code_sizes[num_packed_code_sizes++] = (mz_uint8)(rle_repeat_count - 3);
            }
            // 重置重复计数为0，准备处理下一个编码长度
            rle_repeat_count = 0;
        }
    }
// 定义宏 TDEFL_RLE_ZERO_CODE_SIZE()
#define TDEFL_RLE_ZERO_CODE_SIZE()                                                         \
    {                                                                                      \
        if (rle_z_count)                                                                   \  // 如果存在需要压缩的零码长度
        {                                                                                  \
            if (rle_z_count < 3)                                                           \  // 如果零码长度小于3
            {                                                                              \
                d->m_huff_count[2][0] = (mz_uint16)(d->m_huff_count[2][0] + rle_z_count);  \  // 增加零码计数
                while (rle_z_count--)                                                      \  // 将 rle_z_count 个零码添加到 packed_code_sizes 中
                    packed_code_sizes[num_packed_code_sizes++] = 0;                        \
            }                                                                              \
            else if (rle_z_count <= 10)                                                    \  // 如果零码长度在3到10之间
            {                                                                              \
                d->m_huff_count[2][17] = (mz_uint16)(d->m_huff_count[2][17] + 1);          \  // 增加码长度为17的计数
                packed_code_sizes[num_packed_code_sizes++] = 17;                           \  // 添加码长度17到 packed_code_sizes
                packed_code_sizes[num_packed_code_sizes++] = (mz_uint8)(rle_z_count - 3);  \  // 添加实际零码长度减3到 packed_code_sizes
            }                                                                              \
            else                                                                           \  // 如果零码长度大于10
            {                                                                              \
                d->m_huff_count[2][18] = (mz_uint16)(d->m_huff_count[2][18] + 1);          \  // 增加码长度为18的计数
                packed_code_sizes[num_packed_code_sizes++] = 18;                           \  // 添加码长度18到 packed_code_sizes
                packed_code_sizes[num_packed_code_sizes++] = (mz_uint8)(rle_z_count - 11); \  // 添加实际零码长度减11到 packed_code_sizes
            }                                                                              \
            rle_z_count = 0;                                                               \  // 重置零码计数
        }                                                                                  \
    }

// 静态数组，用于码长度符号的置换
static mz_uint8 s_tdefl_packed_code_size_syms_swizzle[] = { 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };

// 初始化动态块的函数，接受 tdefl_compressor 结构体指针 d 作为参数
static void tdefl_start_dynamic_block(tdefl_compressor *d)
{
    // 声明变量但不初始化：字面量码数、距离码数、位长度码数
    int num_lit_codes, num_dist_codes, num_bit_lengths;
    // 声明变量但不初始化：循环控制变量 i、待压缩码长度总数、已打包码长度数、待压缩的零码长度、重复的 rle 计数、打包码长度的索引
    mz_uint i, total_code_sizes_to_pack, num_packed_code_sizes, rle_z_count, rle_repeat_count, packed_code_sizes_index;
    // 定义三个数组：code_sizes_to_pack、packed_code_sizes、prev_code_size，大小为 TDEFL_MAX_HUFF_SYMBOLS_0 + TDEFL_MAX_HUFF_SYMBOLS_1，并初始化 prev_code_size 为 0xFF
    mz_uint8 code_sizes_to_pack[TDEFL_MAX_HUFF_SYMBOLS_0 + TDEFL_MAX_HUFF_SYMBOLS_1], packed_code_sizes[TDEFL_MAX_HUFF_SYMBOLS_0 + TDEFL_MAX_HUFF_SYMBOLS_1], prev_code_size = 0xFF;

    // 将 d->m_huff_count[0][256] 设为 1，用于表示 Huffman 表的特殊情况
    d->m_huff_count[0][256] = 1;

    // 优化第一个 Huffman 表，通过 tdefl_optimize_huffman_table 函数实现，不强制使用 15 位编码
    tdefl_optimize_huffman_table(d, 0, TDEFL_MAX_HUFF_SYMBOLS_0, 15, MZ_FALSE);
    // 优化第二个 Huffman 表，同样不强制使用 15 位编码
    tdefl_optimize_huffman_table(d, 1, TDEFL_MAX_HUFF_SYMBOLS_1, 15, MZ_FALSE);

    // 计算非零字面码的数目，初始化 num_lit_codes 为 286
    for (num_lit_codes = 286; num_lit_codes > 257; num_lit_codes--)
        if (d->m_huff_code_sizes[0][num_lit_codes - 1])
            break;
    
    // 计算非零距离码的数目，初始化 num_dist_codes 为 30
    for (num_dist_codes = 30; num_dist_codes > 1; num_dist_codes--)
        if (d->m_huff_code_sizes[1][num_dist_codes - 1])
            break;

    // 复制字面码和距离码的 Huffman 编码大小到 code_sizes_to_pack 数组中
    memcpy(code_sizes_to_pack, &d->m_huff_code_sizes[0][0], num_lit_codes);
    memcpy(code_sizes_to_pack + num_lit_codes, &d->m_huff_code_sizes[1][0], num_dist_codes);
    // 计算总共需要打包的 Huffman 编码大小数量
    total_code_sizes_to_pack = num_lit_codes + num_dist_codes;
    // 初始化已打包的编码大小数量为 0
    num_packed_code_sizes = 0;
    // 初始化 RLE 编码的零计数和重复计数为 0
    rle_z_count = 0;
    rle_repeat_count = 0;

    // 将第三个 Huffman 表的计数数组清零
    memset(&d->m_huff_count[2][0], 0, sizeof(d->m_huff_count[2][0]) * TDEFL_MAX_HUFF_SYMBOLS_2);
    
    // 遍历需要打包的 Huffman 编码大小数组
    for (i = 0; i < total_code_sizes_to_pack; i++)
    {
        // 取当前编码大小
        mz_uint8 code_size = code_sizes_to_pack[i];
        // 如果当前编码大小为 0，则执行 RLE 编码，递增 rle_z_count
        if (!code_size)
        {
            TDEFL_RLE_PREV_CODE_SIZE();
            if (++rle_z_count == 138)
            {
                TDEFL_RLE_ZERO_CODE_SIZE();
            }
        }
        else
        {
            // 否则执行非零编码，递增 rle_repeat_count
            TDEFL_RLE_ZERO_CODE_SIZE();
            if (code_size != prev_code_size)
            {
                TDEFL_RLE_PREV_CODE_SIZE();
                // 更新第三个 Huffman 表对应位置的计数值，并记录已打包的编码大小
                d->m_huff_count[2][code_size] = (mz_uint16)(d->m_huff_count[2][code_size] + 1);
                packed_code_sizes[num_packed_code_sizes++] = code_size;
            }
            else if (++rle_repeat_count == 6)
            {
                TDEFL_RLE_PREV_CODE_SIZE();
            }
        }
        // 更新 prev_code_size 为当前编码大小
        prev_code_size = code_size;
    }
    
    // 处理最后一个编码大小的 RLE 编码
    if (rle_repeat_count)
    {
        TDEFL_RLE_PREV_CODE_SIZE();
    }
    else
    {
        TDEFL_RLE_ZERO_CODE_SIZE();
    }

    // 优化第三个 Huffman 表
    tdefl_optimize_huffman_table(d, 2, TDEFL_MAX_HUFF_SYMBOLS_2, 7, MZ_FALSE);

    // 输出 2 位的特定比特流
    TDEFL_PUT_BITS(2, 2);

    // 输出非零字面码和距离码的数量（-257 和 -1 的差值）
    TDEFL_PUT_BITS(num_lit_codes - 257, 5);
    TDEFL_PUT_BITS(num_dist_codes - 1, 5);

    // 计算需要输出的编码比特长度的数量
    for (num_bit_lengths = 18; num_bit_lengths >= 0; num_bit_lengths--)
        if (d->m_huff_code_sizes[2][s_tdefl_packed_code_size_syms_swizzle[num_bit_lengths]])
            break;
    // 确保至少输出 4 个编码比特长度
    num_bit_lengths = MZ_MAX(4, (num_bit_lengths + 1));
    TDEFL_PUT_BITS(num_bit_lengths - 4, 4);
    // 输出具体的编码比特长度
    for (i = 0; (int)i < num_bit_lengths; i++)
        TDEFL_PUT_BITS(d->m_huff_code_sizes[2][s_tdefl_packed_code_size_syms_swizzle[i]], 3);

    // 进入循环处理打包的编码大小数组
    for (packed_code_sizes_index = 0; packed_code_sizes_index < num_packed_code_sizes;)
    {
        // 从 packed_code_sizes 数组中取出一个无符号整数 code，并递增索引 packed_code_sizes_index
        mz_uint code = packed_code_sizes[packed_code_sizes_index++];
        
        // 使用断言确保 code 的值小于 TDEFL_MAX_HUFF_SYMBOLS_2，即 code 的合法性检查
        MZ_ASSERT(code < TDEFL_MAX_HUFF_SYMBOLS_2);
        
        // 将编码值 code 放入 d->m_huff_codes[2] 中，并指定编码长度为 d->m_huff_code_sizes[2][code]
        TDEFL_PUT_BITS(d->m_huff_codes[2][code], d->m_huff_code_sizes[2][code]);
        
        // 如果 code 大于等于 16，则执行以下逻辑
        if (code >= 16)
            // NOLINTNEXTLINE(bugprone-signed-char-misuse)
            // 将 "\02\03\07"[code - 16] 中的字符编码放入输出流，即对长编码的处理
            TDEFL_PUT_BITS(packed_code_sizes[packed_code_sizes_index++], "\02\03\07"[code - 16]);
    }
// 结束静态块初始化函数，初始化压缩器中的哈夫曼编码大小数组
static void tdefl_start_static_block(tdefl_compressor *d)
{
    mz_uint i;  // 声明无符号整数变量 i
    mz_uint8 *p = &d->m_huff_code_sizes[0][0];  // 指向哈夫曼编码大小数组的指针

    // 初始化长度为 8 的哈夫曼编码大小，范围为 0 到 143
    for (i = 0; i <= 143; ++i)
        *p++ = 8;
    // 初始化长度为 9 的哈夫曼编码大小，范围为 144 到 255
    for (; i <= 255; ++i)
        *p++ = 9;
    // 初始化长度为 7 的哈夫曼编码大小，范围为 256 到 279
    for (; i <= 279; ++i)
        *p++ = 7;
    // 初始化长度为 8 的哈夫曼编码大小，范围为 280 到 287
    for (; i <= 287; ++i)
        *p++ = 8;

    // 使用值为 5 初始化第二个哈夫曼编码大小数组的前 32 个元素
    memset(d->m_huff_code_sizes[1], 5, 32);

    // 优化第一个哈夫曼编码表
    tdefl_optimize_huffman_table(d, 0, 288, 15, MZ_TRUE);
    // 优化第二个哈夫曼编码表
    tdefl_optimize_huffman_table(d, 1, 32, 15, MZ_TRUE);

    // 输出两位比特位到输出缓冲区中
    TDEFL_PUT_BITS(1, 2);
}

// 定义常量数组 mz_bitmasks，包含了各种位掩码
static const mz_uint mz_bitmasks[17] = { 0x0000, 0x0001, 0x0003, 0x0007, 0x000F, 0x001F, 0x003F, 0x007F, 0x00FF, 0x01FF, 0x03FF, 0x07FF, 0x0FFF, 0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF };

// 如果满足条件：MINIZ_USE_UNALIGNED_LOADS_AND_STORES 且 MINIZ_LITTLE_ENDIAN 且 MINIZ_HAS_64BIT_REGISTERS，则定义压缩 LZ 编码的函数
static mz_bool tdefl_compress_lz_codes(tdefl_compressor *d)
{
    mz_uint flags;  // 标志位
    mz_uint8 *pLZ_codes;  // 指向 LZ 编码缓冲区的指针
    mz_uint8 *pOutput_buf = d->m_pOutput_buf;  // 指向输出缓冲区的指针
    mz_uint8 *pLZ_code_buf_end = d->m_pLZ_code_buf;  // 指向 LZ 编码缓冲区末尾的指针
    mz_uint64 bit_buffer = d->m_bit_buffer;  // 64 位比特位缓冲区
    mz_uint bits_in = d->m_bits_in;  // 比特位缓冲区中当前的比特位数

    // 定义宏 TDEFL_PUT_BITS_FAST，用于快速向比特位缓冲区中添加比特位
#define TDEFL_PUT_BITS_FAST(b, l)                    \
    {                                                \
        bit_buffer |= (((mz_uint64)(b)) << bits_in); \
        bits_in += (l);                              \
    }

    flags = 1;  // 初始化标志位为 1
    // 遍历 LZ 编码缓冲区中的每个 LZ 编码
    for (pLZ_codes = d->m_lz_code_buf; pLZ_codes < pLZ_code_buf_end; flags >>= 1)
    {
        // 检查标志位是否为1，如果是则设置flags为pLZ_codes指向的值并加上0x100
        if (flags == 1)
            flags = *pLZ_codes++ | 0x100;
    
        // 如果flags的最低位为1，执行以下代码块
        if (flags & 1)
        {
            mz_uint s0, s1, n0, n1, sym, num_extra_bits;
            mz_uint match_len = pLZ_codes[0], match_dist = *(const mz_uint16 *)(pLZ_codes + 1);
            pLZ_codes += 3;
    
            // 断言match_len对应的哈夫曼编码大小不为0
            MZ_ASSERT(d->m_huff_code_sizes[0][s_tdefl_len_sym[match_len]]);
            // 将哈夫曼编码写入输出流，使用快速方法
            TDEFL_PUT_BITS_FAST(d->m_huff_codes[0][s_tdefl_len_sym[match_len]], d->m_huff_code_sizes[0][s_tdefl_len_sym[match_len]]);
            // 将match_len的额外位写入输出流，使用快速方法
            TDEFL_PUT_BITS_FAST(match_len & mz_bitmasks[s_tdefl_len_extra[match_len]], s_tdefl_len_extra[match_len]);
    
            /* This sequence coaxes MSVC into using cmov's vs. jmp's. */
            // 从预定义的数组中获取短距离和长距离的符号及额外位数
            s0 = s_tdefl_small_dist_sym[match_dist & 511];
            n0 = s_tdefl_small_dist_extra[match_dist & 511];
            s1 = s_tdefl_large_dist_sym[match_dist >> 8];
            n1 = s_tdefl_large_dist_extra[match_dist >> 8];
            // 根据match_dist的大小选择符号和额外位数
            sym = (match_dist < 512) ? s0 : s1;
            num_extra_bits = (match_dist < 512) ? n0 : n1;
    
            // 断言sym对应的哈夫曼编码大小不为0
            MZ_ASSERT(d->m_huff_code_sizes[1][sym]);
            // 将哈夫曼编码写入输出流，使用快速方法
            TDEFL_PUT_BITS_FAST(d->m_huff_codes[1][sym], d->m_huff_code_sizes[1][sym]);
            // 将match_dist的额外位数写入输出流，使用快速方法
            TDEFL_PUT_BITS_FAST(match_dist & mz_bitmasks[num_extra_bits], num_extra_bits);
        }
        else
        {
            // 如果flags的最低位不为1，执行以下代码块
            mz_uint lit = *pLZ_codes++;
            // 断言lit对应的哈夫曼编码大小不为0
            MZ_ASSERT(d->m_huff_code_sizes[0][lit]);
            // 将哈夫曼编码写入输出流，使用快速方法
            TDEFL_PUT_BITS_FAST(d->m_huff_codes[0][lit], d->m_huff_code_sizes[0][lit]);
    
            // 如果flags的第二位为0且pLZ_codes指针未超出缓冲区末尾，执行以下嵌套的代码块
            if (((flags & 2) == 0) && (pLZ_codes < pLZ_code_buf_end))
            {
                // 右移flags一位
                flags >>= 1;
                // 获取下一个字面值
                lit = *pLZ_codes++;
                // 断言lit对应的哈夫曼编码大小不为0
                MZ_ASSERT(d->m_huff_code_sizes[0][lit]);
                // 将哈夫曼编码写入输出流，使用快速方法
                TDEFL_PUT_BITS_FAST(d->m_huff_codes[0][lit], d->m_huff_code_sizes[0][lit]);
    
                // 如果flags的第二位为0且pLZ_codes指针未超出缓冲区末尾，执行以下嵌套的代码块
                if (((flags & 2) == 0) && (pLZ_codes < pLZ_code_buf_end))
                {
                    // 右移flags一位
                    flags >>= 1;
                    // 获取下一个字面值
                    lit = *pLZ_codes++;
                    // 断言lit对应的哈夫曼编码大小不为0
                    MZ_ASSERT(d->m_huff_code_sizes[0][lit]);
                    // 将哈夫曼编码写入输出流，使用快速方法
                    TDEFL_PUT_BITS_FAST(d->m_huff_codes[0][lit], d->m_huff_code_sizes[0][lit]);
                }
            }
        }
    
        // 如果输出缓冲区指针超过了输出缓冲区的末尾，返回假
        if (pOutput_buf >= d->m_pOutput_buf_end)
            return MZ_FALSE;
    
        // 将位缓冲写入输出缓冲区
        *(mz_uint64 *)pOutput_buf = bit_buffer;
        // 移动输出缓冲区指针到位缓冲区使用的字节数
        pOutput_buf += (bits_in >> 3);
        // 右移位缓冲以清除已写入输出缓冲区的位
        bit_buffer >>= (bits_in & ~7);
        // 更新位缓冲中剩余的位数
        bits_in &= 7;
    }
// 未定义 TDEFL_PUT_BITS_FAST 符号

    // 设置输出缓冲区指针
    d->m_pOutput_buf = pOutput_buf;
    // 初始化位数统计变量
    d->m_bits_in = 0;
    // 初始化位缓冲区
    d->m_bit_buffer = 0;

    // 处理剩余的比特位
    while (bits_in)
    {
        // 计算要处理的比特数，最多16位
        mz_uint32 n = MZ_MIN(bits_in, 16);
        // 将位缓冲区中的低 n 位写入输出流
        TDEFL_PUT_BITS((mz_uint)bit_buffer & mz_bitmasks[n], n);
        // 右移位缓冲区，丢弃已处理的位数
        bit_buffer >>= n;
        // 更新剩余比特位数
        bits_in -= n;
    }

    // 将终止码写入输出流
    TDEFL_PUT_BITS(d->m_huff_codes[0][256], d->m_huff_code_sizes[0][256]);

    // 返回是否输出缓冲区仍有空间
    return (d->m_pOutput_buf < d->m_pOutput_buf_end);
}
#else
static mz_bool tdefl_compress_lz_codes(tdefl_compressor *d)
{
    // 初始化标志位
    mz_uint flags = 1;
    // 指向 LZ 编码缓冲区的指针
    mz_uint8 *pLZ_codes;

    // 遍历 LZ 编码缓冲区
    for (pLZ_codes = d->m_lz_code_buf; pLZ_codes < d->m_pLZ_code_buf; flags >>= 1)
    {
        // 若标志位为1，则读取下一个字节作为新的标志
        if (flags == 1)
            flags = *pLZ_codes++ | 0x100;
        // 若标志位为真
        if (flags & 1)
        {
            // 初始化变量
            mz_uint sym, num_extra_bits;
            // 读取匹配长度和距离
            mz_uint match_len = pLZ_codes[0], match_dist = (pLZ_codes[1] | (pLZ_codes[2] << 8));
            pLZ_codes += 3;

            // 断言当前匹配长度对应的哈夫曼编码大小
            MZ_ASSERT(d->m_huff_code_sizes[0][s_tdefl_len_sym[match_len]]);
            // 将匹配长度的哈夫曼编码写入输出流
            TDEFL_PUT_BITS(d->m_huff_codes[0][s_tdefl_len_sym[match_len]], d->m_huff_code_sizes[0][s_tdefl_len_sym[match_len]]);
            // 将匹配长度的额外比特写入输出流
            TDEFL_PUT_BITS(match_len & mz_bitmasks[s_tdefl_len_extra[match_len]], s_tdefl_len_extra[match_len]);

            // 根据匹配距离大小选择符号和额外比特数
            if (match_dist < 512)
            {
                sym = s_tdefl_small_dist_sym[match_dist];
                num_extra_bits = s_tdefl_small_dist_extra[match_dist];
            }
            else
            {
                sym = s_tdefl_large_dist_sym[match_dist >> 8];
                num_extra_bits = s_tdefl_large_dist_extra[match_dist >> 8];
            }
            // 断言当前距离符号对应的哈夫曼编码大小
            MZ_ASSERT(d->m_huff_code_sizes[1][sym]);
            // 将距离符号的哈夫曼编码写入输出流
            TDEFL_PUT_BITS(d->m_huff_codes[1][sym], d->m_huff_code_sizes[1][sym]);
            // 将距离值的额外比特写入输出流
            TDEFL_PUT_BITS(match_dist & mz_bitmasks[num_extra_bits], num_extra_bits);
        }
        else
        {
            // 读取文本字面量
            mz_uint lit = *pLZ_codes++;
            // 断言当前字面量对应的哈夫曼编码大小
            MZ_ASSERT(d->m_huff_code_sizes[0][lit]);
            // 将字面量的哈夫曼编码写入输出流
            TDEFL_PUT_BITS(d->m_huff_codes[0][lit], d->m_huff_code_sizes[0][lit]);
        }
    }

    // 将终止码写入输出流
    TDEFL_PUT_BITS(d->m_huff_codes[0][256], d->m_huff_code_sizes[0][256]);

    // 返回是否输出缓冲区仍有空间
    return (d->m_pOutput_buf < d->m_pOutput_buf_end);
}
#endif /* MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN && MINIZ_HAS_64BIT_REGISTERS */

static mz_bool tdefl_compress_block(tdefl_compressor *d, mz_bool static_block)
{
    // 若为静态块，则开始静态块压缩
    if (static_block)
        tdefl_start_static_block(d);
    else
        // 否则开始动态块压缩
        tdefl_start_dynamic_block(d);
    // 返回块压缩结果
    return tdefl_compress_lz_codes(d);
}

static int tdefl_flush_block(tdefl_compressor *d, int flush)
{
    // 初始化保存的位缓冲区和位数
    mz_uint saved_bit_buf, saved_bits_in;
    // 保存输出缓冲区的指针
    mz_uint8 *pSaved_output_buf;
    // 压缩块操作是否成功的标志
    mz_bool comp_block_succeeded = MZ_FALSE;
    // 根据条件设置是否使用原始数据块的标志，根据字典的剩余空间和标志位决定
    int n, use_raw_block = ((d->m_flags & TDEFL_FORCE_ALL_RAW_BLOCKS) != 0) && (d->m_lookahead_pos - d->m_lz_code_buf_dict_pos) <= d->m_dict_size;
    
    // 设置输出缓冲区的起始位置，根据条件选择使用外部缓冲区或内部缓冲区
    mz_uint8 *pOutput_buf_start = ((d->m_pPut_buf_func == NULL) && ((*d->m_pOut_buf_size - d->m_out_buf_ofs) >= TDEFL_OUT_BUF_SIZE)) ? ((mz_uint8 *)d->m_pOut_buf + d->m_out_buf_ofs) : d->m_output_buf;

    // 设置输出缓冲区的起始和结束位置
    d->m_pOutput_buf = pOutput_buf_start;
    d->m_pOutput_buf_end = d->m_pOutput_buf + TDEFL_OUT_BUF_SIZE - 16;

    // 断言当前没有剩余需要刷新的输出数据
    MZ_ASSERT(!d->m_output_flush_remaining);
    d->m_output_flush_ofs = 0;
    d->m_output_flush_remaining = 0;

    // 更新当前字典的标志位
    *d->m_pLZ_flags = (mz_uint8)(*d->m_pLZ_flags >> d->m_num_flags_left);
    d->m_pLZ_code_buf -= (d->m_num_flags_left == 8);

    // 如果需要写入 ZLIB 头，并且当前块是第一个块，则写入 ZLIB 头部
    if ((d->m_flags & TDEFL_WRITE_ZLIB_HEADER) && (!d->m_block_index))
    {
        TDEFL_PUT_BITS(0x78, 8);
        TDEFL_PUT_BITS(0x01, 8);
    }

    // 根据 flush 参数设置是否结束压缩
    TDEFL_PUT_BITS(flush == TDEFL_FINISH, 1);

    // 保存当前输出缓冲区的状态，以便在需要时恢复
    pSaved_output_buf = d->m_pOutput_buf;
    saved_bit_buf = d->m_bit_buffer;
    saved_bits_in = d->m_bits_in;

    // 如果不使用原始数据块，则进行压缩当前块
    if (!use_raw_block)
        comp_block_succeeded = tdefl_compress_block(d, (d->m_flags & TDEFL_FORCE_ALL_STATIC_BLOCKS) || (d->m_total_lz_bytes < 48));

    // 如果当前块需要使用原始数据块，或者压缩后的数据量超出当前输出缓冲区的空间，且未超出字典的限制，则发送一个原始数据块
    else if (((use_raw_block) || ((d->m_total_lz_bytes) && ((d->m_pOutput_buf - pSaved_output_buf + 1U) >= d->m_total_lz_bytes))) &&
        ((d->m_lookahead_pos - d->m_lz_code_buf_dict_pos) <= d->m_dict_size))
    {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        mz_uint i;
        d->m_pOutput_buf = pSaved_output_buf;
        d->m_bit_buffer = saved_bit_buf, d->m_bits_in = saved_bits_in;
        TDEFL_PUT_BITS(0, 2);
        if (d->m_bits_in)
        {
            TDEFL_PUT_BITS(0, 8 - d->m_bits_in);
        }
        // 将总的 LZ 字节写入输出缓冲区
        for (i = 2; i; --i, d->m_total_lz_bytes ^= 0xFFFF)
        {
            TDEFL_PUT_BITS(d->m_total_lz_bytes & 0xFFFF, 16);
        }
        // 将字典中的数据写入输出缓冲区
        for (i = 0; i < d->m_total_lz_bytes; ++i)
        {
            TDEFL_PUT_BITS(d->m_dict[(d->m_lz_code_buf_dict_pos + i) & TDEFL_LZ_DICT_SIZE_MASK], 8);
        }
    }
    // 如果压缩当前块失败，则发送一个原始数据块
    else if (!comp_block_succeeded)
    {
        d->m_pOutput_buf = pSaved_output_buf;
        d->m_bit_buffer = saved_bit_buf, d->m_bits_in = saved_bits_in;
        tdefl_compress_block(d, MZ_TRUE);
    }

    // 如果 flush 参数为真，则刷新输出缓冲区
    if (flush)
    {
        // 如果需要刷新缓冲区
        if (flush == TDEFL_FINISH)
        {
            // 如果缓冲区中还有未处理的位数，补齐到字节边界
            if (d->m_bits_in)
            {
                TDEFL_PUT_BITS(0, 8 - d->m_bits_in);
            }
            // 如果设置了写入ZLIB头部的标志位
            if (d->m_flags & TDEFL_WRITE_ZLIB_HEADER)
            {
                // 计算Adler-32校验和并将其写入输出
                mz_uint i, a = d->m_adler32;
                for (i = 0; i < 4; i++)
                {
                    TDEFL_PUT_BITS((a >> 24) & 0xFF, 8);
                    a <<= 8;
                }
            }
        }
        else
        {
            // 否则（非TDEFL_FINISH情况下），初始化i和z变量
            mz_uint i, z = 0;
            // 将3位的结束块标志写入输出
            TDEFL_PUT_BITS(0, 3);
            // 如果缓冲区中还有未处理的位数，补齐到字节边界
            if (d->m_bits_in)
            {
                TDEFL_PUT_BITS(0, 8 - d->m_bits_in);
            }
            // 循环两次，每次将z值异或0xFFFF后写入16位到输出
            for (i = 2; i; --i, z ^= 0xFFFF)
            {
                TDEFL_PUT_BITS(z & 0xFFFF, 16);
            }
        }
    }
    
    // 断言检查输出缓冲区指针是否仍在有效范围内
    MZ_ASSERT(d->m_pOutput_buf < d->m_pOutput_buf_end);
    
    // 将huff_count数组清零，大小为TDEFL_MAX_HUFF_SYMBOLS_0 * sizeof(d->m_huff_count[0][0])
    memset(&d->m_huff_count[0][0], 0, sizeof(d->m_huff_count[0][0]) * TDEFL_MAX_HUFF_SYMBOLS_0);
    // 将huff_count数组清零，大小为TDEFL_MAX_HUFF_SYMBOLS_1 * sizeof(d->m_huff_count[1][0])
    memset(&d->m_huff_count[1][0], 0, sizeof(d->m_huff_count[1][0]) * TDEFL_MAX_HUFF_SYMBOLS_1);
    
    // 设置LZ编码缓冲区指针，指向m_lz_code_buf的下一个位置
    d->m_pLZ_code_buf = d->m_lz_code_buf + 1;
    // 设置LZ标志缓冲区指针，指向m_lz_code_buf的起始位置
    d->m_pLZ_flags = d->m_lz_code_buf;
    // 设置剩余LZ标志位数量为8
    d->m_num_flags_left = 8;
    // 增加LZ字典位置指针，加上已处理的总LZ字节数
    d->m_lz_code_buf_dict_pos += d->m_total_lz_bytes;
    // 重置已处理的总LZ字节数为0
    d->m_total_lz_bytes = 0;
    // 增加块索引
    d->m_block_index++;
    
    // 如果输出缓冲区中有数据被写入
    if ((n = (int)(d->m_pOutput_buf - pOutput_buf_start)) != 0)
    {
        // 如果存在写入缓冲区的回调函数
        if (d->m_pPut_buf_func)
        {
            // 设置输入缓冲区的大小为当前处理的数据量
            *d->m_pIn_buf_size = d->m_pSrc - (const mz_uint8 *)d->m_pIn_buf;
            // 调用写入缓冲区的回调函数，写入输出缓冲区的数据
            if (!(*d->m_pPut_buf_func)(d->m_output_buf, n, d->m_pPut_buf_user))
                // 如果写入失败，则返回失败状态码
                return (d->m_prev_return_status = TDEFL_STATUS_PUT_BUF_FAILED);
        }
        // 否则，如果输出缓冲区的起始位置等于m_output_buf（初始位置）
        else if (pOutput_buf_start == d->m_output_buf)
        {
            // 计算需要复制的字节数，最多不超过输出缓冲区的剩余空间和当前数据量
            int bytes_to_copy = (int)MZ_MIN((size_t)n, (size_t)(*d->m_pOut_buf_size - d->m_out_buf_ofs));
            // 将数据从输出缓冲区复制到输出缓冲区的指定位置
            memcpy((mz_uint8 *)d->m_pOut_buf + d->m_out_buf_ofs, d->m_output_buf, bytes_to_copy);
            // 更新输出缓冲区偏移量
            d->m_out_buf_ofs += bytes_to_copy;
            // 如果仍有未复制完的数据，更新输出刷新偏移量和剩余数据量
            if ((n -= bytes_to_copy) != 0)
            {
                d->m_output_flush_ofs = bytes_to_copy;
                d->m_output_flush_remaining = n;
            }
        }
        // 否则，只更新输出缓冲区的偏移量
        else
        {
            d->m_out_buf_ofs += n;
        }
    }
    
    // 返回未刷新的输出数据量
    return d->m_output_flush_remaining;
// 如果启用了MINIZ_USE_UNALIGNED_LOADS_AND_STORES宏，则使用memcpy函数从字节流中读取一个未对齐的16位字。
#ifdef MINIZ_USE_UNALIGNED_LOADS_AND_STORES
#ifdef MINIZ_UNALIGNED_USE_MEMCPY
static mz_uint16 TDEFL_READ_UNALIGNED_WORD(const mz_uint8* p)
{
    mz_uint16 ret;
    memcpy(&ret, p, sizeof(mz_uint16));  // 从地址p处复制sizeof(mz_uint16)字节到ret中
    return ret;
}
static mz_uint16 TDEFL_READ_UNALIGNED_WORD2(const mz_uint16* p)
{
    mz_uint16 ret;
    memcpy(&ret, p, sizeof(mz_uint16));  // 从地址p处复制sizeof(mz_uint16)字节到ret中
    return ret;
}
#else
// 否则，使用宏定义直接读取未对齐的16位字
#define TDEFL_READ_UNALIGNED_WORD(p) *(const mz_uint16 *)(p)
#define TDEFL_READ_UNALIGNED_WORD2(p) *(const mz_uint16 *)(p)
#endif
// 定义了一个内联函数tdefl_find_match，用于在压缩器对象d的字典中查找匹配项
static MZ_FORCEINLINE void tdefl_find_match(tdefl_compressor *d, mz_uint lookahead_pos, mz_uint max_dist, mz_uint max_match_len, mz_uint *pMatch_dist, mz_uint *pMatch_len)
{
    // 声明并初始化局部变量：距离(dist)，位置(pos)，匹配长度(match_len)，探测位置(probe_pos)，下一个探测位置(next_probe_pos)，探测长度(probe_len)
    mz_uint dist, pos = lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK, match_len = *pMatch_len, probe_pos = pos, next_probe_pos, probe_len;
    // 使用d的成员变量m_max_probes根据条件选择探测次数上限，存储在num_probes_left中
    mz_uint num_probes_left = d->m_max_probes[match_len >= 32];
    // 声明并初始化指向字典的16位无符号整数指针s
    const mz_uint16 *s = (const mz_uint16 *)(d->m_dict + pos), *p, *q;
    // 从字典中读取未对齐的16位字存储在c01和s01中
    mz_uint16 c01 = TDEFL_READ_UNALIGNED_WORD(&d->m_dict[pos + match_len - 1]), s01 = TDEFL_READ_UNALIGNED_WORD2(s);
    // 断言最大匹配长度不超过TDEFL_MAX_MATCH_LEN
    MZ_ASSERT(max_match_len <= TDEFL_MAX_MATCH_LEN);
    // 如果最大匹配长度小于或等于当前匹配长度，直接返回
    if (max_match_len <= match_len)
        return;
    // 进入无限循环，查找匹配项
    for (;;)
    {
        // 内层循环，不断进行探测
        for (;;)
        {
            // 如果剩余的探测次数减少到0，退出循环
            if (--num_probes_left == 0)
                return;
            // 定义宏TDEFL_PROBE，设置下一个探测位置并检查距离是否超出最大限制
#define TDEFL_PROBE                                                                             \
    next_probe_pos = d->m_next[probe_pos];                                                      \
    if ((!next_probe_pos) || ((dist = (mz_uint16)(lookahead_pos - next_probe_pos)) > max_dist)) \
        return;                                                                                 \
    probe_pos = next_probe_pos & TDEFL_LZ_DICT_SIZE_MASK;                                       \
    // 检查当前位置的字典项是否与给定的c01相等，如果是则跳出循环
    if (TDEFL_READ_UNALIGNED_WORD(&d->m_dict[probe_pos + match_len - 1]) == c01)
        break;
    // 进行探测
    TDEFL_PROBE;
    // 进行探测
    TDEFL_PROBE;
    // 进行探测
    TDEFL_PROBE;
    // 如果距离为0，则跳出整体循环
    }
        if (!dist)
            break;
    // 将q设置为指向probe_pos位置的字典中的16位整数指针
        q = (const mz_uint16 *)(d->m_dict + probe_pos);
    // 如果TDEFL_READ_UNALIGNED_WORD2(q)的值不等于s01，则继续下一个循环
        if (TDEFL_READ_UNALIGNED_WORD2(q) != s01)
            continue;
    // 将p设置为s
        p = s;
    // 设置probe_len为32
        probe_len = 32;
    // 执行循环，直到p和q不相等，或者probe_len减为0
        do
        {
        } while ((TDEFL_READ_UNALIGNED_WORD2(++p) == TDEFL_READ_UNALIGNED_WORD2(++q)) && (TDEFL_READ_UNALIGNED_WORD2(++p) == TDEFL_READ_UNALIGNED_WORD2(++q)) &&
                 (TDEFL_READ_UNALIGNED_WORD2(++p) == TDEFL_READ_UNALIGNED_WORD2(++q)) && (TDEFL_READ_UNALIGNED_WORD2(++p) == TDEFL_READ_UNALIGNED_WORD2(++q)) && (--probe_len > 0));
    // 如果probe_len为0
        if (!probe_len)
    // 将匹配距离设置为dist
            *pMatch_dist = dist;
    // 将匹配长度设置为max_match_len和TDEFL_MAX_MATCH_LEN的较小值
            *pMatch_len = MZ_MIN(max_match_len, (mz_uint)TDEFL_MAX_MATCH_LEN);
    // 跳出循环
            break;
    // 如果probe_len大于match_len
        else if ((probe_len = ((mz_uint)(p - s) * 2) + (mz_uint)(*(const mz_uint8 *)p == *(const mz_uint8 *)q)) > match_len)
    // 将匹配距离设置为dist
            *pMatch_dist = dist;
    // 如果匹配长度等于max_match_len，则跳出循环
            if ((*pMatch_len = match_len = MZ_MIN(max_match_len, probe_len)) == max_match_len)
                break;
    // 设置c01为d->m_dict[pos + match_len - 1]的16位无符号整数
            c01 = TDEFL_READ_UNALIGNED_WORD(&d->m_dict[pos + match_len - 1]);
    }
#else
static MZ_FORCEINLINE void tdefl_find_match(tdefl_compressor *d, mz_uint lookahead_pos, mz_uint max_dist, mz_uint max_match_len, mz_uint *pMatch_dist, mz_uint *pMatch_len)
{
    // 定义变量：距离(dist)，当前位置(pos)，匹配长度(match_len)，探查位置(probe_pos)，下一个探查位置(next_probe_pos)，探查长度(probe_len)，剩余探查次数(num_probes_left)，字典内容指针(s)，以及匹配字符(c0和c1)
    mz_uint dist, pos = lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK, match_len = *pMatch_len, probe_pos = pos, next_probe_pos, probe_len;
    mz_uint num_probes_left = d->m_max_probes[match_len >= 32];
    const mz_uint8 *s = d->m_dict + pos, *p, *q;
    mz_uint8 c0 = d->m_dict[pos + match_len], c1 = d->m_dict[pos + match_len - 1];
    
    // 断言最大匹配长度不超过预设值
    MZ_ASSERT(max_match_len <= TDEFL_MAX_MATCH_LEN);
    
    // 如果最大匹配长度小于等于当前匹配长度，则直接返回
    if (max_match_len <= match_len)
        return;
    
    // 无限循环，用于查找匹配
    for (;;)
    {
        // 内部循环，用于探查匹配位置
        for (;;)
        {
            // 减少剩余探查次数
            if (--num_probes_left == 0)
                return;
            
            // 宏定义探查过程
#define TDEFL_PROBE                                                                               \
    next_probe_pos = d->m_next[probe_pos];                                                        \
    if ((!next_probe_pos) || ((dist = (mz_uint16)(lookahead_pos - next_probe_pos)) > max_dist))   \
        return;                                                                                   \
    probe_pos = next_probe_pos & TDEFL_LZ_DICT_SIZE_MASK;                                         \
    if ((d->m_dict[probe_pos + match_len] == c0) && (d->m_dict[probe_pos + match_len - 1] == c1)) \
        break;
            // 执行探查宏定义三次
            TDEFL_PROBE;
            TDEFL_PROBE;
            TDEFL_PROBE;
        }
        
        // 如果距离为零，则跳出循环
        if (!dist)
            break;
        
        // p和q指向当前位置和探查位置
        p = s;
        q = d->m_dict + probe_pos;
        
        // 比较内容，计算探查长度
        for (probe_len = 0; probe_len < max_match_len; probe_len++)
            if (*p++ != *q++)
                break;
        
        // 如果探查长度大于当前匹配长度，则更新匹配距离和匹配长度
        if (probe_len > match_len)
        {
            *pMatch_dist = dist;
            if ((*pMatch_len = match_len = probe_len) == max_match_len)
                return;
            c0 = d->m_dict[pos + match_len];
            c1 = d->m_dict[pos + match_len - 1];
        }
    }
}
#endif /* #if MINIZ_USE_UNALIGNED_LOADS_AND_STORES */

#if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN
#ifdef MINIZ_UNALIGNED_USE_MEMCPY
static mz_uint32 TDEFL_READ_UNALIGNED_WORD32(const mz_uint8* p)
{
    // 使用memcpy从字节流中读取一个未对齐的32位整数
    mz_uint32 ret;
    memcpy(&ret, p, sizeof(mz_uint32));
    return ret;
}
#else
// 直接读取未对齐的32位整数
#define TDEFL_READ_UNALIGNED_WORD32(p) *(const mz_uint32 *)(p)
#endif

// 压缩函数，优化的LZRW1风格的匹配和解析循环，更好的寄存器利用，适用于注重原始吞吐量高于压缩比的应用。
static mz_bool tdefl_compress_fast(tdefl_compressor *d)
{
    mz_uint lookahead_pos = d->m_lookahead_pos, lookahead_size = d->m_lookahead_size, dict_size = d->m_dict_size, total_lz_bytes = d->m_total_lz_bytes, num_flags_left = d->m_num_flags_left;
    mz_uint8 *pLZ_code_buf = d->m_pLZ_code_buf, *pLZ_flags = d->m_pLZ_flags;
    mz_uint cur_pos = lookahead_pos & TDEFL_LZ_DICT_SIZE_MASK;

    // 循环，处理源缓冲区中剩余的数据，或者在刷新模式下处理前瞻大小不为零的情况
    while ((d->m_src_buf_left) || ((d->m_flush) && (lookahead_size)))
static mz_bool tdefl_compress_normal(tdefl_compressor *d)
{
    // 从压缩器对象中获取源数据指针和剩余源数据大小
    const mz_uint8 *pSrc = d->m_pSrc;
    size_t src_buf_left = d->m_src_buf_left;
    // 从压缩器对象中获取当前的刷新方式
    tdefl_flush flush = d->m_flush;

    // 当源数据剩余量不为零，或者在刷新方式为真且有预看大小时，继续循环
    while ((src_buf_left) || ((flush) && (d->m_lookahead_size)))
    {
        // 省略了处理源数据的具体压缩算法
    }

    // 将处理后的数据更新回压缩器对象
    d->m_pSrc = pSrc;
    d->m_src_buf_left = src_buf_left;
    // 返回压缩成功的标志
    return MZ_TRUE;
}



static tdefl_status tdefl_flush_output_buffer(tdefl_compressor *d)
{
    // 如果输入缓冲区大小不为零，更新输入缓冲区中的大小
    if (d->m_pIn_buf_size)
    {
        *d->m_pIn_buf_size = d->m_pSrc - (const mz_uint8 *)d->m_pIn_buf;
    }

    // 如果输出缓冲区大小不为零，处理输出缓冲区中的数据（省略具体实现）
}
    {
        // 计算要复制的数据大小，取较小值为缓冲区的可用空间和需要刷新的剩余数据大小的最小值
        size_t n = MZ_MIN(*d->m_pOut_buf_size - d->m_out_buf_ofs, d->m_output_flush_remaining);
    
        // 将输出缓冲区中的部分数据从输出缓冲区偏移位置复制到输出缓冲区的实际数据起始位置
        memcpy((mz_uint8 *)d->m_pOut_buf + d->m_out_buf_ofs, d->m_output_buf + d->m_output_flush_ofs, n);
    
        // 更新输出缓冲区偏移量，增加已经刷新的数据大小
        d->m_output_flush_ofs += (mz_uint)n;
    
        // 更新剩余需要刷新的数据大小
        d->m_output_flush_remaining -= (mz_uint)n;
    
        // 更新输出缓冲区偏移量，以便下一次操作
        d->m_out_buf_ofs += n;
    
        // 更新输出缓冲区的大小为当前输出缓冲区的偏移量
        *d->m_pOut_buf_size = d->m_out_buf_ofs;
    }
    
    // 根据状态返回压缩操作的状态，若已经完成且没有剩余需要刷新的数据，则返回完成状态，否则返回正常状态
    return (d->m_finished && !d->m_output_flush_remaining) ? TDEFL_STATUS_DONE : TDEFL_STATUS_OKAY;
}

// 压缩函数 tdefl_compress 的实现
tdefl_status tdefl_compress(tdefl_compressor *d, const void *pIn_buf, size_t *pIn_buf_size, void *pOut_buf, size_t *pOut_buf_size, tdefl_flush flush)
{
    // 检查压缩器对象是否存在
    if (!d)
    {
        // 如果不存在，将输入和输出缓冲区大小设为0，并返回错误状态
        if (pIn_buf_size)
            *pIn_buf_size = 0;
        if (pOut_buf_size)
            *pOut_buf_size = 0;
        return TDEFL_STATUS_BAD_PARAM;
    }

    // 设置压缩器对象的输入和输出缓冲区以及其他相关参数
    d->m_pIn_buf = pIn_buf;
    d->m_pIn_buf_size = pIn_buf_size;
    d->m_pOut_buf = pOut_buf;
    d->m_pOut_buf_size = pOut_buf_size;
    d->m_pSrc = (const mz_uint8 *)(pIn_buf);
    d->m_src_buf_left = pIn_buf_size ? *pIn_buf_size : 0;
    d->m_out_buf_ofs = 0;
    d->m_flush = flush;

    // 检查参数的有效性，如果参数无效，设定输入和输出缓冲区大小为0，并返回错误状态
    if (((d->m_pPut_buf_func != NULL) == ((pOut_buf != NULL) || (pOut_buf_size != NULL))) || 
        (d->m_prev_return_status != TDEFL_STATUS_OKAY) ||
        (d->m_wants_to_finish && (flush != TDEFL_FINISH)) ||
        (pIn_buf_size && *pIn_buf_size && !pIn_buf) ||
        (pOut_buf_size && *pOut_buf_size && !pOut_buf))
    {
        if (pIn_buf_size)
            *pIn_buf_size = 0;
        if (pOut_buf_size)
            *pOut_buf_size = 0;
        return (d->m_prev_return_status = TDEFL_STATUS_BAD_PARAM);
    }

    // 标记是否需要结束压缩
    d->m_wants_to_finish |= (flush == TDEFL_FINISH);

    // 如果有剩余的输出数据或者已经结束压缩，则刷新输出缓冲区
    if ((d->m_output_flush_remaining) || (d->m_finished))
        return (d->m_prev_return_status = tdefl_flush_output_buffer(d));

    // 根据压缩器的配置和状态选择压缩方法进行处理
#if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN
    if (((d->m_flags & TDEFL_MAX_PROBES_MASK) == 1) &&
        ((d->m_flags & TDEFL_GREEDY_PARSING_FLAG) != 0) &&
        ((d->m_flags & (TDEFL_FILTER_MATCHES | TDEFL_FORCE_ALL_RAW_BLOCKS | TDEFL_RLE_MATCHES)) == 0))
    {
        // 使用快速压缩方法进行压缩
        if (!tdefl_compress_fast(d))
            return d->m_prev_return_status;
    }
    else
#endif /* #if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN */
    {
        // 使用普通压缩方法进行压缩
        if (!tdefl_compress_normal(d))
            return d->m_prev_return_status;
    }

    // 如果需要，计算输入数据的 Adler32 校验值
    if ((d->m_flags & (TDEFL_WRITE_ZLIB_HEADER | TDEFL_COMPUTE_ADLER32)) && (pIn_buf))
        d->m_adler32 = (mz_uint32)mz_adler32(d->m_adler32, (const mz_uint8 *)pIn_buf, d->m_pSrc - (const mz_uint8 *)pIn_buf);

    // 如果需要刷新并且没有待处理数据，则执行块刷新操作
    if ((flush) && (!d->m_lookahead_size) && (!d->m_src_buf_left) && (!d->m_output_flush_remaining))
    {
        if (tdefl_flush_block(d, flush) < 0)
            return d->m_prev_return_status;
        d->m_finished = (flush == TDEFL_FINISH);
        // 如果是完全刷新，清空哈希表和预测状态
        if (flush == TDEFL_FULL_FLUSH)
        {
            MZ_CLEAR_OBJ(d->m_hash);
            MZ_CLEAR_OBJ(d->m_next);
            d->m_dict_size = 0;
        }
    }

    // 返回上次操作的状态
    return (d->m_prev_return_status = tdefl_flush_output_buffer(d));
}

// 对输入缓冲区执行压缩，并返回压缩结果状态
tdefl_status tdefl_compress_buffer(tdefl_compressor *d, const void *pIn_buf, size_t in_buf_size, tdefl_flush flush)
{
    // 断言压缩器的输出函数不为空
    MZ_ASSERT(d->m_pPut_buf_func);
    // 调用通用的压缩函数进行压缩
    return tdefl_compress(d, pIn_buf, &in_buf_size, NULL, NULL, flush);
}

// 初始化压缩器对象
tdefl_status tdefl_init(tdefl_compressor *d, tdefl_put_buf_func_ptr pPut_buf_func, void *pPut_buf_user, int flags)
{
    // 设置压缩器对象的输出函数和用户数据
    d->m_pPut_buf_func = pPut_buf_func;
    d->m_pPut_buf_user = pPut_buf_user;
    // ...

}

// 压缩函数 tdefl_compress 的实现
tdefl_status tdefl_compress(tdefl_compressor *d, const void *pIn_buf, size_t *pIn_buf_size, void *pOut_buf, size_t *pOut_buf_size, tdefl_flush flush)
{
    // 检查压缩器对象是否存在
    if (!d)
    {
        // 如果不存在，将输入和输出缓冲区大小设为0，并返回错误状态
        if (pIn_buf_size)
            *pIn_buf_size = 0;
        if (pOut_buf_size)
            *pOut_buf_size = 0;
        return TDEFL_STATUS_BAD_PARAM;
    }

    // 设置压缩器对象的输入和输出缓冲区以及其他相关参数
    d->m_pIn_buf = pIn_buf;
    d->m_pIn_buf_size = pIn_buf_size;
    d->m_pOut_buf = pOut_buf;
    d->m_pOut_buf_size = pOut_buf_size;
    d->m_pSrc = (const mz_uint8 *)(pIn_buf);
    d->m_src_buf_left = pIn_buf_size ? *pIn_buf_size : 0;
    d->m_out_buf_ofs = 0;
    d->m_flush = flush;

    // 检查参数的有效性，如果参数无效，设定输入和输出缓冲区大小为0，并返回错误状态
    if (((d->m_pPut_buf_func != NULL) == ((pOut_buf != NULL) || (pOut_buf_size != NULL))) || 
        (d->m_prev_return_status != TDEFL_STATUS_OKAY) ||
        (d->m_wants_to_finish && (flush != TDEFL_FINISH)) ||
        (pIn_buf_size && *pIn_buf_size && !pIn_buf) ||
        (pOut_buf_size && *pOut_buf_size && !pOut_buf))
    {
        if (pIn_buf_size)
            *pIn_buf_size = 0;
        if (pOut_buf_size)
            *pOut_buf_size = 0;
        return (d->m_prev_return_status = TDEFL_STATUS_BAD_PARAM);
    }

    // 标记是否需要结束压缩
    d->m_wants_to_finish |= (flush == TDEFL_FINISH);

    // 如果有剩余的输出数据或者已经结束压缩，则刷新输出缓冲区
    if ((d->m_output_flush_remaining) || (d->m_finished))
        return (d->m_prev_return_status = tdefl_flush_output_buffer(d));

    // 根据压缩器的配置和状态选择压缩方法进行处理
#if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN
    if (((d->m_flags & TDEFL_MAX_PROBES_MASK) == 1) &&
        ((d->m_flags & TDEFL_GREEDY_PARSING_FLAG) != 0) &&
        ((d->m_flags & (TDEFL_FILTER_MATCHES | TDEFL_FORCE_ALL_RAW_BLOCKS | TDEFL_RLE_MATCHES)) == 0))
    {
        // 使用快速压缩方法进行压缩
        if (!tdefl_compress_fast(d))
            return d->m_prev_return_status;
    }
    else
#endif /* #if MINIZ_USE_UNALIGNED_LOADS_AND_STORES && MINIZ_LITTLE_ENDIAN */
    {
        // 使用普通压缩方法进行压缩
        if (!tdefl_compress_normal(d))
            return d->m_prev_return_status;
    }

    // 如果需要，计算输入数据的 Adler32 校验值
    if ((d->m_flags & (TDEFL_WRITE_ZLIB_HEADER | TDEFL_COMPUTE_ADLER32)) && (pIn_buf))
        d->m_adler32 = (mz_uint32)mz_adler32(d->m_adler32, (const mz_uint8 *)pIn_buf, d->m_pSrc - (const mz_uint8 *)pIn_buf);

    // 如果需要刷新并且没有待处理数据，则执行块刷新操作
    if ((flush) && (!d->m_lookahead_size) && (!d->m_src_buf_left) && (!d->m_output_flush_remaining))
    {
        if (tdefl_flush_block(d, flush) < 0)
            return d->m_prev_return_status;
        d->m_finished = (flush == TDEFL_FINISH);
        // 如果是完全刷新，清空哈希表和预测状态
        if (flush == TDEFL_FULL_FLUSH)
        {
            MZ_CLEAR_OBJ(d->m_hash);
            MZ_CLEAR_OBJ(d->m_next);
            d->m_dict_size = 0;
        }
    }

    // 返回上次操作的状态
    return (d->m_prev_return_status = tdefl_flush_output_buffer(d));
}

// 对输入缓冲区执行压缩，并返回压缩结果状态
tdefl_status tdefl_compress_buffer(tdefl_compressor *d, const void *pIn_buf, size_t in_buf_size, tdefl_flush flush)
{
    // 断言压缩器的输出函数不为空
    MZ_ASSERT(d->m_pPut_buf_func);
    // 调用通用的压缩函数进行压缩
    return tdefl_compress(d, pIn_buf, &in_buf_size, NULL, NULL, flush);
}

// 初始化压缩器对象
tdefl_status tdefl_init(tdefl_compressor *d, tdefl_put_buf_func_ptr pPut_buf_func, void *pPut_buf_user, int flags)
{
    // 设置压缩器对象的输出函数和用户数据
    d->m_pPut_buf_func = pPut_buf_func;
    d->m_pPut_buf_user = pPut_buf_user;
    // ...
    // 将传入的 flags 强制转换为 mz_uint 类型，并赋值给 d->m_flags
    d->m_flags = (mz_uint)(flags);
    
    // 根据 flags 计算并设置 d->m_max_probes[0]，这是一个探测次数的计算
    d->m_max_probes[0] = 1 + ((flags & 0xFFF) + 2) / 3;
    
    // 检查 flags 中是否包含 TDEFL_GREEDY_PARSING_FLAG，设置 d->m_greedy_parsing
    d->m_greedy_parsing = (flags & TDEFL_GREEDY_PARSING_FLAG) != 0;
    
    // 根据 flags 计算并设置 d->m_max_probes[1]，这也是探测次数的计算
    d->m_max_probes[1] = 1 + (((flags & 0xFFF) >> 2) + 2) / 3;
    
    // 如果 flags 不包含 TDEFL_NONDETERMINISTIC_PARSING_FLAG，则清空 d->m_hash 对象
    if (!(flags & TDEFL_NONDETERMINISTIC_PARSING_FLAG))
        MZ_CLEAR_OBJ(d->m_hash);
    
    // 将多个成员变量的值重置为 0，这些成员变量都与压缩算法的状态管理有关
    d->m_lookahead_pos = d->m_lookahead_size = d->m_dict_size = d->m_total_lz_bytes = d->m_lz_code_buf_dict_pos = d->m_bits_in = 0;
    d->m_output_flush_ofs = d->m_output_flush_remaining = d->m_finished = d->m_block_index = d->m_bit_buffer = d->m_wants_to_finish = 0;
    
    // 设置 d->m_pLZ_code_buf 和 d->m_pLZ_flags，用于指向压缩码缓冲区的位置
    d->m_pLZ_code_buf = d->m_lz_code_buf + 1;
    d->m_pLZ_flags = d->m_lz_code_buf;
    
    // 初始化 d->m_num_flags_left，表示剩余的标志位数
    d->m_num_flags_left = 8;
    
    // 设置输出缓冲区的起始和结束位置，指向 d->m_output_buf
    d->m_pOutput_buf = d->m_pOutput_buf_end = d->m_output_buf;
    
    // 初始化其他状态标志和计数器
    d->m_prev_return_status = TDEFL_STATUS_OKAY;
    d->m_saved_match_dist = d->m_saved_match_len = d->m_saved_lit = 0;
    
    // 初始化 Adler-32 校验和为 1
    d->m_adler32 = 1;
    
    // 初始化输入和输出缓冲区指针为空
    d->m_pIn_buf = NULL;
    d->m_pOut_buf = NULL;
    d->m_pIn_buf_size = NULL;
    d->m_pOut_buf_size = NULL;
    
    // 设置刷新策略为 TDEFL_NO_FLUSH
    d->m_flush = TDEFL_NO_FLUSH;
    
    // 如果 flags 不包含 TDEFL_NONDETERMINISTIC_PARSING_FLAG，则清空 d->m_dict 对象
    if (!(flags & TDEFL_NONDETERMINISTIC_PARSING_FLAG))
        MZ_CLEAR_OBJ(d->m_dict);
    
    // 使用 memset 函数将 d->m_huff_count 数组清零，用于哈夫曼编码统计
    memset(&d->m_huff_count[0][0], 0, sizeof(d->m_huff_count[0][0]) * TDEFL_MAX_HUFF_SYMBOLS_0);
    memset(&d->m_huff_count[1][0], 0, sizeof(d->m_huff_count[1][0]) * TDEFL_MAX_HUFF_SYMBOLS_1);
    
    // 返回压缩状态为 TDEFL_STATUS_OKAY，表示初始化完成且无错误
    return TDEFL_STATUS_OKAY;
}

// 返回压缩器对象中保存的前一个返回状态
tdefl_status tdefl_get_prev_return_status(tdefl_compressor *d)
{
    return d->m_prev_return_status;
}

// 返回压缩器对象中保存的 Adler-32 校验和
mz_uint32 tdefl_get_adler32(tdefl_compressor *d)
{
    return d->m_adler32;
}

// 将内存中的数据压缩到输出流中
// pBuf: 要压缩的数据的指针
// buf_len: 数据长度
// pPut_buf_func: 回调函数，用于将压缩后的数据写入输出
// pPut_buf_user: 用户传入的数据，供回调函数使用
// flags: 压缩标志位
// 返回值: 是否成功压缩到输出流中
mz_bool tdefl_compress_mem_to_output(const void *pBuf, size_t buf_len, tdefl_put_buf_func_ptr pPut_buf_func, void *pPut_buf_user, int flags)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    tdefl_compressor *pComp;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_bool succeeded;

    // 检查输入参数的有效性
    if (((buf_len) && (!pBuf)) || (!pPut_buf_func))
        return MZ_FALSE;

    // 分配压缩器对象的内存
    pComp = (tdefl_compressor *)MZ_MALLOC(sizeof(tdefl_compressor));
    if (!pComp)
        return MZ_FALSE;

    // 初始化压缩器对象
    succeeded = (tdefl_init(pComp, pPut_buf_func, pPut_buf_user, flags) == TDEFL_STATUS_OKAY);

    // 压缩数据并检查是否成功
    succeeded = succeeded && (tdefl_compress_buffer(pComp, pBuf, buf_len, TDEFL_FINISH) == TDEFL_STATUS_DONE);

    // 释放压缩器对象的内存
    MZ_FREE(pComp);

    // 返回压缩结果
    return succeeded;
}

// 定义输出缓冲区结构体
typedef struct
{
    size_t m_size, m_capacity;
    mz_uint8 *m_pBuf;
    mz_bool m_expandable;
} tdefl_output_buffer;

// 输出缓冲区的回调函数，用于接收压缩后的数据
static mz_bool tdefl_output_buffer_putter(const void *pBuf, int len, void *pUser)
{
    tdefl_output_buffer *p = (tdefl_output_buffer *)pUser;
    size_t new_size = p->m_size + len;

    // 如果新数据超过当前容量，则扩展缓冲区容量
    if (new_size > p->m_capacity)
    {
        size_t new_capacity = p->m_capacity;
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        mz_uint8 *pNew_buf;

        // 如果不允许扩展，则返回失败
        if (!p->m_expandable)
            return MZ_FALSE;

        // 扩展缓冲区容量
        do
        {
            new_capacity = MZ_MAX(128U, new_capacity << 1U);
        } while (new_size > new_capacity);

        // 重新分配扩展后的缓冲区
        pNew_buf = (mz_uint8 *)MZ_REALLOC(p->m_pBuf, new_capacity);
        if (!pNew_buf)
            return MZ_FALSE;

        // 更新缓冲区指针和容量
        p->m_pBuf = pNew_buf;
        p->m_capacity = new_capacity;
    }

    // 将数据复制到缓冲区末尾
    memcpy((mz_uint8 *)p->m_pBuf + p->m_size, pBuf, len);
    p->m_size = new_size;
    return MZ_TRUE;
}

// 将内存中的数据压缩到堆中，并返回压缩后的数据指针
// pSrc_buf: 要压缩的数据的指针
// src_buf_len: 数据长度
// pOut_len: 输出参数，用于返回压缩后数据的长度
// flags: 压缩标志位
// 返回值: 压缩后的数据指针
void *tdefl_compress_mem_to_heap(const void *pSrc_buf, size_t src_buf_len, size_t *pOut_len, int flags)
{
    tdefl_output_buffer out_buf;

    // 初始化输出缓冲区对象
    MZ_CLEAR_OBJ(out_buf);

    // 如果输出参数为空，则返回失败
    if (!pOut_len)
        return MZ_FALSE;
    else
        *pOut_len = 0;

    // 设置输出缓冲区为可扩展
    out_buf.m_expandable = MZ_TRUE;

    // 将数据压缩到输出缓冲区
    if (!tdefl_compress_mem_to_output(pSrc_buf, src_buf_len, tdefl_output_buffer_putter, &out_buf, flags))
        return NULL;

    // 更新输出数据的长度
    *pOut_len = out_buf.m_size;

    // 返回压缩后的数据指针
    return out_buf.m_pBuf;
}

// 将内存中的数据直接压缩到指定的内存中
// pOut_buf: 输出缓冲区的指针
// out_buf_len: 输出缓冲区的长度
// pSrc_buf: 要压缩的数据的指针
// src_buf_len: 数据长度
// flags: 压缩标志位
// 返回值: 实际压缩的数据长度
size_t tdefl_compress_mem_to_mem(void *pOut_buf, size_t out_buf_len, const void *pSrc_buf, size_t src_buf_len, int flags)
{
    tdefl_output_buffer out_buf;

    // 初始化输出缓冲区对象
    MZ_CLEAR_OBJ(out_buf);

    // 如果输出缓冲区为空，则返回压缩数据长度为 0
    if (!pOut_buf)
        return 0;

    // 设置输出缓冲区的指针和长度
    out_buf.m_pBuf = (mz_uint8 *)pOut_buf;
    out_buf.m_capacity = out_buf_len;

    // 将数据压缩到输出缓冲区
    if (!tdefl_compress_mem_to_output(pSrc_buf, src_buf_len, tdefl_output_buffer_putter, &out_buf, flags))
        return 0;

    // 返回实际压缩的数据长度
    return out_buf.m_size;
}

// 定义用于压缩算法的探测值数组
static const mz_uint s_tdefl_num_probes[11] = { 0, 1, 6, 32, 16, 32, 128, 256, 512, 768, 1500 };
/* 根据给定的压缩级别、窗口位数和策略生成压缩标志。压缩级别范围实际为[0,10]（10是一个“隐藏”的最大级别，在某些文件上可以有更多的压缩，并且在吞吐量上有所下降是可以接受的） */
mz_uint tdefl_create_comp_flags_from_zip_params(int level, int window_bits, int strategy)
{
    // 根据给定的压缩级别选择预定义的探测次数和是否使用贪婪解析标志
    mz_uint comp_flags = s_tdefl_num_probes[(level >= 0) ? MZ_MIN(10, level) : MZ_DEFAULT_LEVEL] | ((level <= 3) ? TDEFL_GREEDY_PARSING_FLAG : 0);

    // 如果窗口位数大于0，设置写入ZLIB头部的标志
    if (window_bits > 0)
        comp_flags |= TDEFL_WRITE_ZLIB_HEADER;

    // 根据压缩级别、策略设置相应的标志
    if (!level)
        comp_flags |= TDEFL_FORCE_ALL_RAW_BLOCKS;
    else if (strategy == MZ_FILTERED)
        comp_flags |= TDEFL_FILTER_MATCHES;
    else if (strategy == MZ_HUFFMAN_ONLY)
        comp_flags &= ~TDEFL_MAX_PROBES_MASK;
    else if (strategy == MZ_FIXED)
        comp_flags |= TDEFL_FORCE_ALL_STATIC_BLOCKS;
    else if (strategy == MZ_RLE)
        comp_flags |= TDEFL_RLE_MATCHES;

    // 返回最终生成的压缩标志
    return comp_flags;
}

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4204) /* nonstandard extension used : non-constant aggregate initializer (also supported by GNU C and C99, so no big deal) */
#endif

/* Alex Evans编写的简单PNG写入函数，2011年。已释放至公共领域：https://gist.github.com/908299，更多背景信息请查看
   http://altdevblogaday.org/2011/04/06/a-smaller-jpg-encoder/。
   这实际上是对Alex原始代码的修改，以便通过pngcheck验证生成的PNG文件。 */
void *tdefl_write_image_to_png_file_in_memory_ex(const void *pImage, int w, int h, int num_chans, size_t *pLen_out, mz_uint level, mz_bool flip)
{
    // 如果MINIZ_NO_ZLIB_APIS被定义，使用本地副本的数组
    static const mz_uint s_tdefl_png_num_probes[11] = { 0, 1, 6, 32, 16, 32, 128, 256, 512, 768, 1500 };
    tdefl_compressor *pComp = (tdefl_compressor *)MZ_MALLOC(sizeof(tdefl_compressor));
    tdefl_output_buffer out_buf;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int i, bpl = w * num_chans, y, z;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint32 c;
    *pLen_out = 0;
    if (!pComp)
        return NULL;
    MZ_CLEAR_OBJ(out_buf);
    out_buf.m_expandable = MZ_TRUE;
    out_buf.m_capacity = 57 + MZ_MAX(64, (1 + bpl) * h);
    if (NULL == (out_buf.m_pBuf = (mz_uint8 *)MZ_MALLOC(out_buf.m_capacity)))
    {
        MZ_FREE(pComp);
        return NULL;
    }
    /* 写入虚拟头部 */
    for (z = 41; z; --z)
        tdefl_output_buffer_putter(&z, 1, &out_buf);
    /* 压缩图像数据 */
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    tdefl_init(pComp, tdefl_output_buffer_putter, &out_buf, s_tdefl_png_num_probes[MZ_MIN(10, level)] | TDEFL_WRITE_ZLIB_HEADER);
    for (y = 0; y < h; ++y)
    {
        tdefl_compress_buffer(pComp, &z, 1, TDEFL_NO_FLUSH);
        tdefl_compress_buffer(pComp, (mz_uint8 *)pImage + (flip ? (h - 1 - y) : y) * bpl, bpl, TDEFL_NO_FLUSH);
    }
    // 如果压缩未完成，释放内存并返回空指针
    if (tdefl_compress_buffer(pComp, NULL, 0, TDEFL_FINISH) != TDEFL_STATUS_DONE)
    {
        MZ_FREE(pComp);  // 释放压缩缓冲区的内存
        MZ_FREE(out_buf.m_pBuf);  // 释放输出缓冲区的内存
        return NULL;  // 返回空指针
    }

    /* write real header */
    // 计算实际头部数据的长度，并更新指针指向的值
    *pLen_out = out_buf.m_size - 41;

    {
        static const mz_uint8 chans[] = { 0x00, 0x00, 0x04, 0x02, 0x06 };
        // PNG 文件头部数据的数组定义，包含 PNG 文件头的固定部分和变化的尺寸信息等
        mz_uint8 pnghdr[41] = { 0x89, 0x50, 0x4e, 0x47, 0x0d,
                                0x0a, 0x1a, 0x0a, 0x00, 0x00,
                                0x00, 0x0d, 0x49, 0x48, 0x44,
                                0x52, 0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x00, 0x08,
                                0x00, 0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x00, 0x00, 0x00,
                                0x00, 0x00, 0x49, 0x44, 0x41,
                                0x54 };
        // 更新 PNG 头部数据中的宽度和高度信息
        pnghdr[18] = (mz_uint8)(w >> 8);
        pnghdr[19] = (mz_uint8)w;
        pnghdr[22] = (mz_uint8)(h >> 8);
        pnghdr[23] = (mz_uint8)h;
        // 更新 PNG 头部数据中的通道信息
        pnghdr[25] = chans[num_chans];
        // 更新 PNG 头部数据中的长度信息
        pnghdr[33] = (mz_uint8)(*pLen_out >> 24);
        pnghdr[34] = (mz_uint8)(*pLen_out >> 16);
        pnghdr[35] = (mz_uint8)(*pLen_out >> 8);
        pnghdr[36] = (mz_uint8)*pLen_out;
        // 计算 PNG 头部数据的 CRC32 校验值
        c = (mz_uint32)mz_crc32(MZ_CRC32_INIT, pnghdr + 12, 17);
        // 将 CRC32 的结果写入到 PNG 头部数据中
        for (i = 0; i < 4; ++i, c <<= 8)
            ((mz_uint8 *)(pnghdr + 29))[i] = (mz_uint8)(c >> 24);
        // 将更新后的 PNG 头部数据拷贝到输出缓冲区的起始位置
        memcpy(out_buf.m_pBuf, pnghdr, 41);
    }

    /* write footer (IDAT CRC-32, followed by IEND chunk) */
    // 写入 PNG 的尾部数据，包括 IDAT 的 CRC-32 校验值和 IEND 块
    if (!tdefl_output_buffer_putter("\0\0\0\0\0\0\0\0\x49\x45\x4e\x44\xae\x42\x60\x82", 16, &out_buf))
    {
        *pLen_out = 0;  // 如果写入失败，将输出长度置为 0
        MZ_FREE(pComp);  // 释放压缩缓冲区的内存
        MZ_FREE(out_buf.m_pBuf);  // 释放输出缓冲区的内存
        return NULL;  // 返回空指针
    }
    // 计算 PNG 尾部数据的 CRC32 校验值
    c = (mz_uint32)mz_crc32(MZ_CRC32_INIT, out_buf.m_pBuf + 41 - 4, *pLen_out + 4);
    // 将计算得到的 CRC32 结果写入 PNG 尾部数据中
    for (i = 0; i < 4; ++i, c <<= 8)
        (out_buf.m_pBuf + out_buf.m_size - 16)[i] = (mz_uint8)(c >> 24);

    /* compute final size of file, grab compressed data buffer and return */
    // 计算最终文件的大小，增加压缩数据的长度
    *pLen_out += 57;
    MZ_FREE(pComp);  // 释放压缩缓冲区的内存
    return out_buf.m_pBuf;  // 返回输出缓冲区的指针
}
/* 
 * 返回一个指针，指向将图像数据写入内存中的 PNG 文件的函数
 * pImage: 图像数据指针
 * w: 图像宽度
 * h: 图像高度
 * num_chans: 图像通道数
 * pLen_out: 返回 PNG 数据长度的指针
 */
void *tdefl_write_image_to_png_file_in_memory(const void *pImage, int w, int h, int num_chans, size_t *pLen_out)
{
    /* Level 6 对应于 TDEFL_DEFAULT_MAX_PROBES 或者 MZ_DEFAULT_LEVEL（但不能依赖于 MZ_DEFAULT_LEVEL 因为 zlib API 可能已经被 #defined 掉） */
    return tdefl_write_image_to_png_file_in_memory_ex(pImage, w, h, num_chans, pLen_out, 6, MZ_FALSE);
}

#ifndef MINIZ_NO_MALLOC
/* 
 * 在 C 中分配 tdefl_compressor 和 tinfl_decompressor 结构体，以便非 C 语言的 tdefL_ 和 tinfl_ API 绑定不需要担心结构体大小和分配机制
 */
tdefl_compressor *tdefl_compressor_alloc(void)
{
    return (tdefl_compressor *)MZ_MALLOC(sizeof(tdefl_compressor));
}

/* 
 * 释放 tdefl_compressor 结构体的内存
 * pComp: 要释放的 tdefl_compressor 结构体指针
 */
void tdefl_compressor_free(tdefl_compressor *pComp)
{
    MZ_FREE(pComp);
}
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#ifdef __cplusplus
}
#endif
    }                                        \
    MZ_MACRO_END


    }                                        \  // 结束宏定义的大括号
    MZ_MACRO_END                              // 标识宏定义结束的宏定义名称
// 定义一个宏函数 TINFL_CR_RETURN_FOREVER，用于无限循环调用 TINFL_CR_RETURN 宏，并返回指定结果
#define TINFL_CR_RETURN_FOREVER(state_index, result) \
    do                                               \
    {                                                \
        for (;;)                                     \
        {                                            \
            TINFL_CR_RETURN(state_index, result);    \
        }                                            \
    }                                                \
    MZ_MACRO_END

// 定义一个宏函数 TINFL_CR_FINISH，没有参数，用于结束一个状态机
#define TINFL_CR_FINISH }

// 定义一个宏函数 TINFL_GET_BYTE，接收 state_index 和 c 两个参数，用于从输入缓冲区中获取一个字节到变量 c 中
#define TINFL_GET_BYTE(state_index, c)                                                                                                                           \
    do                                                                                                                                                           \
    {                                                                                                                                                            \
        while (pIn_buf_cur >= pIn_buf_end)                                                                                                                       \
        {                                                                                                                                                        \
            // 如果输入缓冲区已经读取完毕，根据 decomp_flags 的设置决定返回状态，要求更多输入或者无法继续进展
            TINFL_CR_RETURN(state_index, (decomp_flags & TINFL_FLAG_HAS_MORE_INPUT) ? TINFL_STATUS_NEEDS_MORE_INPUT : TINFL_STATUS_FAILED_CANNOT_MAKE_PROGRESS); \
        }                                                                                                                                                        \
        c = *pIn_buf_cur++;                                                                                                                                      \
    }                                                                                                                                                            \
    MZ_MACRO_END

// 定义一个宏函数 TINFL_NEED_BITS，接收 state_index 和 n 两个参数，用于从输入流中获取至少 n 位的数据
#define TINFL_NEED_BITS(state_index, n)                \
    do                                                 \
    {                                                  \
        mz_uint c;                                     \
        TINFL_GET_BYTE(state_index, c);                \
        bit_buf |= (((tinfl_bit_buf_t)c) << num_bits); \
        num_bits += 8;                                 \
    } while (num_bits < (mz_uint)(n))

// 定义一个宏函数 TINFL_SKIP_BITS，接收 state_index 和 n 两个参数，用于跳过输入流中的 n 位数据
#define TINFL_SKIP_BITS(state_index, n)      \
    do                                       \
    {                                        \
        if (num_bits < (mz_uint)(n))         \
        {                                    \
            TINFL_NEED_BITS(state_index, n); \
        }                                    \
        bit_buf >>= (n);                     \
        num_bits -= (n);                     \
    }                                        \
    MZ_MACRO_END

// 定义一个宏函数 TINFL_GET_BITS，接收 state_index, b 和 n 三个参数，用于从输入流中获取 n 位数据到变量 b 中
#define TINFL_GET_BITS(state_index, b, n)    \
    do                                       \
    {
    {                                        \
        // 检查当前剩余比特数是否小于需要的比特数n，如果是，则需要获取更多比特位
        if (num_bits < (mz_uint)(n))         \
        {                                    \
            // 调用宏 TINFL_NEED_BITS，获取所需的比特位
            TINFL_NEED_BITS(state_index, n); \
        }                                    \
        // 将当前比特流bit_buf按照n位掩码，获取最低的n位作为b的值
        b = bit_buf & ((1 << (n)) - 1);      \
        // 右移bit_buf的内容，丢弃已读取的n位
        bit_buf >>= (n);                     \
        // 减少当前剩余比特数num_bits
        num_bits -= (n);                     \
    }                                        \
    // 宏 MZ_MACRO_END 的结束标记
    MZ_MACRO_END
    
    
    这段代码看起来是嵌入了一个宏定义中的代码片段，其中涉及位操作和条件判断，主要用于处理比特流的解析工作。
/* TINFL_HUFF_BITBUF_FILL() is a macro that fills the Huffman decoding bit buffer with enough bits to decode the next Huffman code. It optimizes by using a fast lookup table
   to quickly determine the code length if possible. If the current bits in the buffer are insufficient, it reads more bytes from the input stream until it has enough bits
   to decode the Huffman code, up to a maximum of 15 bits (deflate's max. Huffman code size). */
#define TINFL_HUFF_BITBUF_FILL(state_index, pHuff)                             \
    do                                                                         \
    {                                                                          \
        temp = (pHuff)->m_look_up[bit_buf & (TINFL_FAST_LOOKUP_SIZE - 1)];     \
        if (temp >= 0)                                                         \
        {                                                                      \
            code_len = temp >> 9;                                              \
            if ((code_len) && (num_bits >= code_len))                          \
                break;                                                         \
        }                                                                      \
        else if (num_bits > TINFL_FAST_LOOKUP_BITS)                            \
        {                                                                      \
            code_len = TINFL_FAST_LOOKUP_BITS;                                 \
            do                                                                 \
            {                                                                  \
                temp = (pHuff)->m_tree[~temp + ((bit_buf >> code_len++) & 1)]; \
            } while ((temp < 0) && (num_bits >= (code_len + 1)));              \
            if (temp >= 0)                                                     \
                break;                                                         \
        }                                                                      \
        TINFL_GET_BYTE(state_index, c);                                        \
        bit_buf |= (((tinfl_bit_buf_t)c) << num_bits);                         \
        num_bits += 8;                                                         \
    } while (num_bits < 15);
/* 定义一个宏，用于解码 Huffman 编码的符号。
   state_index: 解码状态索引
   sym: 解码后的符号值
   pHuff: Huffman 编码表
*/
#define TINFL_HUFF_DECODE(state_index, sym, pHuff)                                                                                  \
    do                                                                                                                              \
    {                                                                                                                               \
        /* 实现 Huffman 解码的细节，根据当前状态索引和 Huffman 表解码出符号 sym */                                                 \
    }                                                                                                                               \
    while (0)                                                                                                                       \
    MZ_MACRO_END

/* tinfl_decompress 函数用于解压缩数据。
   r: 解压器结构体指针
   pIn_buf_next: 输入缓冲区的当前位置
   pIn_buf_size: 输入缓冲区大小的指针
   pOut_buf_start: 输出缓冲区的起始位置
   pOut_buf_next: 输出缓冲区的当前位置
   pOut_buf_size: 输出缓冲区大小的指针
   decomp_flags: 解压标志位
*/
tinfl_status tinfl_decompress(tinfl_decompressor *r, const mz_uint8 *pIn_buf_next, size_t *pIn_buf_size, mz_uint8 *pOut_buf_start, mz_uint8 *pOut_buf_next, size_t *pOut_buf_size, const mz_uint32 decomp_flags)
{
    /* 定义长度码的基础值 */
    static const int s_length_base[31] = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 0, 0 };
    /* 定义长度码的额外位数 */
    static const int s_length_extra[31] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0, 0 };
    /* 定义距离码的基础值 */
    static const int s_dist_base[32] = { 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 0, 0 };
    /* 定义距离码的额外位数 */
    static const int s_dist_extra[32] = { 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };
    /* 用于反向解 zigzag 编码的长度符号 */
    static const mz_uint8 s_length_dezigzag[19] = { 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
    /* 三种 Huffman 表的最小尺寸 */
    static const int s_min_table_sizes[3] = { 257, 1, 4 };

    /* 解压状态，默认为失败 */
    tinfl_status status = TINFL_STATUS_FAILED;
    mz_uint32 num_bits, dist, counter, num_extra;
    tinfl_bit_buf_t bit_buf;
    const mz_uint8 *pIn_buf_cur = pIn_buf_next, *const pIn_buf_end = pIn_buf_next + *pIn_buf_size;
    mz_uint8 *pOut_buf_cur = pOut_buf_next, *const pOut_buf_end = pOut_buf_next + *pOut_buf_size;
    size_t out_buf_size_mask = (decomp_flags & TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF) ? (size_t)-1 : ((pOut_buf_next - pOut_buf_start) + *pOut_buf_size) - 1, dist_from_out_buf_start;

    /* 确保输出缓冲区大小为 2 的幂，除非输出缓冲区足够大以容纳整个输出文件（此时大小无关紧要）。 */
    if (((out_buf_size_mask + 1) & out_buf_size_mask) || (pOut_buf_next < pOut_buf_start))
    {
        *pIn_buf_size = *pOut_buf_size = 0;
        return TINFL_STATUS_BAD_PARAM;
    }

    num_bits = r->m_num_bits;
    bit_buf = r->m_bit_buf;
    dist = r->m_dist;
    counter = r->m_counter;
    num_extra = r->m_num_extra;

    /* 剩余部分，还需添加具体解压缩函数的实现 */
}
    # 将 r 指针中的 m_dist_from_out_buf_start 值赋给 dist_from_out_buf_start
    dist_from_out_buf_start = r->m_dist_from_out_buf_start;
    # TINFL_CR_BEGIN 宏的作用是初始化解压缩器的状态机
    TINFL_CR_BEGIN

    # 将多个变量初始化为零
    bit_buf = num_bits = dist = counter = num_extra = r->m_zhdr0 = r->m_zhdr1 = 0;
    # 初始化 Adler-32 校验值
    r->m_z_adler32 = r->m_check_adler32 = 1;
    # 如果解压缩标志包含 TINFL_FLAG_PARSE_ZLIB_HEADER
    if (decomp_flags & TINFL_FLAG_PARSE_ZLIB_HEADER)
    {
        # 从输入流中读取 zlib 头的两个字节
        TINFL_GET_BYTE(1, r->m_zhdr0);
        TINFL_GET_BYTE(2, r->m_zhdr1);
        # 检查 zlib 头的有效性
        counter = (((r->m_zhdr0 * 256 + r->m_zhdr1) % 31 != 0) || (r->m_zhdr1 & 32) || ((r->m_zhdr0 & 15) != 8));
        # 如果解压缩标志不包含 TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF
        if (!(decomp_flags & TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF))
            // NOLINTNEXTLINE(bugprone-misplaced-widening-cast,cppcoreguidelines-avoid-magic-numbers)
            # 进一步检查 zlib 头的有效性，防止输出缓冲区溢出
            counter |= (((1U << (8U + (r->m_zhdr0 >> 4))) > 32768U) || ((out_buf_size_mask + 1) < (size_t)(1U << (8U + (r->m_zhdr0 >> 4)))));
        # 如果 counter 不为零，则表示 zlib 头无效，返回解压缩失败状态
        if (counter)
        {
            TINFL_CR_RETURN_FOREVER(36, TINFL_STATUS_FAILED);
        }
    }

    # 进入 do 循环
#if TINFL_USE_64BIT_BITBUF
                        # 如果使用64位比特缓冲区，则执行以下代码块
                        if (num_bits < 30)
                        {
                            # 如果当前比特数小于30，则将下一个32位从输入缓冲区中读取并加入比特缓冲区
                            bit_buf |= (((tinfl_bit_buf_t)MZ_READ_LE32(pIn_buf_cur)) << num_bits);
                            pIn_buf_cur += 4;  # 移动输入缓冲区指针4字节
                            num_bits += 32;    # 增加比特缓冲区中的比特数
                        }
#else
                        # 如果不使用64位比特缓冲区，则执行以下代码块
                        if (num_bits < 15)
                        {
                            # 如果当前比特数小于15，则将下一个16位从输入缓冲区中读取并加入比特缓冲区
                            bit_buf |= (((tinfl_bit_buf_t)MZ_READ_LE16(pIn_buf_cur)) << num_bits);
                            pIn_buf_cur += 2;  # 移动输入缓冲区指针2字节
                            num_bits += 16;    # 增加比特缓冲区中的比特数
                        }
#endif
                        # 根据当前比特缓冲区的内容查找对应的符号，更新码长
                        if ((sym2 = r->m_tables[0].m_look_up[bit_buf & (TINFL_FAST_LOOKUP_SIZE - 1)]) >= 0)
                            code_len = sym2 >> 9;  # 如果找到符号，则码长为符号的高9位
                        else
                        {
                            code_len = TINFL_FAST_LOOKUP_BITS;  # 否则，码长为快速查找位数
                            do
                            {
                                sym2 = r->m_tables[0].m_tree[~sym2 + ((bit_buf >> code_len++) & 1)];
                                # 使用树结构继续查找符号，直到找到非负值的符号
                            } while (sym2 < 0);
                        }
                        counter = sym2;  # 将找到的符号赋值给计数器
                        bit_buf >>= code_len;  # 从比特缓冲区中移除已经使用的比特
                        num_bits -= code_len;  # 减去已经使用的比特数
                        if (counter & 256)  # 如果计数器的最高位为1，表示结束
                            break;

#if !TINFL_USE_64BIT_BITBUF
                        # 如果不使用64位比特缓冲区，则执行以下代码块
                        if (num_bits < 15)
                        {
                            # 如果当前比特数小于15，则将下一个16位从输入缓冲区中读取并加入比特缓冲区
                            bit_buf |= (((tinfl_bit_buf_t)MZ_READ_LE16(pIn_buf_cur)) << num_bits);
                            pIn_buf_cur += 2;  # 移动输入缓冲区指针2字节
                            num_bits += 16;    # 增加比特缓冲区中的比特数
                        }
#endif
#else
                        // 如果符号表中包含当前比特缓冲的值，找到对应的码字长度
                        if ((sym2 = r->m_tables[0].m_look_up[bit_buf & (TINFL_FAST_LOOKUP_SIZE - 1)]) >= 0)
                            code_len = sym2 >> 9;
                        else
                        {
                            // 否则设定码字长度为快速查找位数
                            code_len = TINFL_FAST_LOOKUP_BITS;
                            // 使用码字长度读取树结构以获取符号
                            do
                            {
                                sym2 = r->m_tables[0].m_tree[~sym2 + ((bit_buf >> code_len++) & 1)];
                            } while (sym2 < 0);
                        }
                        // 从比特缓冲中移除已使用的比特位数
                        bit_buf >>= code_len;
                        num_bits -= code_len;

                        // 将计数器的值作为输出缓冲的一个字节
                        pOut_buf_cur[0] = (mz_uint8)counter;
                        // 如果符号有256，则表明需要特殊处理
                        if (sym2 & 256)
                        {
                            pOut_buf_cur++;
                            counter = sym2;
                            break;
                        }
                        // 否则将符号的值存储为输出缓冲的下一个字节
                        pOut_buf_cur[1] = (mz_uint8)sym2;
                        pOut_buf_cur += 2;
                    }
                }
                // 如果计数器为256，跳出循环
                if ((counter &= 511) == 256)
                    break;

                // 从长度表中获取额外的比特数和基本计数器值
                num_extra = s_length_extra[counter - 257];
                counter = s_length_base[counter - 257];
                if (num_extra)
                {
                    // 从输入中获取额外的比特数
                    mz_uint extra_bits;
                    TINFL_GET_BITS(25, extra_bits, num_extra);
                    counter += extra_bits;
                }

                // 解码距离符号
                TINFL_HUFF_DECODE(26, dist, &r->m_tables[1]);
                // 从距离表中获取额外的比特数和基本距离值
                num_extra = s_dist_extra[dist];
                dist = s_dist_base[dist];
                if (num_extra)
                {
                    // 从输入中获取额外的比特数
                    mz_uint extra_bits;
                    TINFL_GET_BITS(27, extra_bits, num_extra);
                    dist += extra_bits;
                }

                // 计算距离输出缓冲开始位置的偏移量
                dist_from_out_buf_start = pOut_buf_cur - pOut_buf_start;
                // 如果距离大于当前位置与使用非环绕输出缓冲标志，则返回错误
                if ((dist > dist_from_out_buf_start) && (decomp_flags & TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF))
                {
                    TINFL_CR_RETURN_FOREVER(37, TINFL_STATUS_FAILED);
                }

                // 根据距离从输出缓冲中获取源指针
                pSrc = pOut_buf_start + ((dist_from_out_buf_start - dist) & out_buf_size_mask);

                // 如果输出缓冲区溢出，进行逐个字节复制
                if ((MZ_MAX(pOut_buf_cur, pSrc) + counter) > pOut_buf_end)
                {
                    while (counter--)
                    {
                        while (pOut_buf_cur >= pOut_buf_end)
                        {
                            // 返回状态表明有更多的输出
                            TINFL_CR_RETURN(53, TINFL_STATUS_HAS_MORE_OUTPUT);
                        }
                        // 逐个复制输出缓冲区中的字节
                        *pOut_buf_cur++ = pOut_buf_start[(dist_from_out_buf_start++ - dist) & out_buf_size_mask];
                    }
                    continue;
                }
#if MINIZ_USE_UNALIGNED_LOADS_AND_STORES
                else if ((counter >= 9) && (counter <= dist))
                {
                    // 如果启用了非对齐加载和存储，并且计数器在 9 到 dist 之间
                    const mz_uint8 *pSrc_end = pSrc + (counter & ~7);
                    do
                    {
#ifdef MINIZ_UNALIGNED_USE_MEMCPY
                        // 使用 memcpy 复制两个 mz_uint32 的数据到输出缓冲区当前位置
                        memcpy(pOut_buf_cur, pSrc, sizeof(mz_uint32)*2);
#else
                        // 将 pSrc 指向的两个 mz_uint32 数据拷贝到输出缓冲区当前位置
                        ((mz_uint32 *)pOut_buf_cur)[0] = ((const mz_uint32 *)pSrc)[0];
                        ((mz_uint32 *)pOut_buf_cur)[1] = ((const mz_uint32 *)pSrc)[1];
#endif
                        pOut_buf_cur += 8; // 移动输出缓冲区指针到下一个位置
                    } while ((pSrc += 8) < pSrc_end); // 遍历直到处理完所有数据对齐的部分
                    if ((counter &= 7) < 3)
                    {
                        if (counter)
                        {
                            // 将剩余不足8字节的数据复制到输出缓冲区当前位置
                            pOut_buf_cur[0] = pSrc[0];
                            if (counter > 1)
                                pOut_buf_cur[1] = pSrc[1];
                            pOut_buf_cur += counter; // 移动输出缓冲区指针到下一个位置
                        }
                        continue; // 继续处理下一轮循环
                    }
                }
#endif
                while(counter>2)
                {
                    // 将 pSrc 指向的三个字节的数据复制到输出缓冲区当前位置
                    pOut_buf_cur[0] = pSrc[0];
                    pOut_buf_cur[1] = pSrc[1];
                    pOut_buf_cur[2] = pSrc[2];
                    pOut_buf_cur += 3; // 移动输出缓冲区指针到下一个位置
                    pSrc += 3; // 移动输入缓冲区指针到下一个位置
                    counter -= 3; // 减少计数器
                }
                if (counter > 0)
                {
                    // 将剩余的少于三个字节的数据复制到输出缓冲区当前位置
                    pOut_buf_cur[0] = pSrc[0];
                    if (counter > 1)
                        pOut_buf_cur[1] = pSrc[1];
                    pOut_buf_cur += counter; // 移动输出缓冲区指针到下一个位置
                }
            }
        }
    } while (!(r->m_final & 1)); // 直到处理完所有的块

    /* 确保字节对齐，并且如果我们在处理 gzip 或其他 Deflate 流时向前查看过多，需要将位缓冲区的字节放回。 */
    /* 这里我采取了极其保守的方式。对于字节对齐部分可以进行一些简化，而 Adler32 校验现在不再需要担心从位缓冲区读取数据。 */
    TINFL_SKIP_BITS(32, num_bits & 7);
    while ((pIn_buf_cur > pIn_buf_next) && (num_bits >= 8))
    {
        --pIn_buf_cur; // 逆向移动输入缓冲区指针
        num_bits -= 8; // 减少位数
    }
    bit_buf &= (tinfl_bit_buf_t)((((mz_uint64)1) << num_bits) - (mz_uint64)1); // 更新位缓冲区

    MZ_ASSERT(!num_bits); /* 如果此断言失败，则说明我们读取了非 Deflate/zlib 流后面的数据（如 gzip 流）。 */

    if (decomp_flags & TINFL_FLAG_PARSE_ZLIB_HEADER)
    {
        for (counter = 0; counter < 4; ++counter)
        {
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            mz_uint s;
            if (num_bits)
                TINFL_GET_BITS(41, s, 8);
            else
                TINFL_GET_BYTE(42, s);
            r->m_z_adler32 = (r->m_z_adler32 << 8) | s; // 更新 Adler32 校验和
        }
    }
    TINFL_CR_RETURN_FOREVER(34, TINFL_STATUS_DONE); // 永久返回 DONE 状态

    TINFL_CR_FINISH // 结束解压

common_exit:
    /* 只要状态不是需要更多输入或者无法取得进展的情况下，执行以下操作： */

    /* 如果我们在解析 gzip 或其他 Deflate 流后跟随任意数据时预读了太多内容，将位缓冲区中的字节放回去。 */
    /* 需要非常小心，在这里不能将任何我们确切知道需要继续前进的字节放回去，否则会导致调用者进入无限循环。 */
    if ((status != TINFL_STATUS_NEEDS_MORE_INPUT) && (status != TINFL_STATUS_FAILED_CANNOT_MAKE_PROGRESS))
    {
        while ((pIn_buf_cur > pIn_buf_next) && (num_bits >= 8))
        {
            --pIn_buf_cur;
            num_bits -= 8;
        }
    }

    /* 更新解压器中的位数和位缓冲区 */
    r->m_num_bits = num_bits;
    r->m_bit_buf = bit_buf & (tinfl_bit_buf_t)((((mz_uint64)1) << num_bits) - (mz_uint64)1);
    r->m_dist = dist;
    r->m_counter = counter;
    r->m_num_extra = num_extra;
    r->m_dist_from_out_buf_start = dist_from_out_buf_start;

    /* 更新输入缓冲区和输出缓冲区的大小 */
    *pIn_buf_size = pIn_buf_cur - pIn_buf_next;
    *pOut_buf_size = pOut_buf_cur - pOut_buf_next;

    /* 如果需要解析 ZLIB 头部或者计算 Adler32 校验码，并且状态为成功，则执行以下操作 */
    if ((decomp_flags & (TINFL_FLAG_PARSE_ZLIB_HEADER | TINFL_FLAG_COMPUTE_ADLER32)) && (status >= 0))
    {
        const mz_uint8 *ptr = pOut_buf_next;
        size_t buf_len = *pOut_buf_size;
        mz_uint32 i, s1 = r->m_check_adler32 & 0xffff, s2 = r->m_check_adler32 >> 16;
        size_t block_len = buf_len % 5552;

        /* 计算 Adler32 校验码 */
        while (buf_len)
        {
            for (i = 0; i + 7 < block_len; i += 8, ptr += 8)
            {
                s1 += ptr[0], s2 += s1;
                s1 += ptr[1], s2 += s1;
                s1 += ptr[2], s2 += s1;
                s1 += ptr[3], s2 += s1;
                s1 += ptr[4], s2 += s1;
                s1 += ptr[5], s2 += s1;
                s1 += ptr[6], s2 += s1;
                s1 += ptr[7], s2 += s1;
            }
            for (; i < block_len; ++i)
                s1 += *ptr++, s2 += s1;
            s1 %= 65521U, s2 %= 65521U;
            buf_len -= block_len;
            block_len = 5552;
        }

        /* 更新 Adler32 校验码 */
        r->m_check_adler32 = (s2 << 16) + s1;

        /* 如果解压完成并且需要解析 ZLIB 头部，并且 Adler32 校验码不匹配，则更新状态 */
        if ((status == TINFL_STATUS_DONE) && (decomp_flags & TINFL_FLAG_PARSE_ZLIB_HEADER) && (r->m_check_adler32 != r->m_z_adler32))
            status = TINFL_STATUS_ADLER32_MISMATCH;
    }

    /* 返回解压器的状态 */
    return status;
}

/* Higher level helper functions. */
// 用于将内存中的压缩数据解压到堆内存中，返回解压后数据的指针
void *tinfl_decompress_mem_to_heap(const void *pSrc_buf, size_t src_buf_len, size_t *pOut_len, int flags)
{
    tinfl_decompressor decomp;  // tinfl解压器对象
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    void *pBuf = NULL, *pNew_buf;  // 解压后数据的指针及新分配内存的指针
    size_t src_buf_ofs = 0, out_buf_capacity = 0;  // 源缓冲区偏移量和输出缓冲区容量
    *pOut_len = 0;  // 输出长度初始化为0
    tinfl_init(&decomp);  // 初始化tinfl解压器
    for (;;)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        size_t src_buf_size = src_buf_len - src_buf_ofs, dst_buf_size = out_buf_capacity - *pOut_len, new_out_buf_capacity;  // 源缓冲区剩余大小、目标缓冲区剩余大小及新的输出缓冲区容量
        // 进行解压操作
        tinfl_status status = tinfl_decompress(&decomp, (const mz_uint8 *)pSrc_buf + src_buf_ofs, &src_buf_size, (mz_uint8 *)pBuf, pBuf ? (mz_uint8 *)pBuf + *pOut_len : NULL, &dst_buf_size,
                                               (flags & ~TINFL_FLAG_HAS_MORE_INPUT) | TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF);
        // 如果解压出错或者需要更多输入数据，则释放内存并返回空指针
        if ((status < 0) || (status == TINFL_STATUS_NEEDS_MORE_INPUT))
        {
            MZ_FREE(pBuf);  // 释放已分配的内存
            *pOut_len = 0;  // 输出长度置为0
            return NULL;  // 返回空指针
        }
        src_buf_ofs += src_buf_size;  // 更新源缓冲区偏移量
        *pOut_len += dst_buf_size;  // 更新输出长度
        if (status == TINFL_STATUS_DONE)  // 如果解压完成则退出循环
            break;
        new_out_buf_capacity = out_buf_capacity * 2;  // 扩展输出缓冲区容量
        if (new_out_buf_capacity < 128)
            new_out_buf_capacity = 128;  // 最小扩展到128字节
        pNew_buf = MZ_REALLOC(pBuf, new_out_buf_capacity);  // 重新分配内存
        if (!pNew_buf)
        {
            MZ_FREE(pBuf);  // 内存分配失败，释放已分配的内存
            *pOut_len = 0;  // 输出长度置为0
            return NULL;  // 返回空指针
        }
        pBuf = pNew_buf;  // 更新pBuf指向新分配的内存
        out_buf_capacity = new_out_buf_capacity;  // 更新输出缓冲区容量
    }
    return pBuf;  // 返回解压后的数据指针
}

// 将内存中的压缩数据直接解压到另一段内存中
size_t tinfl_decompress_mem_to_mem(void *pOut_buf, size_t out_buf_len, const void *pSrc_buf, size_t src_buf_len, int flags)
{
    tinfl_decompressor decomp;  // tinfl解压器对象
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    tinfl_status status;  // 解压状态
    tinfl_init(&decomp);  // 初始化tinfl解压器
    status = tinfl_decompress(&decomp, (const mz_uint8 *)pSrc_buf, &src_buf_len, (mz_uint8 *)pOut_buf, (mz_uint8 *)pOut_buf, &out_buf_len, (flags & ~TINFL_FLAG_HAS_MORE_INPUT) | TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF);  // 进行解压操作
    return (status != TINFL_STATUS_DONE) ? TINFL_DECOMPRESS_MEM_TO_MEM_FAILED : out_buf_len;  // 返回解压结果，若失败返回TINFL_DECOMPRESS_MEM_TO_MEM_FAILED
}

// 将内存中的压缩数据解压到回调函数中
int tinfl_decompress_mem_to_callback(const void *pIn_buf, size_t *pIn_buf_size, tinfl_put_buf_func_ptr pPut_buf_func, void *pPut_buf_user, int flags)
{
    int result = 0;  // 函数执行结果
    tinfl_decompressor decomp;  // tinfl解压器对象
    mz_uint8 *pDict = (mz_uint8 *)MZ_MALLOC(TINFL_LZ_DICT_SIZE);  // 分配字典内存
    size_t in_buf_ofs = 0, dict_ofs = 0;  // 输入缓冲区偏移量和字典偏移量
    if (!pDict)
        return TINFL_STATUS_FAILED;  // 字典内存分配失败，返回失败状态
    tinfl_init(&decomp);  // 初始化tinfl解压器
    for (;;)
    {
        // 计算输入缓冲区和目标缓冲区的大小
        size_t in_buf_size = *pIn_buf_size - in_buf_ofs, dst_buf_size = TINFL_LZ_DICT_SIZE - dict_ofs;
        // 调用 tinfl_decompress 函数进行解压缩操作
        tinfl_status status = tinfl_decompress(&decomp, (const mz_uint8 *)pIn_buf + in_buf_ofs, &in_buf_size, pDict, pDict + dict_ofs, &dst_buf_size,
                                               (flags & ~(TINFL_FLAG_HAS_MORE_INPUT | TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF)));
        // 更新输入缓冲区偏移量
        in_buf_ofs += in_buf_size;
        // 如果目标缓冲区有数据且写入操作失败，则跳出循环
        if ((dst_buf_size) && (!(*pPut_buf_func)(pDict + dict_ofs, (int)dst_buf_size, pPut_buf_user)))
            break;
        // 如果解压缩状态不是需要更多输出，则设置结果并跳出循环
        if (status != TINFL_STATUS_HAS_MORE_OUTPUT)
        {
            result = (status == TINFL_STATUS_DONE);
            break;
        }
        // 更新字典偏移量，使用环形缓冲区的方式
        dict_ofs = (dict_ofs + dst_buf_size) & (TINFL_LZ_DICT_SIZE - 1);
    }
    // 释放动态分配的字典内存
    MZ_FREE(pDict);
    // 更新输入缓冲区大小，返回解压缩操作结果
    *pIn_buf_size = in_buf_ofs;
    return result;
}

#ifndef MINIZ_NO_MALLOC
// 分配一个 tinfl 解压缩器的内存空间
tinfl_decompressor *tinfl_decompressor_alloc(void)
{
    // 使用 MZ_MALLOC 分配内存空间
    tinfl_decompressor *pDecomp = (tinfl_decompressor *)MZ_MALLOC(sizeof(tinfl_decompressor));
    // 如果内存分配成功，则初始化 tinfl 解压缩器
    if (pDecomp)
        tinfl_init(pDecomp);
    return pDecomp;
}

// 释放 tinfl 解压缩器占用的内存空间
void tinfl_decompressor_free(tinfl_decompressor *pDecomp)
{
    // 使用 MZ_FREE 释放内存空间
    MZ_FREE(pDecomp);
}
#endif

#ifdef __cplusplus
// 如果是 C++ 环境，则关闭 extern "C" 语句块
}
#endif

/**************************************************************************
 *
 * 版权 2013-2014 RAD Game Tools 和 Valve Software
 * 版权 2010-2014 Rich Geldreich 和 Tenacious Software LLC
 * 版权 2016 Martin Raiber
 * 保留所有权利。
 *
 * 根据以下条件，无偿授予获得此软件及其相关文档文件（“软件”）的任何人使用、复制、修改、合并、发布、分发、再授权及/或销售此软件的权利：
 *
 * 上述版权声明和本许可声明应包含在
 * 所有副本或重要部分的软件中。
 *
 * 本软件按“原样”提供，不提供任何形式的明示或暗示保证，
 * 包括但不限于适销性、特定用途适用性和非侵权性的保证。在
 * 任何情况下，作者或版权持有人对任何索赔、损害或其他责任
 * 不承担责任，无论是在合同行为、侵权行为或其他方面引起的，
 * 与软件或使用或其他交易有关。
 *
 **************************************************************************/


#ifndef MINIZ_NO_ARCHIVE_APIS

#ifdef __cplusplus
// 如果是 C++ 环境，则开启 extern "C" 语句块
extern "C" {
#endif

/* ------------------- .ZIP archive reading */

#ifdef MINIZ_NO_STDIO
// 如果定义了 MINIZ_NO_STDIO，则定义 MZ_FILE 为 void *
#define MZ_FILE void *
#else
// 否则，包含标准库文件 <sys/stat.h>

#if defined(_MSC_VER) || defined(__MINGW64__)
// 如果是 Microsoft Visual Studio 或者 MinGW 64 位编译器

// 定义 mz_fopen 函数，用于打开文件
static FILE *mz_fopen(const char *pFilename, const char *pMode)
{
    FILE *pFile = NULL;
    // 使用 fopen_s 安全地打开文件
    fopen_s(&pFile, pFilename, pMode);
    return pFile;
}

// 定义 mz_freopen 函数，用于重新打开文件
static FILE *mz_freopen(const char *pPath, const char *pMode, FILE *pStream)
{
    FILE *pFile = NULL;
    // 使用 freopen_s 安全地重新打开文件
    if (freopen_s(&pFile, pPath, pMode, pStream))
        return NULL;
    return pFile;
}

// 如果不禁用时间相关功能，则包含 <sys/utime.h>
#ifndef MINIZ_NO_TIME
#include <sys/utime.h>
#endif

// 定义 MZ_FOPEN 为 mz_fopen
#define MZ_FOPEN mz_fopen
// 定义 MZ_FCLOSE 为 fclose
#define MZ_FCLOSE fclose
// 定义 MZ_FREAD 为 fread
#define MZ_FREAD fread
// 定义 MZ_FWRITE 为 fwrite
#define MZ_FWRITE fwrite
// 定义 MZ_FTELL64 为 _ftelli64
#define MZ_FTELL64 _ftelli64
// 定义 MZ_FSEEK64 为 _fseeki64
#define MZ_FSEEK64 _fseeki64
// 定义 MZ_FILE_STAT_STRUCT 为 _stat64
#define MZ_FILE_STAT_STRUCT _stat64
// 定义 MZ_FILE_STAT 为 _stat64
#define MZ_FILE_STAT _stat64
// 定义 MZ_FFLUSH 为 fflush
#define MZ_FFLUSH fflush
// 定义 MZ_FREOPEN 为 mz_freopen
#define MZ_FREOPEN mz_freopen
// 定义 MZ_DELETE_FILE 为 remove
#define MZ_DELETE_FILE remove

#elif defined(__MINGW32__)
// 如果是 MinGW 32 位编译器

// 如果不禁用时间相关功能，则包含 <sys/utime.h>
#ifndef MINIZ_NO_TIME
#include <sys/utime.h>
#endif

// 定义 MZ_FOPEN 为 fopen
#define MZ_FOPEN(f, m) fopen(f, m)
// 定义 MZ_FCLOSE 为 fclose
#define MZ_FCLOSE fclose
// 定义 MZ_FREAD 为 fread
#define MZ_FREAD fread
// 定义 MZ_FWRITE 为 fwrite
#define MZ_FWRITE fwrite
// 定义 MZ_FTELL64 为 ftello64
#define MZ_FTELL64 ftello64
// 定义 MZ_FSEEK64 为 fseeko64
#define MZ_FSEEK64 fseeko64
// 定义 MZ_FILE_STAT_STRUCT 为 _stat
#define MZ_FILE_STAT_STRUCT _stat


这些注释详细解释了每行代码的作用，确保每个功能和定义都清晰明了地被说明。
/* 定义 MZ_FILE_STAT 为 _stat */
#define MZ_FILE_STAT _stat
/* 定义 MZ_FFLUSH 为 fflush */
#define MZ_FFLUSH fflush
/* 定义 MZ_FREOPEN(f, m, s) 为 freopen(f, m, s) */
#define MZ_FREOPEN(f, m, s) freopen(f, m, s)
/* 定义 MZ_DELETE_FILE 为 remove */
#define MZ_DELETE_FILE remove

/* 如果编译环境为 Tiny C Compiler (TCC) */
#elif defined(__TINYC__)
/* 如果未定义 MINIZ_NO_TIME，则包含 <sys/utime.h> */
#ifndef MINIZ_NO_TIME
#include <sys/utime.h>
#endif
/* 定义 MZ_FOPEN(f, m) 为 fopen(f, m) */
#define MZ_FOPEN(f, m) fopen(f, m)
/* 定义 MZ_FCLOSE 为 fclose */
#define MZ_FCLOSE fclose
/* 定义 MZ_FREAD 为 fread */
#define MZ_FREAD fread
/* 定义 MZ_FWRITE 为 fwrite */
#define MZ_FWRITE fwrite
/* 定义 MZ_FTELL64 为 ftell */
#define MZ_FTELL64 ftell
/* 定义 MZ_FSEEK64 为 fseek */
#define MZ_FSEEK64 fseek
/* 定义 MZ_FILE_STAT_STRUCT 为 stat */
#define MZ_FILE_STAT_STRUCT stat
/* 定义 MZ_FILE_STAT 为 stat */
#define MZ_FILE_STAT stat
/* 定义 MZ_FFLUSH 为 fflush */
#define MZ_FFLUSH fflush
/* 定义 MZ_FREOPEN(f, m, s) 为 freopen(f, m, s) */
#define MZ_FREOPEN(f, m, s) freopen(f, m, s)
/* 定义 MZ_DELETE_FILE 为 remove */
#define MZ_DELETE_FILE remove

/* 如果编译环境为 GCC 并且定义了 _LARGEFILE64_SOURCE */
#elif defined(__GNUC__) && defined(_LARGEFILE64_SOURCE)
/* 如果未定义 MINIZ_NO_TIME，则包含 <utime.h> */
#ifndef MINIZ_NO_TIME
#include <utime.h>
#endif
/* 定义 MZ_FOPEN(f, m) 为 fopen64(f, m) */
#define MZ_FOPEN(f, m) fopen64(f, m)
/* 定义 MZ_FCLOSE 为 fclose */
#define MZ_FCLOSE fclose
/* 定义 MZ_FREAD 为 fread */
#define MZ_FREAD fread
/* 定义 MZ_FWRITE 为 fwrite */
#define MZ_FWRITE fwrite
/* 定义 MZ_FTELL64 为 ftello64 */
#define MZ_FTELL64 ftello64
/* 定义 MZ_FSEEK64 为 fseeko64 */
#define MZ_FSEEK64 fseeko64
/* 定义 MZ_FILE_STAT_STRUCT 为 stat64 */
#define MZ_FILE_STAT_STRUCT stat64
/* 定义 MZ_FILE_STAT 为 stat64 */
#define MZ_FILE_STAT stat64
/* 定义 MZ_FFLUSH 为 fflush */
#define MZ_FFLUSH fflush
/* 定义 MZ_FREOPEN(p, m, s) 为 freopen64(p, m, s) */
#define MZ_FREOPEN(p, m, s) freopen64(p, m, s)
/* 定义 MZ_DELETE_FILE 为 remove */
#define MZ_DELETE_FILE remove

/* 如果编译环境为苹果平台 */
#elif defined(__APPLE__)
/* 如果未定义 MINIZ_NO_TIME，则包含 <utime.h> */
#ifndef MINIZ_NO_TIME
#include <utime.h>
#endif
/* 定义 MZ_FOPEN(f, m) 为 fopen(f, m) */
#define MZ_FOPEN(f, m) fopen(f, m)
/* 定义 MZ_FCLOSE 为 fclose */
#define MZ_FCLOSE fclose
/* 定义 MZ_FREAD 为 fread */
#define MZ_FREAD fread
/* 定义 MZ_FWRITE 为 fwrite */
#define MZ_FWRITE fwrite
/* 定义 MZ_FTELL64 为 ftello */
#define MZ_FTELL64 ftello
/* 定义 MZ_FSEEK64 为 fseeko */
#define MZ_FSEEK64 fseeko
/* 定义 MZ_FILE_STAT_STRUCT 为 stat */
#define MZ_FILE_STAT_STRUCT stat
/* 定义 MZ_FILE_STAT 为 stat */
#define MZ_FILE_STAT stat
/* 定义 MZ_FFLUSH 为 fflush */
#define MZ_FFLUSH fflush
/* 定义 MZ_FREOPEN(p, m, s) 为 freopen(p, m, s) */
#define MZ_FREOPEN(p, m, s) freopen(p, m, s)
/* 定义 MZ_DELETE_FILE 为 remove */

/* 如果以上条件都不符合 */
#else
/* 发出消息提示：使用 fopen、ftello、fseeko、stat() 等路径进行文件 I/O 操作，可能不支持大文件 */
#pragma message("Using fopen, ftello, fseeko, stat() etc. path for file I/O - this path may not support large files.")
/* 如果未定义 MINIZ_NO_TIME，则包含 <utime.h> */
#ifndef MINIZ_NO_TIME
#include <utime.h>
#endif
/* 定义 MZ_FOPEN(f, m) 为 fopen(f, m) */
#define MZ_FOPEN(f, m) fopen(f, m)
/* 定义 MZ_FCLOSE 为 fclose */
#define MZ_FCLOSE fclose
/* 定义 MZ_FREAD 为 fread */
#define MZ_FREAD fread
/* 定义 MZ_FWRITE 为 fwrite */
#define MZ_FWRITE fwrite
/* 如果定义了 __STRICT_ANSI__，则定义 MZ_FTELL64 为 ftell */
/* 否则定义 MZ_FTELL64 为 ftello */
#ifdef __STRICT_ANSI__
#define MZ_FTELL64 ftell
#else
#define MZ_FTELL64 ftello
#endif
/* 如果定义了 __STRICT_ANSI__，则定义 MZ_FSEEK64 为 fseek */
/* 否则定义 MZ_FSEEK64 为 fseeko */
#ifdef __STRICT_ANSI__
#define MZ_FSEEK64 fseek
#else
#define MZ_FSEEK64 fseeko
#endif
/* 定义 MZ_FILE_STAT_STRUCT 为 stat */
#define MZ_FILE_STAT_STRUCT stat
/* 定义 MZ_FILE_STAT 为 stat */
#define MZ_FILE_STAT stat
/* 定义 MZ_FFLUSH 为 fflush */
#define MZ_FFLUSH fflush
/* 定义 MZ_FREOPEN(f, m, s) 为 freopen(f, m, s) */
#define MZ_FREOPEN(f, m, s) freopen(f, m, s)
/* 定义 MZ_DELETE_FILE 为 remove */
#define MZ_DELETE_FILE remove
#endif /* #ifdef _MSC_VER */
#endif /* #ifdef MINIZ_NO_STDIO */

/* 定义 MZ_TOLOWER(c) 为将大写字母转换为小写字母的宏定义 */
#define MZ_TOLOWER(c) ((((c) >= 'A') && ((c) <= 'Z')) ? ((c) - 'A' + 'a') : (c))

/* 各种 ZIP 存档的枚举值和大小定义。为了避免跨平台编译器对齐和平台字节序问题，miniz.c 中不使用结构体定义这些内容。 */
enum
{
    /* ZIP 存档标识符和记录大小 */
    MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIG = 0x06054b50,
    MZ_ZIP_CENTRAL_DIR_HEADER_SIG = 0x02014b50,
    MZ_ZIP_LOCAL_DIR_HEADER_SIG = 0x04034b50,
    MZ_ZIP_LOCAL_DIR_HEADER_SIZE = 30,
    MZ_ZIP_CENTRAL_DIR_HEADER_SIZE = 46,
    MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE = 22,

    /* ZIP64 存档标识符和记录大小 */
    MZ_ZIP64_END_OF_CENTRAL_DIR_HEADER_SIG = 0x06064b50,
    MZ_ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIG = 0x07064b50,
    MZ_ZIP64_END_OF_CENTRAL_DIR_HEADER_SIZE = 56,
    MZ_ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIZE = 20,
    MZ_ZIP64_EXTENDED_INFORMATION_FIELD_HEADER_ID = 0x0001,
    MZ_ZIP_DATA_DESCRIPTOR_ID = 0x08074b50,
    MZ_ZIP_DATA_DESCRIPTER_SIZE64 = 24,
    // 定义 ZIP 数据描述器大小（32位），用于指示数据偏移量
    MZ_ZIP_DATA_DESCRIPTER_SIZE32 = 16,

    // 中央目录头记录的偏移量定义
    MZ_ZIP_CDH_SIG_OFS = 0,                             // 中央目录头记录签名的偏移量
    MZ_ZIP_CDH_VERSION_MADE_BY_OFS = 4,                 // 中央目录头记录中创建 ZIP 的版本偏移量
    MZ_ZIP_CDH_VERSION_NEEDED_OFS = 6,                  // 中央目录头记录中所需 ZIP 版本的偏移量
    MZ_ZIP_CDH_BIT_FLAG_OFS = 8,                        // 中央目录头记录中位标志的偏移量
    MZ_ZIP_CDH_METHOD_OFS = 10,                         // 中央目录头记录中压缩方法的偏移量
    MZ_ZIP_CDH_FILE_TIME_OFS = 12,                      // 中央目录头记录中文件时间的偏移量
    MZ_ZIP_CDH_FILE_DATE_OFS = 14,                      // 中央目录头记录中文件日期的偏移量
    MZ_ZIP_CDH_CRC32_OFS = 16,                          // 中央目录头记录中 CRC32 校验和的偏移量
    MZ_ZIP_CDH_COMPRESSED_SIZE_OFS = 20,                // 中央目录头记录中压缩后大小的偏移量
    MZ_ZIP_CDH_DECOMPRESSED_SIZE_OFS = 24,              // 中央目录头记录中解压后大小的偏移量
    MZ_ZIP_CDH_FILENAME_LEN_OFS = 28,                   // 中央目录头记录中文件名长度的偏移量
    MZ_ZIP_CDH_EXTRA_LEN_OFS = 30,                      // 中央目录头记录中额外字段长度的偏移量
    MZ_ZIP_CDH_COMMENT_LEN_OFS = 32,                    // 中央目录头记录中注释长度的偏移量
    MZ_ZIP_CDH_DISK_START_OFS = 34,                     // 中央目录头记录中文件起始磁盘编号的偏移量
    MZ_ZIP_CDH_INTERNAL_ATTR_OFS = 36,                  // 中央目录头记录中内部文件属性的偏移量
    MZ_ZIP_CDH_EXTERNAL_ATTR_OFS = 38,                  // 中央目录头记录中外部文件属性的偏移量
    MZ_ZIP_CDH_LOCAL_HEADER_OFS = 42,                   // 中央目录头记录中本地文件头的偏移量

    // 本地文件头偏移量定义
    MZ_ZIP_LDH_SIG_OFS = 0,                             // 本地文件头签名的偏移量
    MZ_ZIP_LDH_VERSION_NEEDED_OFS = 4,                  // 本地文件头中所需 ZIP 版本的偏移量
    MZ_ZIP_LDH_BIT_FLAG_OFS = 6,                        // 本地文件头中位标志的偏移量
    MZ_ZIP_LDH_METHOD_OFS = 8,                          // 本地文件头中压缩方法的偏移量
    MZ_ZIP_LDH_FILE_TIME_OFS = 10,                      // 本地文件头中文件时间的偏移量
    MZ_ZIP_LDH_FILE_DATE_OFS = 12,                      // 本地文件头中文件日期的偏移量
    MZ_ZIP_LDH_CRC32_OFS = 14,                          // 本地文件头中 CRC32 校验和的偏移量
    MZ_ZIP_LDH_COMPRESSED_SIZE_OFS = 18,                // 本地文件头中压缩后大小的偏移量
    MZ_ZIP_LDH_DECOMPRESSED_SIZE_OFS = 22,              // 本地文件头中解压后大小的偏移量
    MZ_ZIP_LDH_FILENAME_LEN_OFS = 26,                   // 本地文件头中文件名长度的偏移量
    MZ_ZIP_LDH_EXTRA_LEN_OFS = 28,                      // 本地文件头中额外字段长度的偏移量
    MZ_ZIP_LDH_BIT_FLAG_HAS_LOCATOR = 1 << 3,           // 本地文件头位标志中是否有定位器的标志位

    // 中央目录结束结构偏移量定义
    MZ_ZIP_ECDH_SIG_OFS = 0,                            // 中央目录结束结构签名的偏移量
    MZ_ZIP_ECDH_NUM_THIS_DISK_OFS = 4,                  // 中央目录结束结构中本磁盘编号的偏移量
    MZ_ZIP_ECDH_NUM_DISK_CDIR_OFS = 6,                  // 中央目录结束结构中中央目录总数的偏移量
    MZ_ZIP_ECDH_CDIR_NUM_ENTRIES_ON_DISK_OFS = 8,       // 中央目录结束结构中本磁盘中的中央目录条目数的偏移量
    MZ_ZIP_ECDH_CDIR_TOTAL_ENTRIES_OFS = 10,            // 中央目录结束结构中总的中央目录条目数的偏移量
    MZ_ZIP_ECDH_CDIR_SIZE_OFS = 12,                     // 中央目录结束结构中中央目录大小的偏移量
    MZ_ZIP_ECDH_CDIR_OFS_OFS = 16,                      // 中央目录结束结构中中央目录的偏移量的偏移量
    MZ_ZIP_ECDH_COMMENT_SIZE_OFS = 20,                  // 中央目录结束结构中注释长度的偏移量

    // ZIP64 中央目录结束结构定位器偏移量定义
    MZ_ZIP64_ECDL_SIG_OFS = 0,                          // ZIP64 中央目录结束结构定位器签名的偏移量（4字节）
    MZ_ZIP64_ECDL_NUM_DISK_CDIR_OFS = 4,                // ZIP64 中央目录结束结构定位器中中央目录总数的偏移量（4字节）
    MZ_ZIP64_ECDL_REL_OFS_TO_ZIP64_ECDR_OFS = 8,        // ZIP64 中央目录结束结构定位器到 ZIP64 中央目录结束结构的相对偏移量（8字节）
    MZ_ZIP64_ECDL_TOTAL_NUMBER_OF_DISKS_OFS = 16,       // ZIP64 中央目录结束结构定位器中总的磁盘数目（4字节）

    // ZIP64 中央目录结束结构偏移量定义
    MZ_ZIP64_ECDH_SIG_OFS = 0,                          // ZIP64 中央目录结束结构签名的偏移量（4字节）
    MZ_ZIP64_ECDH_SIZE_OF_RECORD_OFS = 4,               // ZIP64 中央目录结束结构记录大小的偏移量（8字节）
    MZ_ZIP64_ECDH_VERSION_MADE_BY_OFS = 12,             // ZIP64 中央目录结束结构中创建 ZIP 的版本偏移量（2字节）
    MZ_ZIP64_ECDH_VERSION_NEEDED_OFS = 14,              // ZIP64 中央目录结束结构中所需 ZIP 版本的偏移量（2字节）
    MZ_ZIP64_ECDH_NUM_THIS_DISK_OFS = 16,               // ZIP64 中央目录结束结构中本磁盘编号的偏移量（4字节）
    MZ_ZIP64_ECDH_NUM_DISK_CDIR_OFS = 20,               // ZIP64 中央目录结束结构中中央目录总数的偏移量（4字节）
    MZ_ZIP64_ECDH_CDIR_NUM_ENTRIES_ON_DISK_OFS = 24,    // ZIP64 中央目录结束结构中本磁盘中的中央目录条目数的偏移量（8字节）
    MZ_ZIP64_ECDH_CDIR_TOTAL_ENTRIES_OFS = 32,          // ZIP64 中央目录结束结构中总的中央目录条目数
};

// 定义一个结构体 mz_zip_array，用于管理动态数组
typedef struct
{
    void *m_p;                    // 指向数组数据的指针
    size_t m_size, m_capacity;    // 数组当前大小和容量
    mz_uint m_element_size;       // 单个元素的大小
} mz_zip_array;

// 定义 mz_zip_internal_state_tag 结构体，用于存储 ZIP 内部状态信息
struct mz_zip_internal_state_tag
{
    mz_zip_array m_central_dir;               // 中央目录的动态数组
    mz_zip_array m_central_dir_offsets;       // 中央目录偏移的动态数组
    mz_zip_array m_sorted_central_dir_offsets;// 排序后的中央目录偏移的动态数组

    /* 初始化时传入的标志位 */
    uint32_t m_init_flags;

    /* 如果存档具有 zip64 的结束中央目录头部等，则为 MZ_TRUE */
    mz_bool m_zip64;

    /* 如果在中央目录中找到 zip64 扩展信息，则为 MZ_TRUE（即使未找到 zip64 结尾中央目录头部等，m_zip64 也将被设置为 true） */
    mz_bool m_zip64_has_extended_info_fields;

    /* 这些字段由文件、FILE、内存和内存/堆读写辅助器使用 */
    MZ_FILE *m_pFile;               // 文件指针
    mz_uint64 m_file_archive_start_ofs;  // 文件存档起始偏移量

    void *m_pMem;                   // 内存指针
    size_t m_mem_size;              // 内存大小
    size_t m_mem_capacity;          // 内存容量
};

// 定义宏 MZ_ZIP_ARRAY_SET_ELEMENT_SIZE，用于设置动态数组的元素大小
#define MZ_ZIP_ARRAY_SET_ELEMENT_SIZE(array_ptr, element_size) (array_ptr)->m_element_size = element_size

// 如果定义了 DEBUG、_DEBUG 或 NDEBUG，则定义静态内联函数 mz_zip_array_range_check
#if defined(DEBUG) || defined(_DEBUG) || defined(NDEBUG)
static MZ_FORCEINLINE mz_uint mz_zip_array_range_check(const mz_zip_array *pArray, mz_uint index)
{
    MZ_ASSERT(index < pArray->m_size);    // 断言索引小于数组当前大小
    return index;                         // 返回索引
}
#define MZ_ZIP_ARRAY_ELEMENT(array_ptr, element_type, index) ((element_type *)((array_ptr)->m_p))[mz_zip_array_range_check(array_ptr, index)]
#else
#define MZ_ZIP_ARRAY_ELEMENT(array_ptr, element_type, index) ((element_type *)((array_ptr)->m_p))[index]
#endif

// 定义静态内联函数 mz_zip_array_init，用于初始化动态数组
static MZ_FORCEINLINE void mz_zip_array_init(mz_zip_array *pArray, mz_uint32 element_size)
{
    memset(pArray, 0, sizeof(mz_zip_array));    // 清空动态数组结构体
    pArray->m_element_size = element_size;      // 设置元素大小
}

// 定义静态内联函数 mz_zip_array_clear，用于清空动态数组
static MZ_FORCEINLINE void mz_zip_array_clear(mz_zip_archive *pZip, mz_zip_array *pArray)
{
    pZip->m_pFree(pZip->m_pAlloc_opaque, pArray->m_p);   // 调用释放内存的函数
    memset(pArray, 0, sizeof(mz_zip_array));             // 清空动态数组结构体
}

// 定义静态函数 mz_zip_array_ensure_capacity，用于确保动态数组容量足够
static mz_bool mz_zip_array_ensure_capacity(mz_zip_archive *pZip, mz_zip_array *pArray, size_t min_new_capacity, mz_uint growing)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    void *pNew_p;
    size_t new_capacity = min_new_capacity;
    MZ_ASSERT(pArray->m_element_size);    // 断言元素大小已设置
    if (pArray->m_capacity >= min_new_capacity)
        return MZ_TRUE;                  // 如果当前容量已经满足需求，则返回真

    if (growing)
    {
        new_capacity = MZ_MAX(1, pArray->m_capacity);   // 计算新容量
        while (new_capacity < min_new_capacity)
            new_capacity *= 2;            // 如果增长标志为真，倍增新容量直到满足需求
    }

    // 重新分配内存并更新动态数组结构体
    if (NULL == (pNew_p = pZip->m_pRealloc(pZip->m_pAlloc_opaque, pArray->m_p, pArray->m_element_size, new_capacity)))
        return MZ_FALSE;                 // 内存分配失败则返回假

    pArray->m_p = pNew_p;                // 更新数组指针
    pArray->m_capacity = new_capacity;   // 更新容量
    return MZ_TRUE;                      // 返回真表示操作成功
}

// 定义静态内联函数 mz_zip_array_reserve，用于预留动态数组的容量
static MZ_FORCEINLINE mz_bool mz_zip_array_reserve(mz_zip_archive *pZip, mz_zip_array *pArray, size_t new_capacity, mz_uint growing)
{
    if (new_capacity > pArray->m_capacity)
    {
        if (!mz_zip_array_ensure_capacity(pZip, pArray, new_capacity, growing))
            return MZ_FALSE;            // 确保容量不小于指定大小，如果失败则返回假
    }
    return MZ_TRUE;                     // 返回真表示操作成功
}
static MZ_FORCEINLINE mz_bool mz_zip_array_resize(mz_zip_archive *pZip, mz_zip_array *pArray, size_t new_size, mz_uint growing)
{
    // 如果新的大小超过当前容量
    if (new_size > pArray->m_capacity)
    {
        // 确保数组能容纳新的大小，如果无法分配则返回失败
        if (!mz_zip_array_ensure_capacity(pZip, pArray, new_size, growing))
            return MZ_FALSE;
    }
    // 更新数组的大小为新的大小
    pArray->m_size = new_size;
    return MZ_TRUE;
}

static MZ_FORCEINLINE mz_bool mz_zip_array_ensure_room(mz_zip_archive *pZip, mz_zip_array *pArray, size_t n)
{
    // 确保数组有足够的空间容纳额外的 n 个元素
    return mz_zip_array_reserve(pZip, pArray, pArray->m_size + n, MZ_TRUE);
}

static MZ_FORCEINLINE mz_bool mz_zip_array_push_back(mz_zip_archive *pZip, mz_zip_array *pArray, const void *pElements, size_t n)
{
    // 保存原始大小
    size_t orig_size = pArray->m_size;
    // 调整数组大小以容纳新增的 n 个元素，如果调整失败则返回失败
    if (!mz_zip_array_resize(pZip, pArray, orig_size + n, MZ_TRUE))
        return MZ_FALSE;
    // 如果 n 大于 0，则将 pElements 指向的数据复制到数组末尾
    if (n > 0)
        memcpy((mz_uint8 *)pArray->m_p + orig_size * pArray->m_element_size, pElements, n * pArray->m_element_size);
    return MZ_TRUE;
}

#ifndef MINIZ_NO_TIME
static MZ_TIME_T mz_zip_dos_to_time_t(int dos_time, int dos_date)
{
    struct tm tm;
    memset(&tm, 0, sizeof(tm));
    tm.tm_isdst = -1;
    // 解析 DOS 时间和日期并转换为标准时间
    tm.tm_year = ((dos_date >> 9) & 127) + 1980 - 1900;
    tm.tm_mon = ((dos_date >> 5) & 15) - 1;
    tm.tm_mday = dos_date & 31;
    tm.tm_hour = (dos_time >> 11) & 31;
    tm.tm_min = (dos_time >> 5) & 63;
    tm.tm_sec = (dos_time << 1) & 62;
    return mktime(&tm);
}

#ifndef MINIZ_NO_ARCHIVE_WRITING_APIS
static void mz_zip_time_t_to_dos_time(MZ_TIME_T time, mz_uint16 *pDOS_time, mz_uint16 *pDOS_date)
{
#ifdef _MSC_VER
    struct tm tm_struct;
    struct tm *tm = &tm_struct;
    // 使用安全的 localtime_s 函数获取本地时间
    errno_t err = localtime_s(tm, &time);
    if (err)
    {
        *pDOS_date = 0;
        *pDOS_time = 0;
        return;
    }
#else
    // 使用 localtime 函数获取本地时间
    struct tm *tm = localtime(&time);
#endif /* #ifdef _MSC_VER */

    // 将标准时间转换为 DOS 时间和日期格式
    *pDOS_time = (mz_uint16)(((tm->tm_hour) << 11) + ((tm->tm_min) << 5) + ((tm->tm_sec) >> 1));
    *pDOS_date = (mz_uint16)(((tm->tm_year + 1900 - 1980) << 9) + ((tm->tm_mon + 1) << 5) + tm->tm_mday);
}
#endif /* MINIZ_NO_ARCHIVE_WRITING_APIS */

#ifndef MINIZ_NO_STDIO
#ifndef MINIZ_NO_ARCHIVE_WRITING_APIS
static mz_bool mz_zip_get_file_modified_time(const char *pFilename, MZ_TIME_T *pTime)
{
    // 获取文件的修改时间
    struct MZ_FILE_STAT_STRUCT file_stat;

    /* On Linux with x86 glibc, this call will fail on large files (I think >= 0x80000000 bytes) unless you compiled with _LARGEFILE64_SOURCE. Argh. */
    // 获取文件的状态信息，如果失败则返回 false
    if (MZ_FILE_STAT(pFilename, &file_stat) != 0)
        return MZ_FALSE;

    *pTime = file_stat.st_mtime;

    return MZ_TRUE;
}
#endif /* #ifndef MINIZ_NO_ARCHIVE_WRITING_APIS*/

static mz_bool mz_zip_set_file_times(const char *pFilename, MZ_TIME_T access_time, MZ_TIME_T modified_time)
{
    // 设置文件的访问时间和修改时间
    struct utimbuf t;

    memset(&t, 0, sizeof(t));
    t.actime = access_time;
    t.modtime = modified_time;

    // 调用 utime 函数来设置文件时间，成功返回 true，否则返回 false
    return !utime(pFilename, &t);
}
#endif /* #ifndef MINIZ_NO_STDIO */
#endif /* #ifndef MINIZ_NO_TIME */

static MZ_FORCEINLINE mz_bool mz_zip_set_error(mz_zip_archive *pZip, mz_zip_error err_num)
{
    // 如果指针 pZip 不为空，则将其成员变量 m_last_error 设置为 err_num
    if (pZip)
        pZip->m_last_error = err_num;
    // 返回 MZ_FALSE，表示操作失败
    return MZ_FALSE;
}

static mz_bool mz_zip_reader_init_internal(mz_zip_archive *pZip, mz_uint flags)
{
    // 忽略 flags 参数

    // 检查 pZip 指针是否为空，或者其状态已经初始化，或者其 zip 模式不是无效模式
    if ((!pZip) || (pZip->m_pState) || (pZip->m_zip_mode != MZ_ZIP_MODE_INVALID))
        // 设置错误状态为 MZ_ZIP_INVALID_PARAMETER，返回失败
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 如果 pZip 的内存分配函数未设置，则使用默认函数
    if (!pZip->m_pAlloc)
        pZip->m_pAlloc = miniz_def_alloc_func;
    if (!pZip->m_pFree)
        pZip->m_pFree = miniz_def_free_func;
    if (!pZip->m_pRealloc)
        pZip->m_pRealloc = miniz_def_realloc_func;

    // 初始化 pZip 的各个状态变量
    pZip->m_archive_size = 0;
    pZip->m_central_directory_file_ofs = 0;
    pZip->m_total_files = 0;
    pZip->m_last_error = MZ_ZIP_NO_ERROR;

    // 分配并初始化 pZip 的内部状态结构体
    if (NULL == (pZip->m_pState = (mz_zip_internal_state *)pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1, sizeof(mz_zip_internal_state))))
        // 如果分配失败，设置错误状态为 MZ_ZIP_ALLOC_FAILED，返回失败
        return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);

    // 清空 pZip 的内部状态结构体
    memset(pZip->m_pState, 0, sizeof(mz_zip_internal_state));
    // 设置 pZip 内部状态结构体中数组的元素大小
    MZ_ZIP_ARRAY_SET_ELEMENT_SIZE(&pZip->m_pState->m_central_dir, sizeof(mz_uint8));
    MZ_ZIP_ARRAY_SET_ELEMENT_SIZE(&pZip->m_pState->m_central_dir_offsets, sizeof(mz_uint32));
    MZ_ZIP_ARRAY_SET_ELEMENT_SIZE(&pZip->m_pState->m_sorted_central_dir_offsets, sizeof(mz_uint32));
    // 设置 pZip 内部状态结构体的初始化标志和 zip64 标志
    pZip->m_pState->m_init_flags = flags;
    pZip->m_pState->m_zip64 = MZ_FALSE;
    pZip->m_pState->m_zip64_has_extended_info_fields = MZ_FALSE;

    // 设置 pZip 的 zip 模式为读取模式
    pZip->m_zip_mode = MZ_ZIP_MODE_READING;

    // 返回成功状态
    return MZ_TRUE;
}

static MZ_FORCEINLINE mz_bool mz_zip_reader_filename_less(const mz_zip_array *pCentral_dir_array, const mz_zip_array *pCentral_dir_offsets, mz_uint l_index, mz_uint r_index)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 获取左右索引的文件名长度和文件名的起始指针
    const mz_uint8 *pL = &MZ_ZIP_ARRAY_ELEMENT(pCentral_dir_array, mz_uint8, MZ_ZIP_ARRAY_ELEMENT(pCentral_dir_offsets, mz_uint32, l_index)), *pE;
    const mz_uint8 *pR = &MZ_ZIP_ARRAY_ELEMENT(pCentral_dir_array, mz_uint8, MZ_ZIP_ARRAY_ELEMENT(pCentral_dir_offsets, mz_uint32, r_index));
    mz_uint l_len = MZ_READ_LE16(pL + MZ_ZIP_CDH_FILENAME_LEN_OFS), r_len = MZ_READ_LE16(pR + MZ_ZIP_CDH_FILENAME_LEN_OFS);
    mz_uint8 l = 0, r = 0;
    // 跳过中央目录头部，比较两个文件名的大小写不敏感排序
    pL += MZ_ZIP_CENTRAL_DIR_HEADER_SIZE;
    pR += MZ_ZIP_CENTRAL_DIR_HEADER_SIZE;
    pE = pL + MZ_MIN(l_len, r_len);
    while (pL < pE)
    {
        // 将字符转换为小写并比较
        if ((l = MZ_TOLOWER(*pL)) != (r = MZ_TOLOWER(*pR)))
            break;
        pL++;
        pR++;
    }
    // 如果前缀相同，则按照文件名长度排序
    return (pL == pE) ? (l_len < r_len) : (l < r);
}

#define MZ_SWAP_UINT32(a, b) \
    do                       \
    {                        \
        // 交换两个 mz_uint32 类型的变量 a 和 b 的值
        mz_uint32 t = a;     \
        a = b;               \
        b = t;               \
    }                        \
    MZ_MACRO_END

/* 使用堆排序对文件名进行小写化处理，加快 mz_zip_reader_locate_file() 的普通中央目录搜索速度。（也可以使用 qsort()，但可能会分配内存。） */
static void mz_zip_reader_sort_central_dir_offsets_by_filename(mz_zip_archive *pZip)
{
    // 获取 pZip 的内部状态结构体指针
    mz_zip_internal_state *pState = pZip->m_pState;
    // 使用指针指向 ZIP 文件状态中的中央目录偏移数组
    const mz_zip_array *pCentral_dir_offsets = &pState->m_central_dir_offsets;
    // 使用指针指向 ZIP 文件状态中的中央目录数组
    const mz_zip_array *pCentral_dir = &pState->m_central_dir;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 声明指向 mz_uint32 的指针 pIndices，但不初始化
    mz_uint32 *pIndices;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 声明 start 和 end 作为 mz_uint32 类型，但不初始化
    mz_uint32 start, end;
    // 获取 ZIP 文件总文件数，并赋值给 size
    const mz_uint32 size = pZip->m_total_files;

    // 若文件总数小于等于 1，则直接返回
    if (size <= 1U)
        return;

    // 将 pIndices 指向排序后的中央目录偏移数组的第一个元素
    pIndices = &MZ_ZIP_ARRAY_ELEMENT(&pState->m_sorted_central_dir_offsets, mz_uint32, 0);

    // 计算堆排序的起始位置
    start = (size - 2U) >> 1U;
    // 开始堆排序的循环
    for (;;)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        // 声明 child 和 root 为 mz_uint64 类型，并初始化 root 为 start
        mz_uint64 child, root = start;
        // 内层循环找出子节点中较大的节点
        for (;;)
        {
            // 如果 child 节点的索引超出文件总数，则退出循环
            if ((child = (root << 1U) + 1U) >= size)
                break;
            // 如果右子节点存在且右子节点比左子节点小，则选择右子节点
            child += (((child + 1U) < size) && (mz_zip_reader_filename_less(pCentral_dir, pCentral_dir_offsets, pIndices[child], pIndices[child + 1U])));
            // 如果根节点不比子节点小，则退出循环
            if (!mz_zip_reader_filename_less(pCentral_dir, pCentral_dir_offsets, pIndices[root], pIndices[child]))
                break;
            // 交换根节点和子节点的索引
            MZ_SWAP_UINT32(pIndices[root], pIndices[child]);
            // 更新根节点索引为子节点索引
            root = child;
        }
        // 如果 start 等于 0，则退出外层循环
        if (!start)
            break;
        // 更新 start 的值
        start--;
    }

    // 设置 end 初始值为 size - 1
    end = size - 1;
    // 开始堆排序的循环，从末尾向前调整堆
    while (end > 0)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        // 声明 child 和 root 为 mz_uint64 类型，并初始化 root 为 0
        mz_uint64 child, root = 0;
        // 将末尾元素与根节点元素交换
        MZ_SWAP_UINT32(pIndices[end], pIndices[0]);
        // 内层循环找出子节点中较大的节点
        for (;;)
        {
            // 如果 child 节点的索引超出 end，则退出循环
            if ((child = (root << 1U) + 1U) >= end)
                break;
            // 如果右子节点存在且右子节点比左子节点小，则选择右子节点
            child += (((child + 1U) < end) && mz_zip_reader_filename_less(pCentral_dir, pCentral_dir_offsets, pIndices[child], pIndices[child + 1U]));
            // 如果根节点不比子节点小，则退出循环
            if (!mz_zip_reader_filename_less(pCentral_dir, pCentral_dir_offsets, pIndices[root], pIndices[child]))
                break;
            // 交换根节点和子节点的索引
            MZ_SWAP_UINT32(pIndices[root], pIndices[child]);
            // 更新根节点索引为子节点索引
            root = child;
        }
        // 更新 end 的值
        end--;
    }
}
    // 如果未能定位到 ZIP 文件的中央目录结束标志，返回查找中央目录失败的错误状态
    if (!mz_zip_reader_locate_header_sig(pZip, MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIG, MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE, &cur_file_ofs))
        return mz_zip_set_error(pZip, MZ_ZIP_FAILED_FINDING_CENTRAL_DIR);

    /* 读取并验证中央目录结束记录。*/
    // 从当前文件偏移处读取指定大小的数据到 pBuf，并检查读取是否成功
    if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pBuf, MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE) != MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE)
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);

    // 检查读取的数据是否包含有效的中央目录结束标志
    if (MZ_READ_LE32(pBuf + MZ_ZIP_ECDH_SIG_OFS) != MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIG)
        return mz_zip_set_error(pZip, MZ_ZIP_NOT_AN_ARCHIVE);

    // 如果当前文件偏移大于等于 ZIP64 中央目录结束定位器和头部的大小
    if (cur_file_ofs >= (MZ_ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIZE + MZ_ZIP64_END_OF_CENTRAL_DIR_HEADER_SIZE))
    {
        // 从当前文件偏移减去 ZIP64 中央目录结束定位器的大小处开始读取 ZIP64 定位器数据
        if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs - MZ_ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIZE, pZip64_locator, MZ_ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIZE) == MZ_ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIZE)
        {
            // 检查读取的 ZIP64 定位器数据是否包含有效的 ZIP64 中央目录结束定位器标志
            if (MZ_READ_LE32(pZip64_locator + MZ_ZIP64_ECDL_SIG_OFS) == MZ_ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIG)
            {
                // 从 ZIP64 中央目录结束定位器偏移处读取 ZIP64 中央目录结束记录的偏移
                zip64_end_of_central_dir_ofs = MZ_READ_LE64(pZip64_locator + MZ_ZIP64_ECDL_REL_OFS_TO_ZIP64_ECDR_OFS);
                // 如果 ZIP64 中央目录结束记录偏移超出文件大小减去 ZIP64 中央目录结束头的大小，返回不是有效的 ZIP 文件的错误状态
                if (zip64_end_of_central_dir_ofs > (pZip->m_archive_size - MZ_ZIP64_END_OF_CENTRAL_DIR_HEADER_SIZE))
                    return mz_zip_set_error(pZip, MZ_ZIP_NOT_AN_ARCHIVE);

                // 从 ZIP64 中央目录结束记录偏移处读取 ZIP64 中央目录结束记录的数据
                if (pZip->m_pRead(pZip->m_pIO_opaque, zip64_end_of_central_dir_ofs, pZip64_end_of_central_dir, MZ_ZIP64_END_OF_CENTRAL_DIR_HEADER_SIZE) == MZ_ZIP64_END_OF_CENTRAL_DIR_HEADER_SIZE)
                {
                    // 检查读取的 ZIP64 中央目录结束记录数据是否包含有效的 ZIP64 中央目录结束记录标志
                    if (MZ_READ_LE32(pZip64_end_of_central_dir + MZ_ZIP64_ECDH_SIG_OFS) == MZ_ZIP64_END_OF_CENTRAL_DIR_HEADER_SIG)
                    {
                        // 设置 ZIP 文件状态为 ZIP64
                        pZip->m_pState->m_zip64 = MZ_TRUE;
                    }
                }
            }
        }
    }

    // 从中央目录结束头部数据中读取总文件数
    pZip->m_total_files = MZ_READ_LE16(pBuf + MZ_ZIP_ECDH_CDIR_TOTAL_ENTRIES_OFS);
    // 从中央目录结束头部数据中读取当前磁盘上的目录项数
    cdir_entries_on_this_disk = MZ_READ_LE16(pBuf + MZ_ZIP_ECDH_CDIR_NUM_ENTRIES_ON_DISK_OFS);
    // 从中央目录结束头部数据中读取当前磁盘编号
    num_this_disk = MZ_READ_LE16(pBuf + MZ_ZIP_ECDH_NUM_THIS_DISK_OFS);
    // 从中央目录结束头部数据中读取中央目录开始位置所在的磁盘编号
    cdir_disk_index = MZ_READ_LE16(pBuf + MZ_ZIP_ECDH_NUM_DISK_CDIR_OFS);
    // 从中央目录结束头部数据中读取中央目录的大小
    cdir_size = MZ_READ_LE32(pBuf + MZ_ZIP_ECDH_CDIR_SIZE_OFS);
    // 从中央目录结束头部数据中读取中央目录的偏移量
    cdir_ofs = MZ_READ_LE32(pBuf + MZ_ZIP_ECDH_CDIR_OFS_OFS);

    // 如果 ZIP 文件状态为 ZIP64，则执行以下操作
    if (pZip->m_pState->m_zip64)
    {
        // 读取 ZIP64 中的总磁盘数目
        mz_uint32 zip64_total_num_of_disks = MZ_READ_LE32(pZip64_locator + MZ_ZIP64_ECDL_TOTAL_NUMBER_OF_DISKS_OFS);
        // 读取 ZIP64 中的中央目录总条目数
        mz_uint64 zip64_cdir_total_entries = MZ_READ_LE64(pZip64_end_of_central_dir + MZ_ZIP64_ECDH_CDIR_TOTAL_ENTRIES_OFS);
        // 读取 ZIP64 中当前磁盘上的中央目录条目数
        mz_uint64 zip64_cdir_total_entries_on_this_disk = MZ_READ_LE64(pZip64_end_of_central_dir + MZ_ZIP64_ECDH_CDIR_NUM_ENTRIES_ON_DISK_OFS);
        // 读取 ZIP64 中结束中央目录记录的大小
        mz_uint64 zip64_size_of_end_of_central_dir_record = MZ_READ_LE64(pZip64_end_of_central_dir + MZ_ZIP64_ECDH_SIZE_OF_RECORD_OFS);
        // 读取 ZIP64 中中央目录的大小
        mz_uint64 zip64_size_of_central_directory = MZ_READ_LE64(pZip64_end_of_central_dir + MZ_ZIP64_ECDH_CDIR_SIZE_OFS);
    
        // 检查结束中央目录记录的大小是否小于最小有效值
        if (zip64_size_of_end_of_central_dir_record < (MZ_ZIP64_END_OF_CENTRAL_DIR_HEADER_SIZE - 12))
            return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);
    
        // 检查 ZIP 文件是否超过一个磁盘
        if (zip64_total_num_of_disks != 1U)
            return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_MULTIDISK);
    
        /* 检查 miniz 的实际限制 */
        // 检查中央目录总条目数是否超过 mz_uint32 的最大值
        if (zip64_cdir_total_entries > MZ_UINT32_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_TOO_MANY_FILES);
    
        // 设置 ZIP 结构体中的总文件数为 ZIP64 中的中央目录总条目数
        pZip->m_total_files = (mz_uint32)zip64_cdir_total_entries;
    
        // 检查当前磁盘上的中央目录条目数是否超过 mz_uint32 的最大值
        if (zip64_cdir_total_entries_on_this_disk > MZ_UINT32_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_TOO_MANY_FILES);
    
        // 将当前磁盘上的中央目录条目数存储到 cdir_entries_on_this_disk 变量中
        cdir_entries_on_this_disk = (mz_uint32)zip64_cdir_total_entries_on_this_disk;
    
        /* 检查 miniz 的当前实际限制 */
        // 检查中央目录的大小是否超过 mz_uint32 的最大值
        if (zip64_size_of_central_directory > MZ_UINT32_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_CDIR_SIZE);
    
        // 将中央目录的大小存储到 cdir_size 变量中
        cdir_size = (mz_uint32)zip64_size_of_central_directory;
    
        // 从 ZIP64 结构中读取当前磁盘编号
        num_this_disk = MZ_READ_LE32(pZip64_end_of_central_dir + MZ_ZIP64_ECDH_NUM_THIS_DISK_OFS);
    
        // 从 ZIP64 结构中读取中央目录所在磁盘编号
        cdir_disk_index = MZ_READ_LE32(pZip64_end_of_central_dir + MZ_ZIP64_ECDH_NUM_DISK_CDIR_OFS);
    
        // 从 ZIP64 结构中读取中央目录的偏移量
        cdir_ofs = MZ_READ_LE64(pZip64_end_of_central_dir + MZ_ZIP64_ECDH_CDIR_OFS_OFS);
    }
    
    // 检查总文件数与当前磁盘上的中央目录条目数是否一致
    if (pZip->m_total_files != cdir_entries_on_this_disk)
        return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_MULTIDISK);
    
    // 检查当前磁盘编号和中央目录所在磁盘编号是否符合规范
    if (((num_this_disk | cdir_disk_index) != 0) && ((num_this_disk != 1) || (cdir_disk_index != 1)))
        return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_MULTIDISK);
    
    // 检查中央目录的大小是否足够大以包含所有文件的信息
    if (cdir_size < pZip->m_total_files * MZ_ZIP_CENTRAL_DIR_HEADER_SIZE)
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);
    
    // 检查中央目录的偏移量加上中央目录的大小是否超过存档文件的大小
    if ((cdir_ofs + (mz_uint64)cdir_size) > pZip->m_archive_size)
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);
    
    // 将中央目录的偏移量存储到 ZIP 结构体中
    pZip->m_central_directory_file_ofs = cdir_ofs;
    
    // 如果需要排序中央目录，则调用相应的函数进行排序
    if (sort_central_dir)
        mz_zip_reader_sort_central_dir_offsets_by_filename(pZip);
    
    // 返回操作成功
    return MZ_TRUE;
}

// 将给定的 mz_zip_archive 结构体清零化
void mz_zip_zero_struct(mz_zip_archive *pZip)
{
    // 如果传入的指针不为空，则清空该结构体
    if (pZip)
        MZ_CLEAR_OBJ(*pZip);
}

// 结束 ZIP 读取器的内部函数，设置最后错误标志
static mz_bool mz_zip_reader_end_internal(mz_zip_archive *pZip, mz_bool set_last_error)
{
    mz_bool status = MZ_TRUE;

    // 如果传入的指针为空，直接返回失败
    if (!pZip)
        return MZ_FALSE;

    // 检查 ZIP 结构体的状态是否满足读取条件，若不满足则设置错误码并返回失败
    if ((!pZip->m_pState) || (!pZip->m_pAlloc) || (!pZip->m_pFree) || (pZip->m_zip_mode != MZ_ZIP_MODE_READING))
    {
        if (set_last_error)
            pZip->m_last_error = MZ_ZIP_INVALID_PARAMETER;

        return MZ_FALSE;
    }

    // 清理 ZIP 内部状态
    if (pZip->m_pState)
    {
        mz_zip_internal_state *pState = pZip->m_pState;
        pZip->m_pState = NULL;

        // 清理中央目录和偏移量数组
        mz_zip_array_clear(pZip, &pState->m_central_dir);
        mz_zip_array_clear(pZip, &pState->m_central_dir_offsets);
        mz_zip_array_clear(pZip, &pState->m_sorted_central_dir_offsets);

        // 如果支持标准输入输出，则关闭文件流
#ifndef MINIZ_NO_STDIO
        if (pState->m_pFile)
        {
            if (pZip->m_zip_type == MZ_ZIP_TYPE_FILE)
            {
                if (MZ_FCLOSE(pState->m_pFile) == EOF)
                {
                    if (set_last_error)
                        pZip->m_last_error = MZ_ZIP_FILE_CLOSE_FAILED;
                    status = MZ_FALSE;
                }
            }
            pState->m_pFile = NULL;
        }
#endif /* #ifndef MINIZ_NO_STDIO */

        // 释放 ZIP 内存状态
        pZip->m_pFree(pZip->m_pAlloc_opaque, pState);
    }
    // 将 ZIP 模式设置为无效
    pZip->m_zip_mode = MZ_ZIP_MODE_INVALID;

    return status;
}

// 结束 ZIP 读取器，设置最后错误标志并返回状态
mz_bool mz_zip_reader_end(mz_zip_archive *pZip)
{
    return mz_zip_reader_end_internal(pZip, MZ_TRUE);
}

// 初始化 ZIP 读取器，设置大小和标志，并进行中央目录的读取
mz_bool mz_zip_reader_init(mz_zip_archive *pZip, mz_uint64 size, mz_uint flags)
{
    // 如果 ZIP 结构体为空或者读取函数指针为空，返回参数错误
    if ((!pZip) || (!pZip->m_pRead))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 初始化内部 ZIP 读取
    if (!mz_zip_reader_init_internal(pZip, flags))
        return MZ_FALSE;

    // 设置 ZIP 类型为用户定义，并设置存档大小
    pZip->m_zip_type = MZ_ZIP_TYPE_USER;
    pZip->m_archive_size = size;

    // 如果中央目录读取失败，则结束 ZIP 读取并返回失败
    if (!mz_zip_reader_read_central_dir(pZip, flags))
    {
        mz_zip_reader_end_internal(pZip, MZ_FALSE);
        return MZ_FALSE;
    }

    return MZ_TRUE;
}

// 内存中 ZIP 读取函数，从内存中读取数据
static size_t mz_zip_mem_read_func(void *pOpaque, mz_uint64 file_ofs, void *pBuf, size_t n)
{
    mz_zip_archive *pZip = (mz_zip_archive *)pOpaque;
    size_t s = (file_ofs >= pZip->m_archive_size) ? 0 : (size_t)MZ_MIN(pZip->m_archive_size - file_ofs, n);
    memcpy(pBuf, (const mz_uint8 *)pZip->m_pState->m_pMem + file_ofs, s);
    return s;
}

// 初始化内存中 ZIP 读取器，设置内存指针和大小，并进行中央目录的读取
mz_bool mz_zip_reader_init_mem(mz_zip_archive *pZip, const void *pMem, size_t size, mz_uint flags)
{
    // 如果内存指针为空，返回参数错误
    if (!pMem)
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 如果大小小于中央目录头部大小，返回不是存档错误
    if (size < MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE)
        return mz_zip_set_error(pZip, MZ_ZIP_NOT_AN_ARCHIVE);

    // 初始化内部 ZIP 读取
    if (!mz_zip_reader_init_internal(pZip, flags))
        return MZ_FALSE;

    // 设置 ZIP 类型为内存，并设置存档大小和读取函数指针
    pZip->m_zip_type = MZ_ZIP_TYPE_MEMORY;
    pZip->m_archive_size = size;
    pZip->m_pRead = mz_zip_mem_read_func;
    pZip->m_pIO_opaque = pZip;
    pZip->m_pNeeds_keepalive = NULL;

#ifdef __cplusplus
    # 将指针 pMem 指向的内存强制转换为 void* 类型，并赋值给 pZip 对象所指向的 m_pState 结构体的 m_pMem 成员变量
    pZip->m_pState->m_pMem = const_cast<void *>(pMem);
#else
    pZip->m_pState->m_pMem = (void *)pMem;
#endif


    // 如果不是以MINIZ_NO_STDIO定义编译，则将内存指针设置为给定的pMem指针
    pZip->m_pState->m_pMem = (void *)pMem;
#endif



    pZip->m_pState->m_mem_size = size;


    // 设置ZIP归档的内存大小
    pZip->m_pState->m_mem_size = size;



    if (!mz_zip_reader_read_central_dir(pZip, flags))
    {
        mz_zip_reader_end_internal(pZip, MZ_FALSE);
        return MZ_FALSE;
    }


    // 读取ZIP归档的中央目录信息，如果失败则结束读取并返回失败标志
    if (!mz_zip_reader_read_central_dir(pZip, flags))
    {
        mz_zip_reader_end_internal(pZip, MZ_FALSE);
        return MZ_FALSE;
    }



    return MZ_TRUE;
}


    // 返回成功标志
    return MZ_TRUE;
}



#ifndef MINIZ_NO_STDIO
static size_t mz_zip_file_read_func(void *pOpaque, mz_uint64 file_ofs, void *pBuf, size_t n)
{
    mz_zip_archive *pZip = (mz_zip_archive *)pOpaque;
    mz_int64 cur_ofs = MZ_FTELL64(pZip->m_pState->m_pFile);

    file_ofs += pZip->m_pState->m_file_archive_start_ofs;

    if (((mz_int64)file_ofs < 0) || (((cur_ofs != (mz_int64)file_ofs)) && (MZ_FSEEK64(pZip->m_pState->m_pFile, (mz_int64)file_ofs, SEEK_SET))))
        return 0;

    return MZ_FREAD(pBuf, 1, n, pZip->m_pState->m_pFile);
}


#ifndef MINIZ_NO_STDIO
// 从文件读取ZIP归档数据的回调函数
static size_t mz_zip_file_read_func(void *pOpaque, mz_uint64 file_ofs, void *pBuf, size_t n)
{
    mz_zip_archive *pZip = (mz_zip_archive *)pOpaque;
    mz_int64 cur_ofs = MZ_FTELL64(pZip->m_pState->m_pFile);

    // 调整文件偏移量到实际的归档开始位置
    file_ofs += pZip->m_pState->m_file_archive_start_ofs;

    // 如果文件偏移小于0或者当前偏移与目标偏移不一致，尝试重新定位到目标位置
    if (((mz_int64)file_ofs < 0) || (((cur_ofs != (mz_int64)file_ofs)) && (MZ_FSEEK64(pZip->m_pState->m_pFile, (mz_int64)file_ofs, SEEK_SET))))
        return 0;

    // 从当前位置读取n个字节到缓冲区pBuf中
    return MZ_FREAD(pBuf, 1, n, pZip->m_pState->m_pFile);
}
#endif



mz_bool mz_zip_reader_init_file(mz_zip_archive *pZip, const char *pFilename, mz_uint32 flags)
{
    return mz_zip_reader_init_file_v2(pZip, pFilename, flags, 0, 0);
}


// 初始化从文件读取的ZIP归档
mz_bool mz_zip_reader_init_file(mz_zip_archive *pZip, const char *pFilename, mz_uint32 flags)
{
    return mz_zip_reader_init_file_v2(pZip, pFilename, flags, 0, 0);
}



mz_bool mz_zip_reader_init_file_v2(mz_zip_archive *pZip, const char *pFilename, mz_uint flags, mz_uint64 file_start_ofs, mz_uint64 archive_size)
{
    mz_uint64 file_size;
    MZ_FILE *pFile;

    if ((!pZip) || (!pFilename) || ((archive_size) && (archive_size < MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE)))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 打开文件以读取ZIP归档
    pFile = MZ_FOPEN(pFilename, "rb");
    if (!pFile)
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_OPEN_FAILED);

    // 确定归档文件的大小
    file_size = archive_size;
    if (!file_size)
    {
        if (MZ_FSEEK64(pFile, 0, SEEK_END))
        {
            MZ_FCLOSE(pFile);
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_SEEK_FAILED);
        }

        file_size = MZ_FTELL64(pFile);
    }

    /* TODO: Better sanity check archive_size and the # of actual remaining bytes */

    // 如果文件大小小于ZIP归档的中央目录头大小，则认为不是一个有效的ZIP归档
    if (file_size < MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE)
    {
        MZ_FCLOSE(pFile);
        return mz_zip_set_error(pZip, MZ_ZIP_NOT_AN_ARCHIVE);
    }

    // 初始化ZIP归档的内部结构
    if (!mz_zip_reader_init_internal(pZip, flags))
    {
        MZ_FCLOSE(pFile);
        return MZ_FALSE;
    }

    // 设置ZIP归档类型为文件类型，并指定文件读取的回调函数及相关参数
    pZip->m_zip_type = MZ_ZIP_TYPE_FILE;
    pZip->m_pRead = mz_zip_file_read_func;
    pZip->m_pIO_opaque = pZip;
    pZip->m_pState->m_pFile = pFile;
    pZip->m_archive_size = file_size;
    pZip->m_pState->m_file_archive_start_ofs = file_start_ofs;

    // 读取ZIP归档的中央目录
    if (!mz_zip_reader_read_central_dir(pZip, flags))
    {
        mz_zip_reader_end_internal(pZip, MZ_FALSE);
        return MZ_FALSE;
    }

    return MZ_TRUE;
}



mz_bool mz_zip_reader_init_cfile(mz_zip_archive *pZip, MZ_FILE *pFile, mz_uint64 archive_size, mz_uint flags)
{
    mz_uint64 cur_file_ofs;

    if ((!pZip) || (!pFile))
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_OPEN_FAILED);

    cur_file_ofs = MZ_FTELL64(pFile);

    // 初始化从C风格文件指针读取的ZIP归档
    if (!archive_size)


// 初始化从C风格文件指针读取的ZIP归档
mz_bool mz_zip_reader_init_cfile(mz_zip_archive *pZip, MZ_FILE *pFile, mz_uint64 archive_size, mz_uint flags)
{
    mz_uint64 cur_file_ofs;

    if ((!pZip) || (!pFile))
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_OPEN_FAILED);

    cur_file_ofs = MZ_FTELL64(pFile);

    // 如果未指定归档大小，则根据当前位置计算文件大小
    if (!archive_size)



    if (!archive_size)


    // 如果未指定归档大小
    if (!archive_size)
    {
        // 如果无法将文件指针定位到文件末尾，返回文件定位失败错误
        if (MZ_FSEEK64(pFile, 0, SEEK_END))
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_SEEK_FAILED);
    
        // 计算整个存档的大小
        archive_size = MZ_FTELL64(pFile) - cur_file_ofs;
    
        // 如果存档大小小于ZIP尾部中央目录头部的大小，返回不是有效存档错误
        if (archive_size < MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE)
            return mz_zip_set_error(pZip, MZ_ZIP_NOT_AN_ARCHIVE);
    }
    
    // 初始化ZIP阅读器内部结构，如果失败返回假
    if (!mz_zip_reader_init_internal(pZip, flags))
        return MZ_FALSE;
    
    // 设置ZIP结构的类型为C文件类型
    pZip->m_zip_type = MZ_ZIP_TYPE_CFILE;
    // 设置ZIP结构的读取函数为mz_zip_file_read_func
    
    pZip->m_pRead = mz_zip_file_read_func;
    
    // 设置ZIP结构的IO透明指针为pZip
    pZip->m_pIO_opaque = pZip;
    // 设置ZIP状态结构的文件指针为pFile
    pZip->m_pState->m_pFile = pFile;
    // 设置ZIP结构的存档大小为archive_size
    pZip->m_archive_size = archive_size;
    // 设置ZIP状态结构的文件存档起始偏移量为cur_file_ofs
    pZip->m_pState->m_file_archive_start_ofs = cur_file_ofs;
    
    // 如果无法读取中央目录信息，结束ZIP阅读器并返回假
    if (!mz_zip_reader_read_central_dir(pZip, flags))
    {
        mz_zip_reader_end_internal(pZip, MZ_FALSE);
        return MZ_FALSE;
    }
    
    // 返回真表示初始化成功
    return MZ_TRUE;
#endif /* #ifndef MINIZ_NO_STDIO */

// 返回指定索引的文件的中央目录条目（Central Directory Header，CDH）
static MZ_FORCEINLINE const mz_uint8 *mz_zip_get_cdh(mz_zip_archive *pZip, mz_uint file_index)
{
    // 如果传入的指针为空，或者指针指向的状态为空，或者文件索引超出总文件数，返回空指针
    if ((!pZip) || (!pZip->m_pState) || (file_index >= pZip->m_total_files))
        return NULL;
    // 返回指向中央目录条目的指针
    return &MZ_ZIP_ARRAY_ELEMENT(&pZip->m_pState->m_central_dir, mz_uint8, MZ_ZIP_ARRAY_ELEMENT(&pZip->m_pState->m_central_dir_offsets, mz_uint32, file_index));
}

// 判断指定文件是否被加密
mz_bool mz_zip_reader_is_file_encrypted(mz_zip_archive *pZip, mz_uint file_index)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint m_bit_flag;
    // 获取指定文件的中央目录条目
    const mz_uint8 *p = mz_zip_get_cdh(pZip, file_index);
    // 如果获取失败，设置 ZIP 错误并返回假
    if (!p)
    {
        mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);
        return MZ_FALSE;
    }

    // 读取文件的通用目的位标志（General Purpose Bit Flag）
    m_bit_flag = MZ_READ_LE16(p + MZ_ZIP_CDH_BIT_FLAG_OFS);
    // 返回文件是否被加密的布尔值
    return (m_bit_flag & (MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_IS_ENCRYPTED | MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_USES_STRONG_ENCRYPTION)) != 0;
}

// 判断指定文件是否被支持
mz_bool mz_zip_reader_is_file_supported(mz_zip_archive *pZip, mz_uint file_index)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint bit_flag;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint method;

    // 获取指定文件的中央目录条目
    const mz_uint8 *p = mz_zip_get_cdh(pZip, file_index);
    // 如果获取失败，设置 ZIP 错误并返回假
    if (!p)
    {
        mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);
        return MZ_FALSE;
    }

    // 读取文件的压缩方法
    method = MZ_READ_LE16(p + MZ_ZIP_CDH_METHOD_OFS);
    // 读取文件的通用目的位标志
    bit_flag = MZ_READ_LE16(p + MZ_ZIP_CDH_BIT_FLAG_OFS);

    // 如果文件的压缩方法既不是 0（不压缩）也不是 8（使用 deflate 压缩）
    if ((method != 0) && (method != MZ_DEFLATED))
    {
        // 设置 ZIP 错误为不支持的方法，并返回假
        mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_METHOD);
        return MZ_FALSE;
    }

    // 如果文件使用了加密或强加密
    if (bit_flag & (MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_IS_ENCRYPTED | MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_USES_STRONG_ENCRYPTION))
    {
        // 设置 ZIP 错误为不支持的加密，并返回假
        mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_ENCRYPTION);
        return MZ_FALSE;
    }

    // 如果文件的通用目的位标志包含压缩补丁标志
    if (bit_flag & MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_COMPRESSED_PATCH_FLAG)
    {
        // 设置 ZIP 错误为不支持的特性，并返回假
        mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_FEATURE);
        return MZ_FALSE;
    }

    // 文件被支持，返回真
    return MZ_TRUE;
}

// 判断指定文件是否是一个目录
mz_bool mz_zip_reader_is_file_a_directory(mz_zip_archive *pZip, mz_uint file_index)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint filename_len, attribute_mapping_id, external_attr;
    // 获取指定文件的中央目录条目
    const mz_uint8 *p = mz_zip_get_cdh(pZip, file_index);
    // 如果获取失败，设置 ZIP 错误并返回假
    if (!p)
    {
        mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);
        return MZ_FALSE;
    }

    // 读取文件名长度
    filename_len = MZ_READ_LE16(p + MZ_ZIP_CDH_FILENAME_LEN_OFS);
    // 如果文件名长度非零
    if (filename_len)
    {
        // 检查文件名的最后一个字符是否为斜杠，如果是则返回真（表示是目录）
        if (*(p + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + filename_len - 1) == '/')
            return MZ_TRUE;
    }

    /* Bugfix: This code was also checking if the internal attribute was non-zero, which wasn't correct. */
    /* Most/all zip writers (hopefully) set DOS file/directory attributes in the low 16-bits, so check for the DOS directory flag and ignore the source OS ID in the created by field. */
    /* FIXME: Remove this check? Is it necessary - we already check the filename. */

    // 如果文件名长度为零或者最后一个字符不是斜杠，则返回假（不是目录）
    return MZ_FALSE;
}
    # 从指针 p 处读取 16 位 Little-Endian 数据，计算 attribute_mapping_id
    attribute_mapping_id = MZ_READ_LE16(p + MZ_ZIP_CDH_VERSION_MADE_BY_OFS) >> 8;
    # 忽略 attribute_mapping_id 变量，不使用其值

    # 从指针 p 处读取 32 位 Little-Endian 数据，获取文件的外部属性
    external_attr = MZ_READ_LE32(p + MZ_ZIP_CDH_EXTERNAL_ATTR_OFS);
    # 检查外部属性中是否设置了 DOS 目录属性位标志
    if ((external_attr & MZ_ZIP_DOS_DIR_ATTRIBUTE_BITFLAG) != 0)
    {
        # 如果设置了 DOS 目录属性位标志，则返回真（表示是目录）
        return MZ_TRUE;
    }

    # 如果外部属性中未设置 DOS 目录属性位标志，则返回假（表示不是目录）
    return MZ_FALSE;
}

static mz_bool mz_zip_file_stat_internal(mz_zip_archive *pZip, mz_uint file_index, const mz_uint8 *pCentral_dir_header, mz_zip_archive_file_stat *pStat, mz_bool *pFound_zip64_extra_data)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint n;
    const mz_uint8 *p = pCentral_dir_header;

    // 如果传入的指针为空或者传入的结构体为空，则返回参数错误
    if ((!p) || (!pStat))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    /* 从中央目录记录中提取字段信息 */
    pStat->m_file_index = file_index;
    pStat->m_central_dir_ofs = MZ_ZIP_ARRAY_ELEMENT(&pZip->m_pState->m_central_dir_offsets, mz_uint32, file_index);
    pStat->m_version_made_by = MZ_READ_LE16(p + MZ_ZIP_CDH_VERSION_MADE_BY_OFS);
    pStat->m_version_needed = MZ_READ_LE16(p + MZ_ZIP_CDH_VERSION_NEEDED_OFS);
    pStat->m_bit_flag = MZ_READ_LE16(p + MZ_ZIP_CDH_BIT_FLAG_OFS);
    pStat->m_method = MZ_READ_LE16(p + MZ_ZIP_CDH_METHOD_OFS);
#ifndef MINIZ_NO_TIME
    // 将 DOS 时间转换为 time_t 格式
    pStat->m_time = mz_zip_dos_to_time_t(MZ_READ_LE16(p + MZ_ZIP_CDH_FILE_TIME_OFS), MZ_READ_LE16(p + MZ_ZIP_CDH_FILE_DATE_OFS));
#endif
    pStat->m_crc32 = MZ_READ_LE32(p + MZ_ZIP_CDH_CRC32_OFS);
    pStat->m_comp_size = MZ_READ_LE32(p + MZ_ZIP_CDH_COMPRESSED_SIZE_OFS);
    pStat->m_uncomp_size = MZ_READ_LE32(p + MZ_ZIP_CDH_DECOMPRESSED_SIZE_OFS);
    pStat->m_internal_attr = MZ_READ_LE16(p + MZ_ZIP_CDH_INTERNAL_ATTR_OFS);
    pStat->m_external_attr = MZ_READ_LE32(p + MZ_ZIP_CDH_EXTERNAL_ATTR_OFS);
    pStat->m_local_header_ofs = MZ_READ_LE32(p + MZ_ZIP_CDH_LOCAL_HEADER_OFS);

    /* 复制尽可能多的文件名和注释 */
    n = MZ_READ_LE16(p + MZ_ZIP_CDH_FILENAME_LEN_OFS);
    n = MZ_MIN(n, MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE - 1);
    memcpy(pStat->m_filename, p + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE, n);
    pStat->m_filename[n] = '\0';

    n = MZ_READ_LE16(p + MZ_ZIP_CDH_COMMENT_LEN_OFS);
    n = MZ_MIN(n, MZ_ZIP_MAX_ARCHIVE_FILE_COMMENT_SIZE - 1);
    pStat->m_comment_size = n;
    memcpy(pStat->m_comment, p + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + MZ_READ_LE16(p + MZ_ZIP_CDH_FILENAME_LEN_OFS) + MZ_READ_LE16(p + MZ_ZIP_CDH_EXTRA_LEN_OFS), n);
    pStat->m_comment[n] = '\0';

    /* 设置一些便利的标志位 */
    pStat->m_is_directory = mz_zip_reader_is_file_a_directory(pZip, file_index);
    pStat->m_is_encrypted = mz_zip_reader_is_file_encrypted(pZip, file_index);
    pStat->m_is_supported = mz_zip_reader_is_file_supported(pZip, file_index);

    /* 看看是否需要读取任何 zip64 扩展信息字段 */
    /* 令人困惑的是，这些 zip64 字段即使在非 zip64 存档中也可能存在（例如 Debian zip 在从标准输入到标准输出上创建的大文件）。 */
    if (MZ_MAX(MZ_MAX(pStat->m_comp_size, pStat->m_uncomp_size), pStat->m_local_header_ofs) == MZ_UINT32_MAX)
    }

    return MZ_TRUE;
}

static MZ_FORCEINLINE mz_bool mz_zip_string_equal(const char *pA, const char *pB, mz_uint len, mz_uint flags)
{
    // 声明一个无符号整数变量 i，用于循环计数
    mz_uint i;
    // 检查 flags 中是否设置了 MZ_ZIP_FLAG_CASE_SENSITIVE 标志
    if (flags & MZ_ZIP_FLAG_CASE_SENSITIVE)
        // 如果设置了大小写敏感标志，则直接比较 pA 和 pB 的内存内容是否相等，返回比较结果
        return 0 == memcmp(pA, pB, len);
    // 如果未设置大小写敏感标志，则进行大小写不敏感比较
    for (i = 0; i < len; ++i)
        // 比较 pA 和 pB 中对应位置字符转换为小写后的结果是否相等
        if (MZ_TOLOWER(pA[i]) != MZ_TOLOWER(pB[i]))
            // 如果有不相等的字符，则返回假（MZ_FALSE）
            return MZ_FALSE;
    // 如果所有字符都相等，则返回真（MZ_TRUE）
    return MZ_TRUE;
}

// 静态内联函数，用于比较两个文件名
static MZ_FORCEINLINE int mz_zip_filename_compare(const mz_zip_array *pCentral_dir_array, const mz_zip_array *pCentral_dir_offsets, mz_uint l_index, const char *pR, mz_uint r_len)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    const mz_uint8 *pL = &MZ_ZIP_ARRAY_ELEMENT(pCentral_dir_array, mz_uint8, MZ_ZIP_ARRAY_ELEMENT(pCentral_dir_offsets, mz_uint32, l_index)), *pE;
    // 从中央目录数组和偏移数组获取左边文件名的长度
    mz_uint l_len = MZ_READ_LE16(pL + MZ_ZIP_CDH_FILENAME_LEN_OFS);
    mz_uint8 l = 0, r = 0;
    // 指向中央目录头部，跳过固定长度
    pL += MZ_ZIP_CENTRAL_DIR_HEADER_SIZE;
    // 右边文件名长度取最小值，用于比较文件名
    pE = pL + MZ_MIN(l_len, r_len);
    // 比较两个文件名的每个字符，直到遇到不同或者其中一个文件名结尾
    while (pL < pE)
    {
        if ((l = MZ_TOLOWER(*pL)) != (r = MZ_TOLOWER(*pR)))
            break;
        pL++;
        pR++;
    }
    // 如果左边文件名已经遍历完，则比较长度差；否则比较第一个不同字符的 ASCII 差值
    return (pL == pE) ? (int)(l_len - r_len) : (l - r);
}

// 通过二分查找定位 ZIP 文件中的文件
static mz_bool mz_zip_locate_file_binary_search(mz_zip_archive *pZip, const char *pFilename, mz_uint32 *pIndex)
{
    // 获取 ZIP 归档的内部状态
    mz_zip_internal_state *pState = pZip->m_pState;
    const mz_zip_array *pCentral_dir_offsets = &pState->m_central_dir_offsets;
    const mz_zip_array *pCentral_dir = &pState->m_central_dir;
    mz_uint32 *pIndices = &MZ_ZIP_ARRAY_ELEMENT(&pState->m_sorted_central_dir_offsets, mz_uint32, 0);
    const uint32_t size = pZip->m_total_files;
    const mz_uint filename_len = (mz_uint)strlen(pFilename);

    // 如果传入的索引指针不为空，则初始化为0
    if (pIndex)
        *pIndex = 0;

    // 如果文件数量不为0
    if (size)
    {
        /* yes I could use uint32_t's, but then we would have to add some special case checks in the loop, argh, and */
        /* honestly the major expense here on 32-bit CPU's will still be the filename compare */
        // 初始化搜索范围的下界和上界
        mz_int64 l = 0, h = (mz_int64)size - 1;

        // 二分查找
        while (l <= h)
        {
            mz_int64 m = l + ((h - l) >> 1);
            uint32_t file_index = pIndices[(uint32_t)m];

            // 比较文件名，返回值为0表示找到匹配文件名
            int comp = mz_zip_filename_compare(pCentral_dir, pCentral_dir_offsets, file_index, pFilename, filename_len);
            if (!comp)
            {
                // 如果传入索引指针不为空，将找到的文件索引赋给它
                if (pIndex)
                    *pIndex = file_index;
                return MZ_TRUE;
            }
            else if (comp < 0)
                l = m + 1;  // 继续向右半边搜索
            else
                h = m - 1;  // 继续向左半边搜索
        }
    }

    // 没有找到匹配文件，返回错误
    return mz_zip_set_error(pZip, MZ_ZIP_FILE_NOT_FOUND);
}

// 定位 ZIP 文件中的文件，返回索引，如果找不到返回-1
int mz_zip_reader_locate_file(mz_zip_archive *pZip, const char *pName, const char *pComment, mz_uint flags)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint32 index;
    // 调用版本2的函数，如果找不到返回-1，否则返回找到的索引
    if (!mz_zip_reader_locate_file_v2(pZip, pName, pComment, flags, &index))
        return -1;
    else
        return (int)index;
}

// 定位 ZIP 文件中的文件，返回索引，如果找不到返回假
mz_bool mz_zip_reader_locate_file_v2(mz_zip_archive *pZip, const char *pName, const char *pComment, mz_uint flags, mz_uint32 *pIndex)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint file_index;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t name_len, comment_len;

    // 如果传入索引指针不为空，初始化为0
    if (pIndex)
        *pIndex = 0;

    // 检查参数是否有效
    if ((!pZip) || (!pZip->m_pState) || (!pName))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);
    /* 看看我们能否使用二分查找 */
    if (((pZip->m_pState->m_init_flags & MZ_ZIP_FLAG_DO_NOT_SORT_CENTRAL_DIRECTORY) == 0) &&
        (pZip->m_zip_mode == MZ_ZIP_MODE_READING) &&
        ((flags & (MZ_ZIP_FLAG_IGNORE_PATH | MZ_ZIP_FLAG_CASE_SENSITIVE)) == 0) && (!pComment) && (pZip->m_pState->m_sorted_central_dir_offsets.m_size))
    {
        // 如果初始化标志不包含不排序中央目录标志，并且处于读取模式，并且不忽略路径和大小写敏感标志，并且没有评论，并且已排序中央目录偏移量大小不为零，则执行二分查找定位文件
        return mz_zip_locate_file_binary_search(pZip, pName, pIndex);
    }

    /* 通过扫描整个中央目录来定位条目 */
    name_len = strlen(pName);
    if (name_len > MZ_UINT16_MAX)
        // 如果文件名长度超过了最大的无符号16位整数值，则返回无效参数错误
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    comment_len = pComment ? strlen(pComment) : 0;
    if (comment_len > MZ_UINT16_MAX)
        // 如果评论长度超过了最大的无符号16位整数值，则返回无效参数错误
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    for (file_index = 0; file_index < pZip->m_total_files; file_index++)
    {
        const mz_uint8 *pHeader = &MZ_ZIP_ARRAY_ELEMENT(&pZip->m_pState->m_central_dir, mz_uint8, MZ_ZIP_ARRAY_ELEMENT(&pZip->m_pState->m_central_dir_offsets, mz_uint32, file_index));
        // 获取文件名长度
        mz_uint filename_len = MZ_READ_LE16(pHeader + MZ_ZIP_CDH_FILENAME_LEN_OFS);
        // 获取文件名指针
        const char *pFilename = (const char *)pHeader + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE;
        if (filename_len < name_len)
            // 如果文件名长度小于要查找的名称长度，则继续下一次循环
            continue;
        if (comment_len)
        {
            // 获取额外长度和评论长度
            mz_uint file_extra_len = MZ_READ_LE16(pHeader + MZ_ZIP_CDH_EXTRA_LEN_OFS), file_comment_len = MZ_READ_LE16(pHeader + MZ_ZIP_CDH_COMMENT_LEN_OFS);
            // 获取文件评论指针
            const char *pFile_comment = pFilename + filename_len + file_extra_len;
            // 如果文件评论长度不等于给定评论长度，或者评论内容不相等，则继续下一次循环
            if ((file_comment_len != comment_len) || (!mz_zip_string_equal(pComment, pFile_comment, file_comment_len, flags)))
                continue;
        }
        if ((flags & MZ_ZIP_FLAG_IGNORE_PATH) && (filename_len))
        {
            // 如果忽略路径标志被设置，并且文件名长度不为零
            // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
            int ofs = filename_len - 1;
            do
            {
                // 如果文件名中的字符为斜杠、反斜杠或冒号，则退出循环
                if ((pFilename[ofs] == '/') || (pFilename[ofs] == '\\') || (pFilename[ofs] == ':'))
                    break;
            } while (--ofs >= 0);
            ofs++;
            pFilename += ofs;
            filename_len -= ofs;
        }
        if ((filename_len == name_len) && (mz_zip_string_equal(pName, pFilename, filename_len, flags)))
        {
            // 如果文件名长度等于给定名称长度，并且文件名与给定名称内容相等，则如果索引指针不为空，将文件索引赋值给它，并返回真
            if (pIndex)
                *pIndex = file_index;
            return MZ_TRUE;
        }
    }

    // 如果未找到匹配的文件，则返回文件未找到错误
    return mz_zip_set_error(pZip, MZ_ZIP_FILE_NOT_FOUND);
    // 初始化变量 status，用于表示解压缩状态
    int status = TINFL_STATUS_DONE;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 声明需要的变量，包括 needed_size, cur_file_ofs, comp_remaining,
    // out_buf_ofs, read_buf_size, read_buf_ofs, read_buf_avail
    mz_uint64 needed_size, cur_file_ofs, comp_remaining, out_buf_ofs = 0, read_buf_size, read_buf_ofs = 0, read_buf_avail;
    // 声明文件状态变量
    mz_zip_archive_file_stat file_stat;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 声明需要的缓冲区变量 pRead_buf
    void *pRead_buf;
    // 声明本地头部的缓冲区
    mz_uint32 local_header_u32[(MZ_ZIP_LOCAL_DIR_HEADER_SIZE + sizeof(mz_uint32) - 1) / sizeof(mz_uint32)];
    mz_uint8 *pLocal_header = (mz_uint8 *)local_header_u32;
    // 声明解压缩器 inflator
    tinfl_decompressor inflator;

    // 检查参数的有效性，如果不满足条件则返回错误
    if ((!pZip) || (!pZip->m_pState) || ((buf_size) && (!pBuf)) || ((user_read_buf_size) && (!pUser_read_buf)) || (!pZip->m_pRead))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 获取文件在 ZIP 中的状态信息，如果失败则返回假
    if (!mz_zip_reader_file_stat(pZip, file_index, &file_stat))
        return MZ_FALSE;

    /* A directory or zero length file */
    // 如果是目录或者文件长度为零，则返回真
    if ((file_stat.m_is_directory) || (!file_stat.m_comp_size))
        return MZ_TRUE;

    /* Encryption and patch files are not supported. */
    // 加密和补丁文件不被支持，如果文件设置了相应标志，则返回相应的加密错误
    if (file_stat.m_bit_flag & (MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_IS_ENCRYPTED | MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_USES_STRONG_ENCRYPTION | MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_COMPRESSED_PATCH_FLAG))
        return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_ENCRYPTION);

    /* This function only supports decompressing stored and deflate. */
    // 本函数仅支持解压缩存储和 deflate 压缩方法，如果文件不符合这些条件，则返回相应的方法不支持错误
    if ((!(flags & MZ_ZIP_FLAG_COMPRESSED_DATA)) && (file_stat.m_method != 0) && (file_stat.m_method != MZ_DEFLATED))
        return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_METHOD);

    /* Ensure supplied output buffer is large enough. */
    // 确保提供的输出缓冲区足够大，如果不足则返回缓冲区过小错误
    needed_size = (flags & MZ_ZIP_FLAG_COMPRESSED_DATA) ? file_stat.m_comp_size : file_stat.m_uncomp_size;
    if (buf_size < needed_size)
        return mz_zip_set_error(pZip, MZ_ZIP_BUF_TOO_SMALL);

    /* Read and parse the local directory entry. */
    // 读取和解析本地目录条目，获取当前文件的偏移量
    cur_file_ofs = file_stat.m_local_header_ofs;
    // 从 ZIP 存档中读取本地头部数据，如果读取失败则返回文件读取失败错误
    if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pLocal_header, MZ_ZIP_LOCAL_DIR_HEADER_SIZE) != MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);

    // 检查读取的本地头部数据的有效性，如果不是有效的本地头部签名则返回无效头部错误
    if (MZ_READ_LE32(pLocal_header) != MZ_ZIP_LOCAL_DIR_HEADER_SIG)
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);

    // 更新当前文件的偏移量，跳过文件名长度和额外字段长度
    cur_file_ofs += MZ_ZIP_LOCAL_DIR_HEADER_SIZE + MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_FILENAME_LEN_OFS) + MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_EXTRA_LEN_OFS);
    // 如果当前文件偏移量加上压缩大小超过了存档大小，则返回无效头部错误
    if ((cur_file_ofs + file_stat.m_comp_size) > pZip->m_archive_size)
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);

    // 如果数据标志表明数据已经压缩，或者文件不是使用存储方法，则继续执行解压缩
    {
        /* 如果文件已经存储或调用者请求压缩数据 */
        // 调用读取函数，从 ZIP 文件中读取数据到缓冲区中
        if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pBuf, (size_t)needed_size) != needed_size)
            // 如果读取的数据大小不符合期望，返回文件读取失败的错误状态
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);
    }
#ifndef MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS
        // 如果未禁用 CRC32 检查并且数据未压缩
        if ((flags & MZ_ZIP_FLAG_COMPRESSED_DATA) == 0)
        {
            // 计算数据的 CRC32 值并与文件头中的 CRC32 值比较，如果不匹配则返回 CRC 校验失败
            if (mz_crc32(MZ_CRC32_INIT, (const mz_uint8 *)pBuf, (size_t)file_stat.m_uncomp_size) != file_stat.m_crc32)
                return mz_zip_set_error(pZip, MZ_ZIP_CRC_CHECK_FAILED);
        }
#endif

        // 返回解压缩成功
        return MZ_TRUE;
    }

    /* Decompress the file either directly from memory or from a file input buffer. */
    // 初始化解压缩器
    tinfl_init(&inflator);

    // 如果存档数据在内存中
    if (pZip->m_pState->m_pMem)
    {
        /* Read directly from the archive in memory. */
        // 设置读取缓冲区为存档中的数据
        pRead_buf = (mz_uint8 *)pZip->m_pState->m_pMem + cur_file_ofs;
        read_buf_size = read_buf_avail = file_stat.m_comp_size;
        comp_remaining = 0;
    }
    else if (pUser_read_buf)
    {
        /* Use a user provided read buffer. */
        // 使用用户提供的读取缓冲区
        if (!user_read_buf_size)
            return MZ_FALSE;
        pRead_buf = (mz_uint8 *)pUser_read_buf;
        read_buf_size = user_read_buf_size;
        read_buf_avail = 0;
        comp_remaining = file_stat.m_comp_size;
    }
    else
    {
        /* Temporarily allocate a read buffer. */
        // 临时分配一个读取缓冲区
        read_buf_size = MZ_MIN(file_stat.m_comp_size, (mz_uint64)MZ_ZIP_MAX_IO_BUF_SIZE);
        // 检查是否出现内部错误
        if (((sizeof(size_t) == sizeof(mz_uint32))) && (read_buf_size > 0x7FFFFFFF))
            return mz_zip_set_error(pZip, MZ_ZIP_INTERNAL_ERROR);

        // 使用分配器分配内存作为读取缓冲区
        if (NULL == (pRead_buf = pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1, (size_t)read_buf_size)))
            return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);

        read_buf_avail = 0;
        comp_remaining = file_stat.m_comp_size;
    }

    do
    {
        /* The size_t cast here should be OK because we've verified that the output buffer is >= file_stat.m_uncomp_size above */
        // 在上面已经验证了输出缓冲区大小 >= file_stat.m_uncomp_size，因此这里对 size_t 进行转换应该是安全的
        size_t in_buf_size, out_buf_size = (size_t)(file_stat.m_uncomp_size - out_buf_ofs);
        if ((!read_buf_avail) && (!pZip->m_pState->m_pMem))
        {
            // 如果读取缓冲区没有可用数据且存档数据不在内存中，则从文件中读取数据到读取缓冲区
            read_buf_avail = MZ_MIN(read_buf_size, comp_remaining);
            // 使用存档的读取函数从存档中读取数据
            if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pRead_buf, (size_t)read_buf_avail) != read_buf_avail)
            {
                // 读取失败，设置错误状态为解压缩失败并退出
                status = TINFL_STATUS_FAILED;
                mz_zip_set_error(pZip, MZ_ZIP_DECOMPRESSION_FAILED);
                break;
            }
            cur_file_ofs += read_buf_avail;
            comp_remaining -= read_buf_avail;
            read_buf_ofs = 0;
        }
        in_buf_size = (size_t)read_buf_avail;
        // 调用解压缩函数进行解压缩操作
        status = tinfl_decompress(&inflator, (mz_uint8 *)pRead_buf + read_buf_ofs, &in_buf_size, (mz_uint8 *)pBuf, (mz_uint8 *)pBuf + out_buf_ofs, &out_buf_size, TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF | (comp_remaining ? TINFL_FLAG_HAS_MORE_INPUT : 0));
        read_buf_avail -= in_buf_size;
        read_buf_ofs += in_buf_size;
        out_buf_ofs += out_buf_size;
    } while (status == TINFL_STATUS_NEEDS_MORE_INPUT);

    // 如果解压缩完成
    if (status == TINFL_STATUS_DONE)
    {
        /* 确保整个文件已解压，并检查其 CRC（循环冗余校验）。 */
        // 如果解压后的数据长度与文件的未压缩大小不符合
        if (out_buf_ofs != file_stat.m_uncomp_size)
        {
            // 设置 ZIP 解压错误状态为“解压后大小与预期不符”
            mz_zip_set_error(pZip, MZ_ZIP_UNEXPECTED_DECOMPRESSED_SIZE);
            // 设置解压状态为失败
            status = TINFL_STATUS_FAILED;
        }
#ifndef MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS
        // 如果未禁用 CRC32 检查，执行以下代码块
        else if (mz_crc32(MZ_CRC32_INIT, (const mz_uint8 *)pBuf, (size_t)file_stat.m_uncomp_size) != file_stat.m_crc32)
        {
            // 计算缓冲区的 CRC32 值，与文件的 CRC32 值进行比较
            mz_zip_set_error(pZip, MZ_ZIP_CRC_CHECK_FAILED);
            // 设置 ZIP 解压缩错误状态为 CRC 检查失败
            status = TINFL_STATUS_FAILED;
        }
#endif
    }

    // 如果没有自定义内存分配器并且没有用户提供读取缓冲区，则释放读取缓冲区
    if ((!pZip->m_pState->m_pMem) && (!pUser_read_buf))
        pZip->m_pFree(pZip->m_pAlloc_opaque, pRead_buf);

    // 返回解压缩状态是否完成
    return status == TINFL_STATUS_DONE;
}

// 将 ZIP 存档中的指定文件提取到指定内存中，不进行动态内存分配
mz_bool mz_zip_reader_extract_file_to_mem_no_alloc(mz_zip_archive *pZip, const char *pFilename, void *pBuf, size_t buf_size, mz_uint flags, void *pUser_read_buf, size_t user_read_buf_size)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint32 file_index;
    // 定位 ZIP 存档中指定文件的索引
    if (!mz_zip_reader_locate_file_v2(pZip, pFilename, NULL, flags, &file_index))
        return MZ_FALSE;
    // 将指定索引的文件内容提取到内存中，不进行动态内存分配
    return mz_zip_reader_extract_to_mem_no_alloc(pZip, file_index, pBuf, buf_size, flags, pUser_read_buf, user_read_buf_size);
}

// 将 ZIP 存档中指定索引的文件提取到内存中，不进行动态内存分配
mz_bool mz_zip_reader_extract_to_mem(mz_zip_archive *pZip, mz_uint file_index, void *pBuf, size_t buf_size, mz_uint flags)
{
    return mz_zip_reader_extract_to_mem_no_alloc(pZip, file_index, pBuf, buf_size, flags, NULL, 0);
}

// 将 ZIP 存档中指定文件名的文件提取到内存中，不进行动态内存分配
mz_bool mz_zip_reader_extract_file_to_mem(mz_zip_archive *pZip, const char *pFilename, void *pBuf, size_t buf_size, mz_uint flags)
{
    return mz_zip_reader_extract_file_to_mem_no_alloc(pZip, pFilename, pBuf, buf_size, flags, NULL, 0);
}

// 将 ZIP 存档中指定索引的文件提取到堆上分配的内存中
void *mz_zip_reader_extract_to_heap(mz_zip_archive *pZip, mz_uint file_index, size_t *pSize, mz_uint flags)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint64 comp_size, uncomp_size, alloc_size;
    // 获取指定文件索引的中央目录头信息
    const mz_uint8 *p = mz_zip_get_cdh(pZip, file_index);
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    void *pBuf;

    // 如果存在 pSize，将其设置为 0
    if (pSize)
        *pSize = 0;

    // 如果获取中央目录头信息失败，则返回空指针
    if (!p)
    {
        mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);
        return NULL;
    }

    // 从中央目录头信息中读取压缩大小和未压缩大小
    comp_size = MZ_READ_LE32(p + MZ_ZIP_CDH_COMPRESSED_SIZE_OFS);
    uncomp_size = MZ_READ_LE32(p + MZ_ZIP_CDH_DECOMPRESSED_SIZE_OFS);

    // 根据标志位确定分配的大小（压缩或未压缩大小）
    alloc_size = (flags & MZ_ZIP_FLAG_COMPRESSED_DATA) ? comp_size : uncomp_size;

    // 如果是 32 位系统且分配大小超过 0x7FFFFFFF，则返回空指针
    if (((sizeof(size_t) == sizeof(mz_uint32))) && (alloc_size > 0x7FFFFFFF))
    {
        mz_zip_set_error(pZip, MZ_ZIP_INTERNAL_ERROR);
        return NULL;
    }

    // 使用 ZIP 存档的分配器分配内存
    if (NULL == (pBuf = pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1, (size_t)alloc_size)))
    {
        mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
        return NULL;
    }

    // 将指定索引的文件内容提取到分配的内存中
    if (!mz_zip_reader_extract_to_mem(pZip, file_index, pBuf, (size_t)alloc_size, flags))
    {
        // 如果提取失败，释放分配的内存并返回空指针
        pZip->m_pFree(pZip->m_pAlloc_opaque, pBuf);
        return NULL;
    }

    // 如果存在 pSize，设置其值为分配的大小
    if (pSize)
        *pSize = (size_t)alloc_size;
    return pBuf;
}

// 将 ZIP 存档中指定文件名的文件提取到堆上分配的内存中
void *mz_zip_reader_extract_file_to_heap(mz_zip_archive *pZip, const char *pFilename, size_t *pSize, mz_uint flags)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint32 file_index;
    // 定位 ZIP 存档中指定文件名的文件的索引
    if (!mz_zip_reader_locate_file_v2(pZip, pFilename, NULL, flags, &file_index))
        return NULL;
    // 将指定索引的文件内容提取到堆上分配的内存中

    return mz_zip_reader_extract_to_heap(pZip, file_index, pSize, flags);
}
    # 如果无法定位 ZIP 文件中指定文件名的文件
    if (!mz_zip_reader_locate_file_v2(pZip, pFilename, NULL, flags, &file_index))
    {
        # 如果传入了 pSize 指针，则将其指向的值设为 0
        if (pSize)
            *pSize = 0;
        # 返回失败状态
        return MZ_FALSE;
    }
    # 使用 ZIP 读取器从指定索引的文件中提取数据到堆中，并返回结果
    return mz_zip_reader_extract_to_heap(pZip, file_index, pSize, flags);
    }
    mz_bool mz_zip_reader_extract_to_callback(mz_zip_archive *pZip, mz_uint file_index, mz_file_write_func pCallback, void *pOpaque, mz_uint flags)
    {
        // 初始化状态为完成
        int status = TINFL_STATUS_DONE;
#ifndef MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS
        // 如果未禁用 CRC32 检查，则初始化文件的 CRC32 值
        mz_uint file_crc32 = MZ_CRC32_INIT;
#endif
        // 读取缓冲区大小和偏移量初始化为0，可读取缓冲区和剩余压缩数据大小，输出缓冲区偏移量初始化为0，当前文件偏移量初始化
        mz_uint64 read_buf_size, read_buf_ofs = 0, read_buf_avail, comp_remaining, out_buf_ofs = 0, cur_file_ofs;
        // 存储文件统计信息的结构体
        mz_zip_archive_file_stat file_stat;
        // 初始化读取和写入缓冲区指针为空
        void *pRead_buf = NULL;
        void *pWrite_buf = NULL;
        // 本地头部信息的数组，确保内存对齐
        mz_uint32 local_header_u32[(MZ_ZIP_LOCAL_DIR_HEADER_SIZE + sizeof(mz_uint32) - 1) / sizeof(mz_uint32)];
        // 本地头部信息的指针
        mz_uint8 *pLocal_header = (mz_uint8 *)local_header_u32;

        // 如果输入参数有误或者必需的指针为空，则返回无效参数错误
        if ((!pZip) || (!pZip->m_pState) || (!pCallback) || (!pZip->m_pRead))
            return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

        // 获取指定文件索引的文件统计信息
        if (!mz_zip_reader_file_stat(pZip, file_index, &file_stat))
            return MZ_FALSE;

        /* A directory or zero length file */
        // 如果是目录或者文件大小为0，则返回成功状态
        if ((file_stat.m_is_directory) || (!file_stat.m_comp_size))
            return MZ_TRUE;

        /* Encryption and patch files are not supported. */
        // 如果文件使用了加密或者补丁压缩标志，则返回不支持的加密错误
        if (file_stat.m_bit_flag & (MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_IS_ENCRYPTED | MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_USES_STRONG_ENCRYPTION | MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_COMPRESSED_PATCH_FLAG))
            return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_ENCRYPTION);

        /* This function only supports decompressing stored and deflate. */
        // 如果标志不包含压缩数据标志且文件的压缩方法不是存储或者 deflate，则返回不支持的压缩方法错误
        if ((!(flags & MZ_ZIP_FLAG_COMPRESSED_DATA)) && (file_stat.m_method != 0) && (file_stat.m_method != MZ_DEFLATED))
            return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_METHOD);

        /* Read and do some minimal validation of the local directory entry (this doesn't crack the zip64 stuff, which we already have from the central dir) */
        // 读取本地目录条目，并进行最小的验证
        cur_file_ofs = file_stat.m_local_header_ofs;
        // 从 ZIP 文件中读取本地头部信息
        if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pLocal_header, MZ_ZIP_LOCAL_DIR_HEADER_SIZE) != MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);

        // 验证本地头部的标志
        if (MZ_READ_LE32(pLocal_header) != MZ_ZIP_LOCAL_DIR_HEADER_SIG)
            return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);

        // 计算当前文件数据的结束偏移量
        cur_file_ofs += MZ_ZIP_LOCAL_DIR_HEADER_SIZE + MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_FILENAME_LEN_OFS) + MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_EXTRA_LEN_OFS);
        // 如果当前文件数据的结束偏移量加上压缩数据大小超出了 ZIP 文件的总大小，则返回无效的头部或者损坏的 ZIP 文件错误
        if ((cur_file_ofs + file_stat.m_comp_size) > pZip->m_archive_size)
            return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);

        /* Decompress the file either directly from memory or from a file input buffer. */
        // 如果 ZIP 存档在内存中
        if (pZip->m_pState->m_pMem)
        {
            // 设置读取缓冲区指针和大小，剩余可读取缓冲区大小，压缩数据剩余大小
            pRead_buf = (mz_uint8 *)pZip->m_pState->m_pMem + cur_file_ofs;
            read_buf_size = read_buf_avail = file_stat.m_comp_size;
            comp_remaining = 0;
        }
        else
    {
        // 计算读取缓冲区的大小，取较小值，确保不超过最大允许的 I/O 缓冲区大小
        read_buf_size = MZ_MIN(file_stat.m_comp_size, (mz_uint64)MZ_ZIP_MAX_IO_BUF_SIZE);
        // 分配内存以创建读取缓冲区，如果分配失败则返回内存分配失败的错误状态
        if (NULL == (pRead_buf = pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1, (size_t)read_buf_size)))
            return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
    
        // 初始化读取缓冲区的可用字节数
        read_buf_avail = 0;
        // 设置压缩文件剩余未读取的字节数
        comp_remaining = file_stat.m_comp_size;
    }
    
    if ((flags & MZ_ZIP_FLAG_COMPRESSED_DATA) || (!file_stat.m_method))
    {
        /* The file is stored or the caller has requested the compressed data. */
        // 如果压缩数据标志被设置，或者文件存储方式为无压缩
        if (pZip->m_pState->m_pMem)
        {
            // 如果 size_t 和 mz_uint32 大小相等，并且压缩文件大小超过了 mz_uint32 的最大值，返回内部错误
            if (((sizeof(size_t) == sizeof(mz_uint32))) && (file_stat.m_comp_size > MZ_UINT32_MAX))
                return mz_zip_set_error(pZip, MZ_ZIP_INTERNAL_ERROR);
    
            // 调用回调函数写入压缩数据到输出缓冲区，如果写入失败则设置写入回调失败的错误状态
            if (pCallback(pOpaque, out_buf_ofs, pRead_buf, (size_t)file_stat.m_comp_size) != file_stat.m_comp_size)
            {
                mz_zip_set_error(pZip, MZ_ZIP_WRITE_CALLBACK_FAILED);
                status = TINFL_STATUS_FAILED;
            }
            // 如果没有请求压缩数据，执行以下操作
            else if (!(flags & MZ_ZIP_FLAG_COMPRESSED_DATA))
            {
// 如果未禁用 ZIP 读取器的 CRC32 检查，计算文件数据的 CRC32 值
#ifndef MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS
    file_crc32 = (mz_uint32)mz_crc32(file_crc32, (const mz_uint8 *)pRead_buf, (size_t)file_stat.m_comp_size);
#endif
}

// 更新当前文件偏移量和输出缓冲区偏移量，忽略 Clang 分析器的未使用变量警告
cur_file_ofs += file_stat.m_comp_size;
out_buf_ofs += file_stat.m_comp_size;
// 忽略 Clang 分析器的未使用变量警告
comp_remaining = 0;
}
else
{
    // 当还有压缩数据剩余时循环处理
    while (comp_remaining)
    {
        // 计算当前读取缓冲区的可用大小，取较小值
        read_buf_avail = MZ_MIN(read_buf_size, comp_remaining);
        // 使用 ZIP 结构体的读取回调函数读取数据到读取缓冲区
        if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pRead_buf, (size_t)read_buf_avail) != read_buf_avail)
        {
            // 设置 ZIP 结构体的错误状态为文件读取失败，并标记解压状态为失败
            mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);
            status = TINFL_STATUS_FAILED;
            break;
        }

        // 如果未禁用 ZIP 读取器的 CRC32 检查，并且未压缩数据标志未设置
#ifndef MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS
        if (!(flags & MZ_ZIP_FLAG_COMPRESSED_DATA))
        {
            // 计算文件数据的 CRC32 值
            file_crc32 = (mz_uint32)mz_crc32(file_crc32, (const mz_uint8 *)pRead_buf, (size_t)read_buf_avail);
        }
#endif

        // 使用回调函数处理输出缓冲区中的数据
        if (pCallback(pOpaque, out_buf_ofs, pRead_buf, (size_t)read_buf_avail) != read_buf_avail)
        {
            // 设置 ZIP 结构体的错误状态为写回调函数失败，并标记解压状态为失败
            mz_zip_set_error(pZip, MZ_ZIP_WRITE_CALLBACK_FAILED);
            status = TINFL_STATUS_FAILED;
            break;
        }

        // 更新当前文件偏移量和输出缓冲区偏移量，减少剩余压缩数据量
        cur_file_ofs += read_buf_avail;
        out_buf_ofs += read_buf_avail;
        comp_remaining -= read_buf_avail;
    }
}
    {
        // 初始化解压缩器
        tinfl_decompressor inflator;
        tinfl_init(&inflator);
    
        // 分配写入缓冲区，大小为 TINFL_LZ_DICT_SIZE，使用 m_pAlloc 函数分配内存
        if (NULL == (pWrite_buf = pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1, TINFL_LZ_DICT_SIZE)))
        {
            // 如果分配失败，则设置 ZIP 错误为分配失败，并返回解压缩失败状态
            mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
            status = TINFL_STATUS_FAILED;
        }
        else
        {
            do
            {
                // 计算当前写入缓冲区的位置
                mz_uint8 *pWrite_buf_cur = (mz_uint8 *)pWrite_buf + (out_buf_ofs & (TINFL_LZ_DICT_SIZE - 1));
                // 计算输入缓冲区大小和输出缓冲区大小
                size_t in_buf_size, out_buf_size = TINFL_LZ_DICT_SIZE - (out_buf_ofs & (TINFL_LZ_DICT_SIZE - 1));
                
                // 如果读取缓冲区无效并且内存状态无效，则更新读取缓冲区
                if ((!read_buf_avail) && (!pZip->m_pState->m_pMem))
                {
                    // 确保读取的缓冲区不超过可读大小和剩余压缩大小
                    read_buf_avail = MZ_MIN(read_buf_size, comp_remaining);
                    
                    // 从文件中读取指定大小的数据到读取缓冲区
                    if (pZip->m_pRead(pZip->m_pIO_opaque, cur_file_ofs, pRead_buf, (size_t)read_buf_avail) != read_buf_avail)
                    {
                        // 如果读取失败，则设置 ZIP 错误为文件读取失败，并返回解压缩失败状态
                        mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);
                        status = TINFL_STATUS_FAILED;
                        break;
                    }
    
                    // 更新当前文件偏移和剩余压缩大小
                    cur_file_ofs += read_buf_avail;
                    comp_remaining -= read_buf_avail;
                    read_buf_ofs = 0;
                }
    
                // 设置输入缓冲区的大小
                in_buf_size = (size_t)read_buf_avail;
                
                // 进行解压缩操作，将输入数据解压到写入缓冲区
                status = tinfl_decompress(&inflator, (const mz_uint8 *)pRead_buf + read_buf_ofs, &in_buf_size, (mz_uint8 *)pWrite_buf, pWrite_buf_cur, &out_buf_size, comp_remaining ? TINFL_FLAG_HAS_MORE_INPUT : 0);
                
                // 更新读取缓冲区状态
                read_buf_avail -= in_buf_size;
                read_buf_ofs += in_buf_size;
    
                // 如果输出缓冲区有数据
                if (out_buf_size)
                {
                    // 调用回调函数写入输出缓冲区的数据
                    if (pCallback(pOpaque, out_buf_ofs, pWrite_buf_cur, out_buf_size) != out_buf_size)
                    {
                        // 如果回调写入失败，则设置 ZIP 错误为写入回调失败，并返回解压缩失败状态
                        mz_zip_set_error(pZip, MZ_ZIP_WRITE_CALLBACK_FAILED);
                        status = TINFL_STATUS_FAILED;
                        break;
                    }
#ifndef MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS
                    // 如果未禁用 CRC32 校验，计算文件的 CRC32 值
                    file_crc32 = (mz_uint32)mz_crc32(file_crc32, pWrite_buf_cur, out_buf_size);
#endif
                    // 如果输出缓冲区偏移量超过了文件的未压缩大小，则标记解压失败
                    if ((out_buf_ofs += out_buf_size) > file_stat.m_uncomp_size)
                    {
                        mz_zip_set_error(pZip, MZ_ZIP_DECOMPRESSION_FAILED);
                        status = TINFL_STATUS_FAILED;
                        break;
                    }
                }
            } while ((status == TINFL_STATUS_NEEDS_MORE_INPUT) || (status == TINFL_STATUS_HAS_MORE_OUTPUT));
        }
    }

    // 如果解压状态为完成，并且不包含压缩数据标志
    if ((status == TINFL_STATUS_DONE) && (!(flags & MZ_ZIP_FLAG_COMPRESSED_DATA)))
    {
        /* 确保整个文件已解压并检查其 CRC */
        if (out_buf_ofs != file_stat.m_uncomp_size)
        {
            mz_zip_set_error(pZip, MZ_ZIP_UNEXPECTED_DECOMPRESSED_SIZE);
            status = TINFL_STATUS_FAILED;
        }
#ifndef MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS
        else if (file_crc32 != file_stat.m_crc32)
        {
            mz_zip_set_error(pZip, MZ_ZIP_DECOMPRESSION_FAILED);
            status = TINFL_STATUS_FAILED;
        }
#endif
    }

    // 如果没有分配内存给 ZIP 状态，则释放读取缓冲区
    if (!pZip->m_pState->m_pMem)
        pZip->m_pFree(pZip->m_pAlloc_opaque, pRead_buf);

    // 如果存在写入缓冲区，则释放其内存
    if (pWrite_buf)
        pZip->m_pFree(pZip->m_pAlloc_opaque, pWrite_buf);

    // 返回解压是否成功的状态
    return status == TINFL_STATUS_DONE;
}

// 从 ZIP 文件中提取特定文件到回调函数
mz_bool mz_zip_reader_extract_file_to_callback(mz_zip_archive *pZip, const char *pFilename, mz_file_write_func pCallback, void *pOpaque, mz_uint flags)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint32 file_index;
    // 定位到 ZIP 文件中的指定文件
    if (!mz_zip_reader_locate_file_v2(pZip, pFilename, NULL, flags, &file_index))
        return MZ_FALSE;

    // 调用函数从 ZIP 中提取文件到指定的回调函数
    return mz_zip_reader_extract_to_callback(pZip, file_index, pCallback, pOpaque, flags);
}

// 创建一个 ZIP 文件读取器的文件提取迭代状态
mz_zip_reader_extract_iter_state* mz_zip_reader_extract_iter_new(mz_zip_archive *pZip, mz_uint file_index, mz_uint flags)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_zip_reader_extract_iter_state *pState;
    // 本地头信息缓冲区
    mz_uint32 local_header_u32[(MZ_ZIP_LOCAL_DIR_HEADER_SIZE + sizeof(mz_uint32) - 1) / sizeof(mz_uint32)];
    mz_uint8 *pLocal_header = (mz_uint8 *)local_header_u32;

    /* 参数合法性检查 */
    if ((!pZip) || (!pZip->m_pState))
        return NULL;

    /* 分配一个迭代器状态结构 */
    pState = (mz_zip_reader_extract_iter_state*)pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1, sizeof(mz_zip_reader_extract_iter_state));
    if (!pState)
    {
        mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
        return NULL;
    }

    /* 获取文件的详细信息 */
    if (!mz_zip_reader_file_stat(pZip, file_index, &pState->file_stat))
    {
        pZip->m_pFree(pZip->m_pAlloc_opaque, pState);
        return NULL;
    }

    /* 不支持加密和补丁文件 */
    # 检查文件状态中的位标志，以确定是否包含不支持的加密或压缩标志位
    if (pState->file_stat.m_bit_flag & (MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_IS_ENCRYPTED | MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_USES_STRONG_ENCRYPTION | MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_COMPRESSED_PATCH_FLAG))
    {
        # 设置 ZIP 对象错误状态为不支持的加密
        mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_ENCRYPTION);
        # 释放 pState 内存并返回 NULL
        pZip->m_pFree(pZip->m_pAlloc_opaque, pState);
        return NULL;
    }

    /* This function only supports decompressing stored and deflate. */
    # 检查压缩标志和压缩方法，仅支持存储和 deflate 方法的解压缩
    if ((!(flags & MZ_ZIP_FLAG_COMPRESSED_DATA)) && (pState->file_stat.m_method != 0) && (pState->file_stat.m_method != MZ_DEFLATED))
    {
        # 设置 ZIP 对象错误状态为不支持的压缩方法
        mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_METHOD);
        # 释放 pState 内存并返回 NULL
        pZip->m_pFree(pZip->m_pAlloc_opaque, pState);
        return NULL;
    }

    /* Init state - save args */
    # 初始化状态 - 保存参数
    pState->pZip = pZip;
    pState->flags = flags;

    /* Init state - reset variables to defaults */
    # 初始化状态 - 将变量重置为默认值
    pState->status = TINFL_STATUS_DONE;
#ifndef MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS
    // 如果未禁用 CRC32 校验，则初始化文件 CRC32 校验值
    pState->file_crc32 = MZ_CRC32_INIT;
#endif

// 重置读取缓冲区和输出缓冲区的偏移量
pState->read_buf_ofs = 0;
pState->out_buf_ofs = 0;
// 将读取和写入缓冲区的指针设为 NULL
pState->pRead_buf = NULL;
pState->pWrite_buf = NULL;
// 初始化输出块的剩余字节数为 0
pState->out_blk_remain = 0;

/* 读取并解析本地目录条目 */
// 设置当前文件的偏移量为本地文件头的偏移量
pState->cur_file_ofs = pState->file_stat.m_local_header_ofs;
// 从 ZIP 文件中读取本地文件头部分到 pLocal_header
if (pZip->m_pRead(pZip->m_pIO_opaque, pState->cur_file_ofs, pLocal_header, MZ_ZIP_LOCAL_DIR_HEADER_SIZE) != MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
{
    // 如果读取失败，设置 ZIP 错误状态，并释放 pState 内存
    mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);
    pZip->m_pFree(pZip->m_pAlloc_opaque, pState);
    return NULL;
}

// 检查本地文件头的签名是否正确
if (MZ_READ_LE32(pLocal_header) != MZ_ZIP_LOCAL_DIR_HEADER_SIG)
{
    // 如果签名不正确，设置 ZIP 错误状态，并释放 pState 内存
    mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);
    pZip->m_pFree(pZip->m_pAlloc_opaque, pState);
    return NULL;
}

// 更新当前文件偏移量，跳过文件名长度和额外字段长度
pState->cur_file_ofs += MZ_ZIP_LOCAL_DIR_HEADER_SIZE + MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_FILENAME_LEN_OFS) + MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_EXTRA_LEN_OFS);

// 检查文件解压后大小加上当前偏移量是否超过 ZIP 存档的大小
if ((pState->cur_file_ofs + pState->file_stat.m_comp_size) > pZip->m_archive_size)
{
    // 如果超过，设置 ZIP 错误状态，并释放 pState 内存
    mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);
    pZip->m_pFree(pZip->m_pAlloc_opaque, pState);
    return NULL;
}

/* 根据需求从内存直接解压文件或者使用文件输入缓冲区解压文件 */
if (pZip->m_pState->m_pMem)
{
    // 如果 ZIP 状态中有内存数据，直接指向内存中对应位置的数据
    pState->pRead_buf = (mz_uint8 *)pZip->m_pState->m_pMem + pState->cur_file_ofs;
    pState->read_buf_size = pState->read_buf_avail = pState->file_stat.m_comp_size;
    pState->comp_remaining = pState->file_stat.m_comp_size;
}
else
{
    // 如果不是从内存解压，根据标志位和压缩方法判断是否需要中间读取缓冲区
    if (!((flags & MZ_ZIP_FLAG_COMPRESSED_DATA) || (!pState->file_stat.m_method)))
    {
        // 如果需要解压，确定中间读取缓冲区的大小，并分配内存
        pState->read_buf_size = MZ_MIN(pState->file_stat.m_comp_size, MZ_ZIP_MAX_IO_BUF_SIZE);
        if (NULL == (pState->pRead_buf = pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1, (size_t)pState->read_buf_size)))
        {
            // 分配内存失败，设置 ZIP 错误状态，并释放 pState 内存
            mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
            pZip->m_pFree(pZip->m_pAlloc_opaque, pState);
            return NULL;
        }
    }
    else
    {
        // 如果不需要解压，直接读取到用户缓冲区，不需要临时缓冲区
        pState->read_buf_size = 0;
    }
    pState->read_buf_avail = 0;
    pState->comp_remaining = pState->file_stat.m_comp_size;
}

// 继续判断是否需要解压数据
if (!((flags & MZ_ZIP_FLAG_COMPRESSED_DATA) || (!pState->file_stat.m_method)))
    {
        /* Decompression required, init decompressor */
        // 对解压缩进行初始化，初始化解压缩器
        tinfl_init(&pState->inflator);
    
        /* Allocate write buffer */
        // 分配写入缓冲区
        if (NULL == (pState->pWrite_buf = pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1, TINFL_LZ_DICT_SIZE)))
        {
            // 如果分配失败，则设置 ZIP 错误码为分配失败
            mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
            // 如果读取缓冲区已分配，则释放它
            if (pState->pRead_buf)
                pZip->m_pFree(pZip->m_pAlloc_opaque, pState->pRead_buf);
            // 释放状态对象内存
            pZip->m_pFree(pZip->m_pAlloc_opaque, pState);
            // 返回空指针，表示初始化失败
            return NULL;
        }
    }
    
    // 返回初始化后的状态对象指针
    return pState;
}

mz_zip_reader_extract_iter_state* mz_zip_reader_extract_file_iter_new(mz_zip_archive *pZip, const char *pFilename, mz_uint flags)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint32 file_index;

    /* Locate file index by name */
    // 使用给定的文件名在 ZIP 归档中定位文件索引
    if (!mz_zip_reader_locate_file_v2(pZip, pFilename, NULL, flags, &file_index))
        return NULL;

    /* Construct iterator */
    // 构造一个文件提取迭代器并返回
    return mz_zip_reader_extract_iter_new(pZip, file_index, flags);
}

size_t mz_zip_reader_extract_iter_read(mz_zip_reader_extract_iter_state* pState, void* pvBuf, size_t buf_size)
{
    size_t copied_to_caller = 0;

    /* Argument sanity check */
    // 参数有效性检查：确保 pState 和 pvBuf 非空，且 pState 所引用的 ZIP 对象和状态都有效
    if ((!pState) || (!pState->pZip) || (!pState->pZip->m_pState) || (!pvBuf))
        return 0;

    if ((pState->flags & MZ_ZIP_FLAG_COMPRESSED_DATA) || (!pState->file_stat.m_method))
    {
        /* The file is stored or the caller has requested the compressed data, calc amount to return. */
        // 文件是存储的或调用者请求的是压缩数据，计算需要返回的数据量
        copied_to_caller = (size_t)MZ_MIN( buf_size, pState->comp_remaining );

        /* Zip is in memory....or requires reading from a file? */
        // ZIP 归档在内存中或需要从文件中读取？
        if (pState->pZip->m_pState->m_pMem)
        {
            /* Copy data to caller's buffer */
            // 将数据复制到调用者提供的缓冲区中
            memcpy( pvBuf, pState->pRead_buf, copied_to_caller );
            pState->pRead_buf = ((mz_uint8*)pState->pRead_buf) + copied_to_caller;
        }
        else
        {
            /* Read directly into caller's buffer */
            // 直接读取到调用者提供的缓冲区中
            if (pState->pZip->m_pRead(pState->pZip->m_pIO_opaque, pState->cur_file_ofs, pvBuf, copied_to_caller) != copied_to_caller)
            {
                /* Failed to read all that was asked for, flag failure and alert user */
                // 未能完全读取所需的数据量，标记失败并通知用户
                mz_zip_set_error(pState->pZip, MZ_ZIP_FILE_READ_FAILED);
                pState->status = TINFL_STATUS_FAILED;
                copied_to_caller = 0;
            }
        }

#ifndef MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS
        /* Compute CRC if not returning compressed data only */
        // 如果不仅返回压缩数据，还需计算 CRC32 校验值
        if (!(pState->flags & MZ_ZIP_FLAG_COMPRESSED_DATA))
            pState->file_crc32 = (mz_uint32)mz_crc32(pState->file_crc32, (const mz_uint8 *)pvBuf, copied_to_caller);
#endif

        /* Advance offsets, dec counters */
        // 推进偏移量，减少计数器
        pState->cur_file_ofs += copied_to_caller;
        pState->out_buf_ofs += copied_to_caller;
        pState->comp_remaining -= copied_to_caller;
    }
    else
    {
        do
        {
            /* 计算写入缓冲区的指针位置 - 根据当前输出位置和块大小计算 */
            mz_uint8 *pWrite_buf_cur = (mz_uint8 *)pState->pWrite_buf + (pState->out_buf_ofs & (TINFL_LZ_DICT_SIZE - 1));
    
            /* 计算最大输出大小 - 根据当前输出位置和块大小计算 */
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            size_t in_buf_size, out_buf_size = TINFL_LZ_DICT_SIZE - (pState->out_buf_ofs & (TINFL_LZ_DICT_SIZE - 1));
    
            if (!pState->out_blk_remain)
            {
                /* 如果没有剩余输出块，则从文件中读取更多数据（如果当前不是从内存中读取） */
                if ((!pState->read_buf_avail) && (!pState->pZip->m_pState->m_pMem))
                {
                    /* 计算读取大小 */
                    pState->read_buf_avail = MZ_MIN(pState->read_buf_size, pState->comp_remaining);
                    if (pState->pZip->m_pRead(pState->pZip->m_pIO_opaque, pState->cur_file_ofs, pState->pRead_buf, (size_t)pState->read_buf_avail) != pState->read_buf_avail)
                    {
                        mz_zip_set_error(pState->pZip, MZ_ZIP_FILE_READ_FAILED);
                        pState->status = TINFL_STATUS_FAILED;
                        break;
                    }
    
                    /* 更新偏移量，减少计数器 */
                    pState->cur_file_ofs += pState->read_buf_avail;
                    pState->comp_remaining -= pState->read_buf_avail;
                    pState->read_buf_ofs = 0;
                }
    
                /* 执行解压缩 */
                in_buf_size = (size_t)pState->read_buf_avail;
                pState->status = tinfl_decompress(&pState->inflator, (const mz_uint8 *)pState->pRead_buf + pState->read_buf_ofs, &in_buf_size, (mz_uint8 *)pState->pWrite_buf, pWrite_buf_cur, &out_buf_size, pState->comp_remaining ? TINFL_FLAG_HAS_MORE_INPUT : 0);
                pState->read_buf_avail -= in_buf_size;
                pState->read_buf_ofs += in_buf_size;
    
                /* 更新当前输出块的剩余大小 */
                pState->out_blk_remain = out_buf_size;
            }
    
            if (pState->out_blk_remain)
            {
                /* 计算要复制的数量 */
                size_t to_copy = MZ_MIN((buf_size - copied_to_caller), pState->out_blk_remain);
    
                /* 将数据复制到调用者的缓冲区 */
                memcpy((uint8_t*)pvBuf + copied_to_caller, pWrite_buf_cur, to_copy);
#ifndef MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS
                /* 执行 CRC 校验 */
                pState->file_crc32 = (mz_uint32)mz_crc32(pState->file_crc32, pWrite_buf_cur, to_copy);
#endif

                /* 减少从块中消耗的数据量 */
                pState->out_blk_remain -= to_copy;

                /* 增加输出偏移量，并进行健全性检查 */
                if ((pState->out_buf_ofs += to_copy) > pState->file_stat.m_uncomp_size)
                {
                    mz_zip_set_error(pState->pZip, MZ_ZIP_DECOMPRESSION_FAILED);
                    pState->status = TINFL_STATUS_FAILED;
                    break;
                }

                /* 增加已复制到调用者的数据计数器 */
                copied_to_caller += to_copy;
            }
        } while ( (copied_to_caller < buf_size) && ((pState->status == TINFL_STATUS_NEEDS_MORE_INPUT) || (pState->status == TINFL_STATUS_HAS_MORE_OUTPUT)) );
    }

    /* 返回复制到用户缓冲区的字节数 */
    return copied_to_caller;
}

mz_bool mz_zip_reader_extract_iter_free(mz_zip_reader_extract_iter_state* pState)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int status;

    /* 参数健全性检查 */
    if ((!pState) || (!pState->pZip) || (!pState->pZip->m_pState))
        return MZ_FALSE;

    /* 解压是否已完成且已请求？ */
    if ((pState->status == TINFL_STATUS_DONE) && (!(pState->flags & MZ_ZIP_FLAG_COMPRESSED_DATA)))
    {
        /* 确保文件已完全解压，并检查其 CRC */
        if (pState->out_buf_ofs != pState->file_stat.m_uncomp_size)
        {
            mz_zip_set_error(pState->pZip, MZ_ZIP_UNEXPECTED_DECOMPRESSED_SIZE);
            pState->status = TINFL_STATUS_FAILED;
        }
#ifndef MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS
        else if (pState->file_crc32 != pState->file_stat.m_crc32)
        {
            mz_zip_set_error(pState->pZip, MZ_ZIP_DECOMPRESSION_FAILED);
            pState->status = TINFL_STATUS_FAILED;
        }
#endif
    }

    /* 释放缓冲区 */
    if (!pState->pZip->m_pState->m_pMem)
        pState->pZip->m_pFree(pState->pZip->m_pAlloc_opaque, pState->pRead_buf);
    if (pState->pWrite_buf)
        pState->pZip->m_pFree(pState->pZip->m_pAlloc_opaque, pState->pWrite_buf);

    /* 保存状态 */
    status = pState->status;

    /* 释放上下文 */
    pState->pZip->m_pFree(pState->pZip->m_pAlloc_opaque, pState);

    return status == TINFL_STATUS_DONE;
}

#ifndef MINIZ_NO_STDIO
static size_t mz_zip_file_write_callback(void *pOpaque, mz_uint64 ofs, const void *pBuf, size_t n)
{
    (void)ofs;

    return MZ_FWRITE(pBuf, 1, n, (MZ_FILE *)pOpaque);
}

mz_bool mz_zip_reader_extract_to_file(mz_zip_archive *pZip, mz_uint file_index, const char *pDst_filename, mz_uint flags)
{
    mz_bool status;
    mz_zip_archive_file_stat file_stat;
    MZ_FILE *pFile;

    if (!mz_zip_reader_file_stat(pZip, file_index, &file_stat))
        return MZ_FALSE;
    # 如果文件状态表明是目录或者不支持的文件类型，则返回不支持的特性错误
    if ((file_stat.m_is_directory) || (!file_stat.m_is_supported))
        return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_FEATURE);

    # 尝试以写入二进制模式打开目标文件
    pFile = MZ_FOPEN(pDst_filename, "wb");
    # 如果文件打开失败，则返回文件打开失败错误
    if (!pFile)
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_OPEN_FAILED);

    # 将指定文件索引的内容从 ZIP 中提取到回调函数中，并将结果写入 pFile
    status = mz_zip_reader_extract_to_callback(pZip, file_index, mz_zip_file_write_callback, pFile, flags);

    # 关闭文件，并检查是否关闭失败
    if (MZ_FCLOSE(pFile) == EOF)
    {
        # 如果关闭文件失败，并且之前操作无误，则设置文件关闭失败的错误
        if (status)
            mz_zip_set_error(pZip, MZ_ZIP_FILE_CLOSE_FAILED);

        # 不管之前的操作结果如何，都将状态标记为假（失败）
        status = MZ_FALSE;
    }
#if !defined(MINIZ_NO_TIME) && !defined(MINIZ_NO_STDIO)
    // 如果没有定义 MINIZ_NO_TIME 和 MINIZ_NO_STDIO 宏，则执行以下代码
    if (status)
        // 如果 status 不为 0，则设置目标文件的修改时间为文件状态的修改时间
        mz_zip_set_file_times(pDst_filename, file_stat.m_time, file_stat.m_time);
#endif

    // 返回函数执行的状态
    return status;
}

mz_bool mz_zip_reader_extract_file_to_file(mz_zip_archive *pZip, const char *pArchive_filename, const char *pDst_filename, mz_uint flags)
{
    mz_uint32 file_index;
    // 定位 ZIP 存档中指定文件名的文件索引，如果失败则返回 MZ_FALSE
    if (!mz_zip_reader_locate_file_v2(pZip, pArchive_filename, NULL, flags, &file_index))
        return MZ_FALSE;

    // 使用文件索引提取文件到目标文件中，返回执行结果
    return mz_zip_reader_extract_to_file(pZip, file_index, pDst_filename, flags);
}

mz_bool mz_zip_reader_extract_to_cfile(mz_zip_archive *pZip, mz_uint file_index, MZ_FILE *pFile, mz_uint flags)
{
    mz_zip_archive_file_stat file_stat;
    // 获取 ZIP 存档中指定文件索引的文件状态信息，如果失败则返回 MZ_FALSE
    if (!mz_zip_reader_file_stat(pZip, file_index, &file_stat))
        return MZ_FALSE;

    // 如果是目录或不支持的文件类型，则返回不支持的错误
    if ((file_stat.m_is_directory) || (!file_stat.m_is_supported))
        return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_FEATURE);

    // 将 ZIP 存档中指定文件索引的文件内容写入到回调文件中，返回执行结果
    return mz_zip_reader_extract_to_callback(pZip, file_index, mz_zip_file_write_callback, pFile, flags);
}

mz_bool mz_zip_reader_extract_file_to_cfile(mz_zip_archive *pZip, const char *pArchive_filename, MZ_FILE *pFile, mz_uint flags)
{
    mz_uint32 file_index;
    // 定位 ZIP 存档中指定文件名的文件索引，如果失败则返回 MZ_FALSE
    if (!mz_zip_reader_locate_file_v2(pZip, pArchive_filename, NULL, flags, &file_index))
        return MZ_FALSE;

    // 使用文件索引提取文件内容到回调文件中，返回执行结果
    return mz_zip_reader_extract_to_cfile(pZip, file_index, pFile, flags);
}
#endif /* #ifndef MINIZ_NO_STDIO */

static size_t mz_zip_compute_crc32_callback(void *pOpaque, mz_uint64 file_ofs, const void *pBuf, size_t n)
{
    mz_uint32 *p = (mz_uint32 *)pOpaque;
    (void)file_ofs;
    // 计算给定数据块的 CRC32 校验值，并将结果存储在 p 指向的变量中
    *p = (mz_uint32)mz_crc32(*p, (const mz_uint8 *)pBuf, n);
    return n;
}

mz_bool mz_zip_validate_file(mz_zip_archive *pZip, mz_uint file_index, mz_uint flags)
{
    mz_zip_archive_file_stat file_stat;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_zip_internal_state *pState;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    const mz_uint8 *pCentral_dir_header;
    mz_bool found_zip64_ext_data_in_cdir = MZ_FALSE;
    mz_bool found_zip64_ext_data_in_ldir = MZ_FALSE;
    mz_uint32 local_header_u32[(MZ_ZIP_LOCAL_DIR_HEADER_SIZE + sizeof(mz_uint32) - 1) / sizeof(mz_uint32)];
    mz_uint8 *pLocal_header = (mz_uint8 *)local_header_u32;
    mz_uint64 local_header_ofs = 0;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint32 local_header_filename_len, local_header_extra_len, local_header_crc32;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint64 local_header_comp_size, local_header_uncomp_size;
    mz_uint32 uncomp_crc32 = MZ_CRC32_INIT;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_bool has_data_descriptor;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint32 local_header_bit_flags;

    mz_zip_array file_data_array;
    mz_zip_array_init(&file_data_array, 1);

    // 此处为函数 mz_zip_validate_file 的其余部分，涉及验证 ZIP 存档文件的完整性和结构
}
    /* 检查指针和关键参数的有效性 */
    if ((!pZip) || (!pZip->m_pState) || (!pZip->m_pAlloc) || (!pZip->m_pFree) || (!pZip->m_pRead))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    /* 检查文件索引是否超出范围 */
    if (file_index > pZip->m_total_files)
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    /* 获取 ZIP 结构体中的状态指针 */
    pState = pZip->m_pState;

    /* 获取文件的中央目录头信息 */
    pCentral_dir_header = mz_zip_get_cdh(pZip, file_index);

    /* 获取文件的统计信息和是否找到了 ZIP64 扩展数据 */
    if (!mz_zip_file_stat_internal(pZip, file_index, pCentral_dir_header, &file_stat, &found_zip64_ext_data_in_cdir))
        return MZ_FALSE;

    /* 检查文件是否是目录或者长度为零 */
    /* A directory or zero length file */
    if ((file_stat.m_is_directory) || (!file_stat.m_uncomp_size))
        return MZ_TRUE;

    /* 检查文件是否加密 */
    /* Encryption and patch files are not supported. */
    if (file_stat.m_is_encrypted)
        return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_ENCRYPTION);

    /* 检查文件的压缩方法是否被支持 */
    /* This function only supports stored and deflate. */
    if ((file_stat.m_method != 0) && (file_stat.m_method != MZ_DEFLATED))
        return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_METHOD);

    /* 检查文件是否支持当前操作 */
    if (!file_stat.m_is_supported)
        return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_FEATURE);

    /* 读取并解析本地目录头 */
    /* Read and parse the local directory entry. */
    local_header_ofs = file_stat.m_local_header_ofs;
    if (pZip->m_pRead(pZip->m_pIO_opaque, local_header_ofs, pLocal_header, MZ_ZIP_LOCAL_DIR_HEADER_SIZE) != MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);

    /* 检查本地目录头的签名是否正确 */
    if (MZ_READ_LE32(pLocal_header) != MZ_ZIP_LOCAL_DIR_HEADER_SIG)
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);

    /* 读取本地头部的文件名长度和额外字段长度 */
    local_header_filename_len = MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_FILENAME_LEN_OFS);
    local_header_extra_len = MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_EXTRA_LEN_OFS);

    /* 读取本地头部的压缩和解压缩大小、CRC32校验和、位标志 */
    local_header_comp_size = MZ_READ_LE32(pLocal_header + MZ_ZIP_LDH_COMPRESSED_SIZE_OFS);
    local_header_uncomp_size = MZ_READ_LE32(pLocal_header + MZ_ZIP_LDH_DECOMPRESSED_SIZE_OFS);
    local_header_crc32 = MZ_READ_LE32(pLocal_header + MZ_ZIP_LDH_CRC32_OFS);
    local_header_bit_flags = MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_BIT_FLAG_OFS);

    /* 检查是否有数据描述符 */
    has_data_descriptor = (local_header_bit_flags & 8) != 0;

    /* 检查本地头部文件名长度是否与统计信息中的文件名长度一致 */
    if (local_header_filename_len != strlen(file_stat.m_filename))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);

    /* 检查本地头部数据是否超出 ZIP 存档的范围 */
    if ((local_header_ofs + MZ_ZIP_LOCAL_DIR_HEADER_SIZE + local_header_filename_len + local_header_extra_len + file_stat.m_comp_size) > pZip->m_archive_size)
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);

    /* 调整数组大小以容纳文件名长度和额外字段长度中的较大者 */
    if (!mz_zip_array_resize(pZip, &file_data_array, MZ_MAX(local_header_filename_len, local_header_extra_len), MZ_FALSE))
        return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);

    /* 如果本地头部文件名长度非零，则继续处理 */
    if (local_header_filename_len)
    {
        // 从本地头偏移量处读取文件名数据，存入file_data_array.m_p，长度为local_header_filename_len
        if (pZip->m_pRead(pZip->m_pIO_opaque, local_header_ofs + MZ_ZIP_LOCAL_DIR_HEADER_SIZE, file_data_array.m_p, local_header_filename_len) != local_header_filename_len)
        {
            // 如果读取的文件名数据长度不符合预期，设置ZIP错误并跳转到错误处理标签
            mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);
            goto handle_failure;
        }
    
        /* 我曾经见过一个归档文件，本地目录使用反斜杠，中心目录使用正斜杠。我们关心这个吗？目前情况下，这种情况会导致验证失败。 */
        // 使用NOLINTNEXTLINE(clang-analyzer-unix.cstring.NullArg)避免对空指针进行null分析
        // 检查本地头中的文件名和读取的文件名数据是否一致，如果不一致，设置ZIP错误并跳转到错误处理标签
        if (memcmp(file_stat.m_filename, file_data_array.m_p, local_header_filename_len) != 0)
        {
            mz_zip_set_error(pZip, MZ_ZIP_VALIDATION_FAILED);
            goto handle_failure;
        }
    }
    
    // 如果本地头中有额外数据，并且压缩大小或未压缩大小为最大值，继续处理
    if ((local_header_extra_len) && ((local_header_comp_size == MZ_UINT32_MAX) || (local_header_uncomp_size == MZ_UINT32_MAX)))
    {
        // 记录剩余额外数据的大小和额外数据的起始指针
        mz_uint32 extra_size_remaining = local_header_extra_len;
        const mz_uint8 *pExtra_data = (const mz_uint8 *)file_data_array.m_p;
    
        // 从本地头中读取额外数据到file_data_array.m_p，并检查读取是否成功
        if (pZip->m_pRead(pZip->m_pIO_opaque, local_header_ofs + MZ_ZIP_LOCAL_DIR_HEADER_SIZE + local_header_filename_len, file_data_array.m_p, local_header_extra_len) != local_header_extra_len)
        {
            // 如果读取额外数据失败，设置ZIP错误并跳转到错误处理标签
            mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);
            goto handle_failure;
        }
    
        do
        {
            // 定义字段ID、字段数据大小和字段总大小
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables) 避免cppcoreguidelines初始化变量规则
            mz_uint32 field_id, field_data_size, field_total_size;
    
            // 如果剩余额外数据不足以容纳两个mz_uint16大小的字段，表示头部无效或损坏
            if (extra_size_remaining < (sizeof(mz_uint16) * 2))
                return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);
    
            // NOLINTNEXTLINE(clang-analyzer-core.NullDereference) 避免clang分析器核心空解除引用
            // 从额外数据中读取字段ID和字段数据大小
            field_id = MZ_READ_LE16(pExtra_data);
            field_data_size = MZ_READ_LE16(pExtra_data + sizeof(mz_uint16));
            field_total_size = field_data_size + sizeof(mz_uint16) * 2;
    
            // 如果字段总大小超过剩余额外数据大小，表示头部无效或损坏
            if (field_total_size > extra_size_remaining)
                return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);
    
            // 如果字段ID为MZ_ZIP64_EXTENDED_INFORMATION_FIELD_HEADER_ID，解析ZIP64扩展信息
            if (field_id == MZ_ZIP64_EXTENDED_INFORMATION_FIELD_HEADER_ID)
            {
                const mz_uint8 *pSrc_field_data = pExtra_data + sizeof(mz_uint32);
    
                // 如果字段数据大小不足以包含两个mz_uint64，表示头部无效或损坏
                if (field_data_size < sizeof(mz_uint64) * 2)
                {
                    mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);
                    goto handle_failure;
                }
    
                // 读取本地头未压缩大小和压缩大小，并标记找到了ZIP64扩展数据
                local_header_uncomp_size = MZ_READ_LE64(pSrc_field_data);
                local_header_comp_size = MZ_READ_LE64(pSrc_field_data + sizeof(mz_uint64));
    
                found_zip64_ext_data_in_ldir = MZ_TRUE;
                break;
            }
    
            // 移动指针到下一个字段，并更新剩余额外数据大小
            pExtra_data += field_total_size;
            extra_size_remaining -= field_total_size;
        } while (extra_size_remaining);
    }
    
    /* TODO: 当local_header_comp_size为0xFFFFFFFF时，解析本地头额外数据！（big_descriptor.zip） */
    /* 检查是否存在数据描述符，同时本地头部的压缩大小和 CRC32 值均为零 */
    if ((has_data_descriptor) && (!local_header_comp_size) && (!local_header_crc32))
    {
        // 分配一个大小为 32 字节的缓冲区用于存储数据描述符
        mz_uint8 descriptor_buf[32];
        // 标志位，表示是否存在数据描述符的标识符
        mz_bool has_id;
        // 源数据指针，根据是否有标识符决定其位置
        const mz_uint8 *pSrc;
        // 文件的 CRC32 校验值
        mz_uint32 file_crc32;
        // 压缩和非压缩大小的变量
        mz_uint64 comp_size = 0, uncomp_size = 0;

        // 确定数据描述符的数量，根据是否支持 ZIP64 或者在局部目录中找到 ZIP64 扩展数据
        mz_uint32 num_descriptor_uint32s = ((pState->m_zip64) || (found_zip64_ext_data_in_ldir)) ? 6 : 4;

        // 从文件中读取数据描述符，并检查是否成功
        if (pZip->m_pRead(pZip->m_pIO_opaque, local_header_ofs + MZ_ZIP_LOCAL_DIR_HEADER_SIZE + local_header_filename_len + local_header_extra_len + file_stat.m_comp_size, descriptor_buf, sizeof(mz_uint32) * num_descriptor_uint32s) != (sizeof(mz_uint32) * num_descriptor_uint32s))
        {
            // 如果读取失败，设置 ZIP 错误并跳转到错误处理部分
            mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);
            goto handle_failure;
        }

        // 检查是否存在数据描述符的标识符
        has_id = (MZ_READ_LE32(descriptor_buf) == MZ_ZIP_DATA_DESCRIPTOR_ID);
        pSrc = has_id ? (descriptor_buf + sizeof(mz_uint32)) : descriptor_buf;

        // 读取文件的 CRC32 校验值
        file_crc32 = MZ_READ_LE32(pSrc);

        // 根据 ZIP64 的情况读取压缩和非压缩大小
        if ((pState->m_zip64) || (found_zip64_ext_data_in_ldir))
        {
            comp_size = MZ_READ_LE64(pSrc + sizeof(mz_uint32));
            uncomp_size = MZ_READ_LE64(pSrc + sizeof(mz_uint32) + sizeof(mz_uint64));
        }
        else
        {
            comp_size = MZ_READ_LE32(pSrc + sizeof(mz_uint32));
            uncomp_size = MZ_READ_LE32(pSrc + sizeof(mz_uint32) + sizeof(mz_uint32));
        }

        // 检查文件的 CRC32 校验值、压缩大小和非压缩大小是否与预期值相符
        if ((file_crc32 != file_stat.m_crc32) || (comp_size != file_stat.m_comp_size) || (uncomp_size != file_stat.m_uncomp_size))
        {
            // 如果校验失败，设置 ZIP 错误并跳转到错误处理部分
            mz_zip_set_error(pZip, MZ_ZIP_VALIDATION_FAILED);
            goto handle_failure;
        }
    }
    else
    {
        // 如果不使用数据描述符，则比较本地头部的 CRC32 校验值、压缩大小和非压缩大小是否与预期值相符
        if ((local_header_crc32 != file_stat.m_crc32) || (local_header_comp_size != file_stat.m_comp_size) || (local_header_uncomp_size != file_stat.m_uncomp_size))
        {
            // 如果校验失败，设置 ZIP 错误并跳转到错误处理部分
            mz_zip_set_error(pZip, MZ_ZIP_VALIDATION_FAILED);
            goto handle_failure;
        }
    }

    // 清空文件数据数组
    mz_zip_array_clear(pZip, &file_data_array);

    // 如果不仅验证头部信息，并且需要提取文件数据
    if ((flags & MZ_ZIP_FLAG_VALIDATE_HEADERS_ONLY) == 0)
    {
        // 调用函数将文件数据提取到回调函数中，并计算非压缩数据的 CRC32 校验值
        if (!mz_zip_reader_extract_to_callback(pZip, file_index, mz_zip_compute_crc32_callback, &uncomp_crc32, 0))
            return MZ_FALSE;

        // 进行额外一次校验，确保非压缩数据的 CRC32 校验值与预期值相符
        if (uncomp_crc32 != file_stat.m_crc32)
        {
            // 如果校验失败，设置 ZIP 错误并返回假值
            mz_zip_set_error(pZip, MZ_ZIP_VALIDATION_FAILED);
            return MZ_FALSE;
        }
    }

    // 如果所有校验通过，则返回真值
    return MZ_TRUE;
handle_failure:
    // 清理文件数据数组，释放 pZip 占用的资源
    mz_zip_array_clear(pZip, &file_data_array);
    // 返回失败标志
    return MZ_FALSE;
}

// 验证 ZIP 存档的有效性
mz_bool mz_zip_validate_archive(mz_zip_archive *pZip, mz_uint flags)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_zip_internal_state *pState;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    uint32_t i;

    // 检查参数的有效性
    if ((!pZip) || (!pZip->m_pState) || (!pZip->m_pAlloc) || (!pZip->m_pFree) || (!pZip->m_pRead))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 获取 ZIP 内部状态
    pState = pZip->m_pState;

    /* 基本的合法性检查 */
    if (!pState->m_zip64)
    {
        // 对于非 ZIP64 格式，检查总文件数是否超出范围
        if (pZip->m_total_files > MZ_UINT16_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_ARCHIVE_TOO_LARGE);

        // 检查存档大小是否超出范围
        if (pZip->m_archive_size > MZ_UINT32_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_ARCHIVE_TOO_LARGE);
    }
    else
    {
        // 对于 ZIP64 格式，检查总文件数是否超出范围
        if (pZip->m_total_files >= MZ_UINT32_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_ARCHIVE_TOO_LARGE);

        // 检查中央目录大小是否超出范围
        if (pState->m_central_dir.m_size >= MZ_UINT32_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_ARCHIVE_TOO_LARGE);
    }

    // 遍历 ZIP 存档中的每个文件
    for (i = 0; i < pZip->m_total_files; i++)
    {
        // 如果标志包含验证定位文件标志，则执行以下操作
        if (MZ_ZIP_FLAG_VALIDATE_LOCATE_FILE_FLAG & flags)
        {
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            mz_uint32 found_index;
            mz_zip_archive_file_stat stat;

            // 获取当前文件的统计信息
            if (!mz_zip_reader_file_stat(pZip, i, &stat))
                return MZ_FALSE;

            // 在 ZIP 存档中查找具有给定文件名的文件
            if (!mz_zip_reader_locate_file_v2(pZip, stat.m_filename, NULL, 0, &found_index))
                return MZ_FALSE;

            /* 如果在存档中存在重复的文件名（写入时不检查，由用户负责）则可能失败 */
            if (found_index != i)
                return mz_zip_set_error(pZip, MZ_ZIP_VALIDATION_FAILED);
        }

        // 验证当前文件的有效性
        if (!mz_zip_validate_file(pZip, i, flags))
            return MZ_FALSE;
    }

    // 返回验证结果
    return MZ_TRUE;
}

// 验证内存中的 ZIP 存档的有效性
mz_bool mz_zip_validate_mem_archive(const void *pMem, size_t size, mz_uint flags, mz_zip_error *pErr)
{
    mz_bool success = MZ_TRUE;
    mz_zip_archive zip;
    mz_zip_error actual_err = MZ_ZIP_NO_ERROR;

    // 检查参数的有效性
    if ((!pMem) || (!size))
    {
        if (pErr)
            *pErr = MZ_ZIP_INVALID_PARAMETER;
        return MZ_FALSE;
    }

    // 初始化 ZIP 结构体
    mz_zip_zero_struct(&zip);

    // 从内存中初始化 ZIP 读取器
    if (!mz_zip_reader_init_mem(&zip, pMem, size, flags))
    {
        if (pErr)
            *pErr = zip.m_last_error;
        return MZ_FALSE;
    }

    // 验证 ZIP 存档的有效性
    if (!mz_zip_validate_archive(&zip, flags))
    {
        actual_err = zip.m_last_error;
        success = MZ_FALSE;
    }

    // 结束 ZIP 读取器的内部操作
    if (!mz_zip_reader_end_internal(&zip, success))
    {
        if (!actual_err)
            actual_err = zip.m_last_error;
        success = MZ_FALSE;
    }

    // 设置实际错误（如果有的话）
    if (pErr)
        *pErr = actual_err;

    // 返回验证结果
    return success;
}

#ifndef MINIZ_NO_STDIO
// 验证文件中的 ZIP 存档的有效性
mz_bool mz_zip_validate_file_archive(const char *pFilename, mz_uint flags, mz_zip_error *pErr)
{
    mz_bool success = MZ_TRUE;
    // 定义一个 mz_zip_archive 结构体实例 zip，用于处理 ZIP 文件操作
    mz_zip_archive zip;
    // 初始化实际错误为 MZ_ZIP_NO_ERROR
    mz_zip_error actual_err = MZ_ZIP_NO_ERROR;

    // 如果传入的文件名指针为空
    if (!pFilename)
    {
        // 如果错误指针不为空，则将错误设置为 MZ_ZIP_INVALID_PARAMETER
        if (pErr)
            *pErr = MZ_ZIP_INVALID_PARAMETER;
        // 返回 MZ_FALSE 表示操作失败
        return MZ_FALSE;
    }

    // 将 zip 结构体清零初始化
    mz_zip_zero_struct(&zip);

    // 使用文件名初始化 ZIP 读取器，返回值为假表示初始化失败
    if (!mz_zip_reader_init_file_v2(&zip, pFilename, flags, 0, 0))
    {
        // 如果错误指针不为空，则将错误设置为 zip 结构体的最后错误
        if (pErr)
            *pErr = zip.m_last_error;
        // 返回 MZ_FALSE 表示操作失败
        return MZ_FALSE;
    }

    // 验证 ZIP 归档的有效性，返回假表示验证失败
    if (!mz_zip_validate_archive(&zip, flags))
    {
        // 记录实际的错误
        actual_err = zip.m_last_error;
        // 设置成功标志为 MZ_FALSE
        success = MZ_FALSE;
    }

    // 结束 ZIP 读取器的内部操作，返回假表示操作失败
    if (!mz_zip_reader_end_internal(&zip, success))
    {
        // 如果实际错误未被设置，则将其设置为 zip 结构体的最后错误
        if (!actual_err)
            actual_err = zip.m_last_error;
        // 设置成功标志为 MZ_FALSE
        success = MZ_FALSE;
    }

    // 如果错误指针不为空，则将错误设置为实际错误值
    if (pErr)
        *pErr = actual_err;

    // 返回操作是否成功的标志
    return success;
    // 如果没有定义 MINIZ_NO_STDIO，则执行以下代码块
    if (pState->m_pFile)
#endif /* #ifndef MINIZ_NO_STDIO */
    {
        // 如果 ZIP 结构体的类型是文件类型
        if (pZip->m_zip_type == MZ_ZIP_TYPE_FILE)
        {
            // 关闭文件指针，如果关闭失败
            if (MZ_FCLOSE(pState->m_pFile) == EOF)
            {
                // 如果设置了错误处理标志，设置 ZIP 结构体的错误状态为文件关闭失败
                if (set_last_error)
                    mz_zip_set_error(pZip, MZ_ZIP_FILE_CLOSE_FAILED);
                // 将状态设为假，表示操作失败
                status = MZ_FALSE;
            }
        }
    
        // 将文件指针设为空，表示已经关闭
        pState->m_pFile = NULL;
    }
#ifdef /* #ifndef MINIZ_NO_STDIO */

    // 如果写函数是堆写入函数并且存在内存状态，则释放内存状态
    if ((pZip->m_pWrite == mz_zip_heap_write_func) && (pState->m_pMem))
    {
        pZip->m_pFree(pZip->m_pAlloc_opaque, pState->m_pMem);
        pState->m_pMem = NULL;
    }

    // 释放状态对象的内存并将 ZIP 模式设置为无效
    pZip->m_pFree(pZip->m_pAlloc_opaque, pState);
    pZip->m_zip_mode = MZ_ZIP_MODE_INVALID;
    return status;
}

// 初始化 ZIP 写入器（增强版）
mz_bool mz_zip_writer_init_v2(mz_zip_archive *pZip, mz_uint64 existing_size, mz_uint flags)
{
    // 检查参数的有效性
    if ((!pZip) || (pZip->m_pState) || (!pZip->m_pWrite) || (pZip->m_zip_mode != MZ_ZIP_MODE_INVALID))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 如果设置了允许读取标志，确保存在读函数
    if (flags & MZ_ZIP_FLAG_WRITE_ALLOW_READING)
    {
        if (!pZip->m_pRead)
            return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);
    }

    // 如果设置了文件偏移对齐，确保其是2的幂次方
    if (pZip->m_file_offset_alignment)
    {
        /* Ensure user specified file offset alignment is a power of 2. */
        if (pZip->m_file_offset_alignment & (pZip->m_file_offset_alignment - 1))
            return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);
    }

    // 如果分配函数为空，则使用默认分配函数
    if (!pZip->m_pAlloc)
        pZip->m_pAlloc = miniz_def_alloc_func;
    if (!pZip->m_pFree)
        pZip->m_pFree = miniz_def_free_func;
    if (!pZip->m_pRealloc)
        pZip->m_pRealloc = miniz_def_realloc_func;

    // 初始化 ZIP 归档对象的一些成员变量
    pZip->m_archive_size = existing_size;
    pZip->m_central_directory_file_ofs = 0;
    pZip->m_total_files = 0;

    // 分配内部状态对象的内存
    if (NULL == (pZip->m_pState = (mz_zip_internal_state *)pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1, sizeof(mz_zip_internal_state))))
        return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);

    // 将分配的内存清零
    memset(pZip->m_pState, 0, sizeof(mz_zip_internal_state));

    // 设置中央目录、中央目录偏移和排序后的中央目录偏移的元素大小
    MZ_ZIP_ARRAY_SET_ELEMENT_SIZE(&pZip->m_pState->m_central_dir, sizeof(mz_uint8));
    MZ_ZIP_ARRAY_SET_ELEMENT_SIZE(&pZip->m_pState->m_central_dir_offsets, sizeof(mz_uint32));
    MZ_ZIP_ARRAY_SET_ELEMENT_SIZE(&pZip->m_pState->m_sorted_central_dir_offsets, sizeof(mz_uint32));

    // 设置 ZIP 内部状态对象的一些属性
    pZip->m_pState->m_zip64 = zip64;
    pZip->m_pState->m_zip64_has_extended_info_fields = zip64;

    // 设置 ZIP 类型和模式为用户定义的写入模式
    pZip->m_zip_type = MZ_ZIP_TYPE_USER;
    pZip->m_zip_mode = MZ_ZIP_MODE_WRITING;

    return MZ_TRUE;
}

// 初始化 ZIP 写入器
mz_bool mz_zip_writer_init(mz_zip_archive *pZip, mz_uint64 existing_size)
{
    return mz_zip_writer_init_v2(pZip, existing_size, 0);
}

// 初始化堆内存中的 ZIP 写入器（增强版）
mz_bool mz_zip_writer_init_heap_v2(mz_zip_archive *pZip, size_t size_to_reserve_at_beginning, size_t initial_allocation_size, mz_uint flags)
{
    // 设置写入函数为堆写入函数
    pZip->m_pWrite = mz_zip_heap_write_func;
    pZip->m_pNeeds_keepalive = NULL;

    // 如果设置了允许读取标志，则设置读函数为内存读函数
    if (flags & MZ_ZIP_FLAG_WRITE_ALLOW_READING)
        pZip->m_pRead = mz_zip_mem_read_func;

    // 设置 IO 透明指针为 ZIP 归档对象本身
    pZip->m_pIO_opaque = pZip;

    // 初始化 ZIP 写入器（增强版），如果失败则返回假
    if (!mz_zip_writer_init_v2(pZip, size_to_reserve_at_beginning, flags))
        return MZ_FALSE;

    // 设置 ZIP 类型为堆内存类型
    pZip->m_zip_type = MZ_ZIP_TYPE_HEAP;

    // 确保初始分配大小不小于预留的开始大小
    if (0 != (initial_allocation_size = MZ_MAX(initial_allocation_size, size_to_reserve_at_beginning)))
    {
        // 检查指向 ZIP 结构体的指针是否为 NULL，如果为 NULL，说明内存分配失败
        if (NULL == (pZip->m_pState->m_pMem = pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1, initial_allocation_size)))
        {
            // 内存分配失败时，结束 ZIP 写入操作，并返回内存分配失败的错误码
            mz_zip_writer_end_internal(pZip, MZ_FALSE);
            return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
        }
        // 设置 ZIP 内部状态的内存容量为初始分配大小
        pZip->m_pState->m_mem_capacity = initial_allocation_size;
    }
    
    // 返回指示内存分配成功的标志 MZ_TRUE
    return MZ_TRUE;
}

// 初始化一个在堆上分配空间的压缩文件写入器
mz_bool mz_zip_writer_init_heap(mz_zip_archive *pZip, size_t size_to_reserve_at_beginning, size_t initial_allocation_size)
{
    // 调用带有额外参数的版本初始化压缩文件写入器
    return mz_zip_writer_init_heap_v2(pZip, size_to_reserve_at_beginning, initial_allocation_size, 0);
}

#ifndef MINIZ_NO_STDIO
// 压缩文件写入函数，用于文件的写入操作
static size_t mz_zip_file_write_func(void *pOpaque, mz_uint64 file_ofs, const void *pBuf, size_t n)
{
    // 将不透明指针转换为压缩文件对象
    mz_zip_archive *pZip = (mz_zip_archive *)pOpaque;
    // 获取当前文件偏移量
    mz_int64 cur_ofs = MZ_FTELL64(pZip->m_pState->m_pFile);

    // 根据压缩文件起始偏移量调整文件偏移量
    file_ofs += pZip->m_pState->m_file_archive_start_ofs;

    // 检查文件偏移量是否小于零，或者当前偏移量与目标偏移量不同并且无法定位到目标位置
    if (((mz_int64)file_ofs < 0) || (((cur_ofs != (mz_int64)file_ofs)) && (MZ_FSEEK64(pZip->m_pState->m_pFile, (mz_int64)file_ofs, SEEK_SET))))
    {
        // 设置压缩操作错误状态为文件定位失败
        mz_zip_set_error(pZip, MZ_ZIP_FILE_SEEK_FAILED);
        return 0;
    }

    // 写入数据到文件
    return MZ_FWRITE(pBuf, 1, n, pZip->m_pState->m_pFile);
}

// 初始化一个将数据写入文件的压缩文件写入器
mz_bool mz_zip_writer_init_file(mz_zip_archive *pZip, const char *pFilename, mz_uint64 size_to_reserve_at_beginning)
{
    // 调用带有额外参数的版本初始化文件压缩写入器
    return mz_zip_writer_init_file_v2(pZip, pFilename, size_to_reserve_at_beginning, 0);
}

// 初始化一个带文件名参数的压缩文件写入器
mz_bool mz_zip_writer_init_file_v2(mz_zip_archive *pZip, const char *pFilename, mz_uint64 size_to_reserve_at_beginning, mz_uint flags)
{
    MZ_FILE *pFile;

    // 设置文件写入函数为 mz_zip_file_write_func
    pZip->m_pWrite = mz_zip_file_write_func;
    // 不需要保持文件句柄的存活状态
    pZip->m_pNeeds_keepalive = NULL;

    // 如果允许读取标志被设置，则设置读取函数为 mz_zip_file_read_func
    if (flags & MZ_ZIP_FLAG_WRITE_ALLOW_READING)
        pZip->m_pRead = mz_zip_file_read_func;

    // 设置文件 I/O 不透明指针
    pZip->m_pIO_opaque = pZip;

    // 初始化压缩写入器的 V2 版本，如果失败则返回假
    if (!mz_zip_writer_init_v2(pZip, size_to_reserve_at_beginning, flags))
        return MZ_FALSE;

    // 尝试打开文件进行写入操作
    if (NULL == (pFile = MZ_FOPEN(pFilename, (flags & MZ_ZIP_FLAG_WRITE_ALLOW_READING) ? "w+b" : "wb")))
    {
        // 结束压缩操作，并设置文件打开失败的错误状态
        mz_zip_writer_end(pZip);
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_OPEN_FAILED);
    }

    // 设置当前压缩状态的文件句柄
    pZip->m_pState->m_pFile = pFile;
    // 设置压缩类型为文件
    pZip->m_zip_type = MZ_ZIP_TYPE_FILE;

    // 如果有预留空间需求，则在文件开头写入空数据块
    if (size_to_reserve_at_beginning)
    {
        mz_uint64 cur_ofs = 0;
        char buf[4096];

        // 清空缓冲区
        MZ_CLEAR_OBJ(buf);

        // 循环写入空数据块，直至满足预留空间需求
        do
        {
            size_t n = (size_t)MZ_MIN(sizeof(buf), size_to_reserve_at_beginning);
            if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_ofs, buf, n) != n)
            {
                // 如果写入失败则结束压缩操作，并设置写入失败的错误状态
                mz_zip_writer_end(pZip);
                return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);
            }
            cur_ofs += n;
            size_to_reserve_at_beginning -= n;
        } while (size_to_reserve_at_beginning);
    }

    return MZ_TRUE;
}

// 初始化一个 C 文件句柄的压缩文件写入器
mz_bool mz_zip_writer_init_cfile(mz_zip_archive *pZip, MZ_FILE *pFile, mz_uint flags)
{
    // 设置文件写入函数为 mz_zip_file_write_func
    pZip->m_pWrite = mz_zip_file_write_func;
    // 不需要保持文件句柄的存活状态
    pZip->m_pNeeds_keepalive = NULL;

    // 如果允许读取标志被设置，则设置读取函数为 mz_zip_file_read_func
    if (flags & MZ_ZIP_FLAG_WRITE_ALLOW_READING)
        pZip->m_pRead = mz_zip_file_read_func;

    // 设置文件 I/O 不透明指针
    pZip->m_pIO_opaque = pZip;

    // 初始化压缩写入器的 V2 版本，如果失败则返回假
    if (!mz_zip_writer_init_v2(pZip, 0, flags))
        return MZ_FALSE;

    // 设置当前压缩状态的文件句柄
    pZip->m_pState->m_pFile = pFile;
    // 获取当前文件句柄的起始偏移量
    pZip->m_pState->m_file_archive_start_ofs = MZ_FTELL64(pZip->m_pState->m_pFile);
    // 设置压缩类型为 C 文件
    pZip->m_zip_type = MZ_ZIP_TYPE_CFILE;

    return MZ_TRUE;
}
#endif /* #ifndef MINIZ_NO_STDIO */
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    // 声明指向 mz_zip_internal_state 结构体的指针 pState
    mz_zip_internal_state *pState;

    // 检查参数是否有效，以及当前 ZIP 归档是否处于读取模式
    if ((!pZip) || (!pZip->m_pState) || (pZip->m_zip_mode != MZ_ZIP_MODE_READING))
        // 如果存在无效参数或 ZIP 不处于读取模式，则设置错误并返回
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 如果标志包含 MZ_ZIP_FLAG_WRITE_ZIP64
    if (flags & MZ_ZIP_FLAG_WRITE_ZIP64)
    {
        /* We don't support converting a non-zip64 file to zip64 - this seems like more trouble than it's worth. (What about the existing 32-bit data descriptors that could follow the compressed data?) */
        // 如果不支持将非 zip64 文件转换为 zip64，返回错误
        if (!pZip->m_pState->m_zip64)
            return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);
    }

    /* No sense in trying to write to an archive that's already at the support max size */
    // 如果 ZIP 归档已经达到支持的最大大小
    if (pZip->m_pState->m_zip64)
    {
        // 如果 ZIP 是 zip64 格式，并且文件数目已达到 UINT32 最大值，则返回错误
        if (pZip->m_total_files == MZ_UINT32_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_TOO_MANY_FILES);
    }
    else
    {
        // 如果 ZIP 不是 zip64 格式
        // 如果文件数目已达到 UINT16 最大值，则返回错误
        if (pZip->m_total_files == MZ_UINT16_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_TOO_MANY_FILES);

        // 如果添加新文件后的 ZIP 归档大小超过 UINT32 最大值，则返回错误
        if ((pZip->m_archive_size + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + MZ_ZIP_LOCAL_DIR_HEADER_SIZE) > MZ_UINT32_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_TOO_LARGE);
    }

    // 将当前 ZIP 归档状态保存到 pState 中
    pState = pZip->m_pState;

    // 如果当前文件指针不为空
    if (pState->m_pFile)
    {
#ifdef MINIZ_NO_STDIO
        (void)pFilename;
        // 如果未定义标准 IO，则设置错误并返回
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);
#else
        // 如果 IO 不透明指针不等于当前 ZIP 归档指针，设置错误并返回
        if (pZip->m_pIO_opaque != pZip)
            return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

        // 如果 ZIP 类型为文件类型
        if (pZip->m_zip_type == MZ_ZIP_TYPE_FILE)
        {
            // 如果文件名为空，设置错误并返回
            if (!pFilename)
                return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

            /* Archive is being read from stdio and was originally opened only for reading. Try to reopen as writable. */
            // 如果归档从标准输入输出读取，并且原始打开方式只读，尝试以读写方式重新打开
            if (NULL == (pState->m_pFile = MZ_FREOPEN(pFilename, "r+b", pState->m_pFile)))
            {
                /* The mz_zip_archive is now in a bogus state because pState->m_pFile is NULL, so just close it. */
                // 因为 pState->m_pFile 为空，当前 mz_zip_archive 处于错误状态，关闭归档并返回错误
                mz_zip_reader_end_internal(pZip, MZ_FALSE);
                return mz_zip_set_error(pZip, MZ_ZIP_FILE_OPEN_FAILED);
            }
        }

        // 设置 ZIP 归档的写入函数为 mz_zip_file_write_func
        pZip->m_pWrite = mz_zip_file_write_func;
        // 不需要保持存活状态，设置为 NULL
        pZip->m_pNeeds_keepalive = NULL;
#endif /* #ifdef MINIZ_NO_STDIO */
    }
    // 如果当前内存指针不为空
    else if (pState->m_pMem)
    {
        /* Archive lives in a memory block. Assume it's from the heap that we can resize using the realloc callback. */
        // 如果 ZIP 归档存储在内存块中，假设是可以使用 realloc 回调函数调整大小的堆内存

        // 如果 IO 不透明指针不等于当前 ZIP 归档指针，设置错误并返回
        if (pZip->m_pIO_opaque != pZip)
            return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

        // 设置 ZIP 归档的写入函数为 mz_zip_heap_write_func
        pZip->m_pWrite = mz_zip_heap_write_func;
        // 不需要保持存活状态，设置为 NULL
        pZip->m_pNeeds_keepalive = NULL;
    }
    // 如果 ZIP 归档的写入函数为空
    /* Archive is being read via a user provided read function - make sure the user has specified a write function too. */
    else if (!pZip->m_pWrite)
        // 如果归档是通过用户提供的读函数读取的，确保用户也指定了写函数，否则设置错误并返回
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);
    /* 开始在存档的当前中央目录位置写入新文件。 */
    /* TODO: 我们可以添加一个标志，让用户可以在现有中央目录之后立即开始写入，这样会更安全。 */
    pZip->m_archive_size = pZip->m_central_directory_file_ofs;
    pZip->m_central_directory_file_ofs = 0;

    /* 清除排序后的中央目录偏移量，它们现在无用且不再维护。 */
    /* 即使我们现在处于写入模式，文件仍然可以被提取和验证，但文件定位会变慢。 */
    /* TODO: 我们可以轻松地维护排序后的中央目录偏移量。 */
    mz_zip_array_clear(pZip, &pZip->m_pState->m_sorted_central_dir_offsets);

    pZip->m_zip_mode = MZ_ZIP_MODE_WRITING;

    return MZ_TRUE;
}

# 初始化一个 ZIP 写入器，从已存在的 ZIP 读取器中初始化，使用默认选项
mz_bool mz_zip_writer_init_from_reader(mz_zip_archive *pZip, const char *pFilename)
{
    return mz_zip_writer_init_from_reader_v2(pZip, pFilename, 0);
}

# 向 ZIP 文件中添加内存中的数据，使用指定的压缩级别和标志
# 注意：这里的参数 pArchive_name 命名不佳
mz_bool mz_zip_writer_add_mem(mz_zip_archive *pZip, const char *pArchive_name, const void *pBuf, size_t buf_size, mz_uint level_and_flags)
{
    return mz_zip_writer_add_mem_ex(pZip, pArchive_name, pBuf, buf_size, NULL, 0, level_and_flags, 0, 0);
}

# 定义 ZIP 写入状态的结构体
typedef struct
{
    mz_zip_archive *m_pZip;
    mz_uint64 m_cur_archive_file_ofs;
    mz_uint64 m_comp_size;
} mz_zip_writer_add_state;

# ZIP 写入回调函数，将数据写入到 ZIP 文件中
static mz_bool mz_zip_writer_add_put_buf_callback(const void *pBuf, int len, void *pUser)
{
    mz_zip_writer_add_state *pState = (mz_zip_writer_add_state *)pUser;
    if ((int)pState->m_pZip->m_pWrite(pState->m_pZip->m_pIO_opaque, pState->m_cur_archive_file_ofs, pBuf, len) != len)
        return MZ_FALSE;

    pState->m_cur_archive_file_ofs += len;
    pState->m_comp_size += len;
    return MZ_TRUE;
}

# 定义 ZIP64 扩展字段的最大大小常量
#define MZ_ZIP64_MAX_LOCAL_EXTRA_FIELD_SIZE (sizeof(mz_uint16) * 2 + sizeof(mz_uint64) * 2)
#define MZ_ZIP64_MAX_CENTRAL_EXTRA_FIELD_SIZE (sizeof(mz_uint16) * 2 + sizeof(mz_uint64) * 3)

# 创建 ZIP64 扩展数据的函数，返回实际写入的字节数
static mz_uint32 mz_zip_writer_create_zip64_extra_data(mz_uint8 *pBuf, mz_uint64 *pUncomp_size, mz_uint64 *pComp_size, mz_uint64 *pLocal_header_ofs)
{
    mz_uint8 *pDst = pBuf;
    mz_uint32 field_size = 0;

    MZ_WRITE_LE16(pDst + 0, MZ_ZIP64_EXTENDED_INFORMATION_FIELD_HEADER_ID);
    MZ_WRITE_LE16(pDst + 2, 0);
    pDst += sizeof(mz_uint16) * 2;

    if (pUncomp_size)
    {
        MZ_WRITE_LE64(pDst, *pUncomp_size);
        pDst += sizeof(mz_uint64);
        field_size += sizeof(mz_uint64);
    }

    if (pComp_size)
    {
        MZ_WRITE_LE64(pDst, *pComp_size);
        pDst += sizeof(mz_uint64);
        field_size += sizeof(mz_uint64);
    }

    if (pLocal_header_ofs)
    {
        MZ_WRITE_LE64(pDst, *pLocal_header_ofs);
        pDst += sizeof(mz_uint64);
        field_size += sizeof(mz_uint64);
    }

    MZ_WRITE_LE16(pBuf + 2, field_size);

    return (mz_uint32)(pDst - pBuf);
}

# 创建本地目录头部的函数，填充 ZIP 文件的本地文件头部信息
static mz_bool mz_zip_writer_create_local_dir_header(mz_zip_archive *pZip, mz_uint8 *pDst, mz_uint16 filename_size, mz_uint16 extra_size, mz_uint64 uncomp_size, mz_uint64 comp_size, mz_uint32 uncomp_crc32, mz_uint16 method, mz_uint16 bit_flags, mz_uint16 dos_time, mz_uint16 dos_date)
{
    (void)pZip;
    memset(pDst, 0, MZ_ZIP_LOCAL_DIR_HEADER_SIZE);
    MZ_WRITE_LE32(pDst + MZ_ZIP_LDH_SIG_OFS, MZ_ZIP_LOCAL_DIR_HEADER_SIG);
    MZ_WRITE_LE16(pDst + MZ_ZIP_LDH_VERSION_NEEDED_OFS, method ? 20 : 0);
    MZ_WRITE_LE16(pDst + MZ_ZIP_LDH_BIT_FLAG_OFS, bit_flags);
    MZ_WRITE_LE16(pDst + MZ_ZIP_LDH_METHOD_OFS, method);
    MZ_WRITE_LE16(pDst + MZ_ZIP_LDH_FILE_TIME_OFS, dos_time);
    MZ_WRITE_LE16(pDst + MZ_ZIP_LDH_FILE_DATE_OFS, dos_date);
    MZ_WRITE_LE32(pDst + MZ_ZIP_LDH_CRC32_OFS, uncomp_crc32);
    MZ_WRITE_LE32(pDst + MZ_ZIP_LDH_COMPRESSED_SIZE_OFS, MZ_MIN(comp_size, MZ_UINT32_MAX));
    # 在目标地址上写入已解压大小（以小端格式），限制在不超过32位无符号整数的范围内
    MZ_WRITE_LE32(pDst + MZ_ZIP_LDH_DECOMPRESSED_SIZE_OFS, MZ_MIN(uncomp_size, MZ_UINT32_MAX));
    # 在目标地址上写入文件名长度（以小端格式）
    MZ_WRITE_LE16(pDst + MZ_ZIP_LDH_FILENAME_LEN_OFS, filename_size);
    # 在目标地址上写入额外字段长度（以小端格式）
    MZ_WRITE_LE16(pDst + MZ_ZIP_LDH_EXTRA_LEN_OFS, extra_size);
    # 返回真值，表示操作成功完成
    return MZ_TRUE;
static mz_bool mz_zip_writer_create_central_dir_header(mz_zip_archive *pZip, mz_uint8 *pDst,
                                                       mz_uint16 filename_size, mz_uint16 extra_size, mz_uint16 comment_size,
                                                       mz_uint64 uncomp_size, mz_uint64 comp_size, mz_uint32 uncomp_crc32,
                                                       mz_uint16 method, mz_uint16 bit_flags, mz_uint16 dos_time, mz_uint16 dos_date,
                                                       mz_uint64 local_header_ofs, mz_uint32 ext_attributes)
{
    // 忽略 pZip 参数（未使用）
    (void)pZip;
    // 用零填充中央目录头部数据块
    memset(pDst, 0, MZ_ZIP_CENTRAL_DIR_HEADER_SIZE);
    // 写入中央目录头部的签名（4字节）
    MZ_WRITE_LE32(pDst + MZ_ZIP_CDH_SIG_OFS, MZ_ZIP_CENTRAL_DIR_HEADER_SIG);
    // 写入版本需要的值（2字节），根据 method 是否为非零决定
    MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_VERSION_NEEDED_OFS, method ? 20 : 0);
    // 写入位标志（2字节）
    MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_BIT_FLAG_OFS, bit_flags);
    // 写入压缩方法（2字节）
    MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_METHOD_OFS, method);
    // 写入文件时间（2字节）
    MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_FILE_TIME_OFS, dos_time);
    // 写入文件日期（2字节）
    MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_FILE_DATE_OFS, dos_date);
    // 写入未压缩数据的 CRC32 校验值（4字节）
    MZ_WRITE_LE32(pDst + MZ_ZIP_CDH_CRC32_OFS, uncomp_crc32);
    // 写入压缩后数据大小（4字节），但不超过 MZ_UINT32_MAX
    MZ_WRITE_LE32(pDst + MZ_ZIP_CDH_COMPRESSED_SIZE_OFS, MZ_MIN(comp_size, MZ_UINT32_MAX));
    // 写入未压缩数据大小（4字节），但不超过 MZ_UINT32_MAX
    MZ_WRITE_LE32(pDst + MZ_ZIP_CDH_DECOMPRESSED_SIZE_OFS, MZ_MIN(uncomp_size, MZ_UINT32_MAX));
    // 写入文件名长度（2字节）
    MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_FILENAME_LEN_OFS, filename_size);
    // 写入额外数据长度（2字节）
    MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_EXTRA_LEN_OFS, extra_size);
    // 写入注释长度（2字节）
    MZ_WRITE_LE16(pDst + MZ_ZIP_CDH_COMMENT_LEN_OFS, comment_size);
    // 写入外部文件属性（4字节）
    MZ_WRITE_LE32(pDst + MZ_ZIP_CDH_EXTERNAL_ATTR_OFS, ext_attributes);
    // 写入本地文件头部的偏移量（4字节），但不超过 MZ_UINT32_MAX
    MZ_WRITE_LE32(pDst + MZ_ZIP_CDH_LOCAL_HEADER_OFS, MZ_MIN(local_header_ofs, MZ_UINT32_MAX));
    // 返回操作成功的标志
    return MZ_TRUE;
}

static mz_bool mz_zip_writer_add_to_central_dir(mz_zip_archive *pZip, const char *pFilename, mz_uint16 filename_size,
                                                const void *pExtra, mz_uint16 extra_size, const void *pComment, mz_uint16 comment_size,
                                                mz_uint64 uncomp_size, mz_uint64 comp_size, mz_uint32 uncomp_crc32,
                                                mz_uint16 method, mz_uint16 bit_flags, mz_uint16 dos_time, mz_uint16 dos_date,
                                                mz_uint64 local_header_ofs, mz_uint32 ext_attributes,
                                                const char *user_extra_data, mz_uint user_extra_data_len)
{
    // 获取 ZIP 内部状态
    mz_zip_internal_state *pState = pZip->m_pState;
    // 获取当前中央目录的偏移量
    mz_uint32 central_dir_ofs = (mz_uint32)pState->m_central_dir.m_size;
    // 记录原始中央目录大小
    size_t orig_central_dir_size = pState->m_central_dir.m_size;
    // 创建中央目录头部数据块
    mz_uint8 central_dir_header[MZ_ZIP_CENTRAL_DIR_HEADER_SIZE];

    // 如果不支持 ZIP64 格式，并且本地头部的偏移量超过了 0xFFFFFFFF
    if (!pZip->m_pState->m_zip64)
    {
        if (local_header_ofs > 0xFFFFFFFF)
            // 返回文件过大的错误
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_TOO_LARGE);
    }

    // miniz 目前不支持中央目录大小超过 MZ_UINT32_MAX 字节
    // 检查如果向中央目录添加当前文件的所有信息会导致中央目录大小超过32位整数的最大值，则返回不支持中央目录大小错误
    if (((mz_uint64)pState->m_central_dir.m_size + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + filename_size + extra_size + user_extra_data_len + comment_size) >= MZ_UINT32_MAX)
        return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_CDIR_SIZE);

    // 创建中央目录头部，包括文件名大小、额外数据大小、用户额外数据长度、注释大小等信息，并写入到中央目录头部数组中
    if (!mz_zip_writer_create_central_dir_header(pZip, central_dir_header, filename_size, (mz_uint16)(extra_size + user_extra_data_len), comment_size, uncomp_size, comp_size, uncomp_crc32, method, bit_flags, dos_time, dos_date, local_header_ofs, ext_attributes))
        return mz_zip_set_error(pZip, MZ_ZIP_INTERNAL_ERROR);

    // 将当前文件的所有信息（中央目录头部、文件名、额外数据、用户额外数据、注释、中央目录偏移量）依次加入到中央目录和中央目录偏移量数组中
    if ((!mz_zip_array_push_back(pZip, &pState->m_central_dir, central_dir_header, MZ_ZIP_CENTRAL_DIR_HEADER_SIZE)) ||
        (!mz_zip_array_push_back(pZip, &pState->m_central_dir, pFilename, filename_size)) ||
        (!mz_zip_array_push_back(pZip, &pState->m_central_dir, pExtra, extra_size)) ||
        (!mz_zip_array_push_back(pZip, &pState->m_central_dir, user_extra_data, user_extra_data_len)) ||
        (!mz_zip_array_push_back(pZip, &pState->m_central_dir, pComment, comment_size)) ||
        (!mz_zip_array_push_back(pZip, &pState->m_central_dir_offsets, &central_dir_ofs, 1)))
    {
        /* 尝试将中央目录数组调整回原始大小 */
        mz_zip_array_resize(pZip, &pState->m_central_dir, orig_central_dir_size, MZ_FALSE);
        // 返回内存分配失败错误
        return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
    }

    // 操作成功，返回真值表示成功添加当前文件的信息到中央目录
    return MZ_TRUE;
}

static mz_bool mz_zip_writer_validate_archive_name(const char *pArchive_name)
{
    /* 基本的 ZIP 存档文件名有效性检查：有效的文件名不能以斜杠开头，不能包含驱动器号，也不能使用 DOS 风格的反斜杠。 */
    if (*pArchive_name == '/')
        return MZ_FALSE;

    /* 确保名称不包含驱动器号或 DOS 风格的反斜杠是使用 miniz 的程序的责任 */

    return MZ_TRUE;
}

static mz_uint mz_zip_writer_compute_padding_needed_for_file_alignment(mz_zip_archive *pZip)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint32 n;
    if (!pZip->m_file_offset_alignment)
        return 0;
    n = (mz_uint32)(pZip->m_archive_size & (pZip->m_file_offset_alignment - 1));
    return (mz_uint)((pZip->m_file_offset_alignment - n) & (pZip->m_file_offset_alignment - 1));
}

static mz_bool mz_zip_writer_write_zeros(mz_zip_archive *pZip, mz_uint64 cur_file_ofs, mz_uint32 n)
{
    char buf[4096];
    memset(buf, 0, MZ_MIN(sizeof(buf), n));
    while (n)
    {
        mz_uint32 s = MZ_MIN(sizeof(buf), n);
        if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_file_ofs, buf, s) != s)
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);

        cur_file_ofs += s;
        n -= s;
    }
    return MZ_TRUE;
}

mz_bool mz_zip_writer_add_mem_ex(mz_zip_archive *pZip, const char *pArchive_name, const void *pBuf, size_t buf_size, const void *pComment, mz_uint16 comment_size, mz_uint level_and_flags,
                                 mz_uint64 uncomp_size, mz_uint32 uncomp_crc32)
{
    return mz_zip_writer_add_mem_ex_v2(pZip, pArchive_name, pBuf, buf_size, pComment, comment_size, level_and_flags, uncomp_size, uncomp_crc32, NULL, NULL, 0, NULL, 0);
}

mz_bool mz_zip_writer_add_mem_ex_v2(mz_zip_archive *pZip, const char *pArchive_name, const void *pBuf, size_t buf_size, const void *pComment, mz_uint16 comment_size,
                                    mz_uint level_and_flags, mz_uint64 uncomp_size, mz_uint32 uncomp_crc32, MZ_TIME_T *last_modified,
                                    const char *user_extra_data, mz_uint user_extra_data_len, const char *user_extra_data_central, mz_uint user_extra_data_central_len)
{
    mz_uint16 method = 0, dos_time = 0, dos_date = 0;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint level, ext_attributes = 0, num_alignment_padding_bytes;
    mz_uint64 local_dir_header_ofs = pZip->m_archive_size, cur_archive_file_ofs = pZip->m_archive_size, comp_size = 0;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t archive_name_size;
    mz_uint8 local_dir_header[MZ_ZIP_LOCAL_DIR_HEADER_SIZE];
    tdefl_compressor *pComp = NULL;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_bool store_data_uncompressed;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_zip_internal_state *pState;
    mz_uint8 *pExtra_data = NULL;
    mz_uint32 extra_size = 0;
    // 定义一个数组，用于存储 ZIP64 最大中央扩展字段大小的无符号 8 位整数
    mz_uint8 extra_data[MZ_ZIP64_MAX_CENTRAL_EXTRA_FIELD_SIZE];
    // 定义一个 16 位无符号整数，用于存储位标志（bit flags）
    mz_uint16 bit_flags = 0;
    // 根据缓冲区大小和是否有缓冲区内容来确定是否仅写入元数据
    mz_bool write_metadata_only = buf_size && !pBuf;

    // 如果传入的压缩等级和标志小于 0，则将其设置为默认压缩等级
    if ((int)level_and_flags < 0)
        level_and_flags = MZ_DEFAULT_LEVEL;

    // 如果未压缩大小不为零，或者有缓冲区且不使用压缩数据标志，则设置位标志表明具有定位器
    if (uncomp_size || (buf_size && !(level_and_flags & MZ_ZIP_FLAG_COMPRESSED_DATA)))
        bit_flags |= MZ_ZIP_LDH_BIT_FLAG_HAS_LOCATOR;

    // 如果不使用 ASCII 文件名，则设置通用目的位标志表明使用 UTF-8 编码文件名
    if (!(level_and_flags & MZ_ZIP_FLAG_ASCII_FILENAME))
        bit_flags |= MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_UTF8;

    // 提取压缩等级（低四位）并检查是否存储未压缩数据
    level = level_and_flags & 0xF;
    store_data_uncompressed = ((!level) || (level_and_flags & MZ_ZIP_FLAG_COMPRESSED_DATA));

    // 检查 ZIP 写入器和其状态是否存在，以及是否处于写入模式，并检查其他参数的有效性
    if ((!pZip) || (!pZip->m_pState) || (pZip->m_zip_mode != MZ_ZIP_MODE_WRITING) || (!pArchive_name) || ((comment_size) && (!pComment)) || (level > MZ_UBER_COMPRESSION))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 获取 ZIP 写入器的状态
    pState = pZip->m_pState;

    // 如果支持 ZIP64，并且总文件数达到 MZ_UINT32_MAX，返回文件过多错误
    if (pState->m_zip64)
    {
        if (pZip->m_total_files == MZ_UINT32_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_TOO_MANY_FILES);
    }
    else
    {
        // 如果不支持 ZIP64，并且总文件数达到 MZ_UINT16_MAX，则启用 ZIP64 并返回文件过多错误
        if (pZip->m_total_files == MZ_UINT16_MAX)
        {
            pState->m_zip64 = MZ_TRUE;
            /*return mz_zip_set_error(pZip, MZ_ZIP_TOO_MANY_FILES); */
        }
        // 如果缓冲区大小或未压缩大小超过 0xFFFFFFFF，启用 ZIP64 并返回存档过大错误
        if ((buf_size > 0xFFFFFFFF) || (uncomp_size > 0xFFFFFFFF))
        {
            pState->m_zip64 = MZ_TRUE;
            /*return mz_zip_set_error(pZip, MZ_ZIP_ARCHIVE_TOO_LARGE); */
        }
    }

    // 如果不使用压缩数据标志且存在未压缩大小，返回无效参数错误
    if ((!(level_and_flags & MZ_ZIP_FLAG_COMPRESSED_DATA)) && (uncomp_size))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 验证归档文件名的有效性，如果无效则返回无效文件名错误
    if (!mz_zip_writer_validate_archive_name(pArchive_name))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_FILENAME);
#ifndef MINIZ_NO_TIME
    // 如果 last_modified 不为 NULL，则将其转换为 DOS 时间格式
    if (last_modified != NULL)
    {
        mz_zip_time_t_to_dos_time(*last_modified, &dos_time, &dos_date);
    }
    else
    {
        // 否则获取当前时间，并将其转换为 DOS 时间格式
        MZ_TIME_T cur_time;
        time(&cur_time);
        mz_zip_time_t_to_dos_time(cur_time, &dos_time, &dos_date);
    }
#endif /* #ifndef MINIZ_NO_TIME */

    // 如果不需要压缩数据
    if (!(level_and_flags & MZ_ZIP_FLAG_COMPRESSED_DATA))
    {
        // 如果不仅写入元数据，则计算未压缩数据的 CRC32 校验和
        if (!write_metadata_only) {
            uncomp_crc32 = (mz_uint32)mz_crc32(MZ_CRC32_INIT, (const mz_uint8 *)pBuf, buf_size);
        }
        // 设置未压缩数据的大小
        uncomp_size = buf_size;
        // 如果未压缩数据大小小于等于3，则选择不压缩存储
        if (uncomp_size <= 3)
        {
            level = 0;
            store_data_uncompressed = MZ_TRUE;
        }
    }

    // 计算归档文件名的长度
    archive_name_size = strlen(pArchive_name);
    // 如果归档文件名长度超过 MZ_UINT16_MAX，则返回无效文件名错误
    if (archive_name_size > MZ_UINT16_MAX)
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_FILENAME);

    // 计算需要的字节对齐填充字节数
    num_alignment_padding_bytes = mz_zip_writer_compute_padding_needed_for_file_alignment(pZip);

    /* miniz 目前不支持中央目录 >= MZ_UINT32_MAX 字节 */
    // 如果中央目录加上各种头信息超过 MZ_UINT32_MAX，则返回不支持的中央目录大小错误
    if (((mz_uint64)pState->m_central_dir.m_size + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + archive_name_size + MZ_ZIP64_MAX_CENTRAL_EXTRA_FIELD_SIZE + comment_size) >= MZ_UINT32_MAX)
        return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_CDIR_SIZE);

    // 如果不使用 ZIP64 格式
    if (!pState->m_zip64)
    {
        /* 如果归档文件显然太大，则提前退出 */
        if ((pZip->m_archive_size + num_alignment_padding_bytes + MZ_ZIP_LOCAL_DIR_HEADER_SIZE + archive_name_size
            + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + archive_name_size + comment_size + user_extra_data_len +
            pState->m_central_dir.m_size + MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE + user_extra_data_central_len
            + MZ_ZIP_DATA_DESCRIPTER_SIZE32) > 0xFFFFFFFF)
        {
            // 标记为 ZIP64 格式，虽然当前代码段注释了可能会过大的错误
            pState->m_zip64 = MZ_TRUE;
            /*return mz_zip_set_error(pZip, MZ_ZIP_ARCHIVE_TOO_LARGE); */
        }
    }

    // 如果归档文件名不为空且最后一个字符是 '/'，表示是一个 DOS 子目录
    if ((archive_name_size) && (pArchive_name[archive_name_size - 1] == '/'))
    {
        // 设置 DOS 子目录属性位
        ext_attributes |= MZ_ZIP_DOS_DIR_ATTRIBUTE_BITFLAG;

        // 如果子目录含有数据，则返回无效参数错误
        if ((buf_size) || (uncomp_size))
            return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);
    }

    /* 尝试在写入归档文件之前进行任何分配，以便在分配失败时文件保持未修改状态（如果正在进行原地修改，则是一个好主意）。 */
    // 如果无法确保中央目录和中央目录偏移的空间，则返回分配失败错误
    if ((!mz_zip_array_ensure_room(pZip, &pState->m_central_dir, MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + archive_name_size + comment_size + (pState->m_zip64 ? MZ_ZIP64_MAX_CENTRAL_EXTRA_FIELD_SIZE : 0))) || (!mz_zip_array_ensure_room(pZip, &pState->m_central_dir_offsets, 1)))
        return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);

    // 如果不是存储未压缩数据且 buf_size 不为 0
    if ((!store_data_uncompressed) && (buf_size))
    {
        // 分配内存用于压缩器
        if (NULL == (pComp = (tdefl_compressor *)pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1, sizeof(tdefl_compressor))))
            return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
    }


    // 关闭 if 语句块的结束标志，准备进入下一段代码逻辑

    if (!mz_zip_writer_write_zeros(pZip, cur_archive_file_ofs, num_alignment_padding_bytes))
    {
        // 如果写入指定数量的零字节失败，则释放资源并返回失败标志
        pZip->m_pFree(pZip->m_pAlloc_opaque, pComp);
        return MZ_FALSE;
    }

    // 更新本地目录头的偏移量，并根据文件偏移量对齐要求进行断言验证
    local_dir_header_ofs += num_alignment_padding_bytes;
    if (pZip->m_file_offset_alignment)
    {
        MZ_ASSERT((local_dir_header_ofs & (pZip->m_file_offset_alignment - 1)) == 0);
    }
    cur_archive_file_ofs += num_alignment_padding_bytes;

    // 清空本地目录头对象的内容
    MZ_CLEAR_OBJ(local_dir_header);

    // 如果存储数据为非压缩或设置了压缩标志，则使用 DEFLATED 方法压缩数据
    if (!store_data_uncompressed || (level_and_flags & MZ_ZIP_FLAG_COMPRESSED_DATA))
    {
        method = MZ_DEFLATED;
    }

    // 如果启用 ZIP64 扩展
    if (pState->m_zip64)
    {
        // 如果无压缩大小或本地目录头偏移量超过了32位无符号整数的最大值
        if (uncomp_size >= MZ_UINT32_MAX || local_dir_header_ofs >= MZ_UINT32_MAX)
        {
            // 创建 ZIP64 扩展数据
            pExtra_data = extra_data;
            extra_size = mz_zip_writer_create_zip64_extra_data(extra_data, (uncomp_size >= MZ_UINT32_MAX) ? &uncomp_size : NULL,
                                                               (uncomp_size >= MZ_UINT32_MAX) ? &comp_size : NULL, (local_dir_header_ofs >= MZ_UINT32_MAX) ? &local_dir_header_ofs : NULL);
        }

        // 创建本地目录头，并写入到文件中
        if (!mz_zip_writer_create_local_dir_header(pZip, local_dir_header, (mz_uint16)archive_name_size, (mz_uint16)(extra_size + user_extra_data_len), 0, 0, 0, method, bit_flags, dos_time, dos_date))
            return mz_zip_set_error(pZip, MZ_ZIP_INTERNAL_ERROR);

        // 将本地目录头写入到文件中
        if (pZip->m_pWrite(pZip->m_pIO_opaque, local_dir_header_ofs, local_dir_header, sizeof(local_dir_header)) != sizeof(local_dir_header))
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);

        // 更新当前文件偏移量
        cur_archive_file_ofs += sizeof(local_dir_header);

        // 将归档文件名写入到文件中
        if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, pArchive_name, archive_name_size) != archive_name_size)
        {
            pZip->m_pFree(pZip->m_pAlloc_opaque, pComp);
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);
        }
        cur_archive_file_ofs += archive_name_size;

        // 如果有额外数据，则将额外数据写入到文件中
        if (pExtra_data != NULL)
        {
            if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, extra_data, extra_size) != extra_size)
                return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);

            cur_archive_file_ofs += extra_size;
        }
    }
    else
    {
        // 检查压缩大小和当前文件偏移量是否超过最大值，如果超过则返回归档文件过大的错误
        if ((comp_size > MZ_UINT32_MAX) || (cur_archive_file_ofs > MZ_UINT32_MAX))
            return mz_zip_set_error(pZip, MZ_ZIP_ARCHIVE_TOO_LARGE);
        
        // 创建本地目录头部并写入到 ZIP 文件，如果失败则返回内部错误
        if (!mz_zip_writer_create_local_dir_header(pZip, local_dir_header, (mz_uint16)archive_name_size, (mz_uint16)user_extra_data_len, 0, 0, 0, method, bit_flags, dos_time, dos_date))
            return mz_zip_set_error(pZip, MZ_ZIP_INTERNAL_ERROR);

        // 将本地目录头部写入到文件中，如果写入失败则返回文件写入失败的错误
        if (pZip->m_pWrite(pZip->m_pIO_opaque, local_dir_header_ofs, local_dir_header, sizeof(local_dir_header)) != sizeof(local_dir_header))
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);

        // 更新当前归档文件偏移量
        cur_archive_file_ofs += sizeof(local_dir_header);

        // 将归档文件名写入到文件中，如果写入失败则释放资源并返回文件写入失败的错误
        if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, pArchive_name, archive_name_size) != archive_name_size)
        {
            pZip->m_pFree(pZip->m_pAlloc_opaque, pComp);
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);
        }
        // 更新当前归档文件偏移量
        cur_archive_file_ofs += archive_name_size;
    }

    // 如果存在用户额外数据，则写入到文件中
    if (user_extra_data_len > 0)
    {
        // 将用户额外数据写入到文件中，如果写入失败则返回文件写入失败的错误
        if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, user_extra_data, user_extra_data_len) != user_extra_data_len)
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);

        // 更新当前归档文件偏移量
        cur_archive_file_ofs += user_extra_data_len;
    }

    // 如果需要存储未压缩的数据，则将数据直接写入到文件中
    if (store_data_uncompressed)
    {
        // 将未压缩的数据写入到文件中，如果写入失败则释放资源并返回文件写入失败的错误
        if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, pBuf, buf_size) != buf_size)
        {
            pZip->m_pFree(pZip->m_pAlloc_opaque, pComp);
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);
        }

        // 更新当前归档文件偏移量和压缩大小
        cur_archive_file_ofs += buf_size;
        comp_size = buf_size;
    }
    else if (buf_size)
    {
        // 使用 zlib 进行压缩的状态变量
        mz_zip_writer_add_state state;

        // 初始化压缩状态
        state.m_pZip = pZip;
        state.m_cur_archive_file_ofs = cur_archive_file_ofs;
        state.m_comp_size = 0;

        // 初始化压缩器，如果初始化或压缩失败则释放资源并返回压缩失败的错误
        if ((tdefl_init(pComp, mz_zip_writer_add_put_buf_callback, &state, tdefl_create_comp_flags_from_zip_params(level, -15, MZ_DEFAULT_STRATEGY)) != TDEFL_STATUS_OKAY) ||
            (tdefl_compress_buffer(pComp, pBuf, buf_size, TDEFL_FINISH) != TDEFL_STATUS_DONE))
        {
            pZip->m_pFree(pZip->m_pAlloc_opaque, pComp);
            return mz_zip_set_error(pZip, MZ_ZIP_COMPRESSION_FAILED);
        }

        // 获取压缩后的数据大小和更新当前归档文件偏移量
        comp_size = state.m_comp_size;
        cur_archive_file_ofs = state.m_cur_archive_file_ofs;
    }

    // 释放压缩器资源
    pZip->m_pFree(pZip->m_pAlloc_opaque, pComp);
    pComp = NULL;

    // 如果存在未压缩数据大小，则继续执行
    if (uncomp_size)
    {
        // 定义本地目录尾部数据结构，用于存储数据描述符信息
        mz_uint8 local_dir_footer[MZ_ZIP_DATA_DESCRIPTER_SIZE64];
        // 定义本地目录尾部数据结构的大小，默认为32位
        mz_uint32 local_dir_footer_size = MZ_ZIP_DATA_DESCRIPTER_SIZE32;
    
        // 断言标志位表明具有定位器位
        MZ_ASSERT(bit_flags & MZ_ZIP_LDH_BIT_FLAG_HAS_LOCATOR);
    
        // 在本地目录尾部数据结构中写入数据描述符标识符
        MZ_WRITE_LE32(local_dir_footer + 0, MZ_ZIP_DATA_DESCRIPTOR_ID);
        // 在本地目录尾部数据结构中写入未压缩数据的 CRC32 校验和
        MZ_WRITE_LE32(local_dir_footer + 4, uncomp_crc32);
    
        // 如果额外数据为空
        if (pExtra_data == NULL)
        {
            // 如果压缩大小超过32位无符号整数的最大值，则返回压缩文件过大的错误
            if (comp_size > MZ_UINT32_MAX)
                return mz_zip_set_error(pZip, MZ_ZIP_ARCHIVE_TOO_LARGE);
    
            // 否则，写入压缩大小和未压缩大小到本地目录尾部数据结构中
            MZ_WRITE_LE32(local_dir_footer + 8, comp_size);
            MZ_WRITE_LE32(local_dir_footer + 12, uncomp_size);
        }
        else
        {
            // 如果有额外数据，则写入64位的压缩大小和未压缩大小到本地目录尾部数据结构中
            MZ_WRITE_LE64(local_dir_footer + 8, comp_size);
            MZ_WRITE_LE64(local_dir_footer + 16, uncomp_size);
            // 更新本地目录尾部数据结构的大小为64位
            local_dir_footer_size = MZ_ZIP_DATA_DESCRIPTER_SIZE64;
        }
    
        // 将本地目录尾部数据结构写入到 ZIP 文件中的当前位置，并检查是否写入成功
        if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, local_dir_footer, local_dir_footer_size) != local_dir_footer_size)
            return MZ_FALSE;
    
        // 更新 ZIP 文件中的当前位置
        cur_archive_file_ofs += local_dir_footer_size;
    }
    
    // 如果存在额外数据，则创建并获取ZIP64额外数据的大小
    if (pExtra_data != NULL)
    {
        extra_size = mz_zip_writer_create_zip64_extra_data(extra_data, (uncomp_size >= MZ_UINT32_MAX) ? &uncomp_size : NULL,
                                                           (uncomp_size >= MZ_UINT32_MAX) ? &comp_size : NULL, (local_dir_header_ofs >= MZ_UINT32_MAX) ? &local_dir_header_ofs : NULL);
    }
    
    // 将文件信息添加到中央目录中，包括文件名、额外数据、评论等信息，并检查是否添加成功
    if (!mz_zip_writer_add_to_central_dir(pZip, pArchive_name, (mz_uint16)archive_name_size, pExtra_data, (mz_uint16)extra_size, pComment,
                                          comment_size, uncomp_size, comp_size, uncomp_crc32, method, bit_flags, dos_time, dos_date, local_dir_header_ofs, ext_attributes,
                                          user_extra_data_central, user_extra_data_central_len))
        return MZ_FALSE;
    
    // 增加 ZIP 文件中的文件总数计数
    pZip->m_total_files++;
    // 更新 ZIP 文件的总大小
    pZip->m_archive_size = cur_archive_file_ofs;
    
    // 返回操作成功标志
    return MZ_TRUE;
    mz_uint16 gen_flags = MZ_ZIP_LDH_BIT_FLAG_HAS_LOCATOR;
    // 初始化通用标志为具有定位器的标志

    mz_uint uncomp_crc32 = MZ_CRC32_INIT, level, num_alignment_padding_bytes;
    // 初始化未压缩数据的 CRC32 校验值为初始值，以及压缩级别和对齐填充字节数

    mz_uint16 method = 0, dos_time = 0, dos_date = 0, ext_attributes = 0;
    // 初始化方法、DOS 时间、DOS 日期和扩展属性

    mz_uint64 local_dir_header_ofs, cur_archive_file_ofs = pZip->m_archive_size, uncomp_size = size_to_add, comp_size = 0;
    // 初始化本地目录头偏移量，当前存档文件偏移量为存档大小，未压缩大小为待添加数据大小，压缩大小为0

    size_t archive_name_size;
    // 存档名的大小

    mz_uint8 local_dir_header[MZ_ZIP_LOCAL_DIR_HEADER_SIZE];
    // 本地目录头的字节缓冲区

    mz_uint8 *pExtra_data = NULL;
    // 额外数据的指针，初始化为空

    mz_uint32 extra_size = 0;
    // 额外数据的大小，初始化为0

    mz_uint8 extra_data[MZ_ZIP64_MAX_CENTRAL_EXTRA_FIELD_SIZE];
    // 额外数据的字节缓冲区，大小为 ZIP64 最大中央额外字段大小

    mz_zip_internal_state *pState;
    // ZIP 内部状态指针

    mz_uint64 file_ofs = 0;
    // 文件偏移量初始化为0

    if (!(level_and_flags & MZ_ZIP_FLAG_ASCII_FILENAME))
        gen_flags |= MZ_ZIP_GENERAL_PURPOSE_BIT_FLAG_UTF8;
    // 如果文件名不是 ASCII，则设置通用标志为 UTF-8 格式

    if ((int)level_and_flags < 0)
        level_and_flags = MZ_DEFAULT_LEVEL;
    level = level_and_flags & 0xF;
    // 如果压缩级别和标志小于0，则使用默认级别；否则从标志中提取级别

    /* Sanity checks */
    // 合法性检查
    if ((!pZip) || (!pZip->m_pState) || (pZip->m_zip_mode != MZ_ZIP_MODE_WRITING) || (!pArchive_name) || ((comment_size) && (!pComment)) || (level > MZ_UBER_COMPRESSION))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);
    // 如果存在无效参数（ZIP 对象不存在、ZIP 状态不是写入模式、存档名为空、存在注释但注释为空、级别超出最大压缩级别），则设置 ZIP 错误并返回

    pState = pZip->m_pState;
    // 获取 ZIP 内部状态指针

    if ((!pState->m_zip64) && (uncomp_size > MZ_UINT32_MAX))
    {
        /* Source file is too large for non-zip64 */
        // 对于非 ZIP64，源文件太大
        pState->m_zip64 = MZ_TRUE;
        // 设置 ZIP64 标志为真
    }

    /* We could support this, but why? */
    // 我们可以支持这个，但为什么要呢？
    if (level_and_flags & MZ_ZIP_FLAG_COMPRESSED_DATA)
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);
    // 如果标志指示压缩数据，则设置 ZIP 错误并返回

    if (!mz_zip_writer_validate_archive_name(pArchive_name))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_FILENAME);
    // 如果存档名无效，则设置 ZIP 错误并返回

    if (pState->m_zip64)
    {
        if (pZip->m_total_files == MZ_UINT32_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_TOO_MANY_FILES);
        // 如果已使用 ZIP64 并且文件数达到最大值，则设置 ZIP 错误并返回
    }
    else
    {
        if (pZip->m_total_files == MZ_UINT16_MAX)
        {
            pState->m_zip64 = MZ_TRUE;
            /*return mz_zip_set_error(pZip, MZ_ZIP_TOO_MANY_FILES); */
            // 如果文件数达到非 ZIP64 最大值，则设置 ZIP64 标志为真
        }
    }

    archive_name_size = strlen(pArchive_name);
    // 计算存档名的长度

    if (archive_name_size > MZ_UINT16_MAX)
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_FILENAME);
    // 如果存档名长度超过最大值，则设置 ZIP 错误并返回

    num_alignment_padding_bytes = mz_zip_writer_compute_padding_needed_for_file_alignment(pZip);
    // 计算为文件对齐所需的填充字节数

    /* miniz doesn't support central dirs >= MZ_UINT32_MAX bytes yet */
    // miniz 目前不支持大于或等于 MZ_UINT32_MAX 字节的中央目录
    // 检查如果中央目录大小加上其他额外的字节大小超过了32位无符号整数的最大值，则报告不支持的中央目录大小错误
    if (((mz_uint64)pState->m_central_dir.m_size + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + archive_name_size + MZ_ZIP64_MAX_CENTRAL_EXTRA_FIELD_SIZE + comment_size) >= MZ_UINT32_MAX)
        return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_CDIR_SIZE);

    // 如果不是使用ZIP64格式，并且估算的存档大小超出了32位整数的范围，则启用ZIP64格式
    if (!pState->m_zip64)
    {
        /* 如果存档显然会变得过大，提前结束处理 */
        if ((pZip->m_archive_size + num_alignment_padding_bytes + MZ_ZIP_LOCAL_DIR_HEADER_SIZE + archive_name_size + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE
            + archive_name_size + comment_size + user_extra_data_len + pState->m_central_dir.m_size + MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE + 1024
            + MZ_ZIP_DATA_DESCRIPTER_SIZE32 + user_extra_data_central_len) > 0xFFFFFFFF)
        {
            pState->m_zip64 = MZ_TRUE;
            /* 如果存档过大，可以选择报告错误，此处是注释掉的代码 */
            /*return mz_zip_set_error(pZip, MZ_ZIP_ARCHIVE_TOO_LARGE); */
        }
    }
#ifndef MINIZ_NO_TIME
    // 如果传入了文件时间指针，则将文件时间转换为 DOS 时间和日期
    if (pFile_time)
    {
        mz_zip_time_t_to_dos_time(*pFile_time, &dos_time, &dos_date);
    }
#endif

    // 如果未压缩大小小于等于3，则设置压缩级别为0
    if (uncomp_size <= 3)
        level = 0;

    // 在 ZIP 文件中写入指定数量的零字节，用于对齐文件偏移量
    if (!mz_zip_writer_write_zeros(pZip, cur_archive_file_ofs, num_alignment_padding_bytes))
    {
        // 写入失败时设置 ZIP 错误并返回
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);
    }

    // 更新当前归档文件偏移量
    cur_archive_file_ofs += num_alignment_padding_bytes;

    // 设置本地目录头的偏移量
    local_dir_header_ofs = cur_archive_file_ofs;

    // 如果设置了文件偏移对齐要求，则进行断言验证
    if (pZip->m_file_offset_alignment)
    {
        MZ_ASSERT((cur_archive_file_ofs & (pZip->m_file_offset_alignment - 1)) == 0);
    }

    // 如果有未压缩大小且设置了压缩级别，则选择压缩方法为 MZ_DEFLATED
    if (uncomp_size && level)
    {
        method = MZ_DEFLATED;
    }

    // 清空本地目录头对象
    MZ_CLEAR_OBJ(local_dir_header);

    // 如果需要支持 ZIP64 格式
    if (pState->m_zip64)
    {
        // 如果未压缩大小或本地目录头偏移量超过了 32 位无符号整数的最大值
        if (uncomp_size >= MZ_UINT32_MAX || local_dir_header_ofs >= MZ_UINT32_MAX)
        {
            // 创建 ZIP64 额外数据，并计算额外数据的大小
            pExtra_data = extra_data;
            extra_size = mz_zip_writer_create_zip64_extra_data(extra_data,
                                                               (uncomp_size >= MZ_UINT32_MAX) ? &uncomp_size : NULL,
                                                               (uncomp_size >= MZ_UINT32_MAX) ? &comp_size : NULL,
                                                               (local_dir_header_ofs >= MZ_UINT32_MAX) ? &local_dir_header_ofs : NULL);
        }

        // 创建本地目录头并写入 ZIP 文件
        if (!mz_zip_writer_create_local_dir_header(pZip, local_dir_header,
                                                  (mz_uint16)archive_name_size,
                                                  (mz_uint16)(extra_size + user_extra_data_len),
                                                  0, 0, 0, method, gen_flags, dos_time, dos_date))
            return mz_zip_set_error(pZip, MZ_ZIP_INTERNAL_ERROR);

        if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, local_dir_header, sizeof(local_dir_header)) != sizeof(local_dir_header))
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);

        cur_archive_file_ofs += sizeof(local_dir_header);

        if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, pArchive_name, archive_name_size) != archive_name_size)
        {
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);
        }

        cur_archive_file_ofs += archive_name_size;

        if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, extra_data, extra_size) != extra_size)
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);

        cur_archive_file_ofs += extra_size;
    }
    else
    {
        // 检查压缩大小和当前存档文件偏移是否超过32位整数的最大值，如果超过则返回存档过大错误
        if ((comp_size > MZ_UINT32_MAX) || (cur_archive_file_ofs > MZ_UINT32_MAX))
            return mz_zip_set_error(pZip, MZ_ZIP_ARCHIVE_TOO_LARGE);
        // 创建本地目录头部，如果失败则返回内部错误
        if (!mz_zip_writer_create_local_dir_header(pZip, local_dir_header, (mz_uint16)archive_name_size, (mz_uint16)user_extra_data_len, 0, 0, 0, method, gen_flags, dos_time, dos_date))
            return mz_zip_set_error(pZip, MZ_ZIP_INTERNAL_ERROR);

        // 将本地目录头部写入存档文件，如果写入失败则返回文件写入失败错误
        if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, local_dir_header, sizeof(local_dir_header)) != sizeof(local_dir_header))
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);

        // 更新当前存档文件偏移
        cur_archive_file_ofs += sizeof(local_dir_header);

        // 将存档名称写入存档文件，如果写入失败则返回文件写入失败错误
        if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, pArchive_name, archive_name_size) != archive_name_size)
        {
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);
        }

        // 更新当前存档文件偏移
        cur_archive_file_ofs += archive_name_size;
    }

    // 如果存在用户额外数据
    if (user_extra_data_len > 0)
    {
        // 将用户额外数据写入存档文件，如果写入失败则返回文件写入失败错误
        if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, user_extra_data, user_extra_data_len) != user_extra_data_len)
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);

        // 更新当前存档文件偏移
        cur_archive_file_ofs += user_extra_data_len;
    }

    // 如果存在未压缩大小
    if (uncomp_size)
    {
        {
            // 创建本地目录尾部，用于存储数据描述符信息
            mz_uint8 local_dir_footer[MZ_ZIP_DATA_DESCRIPTER_SIZE64];
            mz_uint32 local_dir_footer_size = MZ_ZIP_DATA_DESCRIPTER_SIZE32;

            // 写入数据描述符标识符
            MZ_WRITE_LE32(local_dir_footer + 0, MZ_ZIP_DATA_DESCRIPTOR_ID);
            // 写入未压缩数据的 CRC32 校验码
            MZ_WRITE_LE32(local_dir_footer + 4, uncomp_crc32);

            // 如果额外数据为空
            if (pExtra_data == NULL)
            {
                // 如果压缩大小超过32位整数的最大值，则返回存档过大错误
                if (comp_size > MZ_UINT32_MAX)
                    return mz_zip_set_error(pZip, MZ_ZIP_ARCHIVE_TOO_LARGE);

                // 写入压缩大小和未压缩大小
                MZ_WRITE_LE32(local_dir_footer + 8, comp_size);
                MZ_WRITE_LE32(local_dir_footer + 12, uncomp_size);
            }
            else
            {
                // 写入64位压缩大小和未压缩大小
                MZ_WRITE_LE64(local_dir_footer + 8, comp_size);
                MZ_WRITE_LE64(local_dir_footer + 16, uncomp_size);
                // 更新本地目录尾部大小为64位
                local_dir_footer_size = MZ_ZIP_DATA_DESCRIPTER_SIZE64;
            }

            // 将本地目录尾部写入存档文件，如果写入失败则返回假
            if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_archive_file_ofs, local_dir_footer, local_dir_footer_size) != local_dir_footer_size)
                return MZ_FALSE;

            // 更新当前存档文件偏移
            cur_archive_file_ofs += local_dir_footer_size;
        }
    }

    // 如果存在额外数据
    if (pExtra_data != NULL)
    {
        // 创建 ZIP64 额外数据，并获取额外数据的大小
        extra_size = mz_zip_writer_create_zip64_extra_data(extra_data, (uncomp_size >= MZ_UINT32_MAX) ? &uncomp_size : NULL,
                                                           (uncomp_size >= MZ_UINT32_MAX) ? &comp_size : NULL, (local_dir_header_ofs >= MZ_UINT32_MAX) ? &local_dir_header_ofs : NULL);
    }
    // 如果向 ZIP 文件的中央目录添加条目失败，则返回 MZ_FALSE
    if (!mz_zip_writer_add_to_central_dir(pZip, pArchive_name, (mz_uint16)archive_name_size, pExtra_data, (mz_uint16)extra_size, pComment, comment_size,
                                          uncomp_size, comp_size, uncomp_crc32, method, gen_flags, dos_time, dos_date, local_dir_header_ofs, ext_attributes,
                                          user_extra_data_central, user_extra_data_central_len))
        return MZ_FALSE;

    // 增加 ZIP 文件总文件数目计数
    pZip->m_total_files++;

    // 更新 ZIP 归档文件的大小偏移量
    pZip->m_archive_size = cur_archive_file_ofs;

    // 添加条目成功，返回 MZ_TRUE
    return MZ_TRUE;
#ifndef MINIZ_NO_STDIO
// 如果未定义 MINIZ_NO_STDIO 宏，则编译以下内容

// 定义一个函数，从标准IO文件中读取数据，用于ZIP文件操作
static size_t mz_file_read_func_stdio(void *pOpaque, mz_uint64 file_ofs, void *pBuf, size_t n)
{
    // 将不透明指针转换为标准IO文件指针
    MZ_FILE *pSrc_file = (MZ_FILE *)pOpaque;
    // 获取当前文件指针位置
    mz_int64 cur_ofs = MZ_FTELL64(pSrc_file);

    // 如果请求的文件偏移量小于0，或者当前文件指针位置与请求的偏移量不一致且无法重新定位到请求位置，则返回0
    if (((mz_int64)file_ofs < 0) || (((cur_ofs != (mz_int64)file_ofs)) && (MZ_FSEEK64(pSrc_file, (mz_int64)file_ofs, SEEK_SET))))
        return 0;

    // 从文件中读取数据到缓冲区
    return MZ_FREAD(pBuf, 1, n, pSrc_file);
}

// 向ZIP归档中添加使用标准IO文件的文件
mz_bool mz_zip_writer_add_cfile(mz_zip_archive *pZip, const char *pArchive_name, MZ_FILE *pSrc_file, mz_uint64 size_to_add, const MZ_TIME_T *pFile_time, const void *pComment, mz_uint16 comment_size, mz_uint level_and_flags,
    const char *user_extra_data, mz_uint user_extra_data_len, const char *user_extra_data_central, mz_uint user_extra_data_central_len)
{
    // 调用通用函数，将标准IO文件中的数据添加到ZIP归档中
    return mz_zip_writer_add_read_buf_callback(pZip, pArchive_name, mz_file_read_func_stdio, pSrc_file, size_to_add, pFile_time, pComment, comment_size, level_and_flags,
        user_extra_data, user_extra_data_len, user_extra_data_central, user_extra_data_central_len);
}

// 向ZIP归档中添加指定文件
mz_bool mz_zip_writer_add_file(mz_zip_archive *pZip, const char *pArchive_name, const char *pSrc_filename, const void *pComment, mz_uint16 comment_size, mz_uint level_and_flags)
{
    // 定义变量
    MZ_FILE *pSrc_file = NULL;
    mz_uint64 uncomp_size = 0;
    MZ_TIME_T file_modified_time;
    MZ_TIME_T *pFile_time = NULL;
    mz_bool status;

    // 初始化文件修改时间
    memset(&file_modified_time, 0, sizeof(file_modified_time));

    // 如果未定义 MINIZ_NO_TIME 和 MINIZ_NO_STDIO 宏，则获取文件的修改时间
    pFile_time = &file_modified_time;
    if (!mz_zip_get_file_modified_time(pSrc_filename, &file_modified_time))
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_STAT_FAILED);

    // 打开要添加的源文件
    pSrc_file = MZ_FOPEN(pSrc_filename, "rb");
    if (!pSrc_file)
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_OPEN_FAILED);

    // 定位到文件末尾获取未压缩大小，并将文件指针重新定位到文件开头
    MZ_FSEEK64(pSrc_file, 0, SEEK_END);
    uncomp_size = MZ_FTELL64(pSrc_file);
    MZ_FSEEK64(pSrc_file, 0, SEEK_SET);

    // 调用添加标准IO文件的函数，并添加到ZIP归档中
    status = mz_zip_writer_add_cfile(pZip, pArchive_name, pSrc_file, uncomp_size, pFile_time, pComment, comment_size, level_and_flags, NULL, 0, NULL, 0);

    // 关闭文件
    MZ_FCLOSE(pSrc_file);

    // 返回添加操作的状态
    return status;
}
#endif /* #ifndef MINIZ_NO_STDIO */

// 更新ZIP64扩展块
static mz_bool mz_zip_writer_update_zip64_extension_block(mz_zip_array *pNew_ext, mz_zip_archive *pZip, const mz_uint8 *pExt, uint32_t ext_len, mz_uint64 *pComp_size, mz_uint64 *pUncomp_size, mz_uint64 *pLocal_header_ofs, mz_uint32 *pDisk_start)
{
    // 为新的ZIP64数据预留足够的空间
    // + 64 应足够容纳任何新的ZIP64数据
    if (!mz_zip_array_reserve(pZip, pNew_ext, ext_len + 64, MZ_FALSE))
        return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);

    // 调整数组大小
    mz_zip_array_resize(pZip, pNew_ext, 0, MZ_FALSE);

    // 如果指定了pUncomp_size、pComp_size、pLocal_header_ofs或pDisk_start，则执行以下操作
    if ((pUncomp_size) || (pComp_size) || (pLocal_header_ofs) || (pDisk_start))
    {
        // 定义一个新的扩展数据块，长度为64字节
        mz_uint8 new_ext_block[64];
        // 指向新扩展数据块的指针
        mz_uint8 *pDst = new_ext_block;
        // 写入ZIP64扩展信息字段的标识（2字节，小端序）
        mz_write_le16(pDst, MZ_ZIP64_EXTENDED_INFORMATION_FIELD_HEADER_ID);
        // 写入占位符0（2字节，小端序）
        mz_write_le16(pDst + sizeof(mz_uint16), 0);
        // 指针移动到新位置
        pDst += sizeof(mz_uint16) * 2;
    
        // 如果存在未压缩大小的指针
        if (pUncomp_size)
        {
            // 写入未压缩大小（8字节，小端序）
            mz_write_le64(pDst, *pUncomp_size);
            // 指针移动到新位置
            pDst += sizeof(mz_uint64);
        }
    
        // 如果存在压缩大小的指针
        if (pComp_size)
        {
            // 写入压缩大小（8字节，小端序）
            mz_write_le64(pDst, *pComp_size);
            // 指针移动到新位置
            pDst += sizeof(mz_uint64);
        }
    
        // 如果存在本地头偏移的指针
        if (pLocal_header_ofs)
        {
            // 写入本地头偏移（8字节，小端序）
            mz_write_le64(pDst, *pLocal_header_ofs);
            // 指针移动到新位置
            pDst += sizeof(mz_uint64);
        }
    
        // 如果存在起始磁盘号的指针
        if (pDisk_start)
        {
            // 写入起始磁盘号（4字节，小端序）
            mz_write_le32(pDst, *pDisk_start);
            // 指针移动到新位置
            pDst += sizeof(mz_uint32);
        }
    
        // 写入ZIP64扩展信息字段的总长度（2字节，小端序）
        mz_write_le16(new_ext_block + sizeof(mz_uint16), (mz_uint16)((pDst - new_ext_block) - sizeof(mz_uint16) * 2));
    
        // 将新扩展数据块添加到ZIP文件中
        if (!mz_zip_array_push_back(pZip, pNew_ext, new_ext_block, pDst - new_ext_block))
            // 如果添加失败，则返回分配内存失败错误
            return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
    }
    
    // 如果存在额外数据和额外数据长度
    if ((pExt) && (ext_len))
    {
        // 剩余额外数据的大小
        mz_uint32 extra_size_remaining = ext_len;
        // 指向额外数据的指针
        const mz_uint8 *pExtra_data = pExt;
    
        // 循环处理所有额外数据
        do
        {
            // 定义字段ID、字段数据大小和字段总大小
            mz_uint32 field_id, field_data_size, field_total_size;
    
            // 如果剩余额外数据大小小于2个字节，则表示头部无效或损坏
            if (extra_size_remaining < (sizeof(mz_uint16) * 2))
                return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);
    
            // 从额外数据中读取字段ID（2字节，小端序）
            field_id = MZ_READ_LE16(pExtra_data);
            // 从额外数据中读取字段数据大小（2字节，小端序）
            field_data_size = MZ_READ_LE16(pExtra_data + sizeof(mz_uint16));
            // 计算字段总大小
            field_total_size = field_data_size + sizeof(mz_uint16) * 2;
    
            // 如果字段总大小大于剩余额外数据大小，则表示头部无效或损坏
            if (field_total_size > extra_size_remaining)
                return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);
    
            // 如果字段ID不是ZIP64扩展信息字段标识
            if (field_id != MZ_ZIP64_EXTENDED_INFORMATION_FIELD_HEADER_ID)
            {
                // 将字段数据添加到ZIP文件的额外数据中
                if (!mz_zip_array_push_back(pZip, pNew_ext, pExtra_data, field_total_size))
                    // 如果添加失败，则返回分配内存失败错误
                    return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
            }
    
            // 移动额外数据指针到下一个字段的起始位置
            pExtra_data += field_total_size;
            // 减少剩余额外数据大小
            extra_size_remaining -= field_total_size;
        } while (extra_size_remaining);  // 继续处理直到额外数据全部处理完毕
    }
    
    // 返回操作成功
    return MZ_TRUE;
/* TODO: This func is now pretty freakin complex due to zip64, split it up? */
mz_bool mz_zip_writer_add_from_zip_reader(mz_zip_archive *pZip, mz_zip_archive *pSource_zip, mz_uint src_file_index)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint n, bit_flags, num_alignment_padding_bytes, src_central_dir_following_data_size;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint64 src_archive_bytes_remaining, local_dir_header_ofs;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint64 cur_src_file_ofs, cur_dst_file_ofs;
    mz_uint32 local_header_u32[(MZ_ZIP_LOCAL_DIR_HEADER_SIZE + sizeof(mz_uint32) - 1) / sizeof(mz_uint32)];
    mz_uint8 *pLocal_header = (mz_uint8 *)local_header_u32;
    mz_uint8 new_central_header[MZ_ZIP_CENTRAL_DIR_HEADER_SIZE];
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    size_t orig_central_dir_size;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_zip_internal_state *pState;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    void *pBuf;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    const mz_uint8 *pSrc_central_header;
    mz_zip_archive_file_stat src_file_stat;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint32 src_filename_len, src_comment_len, src_ext_len;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint32 local_header_filename_size, local_header_extra_len;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint64 local_header_comp_size, local_header_uncomp_size;
    mz_bool found_zip64_ext_data_in_ldir = MZ_FALSE;

    /* Sanity checks */
    // 检查参数是否有效，以及写入模式是否正确
    if ((!pZip) || (!pZip->m_pState) || (pZip->m_zip_mode != MZ_ZIP_MODE_WRITING) || (!pSource_zip->m_pRead))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    pState = pZip->m_pState;

    /* Don't support copying files from zip64 archives to non-zip64, even though in some cases this is possible */
    // 如果源 ZIP 是 zip64 格式，而目标 ZIP 不是，不支持复制
    if ((pSource_zip->m_pState->m_zip64) && (!pZip->m_pState->m_zip64))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    /* Get pointer to the source central dir header and crack it */
    // 获取源 ZIP 文件的中央目录头指针，并解析其内容
    if (NULL == (pSrc_central_header = mz_zip_get_cdh(pSource_zip, src_file_index)))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 检查中央目录头的签名是否正确
    if (MZ_READ_LE32(pSrc_central_header + MZ_ZIP_CDH_SIG_OFS) != MZ_ZIP_CENTRAL_DIR_HEADER_SIG)
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);

    src_filename_len = MZ_READ_LE16(pSrc_central_header + MZ_ZIP_CDH_FILENAME_LEN_OFS);
    src_comment_len = MZ_READ_LE16(pSrc_central_header + MZ_ZIP_CDH_COMMENT_LEN_OFS);
    src_ext_len = MZ_READ_LE16(pSrc_central_header + MZ_ZIP_CDH_EXTRA_LEN_OFS);
    src_central_dir_following_data_size = src_filename_len + src_ext_len + src_comment_len;

    /* TODO: We don't support central dir's >= MZ_UINT32_MAX bytes right now (+32 fudge factor in case we need to add more extra data) */
    // TODO: 目前不支持大于等于 MZ_UINT32_MAX 字节的中央目录，（+32 偏移以防需要添加更多的额外数据）
    # 检查中央目录的大小是否超出支持范围，如果是则返回错误
    if ((pState->m_central_dir.m_size + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + src_central_dir_following_data_size + 32) >= MZ_UINT32_MAX)
        return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_CDIR_SIZE);

    # 计算需要的对齐填充字节数，以确保文件对齐
    num_alignment_padding_bytes = mz_zip_writer_compute_padding_needed_for_file_alignment(pZip);

    # 如果不是使用zip64格式，并且文件总数已达到最大16位整数的限制，则返回错误
    if (!pState->m_zip64)
    {
        if (pZip->m_total_files == MZ_UINT16_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_TOO_MANY_FILES);
    }
    else
    {
        # TODO: 我们的zip64支持仍然存在一些32位限制，可能不值得修复
        # 如果使用zip64格式，并且文件总数已达到最大32位整数的限制，则返回错误
        if (pZip->m_total_files == MZ_UINT32_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_TOO_MANY_FILES);
    }

    # 获取源zip文件中指定文件的统计信息，如果失败则返回假
    if (!mz_zip_file_stat_internal(pSource_zip, src_file_index, pSrc_central_header, &src_file_stat, NULL))
        return MZ_FALSE;

    # 设置当前源文件偏移和目标文件偏移
    cur_src_file_ofs = src_file_stat.m_local_header_ofs;
    cur_dst_file_ofs = pZip->m_archive_size;

    # 读取源存档的本地目录头部信息
    if (pSource_zip->m_pRead(pSource_zip->m_pIO_opaque, cur_src_file_ofs, pLocal_header, MZ_ZIP_LOCAL_DIR_HEADER_SIZE) != MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);

    # 检查本地目录头部的签名是否有效
    if (MZ_READ_LE32(pLocal_header) != MZ_ZIP_LOCAL_DIR_HEADER_SIG)
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);

    # 更新当前源文件偏移
    cur_src_file_ofs += MZ_ZIP_LOCAL_DIR_HEADER_SIZE;

    # 计算需要复制的总大小（文件名长度+额外数据+压缩数据）
    local_header_filename_size = MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_FILENAME_LEN_OFS);
    local_header_extra_len = MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_EXTRA_LEN_OFS);
    local_header_comp_size = MZ_READ_LE32(pLocal_header + MZ_ZIP_LDH_COMPRESSED_SIZE_OFS);
    local_header_uncomp_size = MZ_READ_LE32(pLocal_header + MZ_ZIP_LDH_DECOMPRESSED_SIZE_OFS);
    src_archive_bytes_remaining = local_header_filename_size + local_header_extra_len + src_file_stat.m_comp_size;

    # 尝试查找zip64扩展信息字段
    if ((local_header_extra_len) && ((local_header_comp_size == MZ_UINT32_MAX) || (local_header_uncomp_size == MZ_UINT32_MAX)))
    {
        // 定义存储文件数据的数组
        mz_zip_array file_data_array;
        // 指向额外数据的指针
        const mz_uint8 *pExtra_data;
        // 剩余的额外数据大小
        mz_uint32 extra_size_remaining = local_header_extra_len;
    
        // 初始化文件数据数组
        mz_zip_array_init(&file_data_array, 1);
        // 尝试调整文件数据数组的大小以匹配本地头部的额外数据长度
        if (!mz_zip_array_resize(pZip, &file_data_array, local_header_extra_len, MZ_FALSE))
        {
            // 调整大小失败，返回内存分配失败的错误
            return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
        }
    
        // 从源 ZIP 中读取本地头部后的额外数据
        if (pSource_zip->m_pRead(pSource_zip->m_pIO_opaque, src_file_stat.m_local_header_ofs + MZ_ZIP_LOCAL_DIR_HEADER_SIZE + local_header_filename_size, file_data_array.m_p, local_header_extra_len) != local_header_extra_len)
        {
            // 读取失败，清除文件数据数组并返回文件读取失败的错误
            mz_zip_array_clear(pZip, &file_data_array);
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);
        }
    
        // 将文件数据数组的指针转换为无符号整数类型的额外数据指针
        pExtra_data = (const mz_uint8 *)file_data_array.m_p;
    
        // 循环处理额外数据中的字段
        do
        {
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            mz_uint32 field_id, field_data_size, field_total_size;
    
            // 如果剩余的额外数据小于两个 mz_uint16 的大小，则视为无效头部或损坏的 ZIP 文件
            if (extra_size_remaining < (sizeof(mz_uint16) * 2))
            {
                mz_zip_array_clear(pZip, &file_data_array);
                return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);
            }
    
            // 读取字段 ID 和字段数据大小
            field_id = MZ_READ_LE16(pExtra_data);
            field_data_size = MZ_READ_LE16(pExtra_data + sizeof(mz_uint16));
            // 计算字段总大小
            field_total_size = field_data_size + sizeof(mz_uint16) * 2;
    
            // 如果字段总大小大于剩余的额外数据大小，则视为无效头部或损坏的 ZIP 文件
            if (field_total_size > extra_size_remaining)
            {
                mz_zip_array_clear(pZip, &file_data_array);
                return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);
            }
    
            // 如果字段 ID 是 ZIP64 扩展信息字段的 ID
            if (field_id == MZ_ZIP64_EXTENDED_INFORMATION_FIELD_HEADER_ID)
            {
                const mz_uint8 *pSrc_field_data = pExtra_data + sizeof(mz_uint32);
    
                // 如果字段数据大小小于两个 mz_uint64 的大小，则视为无效头部或损坏的 ZIP 文件
                if (field_data_size < sizeof(mz_uint64) * 2)
                {
                    mz_zip_array_clear(pZip, &file_data_array);
                    return mz_zip_set_error(pZip, MZ_ZIP_INVALID_HEADER_OR_CORRUPTED);
                }
    
                // 读取本地头部中未压缩大小和压缩大小（如果有描述符，则压缩大小可能为 0）
                local_header_uncomp_size = MZ_READ_LE64(pSrc_field_data);
                local_header_comp_size = MZ_READ_LE64(pSrc_field_data + sizeof(mz_uint64));
    
                // 在本地文件目录记录中找到 ZIP64 扩展数据
                found_zip64_ext_data_in_ldir = MZ_TRUE;
                break;
            }
    
            // 更新额外数据指针和剩余额外数据大小
            pExtra_data += field_total_size;
            extra_size_remaining -= field_total_size;
        } while (extra_size_remaining);
    
        // 清除文件数据数组
        mz_zip_array_clear(pZip, &file_data_array);
    }
    
    // 如果不是 ZIP64 格式
    if (!pState->m_zip64)
    {
        /* 尝试检测新归档文件的大致大小，如果可能会超出限制则提前终止（+(sizeof(mz_uint32) * 4) 是用于可能存在的可选描述符，+64 是一个修正因子）。 */
        /* 我们还会在归档最终完成时进行检查，所以这并不需要完全准确。 */
        // 计算新归档文件的大致大小
        mz_uint64 approx_new_archive_size = cur_dst_file_ofs + num_alignment_padding_bytes + MZ_ZIP_LOCAL_DIR_HEADER_SIZE + src_archive_bytes_remaining + (sizeof(mz_uint32) * 4) +
                                            pState->m_central_dir.m_size + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + src_central_dir_following_data_size + MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE + 64;

        // 如果估计的新归档文件大小超过了 MZ_UINT32_MAX，则返回归档文件过大的错误
        if (approx_new_archive_size >= MZ_UINT32_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_ARCHIVE_TOO_LARGE);
    }

    /* 写入目标归档文件的填充区域 */
    // 使用零填充方式写入目标归档文件的填充区域
    if (!mz_zip_writer_write_zeros(pZip, cur_dst_file_ofs, num_alignment_padding_bytes))
        return MZ_FALSE;

    cur_dst_file_ofs += num_alignment_padding_bytes;

    // 记录本地目录头的偏移量
    local_dir_header_ofs = cur_dst_file_ofs;
    // 如果定义了文件偏移对齐值，则确保本地目录头的偏移量是其倍数
    if (pZip->m_file_offset_alignment)
    {
        MZ_ASSERT((local_dir_header_ofs & (pZip->m_file_offset_alignment - 1)) == 0);
    }

    /* 原始 ZIP 文件的本地头部和扩展块即使在使用 zip64 时也不会改变，因此我们可以直接复制到目标 ZIP 文件 */
    // 将原始 ZIP 文件的本地头部数据复制到目标 ZIP 文件
    if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_dst_file_ofs, pLocal_header, MZ_ZIP_LOCAL_DIR_HEADER_SIZE) != MZ_ZIP_LOCAL_DIR_HEADER_SIZE)
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);

    cur_dst_file_ofs += MZ_ZIP_LOCAL_DIR_HEADER_SIZE;

    /* 复制源归档文件的字节到目标归档文件，同时确保有足够的缓冲空间来处理可选的数据描述符 */
    // 分配足够的缓冲区空间来读取源归档文件的数据
    if (NULL == (pBuf = pZip->m_pAlloc(pZip->m_pAlloc_opaque, 1, (size_t)MZ_MAX(32U, MZ_MIN((mz_uint64)MZ_ZIP_MAX_IO_BUF_SIZE, src_archive_bytes_remaining)))))
        return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);

    while (src_archive_bytes_remaining)
    {
        // 计算每次读取的字节数，取较小值为 MZ_ZIP_MAX_IO_BUF_SIZE 或剩余的归档文件字节数
        n = (mz_uint)MZ_MIN((mz_uint64)MZ_ZIP_MAX_IO_BUF_SIZE, src_archive_bytes_remaining);
        // 从源 ZIP 文件读取数据到缓冲区
        if (pSource_zip->m_pRead(pSource_zip->m_pIO_opaque, cur_src_file_ofs, pBuf, n) != n)
        {
            pZip->m_pFree(pZip->m_pAlloc_opaque, pBuf);
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);
        }
        cur_src_file_ofs += n;

        // 将缓冲区中的数据写入目标 ZIP 文件
        if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_dst_file_ofs, pBuf, n) != n)
        {
            pZip->m_pFree(pZip->m_pAlloc_opaque, pBuf);
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);
        }
        cur_dst_file_ofs += n;

        src_archive_bytes_remaining -= n;
    }

    /* 处理可选的数据描述符 */
    // 读取位标志，确定是否存在数据描述符
    bit_flags = MZ_READ_LE16(pLocal_header + MZ_ZIP_LDH_BIT_FLAG_OFS);
    if (bit_flags & 8)
    {
        /* 复制数据描述符 */
        if ((pSource_zip->m_pState->m_zip64) || (found_zip64_ext_data_in_ldir))
        {
            /* 如果源文件是zip64格式或者在本地目录中找到了zip64的扩展数据 */
    
            /* 源文件是zip64格式，目标文件也必须是zip64格式 */
    
            /* name            uint32_t's */
            /* id                1 (zip64中可选) */
            /* crc            1 */
            /* comp_size    2 */
            /* uncomp_size 2 */
    
            /* 从源文件读取数据描述符 */
            if (pSource_zip->m_pRead(pSource_zip->m_pIO_opaque, cur_src_file_ofs, pBuf, (sizeof(mz_uint32) * 6)) != (sizeof(mz_uint32) * 6))
            {
                /* 读取失败时释放缓冲区并返回错误 */
                pZip->m_pFree(pZip->m_pAlloc_opaque, pBuf);
                return mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);
            }
    
            /* 确定数据描述符的实际长度 */
            n = sizeof(mz_uint32) * ((MZ_READ_LE32(pBuf) == MZ_ZIP_DATA_DESCRIPTOR_ID) ? 6 : 5);
        }
        else
        {
            /* 如果源文件不是zip64格式 */
    
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            mz_bool has_id;
    
            /* 从源文件读取4个uint32_t数据 */
            if (pSource_zip->m_pRead(pSource_zip->m_pIO_opaque, cur_src_file_ofs, pBuf, sizeof(mz_uint32) * 4) != sizeof(mz_uint32) * 4)
            {
                /* 读取失败时释放缓冲区并返回错误 */
                pZip->m_pFree(pZip->m_pAlloc_opaque, pBuf);
                return mz_zip_set_error(pZip, MZ_ZIP_FILE_READ_FAILED);
            }
    
            /* 检查是否存在数据描述符ID */
            has_id = (MZ_READ_LE32(pBuf) == MZ_ZIP_DATA_DESCRIPTOR_ID);
    
            if (pZip->m_pState->m_zip64)
            {
                /* 目标文件是zip64格式，升级数据描述符 */
                const mz_uint32 *pSrc_descriptor = (const mz_uint32 *)((const mz_uint8 *)pBuf + (has_id ? sizeof(mz_uint32) : 0));
                const mz_uint32 src_crc32 = pSrc_descriptor[0];
                const mz_uint64 src_comp_size = pSrc_descriptor[1];
                const mz_uint64 src_uncomp_size = pSrc_descriptor[2];
    
                /* 将数据描述符写入缓冲区 */
                mz_write_le32((mz_uint8 *)pBuf, MZ_ZIP_DATA_DESCRIPTOR_ID);
                mz_write_le32((mz_uint8 *)pBuf + sizeof(mz_uint32) * 1, src_crc32);
                mz_write_le64((mz_uint8 *)pBuf + sizeof(mz_uint32) * 2, src_comp_size);
                mz_write_le64((mz_uint8 *)pBuf + sizeof(mz_uint32) * 4, src_uncomp_size);
    
                /* 确定写入数据的长度 */
                n = sizeof(mz_uint32) * 6;
            }
            else
            {
                /* 目标文件不是zip64格式，直接复制数据 */
                n = sizeof(mz_uint32) * (has_id ? 4 : 3);
            }
        }
    
        /* 将数据写入目标文件 */
        if (pZip->m_pWrite(pZip->m_pIO_opaque, cur_dst_file_ofs, pBuf, n) != n)
        {
            /* 写入失败时释放缓冲区并返回错误 */
            pZip->m_pFree(pZip->m_pAlloc_opaque, pBuf);
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);
        }
    
        // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
        /* 更新源文件和目标文件的偏移量 */
        cur_src_file_ofs += n;
        cur_dst_file_ofs += n;
    }
    pZip->m_pFree(pZip->m_pAlloc_opaque, pBuf);
    
    /* 最后，添加新的中央目录头部 */
    orig_central_dir_size = pState->m_central_dir.m_size;
    
    /* 复制新的中央目录头部数据 */
    memcpy(new_central_header, pSrc_central_header, MZ_ZIP_CENTRAL_DIR_HEADER_SIZE);
    
    if (pState->m_zip64)
    {
        /* 这是痛苦的部分：我们需要编写新的中央目录头和扩展块，包含更新的zip64字段，并确保不包含旧字段（如果有）。 */
        const mz_uint8 *pSrc_ext = pSrc_central_header + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + src_filename_len;
        // 创建一个新的扩展块数组
        mz_zip_array new_ext_block;
        
        // 初始化新的扩展块数组，每个元素大小为1字节
        mz_zip_array_init(&new_ext_block, sizeof(mz_uint8));
        
        // 将最大的32位整数写入新中央目录头的压缩大小、解压大小和本地头偏移位置字段
        MZ_WRITE_LE32(new_central_header + MZ_ZIP_CDH_COMPRESSED_SIZE_OFS, MZ_UINT32_MAX);
        MZ_WRITE_LE32(new_central_header + MZ_ZIP_CDH_DECOMPRESSED_SIZE_OFS, MZ_UINT32_MAX);
        MZ_WRITE_LE32(new_central_header + MZ_ZIP_CDH_LOCAL_HEADER_OFS, MZ_UINT32_MAX);
        
        // 更新zip64扩展块，如果失败则清理资源并返回假
        if (!mz_zip_writer_update_zip64_extension_block(&new_ext_block, pZip, pSrc_ext, src_ext_len, &src_file_stat.m_comp_size, &src_file_stat.m_uncomp_size, &local_dir_header_ofs, NULL))
        {
            mz_zip_array_clear(pZip, &new_ext_block);
            return MZ_FALSE;
        }
        
        // 写入新中央目录头的额外数据长度字段
        MZ_WRITE_LE16(new_central_header + MZ_ZIP_CDH_EXTRA_LEN_OFS, new_ext_block.m_size);
        
        // 将新中央目录头推入中央目录数组，如果失败则清理资源并返回分配失败错误
        if (!mz_zip_array_push_back(pZip, &pState->m_central_dir, new_central_header, MZ_ZIP_CENTRAL_DIR_HEADER_SIZE))
        {
            mz_zip_array_clear(pZip, &new_ext_block);
            return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
        }
        
        // 将源中央目录头的文件名部分推入中央目录数组，如果失败则清理资源并返回分配失败错误
        if (!mz_zip_array_push_back(pZip, &pState->m_central_dir, pSrc_central_header + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE, src_filename_len))
        {
            mz_zip_array_clear(pZip, &new_ext_block);
            mz_zip_array_resize(pZip, &pState->m_central_dir, orig_central_dir_size, MZ_FALSE);
            return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
        }
        
        // 将新扩展块的数据推入中央目录数组，如果失败则清理资源并返回分配失败错误
        if (!mz_zip_array_push_back(pZip, &pState->m_central_dir, new_ext_block.m_p, new_ext_block.m_size))
        {
            mz_zip_array_clear(pZip, &new_ext_block);
            mz_zip_array_resize(pZip, &pState->m_central_dir, orig_central_dir_size, MZ_FALSE);
            return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
        }
        
        // 将源中央目录头的注释部分推入中央目录数组，如果失败则清理资源并返回分配失败错误
        if (!mz_zip_array_push_back(pZip, &pState->m_central_dir, pSrc_central_header + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE + src_filename_len + src_ext_len, src_comment_len))
        {
            mz_zip_array_clear(pZip, &new_ext_block);
            mz_zip_array_resize(pZip, &pState->m_central_dir, orig_central_dir_size, MZ_FALSE);
            return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
        }
        
        // 清理新扩展块的资源
        mz_zip_array_clear(pZip, &new_ext_block);
    }
    else
    {
        /* 检查是否超过最大文件偏移量限制 */
        if (cur_dst_file_ofs > MZ_UINT32_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_ARCHIVE_TOO_LARGE);
    
        /* 检查本地目录头偏移量是否超过最大限制 */
        if (local_dir_header_ofs >= MZ_UINT32_MAX)
            return mz_zip_set_error(pZip, MZ_ZIP_ARCHIVE_TOO_LARGE);
    
        /* 将本地目录头偏移量写入新的中央目录头部 */
        MZ_WRITE_LE32(new_central_header + MZ_ZIP_CDH_LOCAL_HEADER_OFS, local_dir_header_ofs);
    
        /* 将新的中央目录头部添加到中央目录数组 */
        if (!mz_zip_array_push_back(pZip, &pState->m_central_dir, new_central_header, MZ_ZIP_CENTRAL_DIR_HEADER_SIZE))
            return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
    
        /* 将源文件后续数据添加到中央目录数组 */
        if (!mz_zip_array_push_back(pZip, &pState->m_central_dir, pSrc_central_header + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE, src_central_dir_following_data_size))
        {
            /* 如果失败，回滚中央目录数组的大小并报内存分配错误 */
            mz_zip_array_resize(pZip, &pState->m_central_dir, orig_central_dir_size, MZ_FALSE);
            return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
        }
    }
    
    /* 如果中央目录数组大小超过最大限制，报不支持的中央目录大小 */
    if (pState->m_central_dir.m_size >= MZ_UINT32_MAX)
    {
        /* TODO: 支持大于等于32位大小的中央目录 */
        mz_zip_array_resize(pZip, &pState->m_central_dir, orig_central_dir_size, MZ_FALSE);
        return mz_zip_set_error(pZip, MZ_ZIP_UNSUPPORTED_CDIR_SIZE);
    }
    
    /* 将原始中央目录大小转换为32位无符号整数，并添加到中央目录偏移量数组 */
    n = (mz_uint32)orig_central_dir_size;
    if (!mz_zip_array_push_back(pZip, &pState->m_central_dir_offsets, &n, 1))
    {
        /* 如果失败，回滚中央目录数组的大小并报内存分配错误 */
        mz_zip_array_resize(pZip, &pState->m_central_dir, orig_central_dir_size, MZ_FALSE);
        return mz_zip_set_error(pZip, MZ_ZIP_ALLOC_FAILED);
    }
    
    /* 增加 ZIP 文件的总文件数 */
    pZip->m_total_files++;
    /* 更新 ZIP 文件的总大小 */
    pZip->m_archive_size = cur_dst_file_ofs;
    
    /* 操作成功返回真 */
    return MZ_TRUE;
}

mz_bool mz_zip_writer_finalize_archive(mz_zip_archive *pZip)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_zip_internal_state *pState;   // 定义指向内部状态结构体的指针变量 pState

    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint64 central_dir_ofs, central_dir_size;   // 定义两个 mz_uint64 类型的变量 central_dir_ofs 和 central_dir_size
    mz_uint8 hdr[256];   // 定义一个长度为 256 的 mz_uint8 类型数组 hdr，用于存储头部信息

    // 检查参数有效性和 ZIP 模式是否为写入模式
    if ((!pZip) || (!pZip->m_pState) || (pZip->m_zip_mode != MZ_ZIP_MODE_WRITING))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    pState = pZip->m_pState;   // 获取 ZIP 结构体中的内部状态指针 pState

    // 如果需要 ZIP64 格式
    if (pState->m_zip64)
    {
        // 检查文件总数或中央目录大小是否超过 32 位无符号整数的最大值
        if ((pZip->m_total_files > MZ_UINT32_MAX) || (pState->m_central_dir.m_size >= MZ_UINT32_MAX))
            return mz_zip_set_error(pZip, MZ_ZIP_TOO_MANY_FILES);
    }
    else
    {
        // 如果不需要 ZIP64 格式，检查文件总数或整个归档大小是否超过 16 位无符号整数的最大值
        if ((pZip->m_total_files > MZ_UINT16_MAX) || ((pZip->m_archive_size + pState->m_central_dir.m_size + MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE) > MZ_UINT32_MAX))
            return mz_zip_set_error(pZip, MZ_ZIP_TOO_MANY_FILES);
    }

    central_dir_ofs = 0;   // 初始化中央目录偏移量为 0
    central_dir_size = 0;   // 初始化中央目录大小为 0
    if (pZip->m_total_files)
    {
        /* Write central directory */
        // 写入中央目录信息
        central_dir_ofs = pZip->m_archive_size;   // 设置中央目录的起始偏移量为当前归档大小
        central_dir_size = pState->m_central_dir.m_size;   // 设置中央目录的大小为内部状态结构体中的中央目录大小
        pZip->m_central_directory_file_ofs = central_dir_ofs;   // 更新 ZIP 结构体中的中央目录文件偏移量

        // 调用写入函数将中央目录数据写入到文件中，如果写入失败则返回错误
        if (pZip->m_pWrite(pZip->m_pIO_opaque, central_dir_ofs, pState->m_central_dir.m_p, (size_t)central_dir_size) != central_dir_size)
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);

        // 更新归档大小，加上中央目录的大小
        pZip->m_archive_size += central_dir_size;
    }

    // 如果需要 ZIP64 格式，则继续处理 ZIP64 相关信息
    if (pState->m_zip64)
    {
        /* 写入zip64中央目录结束头部 */
    
        // 记录相对偏移量到zip64结束中央目录记录
        mz_uint64 rel_ofs_to_zip64_ecdr = pZip->m_archive_size;
    
        // 清空hdr结构体
        MZ_CLEAR_OBJ(hdr);
        // 写入zip64结束中央目录头部的标识
        MZ_WRITE_LE32(hdr + MZ_ZIP64_ECDH_SIG_OFS, MZ_ZIP64_END_OF_CENTRAL_DIR_HEADER_SIG);
        // 写入zip64结束中央目录头部的记录大小
        MZ_WRITE_LE64(hdr + MZ_ZIP64_ECDH_SIZE_OF_RECORD_OFS, MZ_ZIP64_END_OF_CENTRAL_DIR_HEADER_SIZE - sizeof(mz_uint32) - sizeof(mz_uint64));
        // 写入zip64结束中央目录头部的创建版本（始终为Unix）
        MZ_WRITE_LE16(hdr + MZ_ZIP64_ECDH_VERSION_MADE_BY_OFS, 0x031E); /* TODO: always Unix */
        // 写入zip64结束中央目录头部的所需版本
        MZ_WRITE_LE16(hdr + MZ_ZIP64_ECDH_VERSION_NEEDED_OFS, 0x002D);
        // 写入zip64结束中央目录头部的磁盘上条目数
        MZ_WRITE_LE64(hdr + MZ_ZIP64_ECDH_CDIR_NUM_ENTRIES_ON_DISK_OFS, pZip->m_total_files);
        // 写入zip64结束中央目录头部的总条目数
        MZ_WRITE_LE64(hdr + MZ_ZIP64_ECDH_CDIR_TOTAL_ENTRIES_OFS, pZip->m_total_files);
        // 写入zip64结束中央目录头部的中央目录大小
        MZ_WRITE_LE64(hdr + MZ_ZIP64_ECDH_CDIR_SIZE_OFS, central_dir_size);
        // 写入zip64结束中央目录头部的中央目录偏移量
        MZ_WRITE_LE64(hdr + MZ_ZIP64_ECDH_CDIR_OFS_OFS, central_dir_ofs);
    
        // 如果写入操作失败，则返回文件写入失败错误
        if (pZip->m_pWrite(pZip->m_pIO_opaque, pZip->m_archive_size, hdr, MZ_ZIP64_END_OF_CENTRAL_DIR_HEADER_SIZE) != MZ_ZIP64_END_OF_CENTRAL_DIR_HEADER_SIZE)
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);
    
        // 增加存档大小
        pZip->m_archive_size += MZ_ZIP64_END_OF_CENTRAL_DIR_HEADER_SIZE;
    
        /* 写入zip64中央目录结束定位器 */
    
        // 清空hdr结构体
        MZ_CLEAR_OBJ(hdr);
        // 写入zip64中央目录结束定位器的标识
        MZ_WRITE_LE32(hdr + MZ_ZIP64_ECDL_SIG_OFS, MZ_ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIG);
        // 写入zip64中央目录结束定位器的相对偏移量到zip64结束中央目录记录
        MZ_WRITE_LE64(hdr + MZ_ZIP64_ECDL_REL_OFS_TO_ZIP64_ECDR_OFS, rel_ofs_to_zip64_ecdr);
        // 写入zip64中央目录结束定位器的总磁盘数
        MZ_WRITE_LE32(hdr + MZ_ZIP64_ECDL_TOTAL_NUMBER_OF_DISKS_OFS, 1);
    
        // 如果写入操作失败，则返回文件写入失败错误
        if (pZip->m_pWrite(pZip->m_pIO_opaque, pZip->m_archive_size, hdr, MZ_ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIZE) != MZ_ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIZE)
            return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);
    
        // 增加存档大小
        pZip->m_archive_size += MZ_ZIP64_END_OF_CENTRAL_DIR_LOCATOR_SIZE;
    }
    
    /* 写入中央目录结束记录 */
    
    // 清空hdr结构体
    MZ_CLEAR_OBJ(hdr);
    // 写入中央目录结束记录的标识
    MZ_WRITE_LE32(hdr + MZ_ZIP_ECDH_SIG_OFS, MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIG);
    // 写入中央目录结束记录的磁盘上条目数（限制为16位整数的最大值）
    MZ_WRITE_LE16(hdr + MZ_ZIP_ECDH_CDIR_NUM_ENTRIES_ON_DISK_OFS, MZ_MIN(MZ_UINT16_MAX, pZip->m_total_files));
    // 写入中央目录结束记录的总条目数（限制为16位整数的最大值）
    MZ_WRITE_LE16(hdr + MZ_ZIP_ECDH_CDIR_TOTAL_ENTRIES_OFS, MZ_MIN(MZ_UINT16_MAX, pZip->m_total_files));
    // 写入中央目录结束记录的中央目录大小（限制为32位整数的最大值）
    MZ_WRITE_LE32(hdr + MZ_ZIP_ECDH_CDIR_SIZE_OFS, MZ_MIN(MZ_UINT32_MAX, central_dir_size));
    // 写入中央目录结束记录的中央目录偏移量（限制为32位整数的最大值）
    MZ_WRITE_LE32(hdr + MZ_ZIP_ECDH_CDIR_OFS_OFS, MZ_MIN(MZ_UINT32_MAX, central_dir_ofs));
    
    // 如果写入操作失败，则返回文件写入失败错误
    if (pZip->m_pWrite(pZip->m_pIO_opaque, pZip->m_archive_size, hdr, MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE) != MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE)
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_WRITE_FAILED);
#ifndef MINIZ_NO_STDIO
    // 如果已定义 MINIZ_NO_STDIO 宏，则跳过下面的代码块
    if ((pState->m_pFile) && (MZ_FFLUSH(pState->m_pFile) == EOF))
        // 如果文件指针存在且刷新文件流失败，则返回文件关闭失败的错误状态
        return mz_zip_set_error(pZip, MZ_ZIP_FILE_CLOSE_FAILED);
#endif /* #ifndef MINIZ_NO_STDIO */

    // 增加 ZIP 存档的总大小，加上结束中央目录头的大小
    pZip->m_archive_size += MZ_ZIP_END_OF_CENTRAL_DIR_HEADER_SIZE;

    // 将 ZIP 模式设置为写入已经完成
    pZip->m_zip_mode = MZ_ZIP_MODE_WRITING_HAS_BEEN_FINALIZED;
    // 返回操作成功状态
    return MZ_TRUE;
}

// 最终化堆存档的 ZIP 写入器
mz_bool mz_zip_writer_finalize_heap_archive(mz_zip_archive *pZip, void **ppBuf, size_t *pSize)
{
    // 如果传入的缓冲区指针或大小指针为空，则返回无效参数错误
    if ((!ppBuf) || (!pSize))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 将传出的缓冲区指针和大小初始化为零
    *ppBuf = NULL;
    *pSize = 0;

    // 如果 ZIP 存档或其状态为空，则返回无效参数错误
    if ((!pZip) || (!pZip->m_pState))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 如果写入函数不是堆写入函数，则返回无效参数错误
    if (pZip->m_pWrite != mz_zip_heap_write_func)
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 如果最终化 ZIP 存档失败，则返回操作失败状态
    if (!mz_zip_writer_finalize_archive(pZip))
        return MZ_FALSE;

    // 将传出的缓冲区指针设置为存档状态的内存指针，大小为存档状态的内存大小
    *ppBuf = pZip->m_pState->m_pMem;
    *pSize = pZip->m_pState->m_mem_size;
    // 清空存档状态的内存指针，并将大小和容量都设置为零
    pZip->m_pState->m_pMem = NULL;
    pZip->m_pState->m_mem_size = pZip->m_pState->m_mem_capacity = 0;

    // 返回操作成功状态
    return MZ_TRUE;
}

// 结束 ZIP 写入器
mz_bool mz_zip_writer_end(mz_zip_archive *pZip)
{
    // 调用内部函数结束 ZIP 写入器，并返回操作状态
    return mz_zip_writer_end_internal(pZip, MZ_TRUE);
}

#ifndef MINIZ_NO_STDIO
// 将内存添加到原地存档文件的 ZIP
mz_bool mz_zip_add_mem_to_archive_file_in_place(const char *pZip_filename, const char *pArchive_name, const void *pBuf, size_t buf_size, const void *pComment, mz_uint16 comment_size, mz_uint level_and_flags)
{
    // 调用带错误处理的 v2 版本函数，并返回其结果
    return mz_zip_add_mem_to_archive_file_in_place_v2(pZip_filename, pArchive_name, pBuf, buf_size, pComment, comment_size, level_and_flags, NULL);
}

// 带错误处理的 v2 版本将内存添加到原地存档文件的 ZIP
mz_bool mz_zip_add_mem_to_archive_file_in_place_v2(const char *pZip_filename, const char *pArchive_name, const void *pBuf, size_t buf_size, const void *pComment, mz_uint16 comment_size, mz_uint level_and_flags, mz_zip_error *pErr)
{
    // 定义状态和是否创建新存档的标志，并初始化 ZIP 存档结构体和文件状态结构体
    mz_bool status, created_new_archive = MZ_FALSE;
    mz_zip_archive zip_archive;
    struct MZ_FILE_STAT_STRUCT file_stat;
    mz_zip_error actual_err = MZ_ZIP_NO_ERROR;

    // 将 ZIP 存档结构体清零
    mz_zip_zero_struct(&zip_archive);
    // 如果传入的压缩级别小于零，则使用默认级别
    if ((int)level_and_flags < 0)
        level_and_flags = MZ_DEFAULT_LEVEL;

    // 检查必须的参数和条件，如果有无效的参数或条件不满足，则设置错误并返回操作失败状态
    if ((!pZip_filename) || (!pArchive_name) || ((buf_size) && (!pBuf)) || ((comment_size) && (!pComment)) || ((level_and_flags & 0xF) > MZ_UBER_COMPRESSION))
    {
        if (pErr)
            *pErr = MZ_ZIP_INVALID_PARAMETER;
        return MZ_FALSE;
    }

    // 验证存档文件名的有效性，如果无效则设置错误并返回操作失败状态
    if (!mz_zip_writer_validate_archive_name(pArchive_name))
    {
        if (pErr)
            *pErr = MZ_ZIP_INVALID_FILENAME;
        return MZ_FALSE;
    }

    // 使用 stat 获取文件的信息，如果失败则设置错误并返回操作失败状态
    /* Important: The regular non-64 bit version of stat() can fail here if the file is very large, which could cause the archive to be overwritten. */
    /* So be sure to compile with _LARGEFILE64_SOURCE 1 */
    if (MZ_FILE_STAT(pZip_filename, &file_stat) != 0)
        // 设置实际错误并返回操作失败状态
        actual_err = MZ_ZIP_CANNOT_OPEN_FILE;
    {
        /* 创建一个新的归档文件。 */
        if (!mz_zip_writer_init_file_v2(&zip_archive, pZip_filename, 0, level_and_flags))
        {
            /* 如果初始化失败，设置错误信息并返回失败状态。 */
            if (pErr)
                *pErr = zip_archive.m_last_error;
            return MZ_FALSE;
        }
    
        created_new_archive = MZ_TRUE;
    }
    else
    {
        /* 追加到现有的归档文件。 */
        if (!mz_zip_reader_init_file_v2(&zip_archive, pZip_filename, level_and_flags | MZ_ZIP_FLAG_DO_NOT_SORT_CENTRAL_DIRECTORY, 0, 0))
        {
            /* 如果初始化读取器失败，设置错误信息并返回失败状态。 */
            if (pErr)
                *pErr = zip_archive.m_last_error;
            return MZ_FALSE;
        }
    
        /* 尝试从现有归档文件初始化写入器。 */
        if (!mz_zip_writer_init_from_reader_v2(&zip_archive, pZip_filename, level_and_flags))
        {
            /* 如果初始化写入器失败，设置错误信息，结束读取器，并返回失败状态。 */
            if (pErr)
                *pErr = zip_archive.m_last_error;
    
            mz_zip_reader_end_internal(&zip_archive, MZ_FALSE);
    
            return MZ_FALSE;
        }
    }
    
    /* 向归档文件添加内存中的数据。 */
    status = mz_zip_writer_add_mem_ex(&zip_archive, pArchive_name, pBuf, buf_size, pComment, comment_size, level_and_flags, 0, 0);
    actual_err = zip_archive.m_last_error;
    
    /* 总是完成归档操作，即使添加失败，确保有一个有效的中央目录。（这可能不总是成功，但我们尝试一下。） */
    if (!mz_zip_writer_finalize_archive(&zip_archive))
    {
        /* 如果最终化归档失败，记录错误信息。 */
        if (!actual_err)
            actual_err = zip_archive.m_last_error;
    
        status = MZ_FALSE;
    }
    
    /* 结束归档操作，并处理结束内部操作。 */
    if (!mz_zip_writer_end_internal(&zip_archive, status))
    {
        /* 如果结束内部操作失败，记录错误信息。 */
        if (!actual_err)
            actual_err = zip_archive.m_last_error;
    
        status = MZ_FALSE;
    }
    
    /* 如果操作失败且是新建的归档文件，删除该文件。 */
    if ((!status) && (created_new_archive))
    {
        /* 是一个新建的归档文件，但由于某些原因失败，所以删除它。 */
        int ignoredStatus = MZ_DELETE_FILE(pZip_filename);
        (void)ignoredStatus;
    }
    
    /* 如果指定了错误信息指针，将实际错误信息返回。 */
    if (pErr)
        *pErr = actual_err;
    
    /* 返回操作的最终状态。 */
    return status;
}

// 从指定 ZIP 文件中提取指定存档名字的内容到堆中，并返回指针
void *mz_zip_extract_archive_file_to_heap_v2(const char *pZip_filename, const char *pArchive_name, const char *pComment, size_t *pSize, mz_uint flags, mz_zip_error *pErr)
{
    // 文件索引
    mz_uint32 file_index;
    // ZIP 归档对象
    mz_zip_archive zip_archive;
    // 初始化指针为 NULL
    void *p = NULL;

    // 如果传入了 pSize，则将其设置为 0
    if (pSize)
        *pSize = 0;

    // 如果 pZip_filename 或 pArchive_name 为空，则返回无效参数错误
    if ((!pZip_filename) || (!pArchive_name))
    {
        // 如果传入了 pErr，则设置错误类型为无效参数
        if (pErr)
            *pErr = MZ_ZIP_INVALID_PARAMETER;

        return NULL;
    }

    // 将 zip_archive 结构体清零
    mz_zip_zero_struct(&zip_archive);
    // 使用文件初始化 ZIP 读取器，指定不排序中央目录
    if (!mz_zip_reader_init_file_v2(&zip_archive, pZip_filename, flags | MZ_ZIP_FLAG_DO_NOT_SORT_CENTRAL_DIRECTORY, 0, 0))
    {
        // 如果传入了 pErr，则设置错误类型为 ZIP 归档的最后一个错误
        if (pErr)
            *pErr = zip_archive.m_last_error;

        return NULL;
    }

    // 定位指定文件名和注释的文件在 ZIP 归档中的位置
    if (mz_zip_reader_locate_file_v2(&zip_archive, pArchive_name, pComment, flags, &file_index))
    {
        // 提取文件到堆中，并返回指向数据的指针，同时更新文件大小到 pSize
        p = mz_zip_reader_extract_to_heap(&zip_archive, file_index, pSize, flags);
    }

    // 结束 ZIP 读取器的内部操作，根据 p 是否为空来决定是否解压成功
    mz_zip_reader_end_internal(&zip_archive, p != NULL);

    // 如果传入了 pErr，则设置错误类型为 ZIP 归档的最后一个错误
    if (pErr)
        *pErr = zip_archive.m_last_error;

    return p;
}

// 从指定 ZIP 文件中提取指定存档名字的内容到堆中，并返回指针（简化版本，不带注释）
void *mz_zip_extract_archive_file_to_heap(const char *pZip_filename, const char *pArchive_name, size_t *pSize, mz_uint flags)
{
    return mz_zip_extract_archive_file_to_heap_v2(pZip_filename, pArchive_name, NULL, pSize, flags, NULL);
}

#endif /* #ifndef MINIZ_NO_STDIO */

#endif /* #ifndef MINIZ_NO_ARCHIVE_WRITING_APIS */

/* ------------------- Misc utils */

// 获取 ZIP 归档的模式
mz_zip_mode mz_zip_get_mode(mz_zip_archive *pZip)
{
    return pZip ? pZip->m_zip_mode : MZ_ZIP_MODE_INVALID;
}

// 获取 ZIP 归档的类型
mz_zip_type mz_zip_get_type(mz_zip_archive *pZip)
{
    return pZip ? pZip->m_zip_type : MZ_ZIP_TYPE_INVALID;
}

// 设置 ZIP 归档的最后一个错误，并返回先前的错误
mz_zip_error mz_zip_set_last_error(mz_zip_archive *pZip, mz_zip_error err_num)
{
    // 上一个错误
    mz_zip_error prev_err;

    // 如果 pZip 为空，则返回无效参数错误
    if (!pZip)
        return MZ_ZIP_INVALID_PARAMETER;

    // 保存当前的错误到 prev_err
    prev_err = pZip->m_last_error;

    // 设置新的错误类型到 pZip
    pZip->m_last_error = err_num;
    return prev_err;
}

// 获取 ZIP 归档的最后一个错误
mz_zip_error mz_zip_peek_last_error(mz_zip_archive *pZip)
{
    // 如果 pZip 为空，则返回无效参数错误
    if (!pZip)
        return MZ_ZIP_INVALID_PARAMETER;

    return pZip->m_last_error;
}

// 清除 ZIP 归档的最后一个错误
mz_zip_error mz_zip_clear_last_error(mz_zip_archive *pZip)
{
    return mz_zip_set_last_error(pZip, MZ_ZIP_NO_ERROR);
}

// 获取 ZIP 归档的最后一个错误，并将其重置为无错误状态
mz_zip_error mz_zip_get_last_error(mz_zip_archive *pZip)
{
    // 上一个错误
    mz_zip_error prev_err;

    // 如果 pZip 为空，则返回无效参数错误
    if (!pZip)
        return MZ_ZIP_INVALID_PARAMETER;

    // 保存当前的错误到 prev_err
    prev_err = pZip->m_last_error;

    // 将 pZip 的错误状态重置为无错误状态
    pZip->m_last_error = MZ_ZIP_NO_ERROR;
    return prev_err;
}

// 获取指定 ZIP 错误类型对应的错误字符串
const char *mz_zip_get_error_string(mz_zip_error mz_err)
{
    switch (mz_err)
    {
        // 根据返回的错误码选择对应的错误信息并返回
        case MZ_ZIP_NO_ERROR:
            return "no error";
        case MZ_ZIP_UNDEFINED_ERROR:
            return "undefined error";
        case MZ_ZIP_TOO_MANY_FILES:
            return "too many files";
        case MZ_ZIP_FILE_TOO_LARGE:
            return "file too large";
        case MZ_ZIP_UNSUPPORTED_METHOD:
            return "unsupported method";
        case MZ_ZIP_UNSUPPORTED_ENCRYPTION:
            return "unsupported encryption";
        case MZ_ZIP_UNSUPPORTED_FEATURE:
            return "unsupported feature";
        case MZ_ZIP_FAILED_FINDING_CENTRAL_DIR:
            return "failed finding central directory";
        case MZ_ZIP_NOT_AN_ARCHIVE:
            return "not a ZIP archive";
        case MZ_ZIP_INVALID_HEADER_OR_CORRUPTED:
            return "invalid header or archive is corrupted";
        case MZ_ZIP_UNSUPPORTED_MULTIDISK:
            return "unsupported multidisk archive";
        case MZ_ZIP_DECOMPRESSION_FAILED:
            return "decompression failed or archive is corrupted";
        case MZ_ZIP_COMPRESSION_FAILED:
            return "compression failed";
        case MZ_ZIP_UNEXPECTED_DECOMPRESSED_SIZE:
            return "unexpected decompressed size";
        case MZ_ZIP_CRC_CHECK_FAILED:
            return "CRC-32 check failed";
        case MZ_ZIP_UNSUPPORTED_CDIR_SIZE:
            return "unsupported central directory size";
        case MZ_ZIP_ALLOC_FAILED:
            return "allocation failed";
        case MZ_ZIP_FILE_OPEN_FAILED:
            return "file open failed";
        case MZ_ZIP_FILE_CREATE_FAILED:
            return "file create failed";
        case MZ_ZIP_FILE_WRITE_FAILED:
            return "file write failed";
        case MZ_ZIP_FILE_READ_FAILED:
            return "file read failed";
        case MZ_ZIP_FILE_CLOSE_FAILED:
            return "file close failed";
        case MZ_ZIP_FILE_SEEK_FAILED:
            return "file seek failed";
        case MZ_ZIP_FILE_STAT_FAILED:
            return "file stat failed";
        case MZ_ZIP_INVALID_PARAMETER:
            return "invalid parameter";
        case MZ_ZIP_INVALID_FILENAME:
            return "invalid filename";
        case MZ_ZIP_BUF_TOO_SMALL:
            return "buffer too small";
        case MZ_ZIP_INTERNAL_ERROR:
            return "internal error";
        case MZ_ZIP_FILE_NOT_FOUND:
            return "file not found";
        case MZ_ZIP_ARCHIVE_TOO_LARGE:
            return "archive is too large";
        case MZ_ZIP_VALIDATION_FAILED:
            return "validation failed";
        case MZ_ZIP_WRITE_CALLBACK_FAILED:
            return "write calledback failed";
        default:
            break;
    }
    
    // 如果错误码未匹配到任何已知错误，则返回未知错误
    return "unknown error";
/* 注意：这个函数用来检查给定的 mz_zip_archive 结构体是否启用了 Zip64 扩展信息字段。*/
mz_bool mz_zip_is_zip64(mz_zip_archive *pZip)
{
    // 如果传入的指针为空或者指向的状态为空，则返回 MZ_FALSE
    if ((!pZip) || (!pZip->m_pState))
        return MZ_FALSE;

    // 返回结构体中的 m_zip64 字段，表示是否使用了 Zip64 扩展
    return pZip->m_pState->m_zip64;
}

/* 注意：这个函数用来获取中央目录的大小。*/
size_t mz_zip_get_central_dir_size(mz_zip_archive *pZip)
{
    // 如果传入的指针为空或者指向的状态为空，则返回 0
    if ((!pZip) || (!pZip->m_pState))
        return 0;

    // 返回中央目录的大小，存储在结构体的 m_central_dir.m_size 中
    return pZip->m_pState->m_central_dir.m_size;
}

/* 注意：这个函数用来获取 ZIP 文件中的文件数量。*/
mz_uint mz_zip_reader_get_num_files(mz_zip_archive *pZip)
{
    // 如果传入的指针为空，则返回 0；否则返回结构体中的 m_total_files 字段
    return pZip ? pZip->m_total_files : 0;
}

/* 注意：这个函数用来获取整个 ZIP 文件的大小。*/
mz_uint64 mz_zip_get_archive_size(mz_zip_archive *pZip)
{
    // 如果传入的指针为空，则返回 0；否则返回结构体中的 m_archive_size 字段
    if (!pZip)
        return 0;
    return pZip->m_archive_size;
}

/* 注意：这个函数用来获取 ZIP 文件的起始偏移量。*/
mz_uint64 mz_zip_get_archive_file_start_offset(mz_zip_archive *pZip)
{
    // 如果传入的指针为空或者指向的状态为空，则返回 0
    if ((!pZip) || (!pZip->m_pState))
        return 0;
    
    // 返回文件存档起始偏移量，存储在结构体的 m_file_archive_start_ofs 中
    return pZip->m_pState->m_file_archive_start_ofs;
}

/* 注意：这个函数用来获取 ZIP 文件的底层文件句柄。*/
MZ_FILE *mz_zip_get_cfile(mz_zip_archive *pZip)
{
    // 如果传入的指针为空或者指向的状态为空，则返回 0
    if ((!pZip) || (!pZip->m_pState))
        return 0;

    // 返回结构体中的 m_pFile 字段，表示 ZIP 文件的底层文件句柄
    return pZip->m_pState->m_pFile;
}

/* 注意：这个函数用来从 ZIP 文件中读取数据。*/
size_t mz_zip_read_archive_data(mz_zip_archive *pZip, mz_uint64 file_ofs, void *pBuf, size_t n)
{
    // 如果传入的指针为空，或者指向的状态为空，或者缓冲区为空，或者读取函数为空，则返回一个错误代码
    if ((!pZip) || (!pZip->m_pState) || (!pBuf) || (!pZip->m_pRead))
        return mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);

    // 调用读取函数读取指定位置的数据，并返回实际读取的字节数
    return pZip->m_pRead(pZip->m_pIO_opaque, file_ofs, pBuf, n);
}

/* 注意：这个函数用来获取 ZIP 文件中指定索引的文件名。*/
mz_uint mz_zip_reader_get_filename(mz_zip_archive *pZip, mz_uint file_index, char *pFilename, mz_uint filename_buf_size)
{
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    mz_uint n;
    // 获取中央目录头部的指针
    const mz_uint8 *p = mz_zip_get_cdh(pZip, file_index);
    // 如果指针为空，则表示参数无效，设置错误并返回 0
    if (!p)
    {
        if (filename_buf_size)
            pFilename[0] = '\0';
        mz_zip_set_error(pZip, MZ_ZIP_INVALID_PARAMETER);
        return 0;
    }
    // 从中央目录头部中读取文件名的长度
    n = MZ_READ_LE16(p + MZ_ZIP_CDH_FILENAME_LEN_OFS);
    // 如果指定了文件名缓冲区，将文件名拷贝到缓冲区中
    if (filename_buf_size)
    {
        n = MZ_MIN(n, filename_buf_size - 1);
        memcpy(pFilename, p + MZ_ZIP_CENTRAL_DIR_HEADER_SIZE, n);
        pFilename[n] = '\0';
    }
    // 返回文件名的长度加一（包括字符串末尾的空字符）
    return n + 1;
}

/* 注意：这个函数用来获取 ZIP 文件中指定索引的文件的详细信息。*/
mz_bool mz_zip_reader_file_stat(mz_zip_archive *pZip, mz_uint file_index, mz_zip_archive_file_stat *pStat)
{
    // 调用内部函数获取文件状态，传入 ZIP 结构体、文件索引、中央目录头部指针和文件状态结构体指针
    return mz_zip_file_stat_internal(pZip, file_index, mz_zip_get_cdh(pZip, file_index), pStat, NULL);
}

/* 注意：这个函数用来结束 ZIP 操作，释放相关资源。*/
mz_bool mz_zip_end(mz_zip_archive *pZip)
{
    // 如果传入的指针为空，则返回 MZ_FALSE
    if (!pZip)
        return MZ_FALSE;

    // 如果 ZIP 模式为读取模式，则调用读取结束函数；如果支持写入，则调用写入结束函数；否则返回 MZ_FALSE
    if (pZip->m_zip_mode == MZ_ZIP_MODE_READING)
        return mz_zip_reader_end(pZip);
#ifndef MINIZ_NO_ARCHIVE_WRITING_APIS
    else if ((pZip->m_zip_mode == MZ_ZIP_MODE_WRITING) || (pZip->m_zip_mode == MZ_ZIP_MODE_WRITING_HAS_BEEN_FINALIZED))
        return mz_zip_writer_end(pZip);
#endif

    return MZ_FALSE;
}

#ifdef __cplusplus
}
#endif

#endif /*#ifndef MINIZ_NO_ARCHIVE_APIS*/
```