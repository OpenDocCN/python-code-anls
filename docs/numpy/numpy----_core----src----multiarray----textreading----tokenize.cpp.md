# `.\numpy\numpy\_core\src\multiarray\textreading\tokenize.cpp`

```
/*
    How parsing quoted fields works:

    For quoting to be activated, the first character of the field
    must be the quote character (after taking into account
    ignore_leading_spaces).  While quoting is active, delimiters
    are treated as regular characters, not delimiters.  Quoting is
    deactivated by the second occurrence of the quote character.  An
    exception is the occurrence of two consecutive quote characters,
    which is treated as a literal occurrence of a single quote character.
    E.g. (with delimiter=',' and quote='"'):
        12.3,"New York, NY","3'2"""
    The second and third fields are `New York, NY` and `3'2"`.

    If a non-delimiter occurs after the closing quote, the quote is
    ignored and parsing continues with quoting deactivated.  Quotes
    that occur while quoting is not activated are not handled specially;
    they become part of the data.
    E.g:
        12.3,"ABC"DEF,XY"Z
    The second and third fields are `ABCDEF` and `XY"Z`.

    Note that the second field of
        12.3,"ABC"   ,4.5
    is `ABC   `.  Currently there is no option to ignore whitespace
    at the end of a field.
*/

template <typename UCS>
static inline int
copy_to_field_buffer(tokenizer_state *ts,
        const UCS *chunk_start, const UCS *chunk_end)
{
    // 计算块的长度
    npy_intp chunk_length = chunk_end - chunk_start;
    // 计算需要的内存大小，包括长度和终止符，以及额外的空间
    npy_intp size = chunk_length + ts->field_buffer_pos + 3;

    // 如果当前分配的缓冲区不足以容纳新的数据
    if (NPY_UNLIKELY(ts->field_buffer_length < size)) {
        // 计算需要增长的内存大小
        npy_intp alloc_size = grow_size_and_multiply(&size, 32, sizeof(Py_UCS4));
        // 如果增长大小为负数，表示无法处理如此长的行
        if (alloc_size < 0) {
            PyErr_Format(PyExc_ValueError,
                    "line too long to handle while reading file.");
            return -1;
        }
        // 重新分配内存
        Py_UCS4 *grown = (Py_UCS4 *)PyMem_Realloc(ts->field_buffer, alloc_size);
        // 如果分配失败
        if (grown == nullptr) {
            PyErr_NoMemory();
            return -1;
        }
        // 更新缓冲区长度和指针
        ts->field_buffer_length = size;
        ts->field_buffer = grown;
    }

    // 将块数据复制到字段缓冲区中
    Py_UCS4 *write_pos = ts->field_buffer + ts->field_buffer_pos;
    for (; chunk_start < chunk_end; chunk_start++, write_pos++) {
        *write_pos = (Py_UCS4)*chunk_start;
    }
    // 确保以NUL结尾
    *write_pos = '\0';
    // 更新字段缓冲区位置
    ts->field_buffer_pos += chunk_length;
    return 0;
}

static inline int
add_field(tokenizer_state *ts)
{
    // 上一个字段已完成，增加一个NUL字节作为结束符
    ts->field_buffer_pos += 1;
    # 检查字段数量是否超过当前数组大小，如果是则需要扩展数组
    if (NPY_UNLIKELY(ts->num_fields + 1 > ts->fields_size)) {
        # 记录当前字段数量
        npy_intp size = ts->num_fields;

        # 计算需要分配的新数组大小
        npy_intp alloc_size = grow_size_and_multiply(
                &size, 4, sizeof(field_info));
        
        # 检查分配大小是否小于0，通常情况下不可能发生
        if (alloc_size < 0) {
            # 报错，提示列数过多，无法读取文件
            PyErr_Format(PyExc_ValueError,
                    "too many columns found; cannot read file.");
            return -1;
        }
        
        # 重新分配内存以扩展字段数组
        field_info *fields = (field_info *)PyMem_Realloc(ts->fields, alloc_size);
        
        # 检查内存重新分配是否成功
        if (fields == nullptr) {
            # 内存分配失败，抛出内存错误异常
            PyErr_NoMemory();
            return -1;
        }
        
        # 更新字段数组和大小
        ts->fields = fields;
        ts->fields_size = size;
    }

    # 设置当前字段的偏移量和引用
    ts->fields[ts->num_fields].offset = ts->field_buffer_pos;
    ts->fields[ts->num_fields].quoted = false;
    
    # 增加字段数量
    ts->num_fields += 1;
    
    # 确保当前字段缓冲区的末尾是空字符
    /* Ensure this (currently empty) word is NUL terminated. */
    ts->field_buffer[ts->field_buffer_pos] = '\0';
    
    # 断言当前字段缓冲区长度大于字段位置，确保位置正确
    assert(ts->field_buffer_length > ts->field_buffer_pos);
    
    # 返回操作成功
    return 0;
}

/*
 * tokenizer_core 函数负责实现通用的分词器逻辑，用于处理特定 UCS 类型的文本流
 * 参数 ts: 分词器状态结构体指针，用于保存分词器的状态信息
 * 参数 config: 解析器配置结构体指针，包含配置信息，如是否忽略前导空白字符和引号类型
 */
template <typename UCS>
static inline int
tokenizer_core(tokenizer_state *ts, parser_config *const config)
{
    // 将指针 pos 设置为当前位置 ts->pos 的 UCS 类型指针
    UCS *pos = (UCS *)ts->pos;
    // 将指针 stop 设置为结束位置 ts->end 的 UCS 类型指针
    UCS *stop = (UCS *)ts->end;
    UCS *chunk_start;

    // 如果分词器的状态为 TOKENIZE_CHECK_QUOTED
    if (ts->state == TOKENIZE_CHECK_QUOTED) {
        /* before we can check for quotes, strip leading whitespace */
        // 如果配置要求忽略前导空白字符
        if (config->ignore_leading_whitespace) {
            // 跳过前导空白字符，直到遇到非空白字符、'\r' 或 '\n'
            while (pos < stop && Py_UNICODE_ISSPACE(*pos) &&
                        *pos != '\r' && *pos != '\n') {
                pos++;
            }
            // 如果已经到达结束位置，则更新分词器状态并返回
            if (pos == stop) {
                ts->pos = (char *)pos;
                return 0;
            }
        }

        /* Setting chunk effectively starts the field */
        // 如果当前位置的字符为配置中指定的引号字符
        if (*pos == config->quote) {
            // 标记当前字段为带引号的字段，并设置分词器状态为 TOKENIZE_QUOTED
            ts->fields[ts->num_fields - 1].quoted = true;
            ts->state = TOKENIZE_QUOTED;
            // 移动 pos 指针到下一个位置，即跳过引号字符
            pos++;  /* TOKENIZE_QUOTED is OK with pos == stop */
        }
        else {
            /* Set to TOKENIZE_QUOTED or TOKENIZE_QUOTED_WHITESPACE */
            // 否则根据当前未引用的状态设置分词器状态
            ts->state = ts->unquoted_state;
        }
    }

    // 更新分词器位置指针为当前 pos 指针位置的字符指针
    ts->pos = (char *)pos;
    // 返回 0 表示处理完毕
    return 0;
}

/*
 * 此分词器总是复制完整的 "row"（所有标记）。这样做有两个好处：
 * 1. 确保每个单词后面都有一个 NUL 字符（尽管它也可能包含一个）。
 * 2. 如果使用了 usecols，我们可以通过完全解析第一行更轻松地嗅探出它。此外，usecols 可能是负数，因此我们可能事先不知道需要哪一行。
 *
 * 分词器可以增加跳过字段和在已知情况下检查最大字段数的能力，目前还不清楚这是否值得做。
 *
 * 与一些分词器不同，此分词器尝试按块工作并按块复制数据。希望这样做可以多次轻量循环，而不是单个重型循环，例如快速扫描字段的结束。复制块还意味着我们通常只需检查一次缓冲区是否足够大。
 * 不同的选择是可能的，这个选择似乎效果不错。
 *
 * 分词器的核心部分为三种 Python Unicode 类型 UCS1、UCS2 和 UCS4 进行了专门优化。
 */
NPY_NO_EXPORT int
npy_tokenize(stream *s, tokenizer_state *ts, parser_config *const config)
{
    // 断言确保字段大小至少为 2
    assert(ts->fields_size >= 2);
    // 断言确保字段缓冲区长度至少为两倍 UCS4 的大小
    assert(ts->field_buffer_length >= 2*(Py_ssize_t)sizeof(Py_UCS4));

    // 标记文件读取是否完成的标志，初始为 0 表示未完成
    int finished_reading_file = 0;

    /* Reset to start of buffer */
    // 重置字段缓冲区位置为起始位置
    ts->field_buffer_pos = 0;
    // 重置字段数量为 0
    ts->num_fields = 0;
}
    while (true) {
        /*
         * This loop adds new fields to the result (to make up a full row)
         * until the row ends (typically a line end or the file end)
         */
        // 如果当前状态为TOKENIZE_INIT，表示需要开始一个新字段的处理
        if (ts->state == TOKENIZE_INIT) {
            /* Start a new field */
            // 调用add_field函数开始一个新字段的处理，如果返回小于0则表示出错
            if (add_field(ts) < 0) {
                return -1;
            }
            // 设置状态为TOKENIZE_CHECK_QUOTED，表示需要检查是否为带引号的字段
            ts->state = TOKENIZE_CHECK_QUOTED;
        }

        // 如果当前位置已经超过了缓冲区的末尾
        if (NPY_UNLIKELY(ts->pos >= ts->end)) {
            // 如果缓冲区状态为BUFFER_IS_LINEND并且不处于TOKENIZE_QUOTED状态，表示当前行已结束
            if (ts->buf_state == BUFFER_IS_LINEND &&
                    ts->state != TOKENIZE_QUOTED) {
                /*
                 * Finished line, do not read anymore (also do not eat \n).
                 * If we are in a quoted field and the "line" does not end with
                 * a newline, the quoted field will not have it either.
                 * I.e. `np.loadtxt(['"a', 'b"'], dtype="S2", quotechar='"')`
                 * reads "ab". This matches `next(csv.reader(['"a', 'b"']))`.
                 */
                // 如果在带引号的字段中，且行未以换行符结束，则带引号字段也不会包含换行符
                break;
            }
            /* fetch new data */
            // 获取新的数据块，并更新缓冲区状态
            ts->buf_state = stream_nextbuf(s,
                    &ts->pos, &ts->end, &ts->unicode_kind);
            // 如果获取数据出错，返回-1
            if (ts->buf_state < 0) {
                return -1;
            }
            // 如果已到达文件末尾
            if (ts->buf_state == BUFFER_IS_FILEEND) {
                finished_reading_file = 1;
                ts->pos = ts->end;  /* stream should ensure this. */
                break;
            }
            // 如果当前位置等于缓冲区末尾，表示遇到空行
            else if (ts->pos == ts->end) {
                /* This must be an empty line (and it must be indicated!). */
                assert(ts->buf_state == BUFFER_IS_LINEND);
                break;
            }
        }
        // 根据当前数据块的编码类型选择合适的分词器，并进行分词
        int status;
        if (ts->unicode_kind == PyUnicode_1BYTE_KIND) {
            status = tokenizer_core<Py_UCS1>(ts, config);
        }
        else if (ts->unicode_kind == PyUnicode_2BYTE_KIND) {
            status = tokenizer_core<Py_UCS2>(ts, config);
        }
        else {
            assert(ts->unicode_kind == PyUnicode_4BYTE_KIND);
            status = tokenizer_core<Py_UCS4>(ts, config);
        }
        // 如果分词过程中出错，返回-1
        if (status < 0) {
            return -1;
        }

        // 如果当前状态为TOKENIZE_LINE_END，表示已完成一行的分词
        if (ts->state == TOKENIZE_LINE_END) {
            break;
        }
    }

    /*
     * We have finished tokenizing a full row into fields, finalize result
     */
    // 如果缓冲区状态为BUFFER_IS_LINEND，表示当前行已处理完毕
    if (ts->buf_state == BUFFER_IS_LINEND) {
        /* This line is "finished", make sure we don't touch it again: */
        // 将缓冲区状态更新为BUFFER_MAY_CONTAIN_NEWLINE，避免再次处理当前行
        ts->buf_state = BUFFER_MAY_CONTAIN_NEWLINE;
        // 如果当前位置小于末尾，表示发现未引用的嵌入换行符，抛出异常
        if (NPY_UNLIKELY(ts->pos < ts->end)) {
            PyErr_SetString(PyExc_ValueError,
                    "Found an unquoted embedded newline within a single line of "
                    "input.  This is currently not supported.");
            return -1;
        }
    }

    /* Finish the last field (we "append" one to store the last ones length) */
    // 处理最后一个字段，添加长度信息
    if (add_field(ts) < 0) {
        return -1;
    }
    // 减去最后一个空字段
    ts->num_fields -= 1;
}
    /*
     * 我们总是从新字段开始（从开头开始，以及每次找到分隔符时）。
     * 这给了我们两种情况需要忽略最后一个字段如果它为空：
     * 1. 如果恰好有一个空的（未引用的）字段，整行就是空的。
     * 2. 如果我们在分割空白字符上，我们总是忽略最后一个空字段以匹配Python的分割行为：" 1 ".split()。
     *    （当我们只跳过行时，零个字段是可能的）
     */
    if (ts->num_fields == 1 || (ts->num_fields > 0
                && ts->unquoted_state == TOKENIZE_UNQUOTED_WHITESPACE)) {
        // 获取最后一个字段的偏移量和结束位置
        size_t offset_last = ts->fields[ts->num_fields-1].offset;
        size_t end_last = ts->fields[ts->num_fields].offset;
        // 如果最后一个字段不是引用的且长度为1，则将其忽略
        if (!ts->fields->quoted && end_last - offset_last == 1) {
            ts->num_fields--; // 减少字段计数，忽略最后一个字段
        }
    }
    ts->state = TOKENIZE_INIT; // 设置解析状态为初始状态
    return finished_reading_file; // 返回文件读取完成状态
/*
 * 清理 tokenizer_state 结构体中的资源，包括释放动态分配的内存和重置相关变量。
 */
NPY_NO_EXPORT void
npy_tokenizer_clear(tokenizer_state *ts)
{
    // 释放 field_buffer 指针指向的内存，并将其置为 nullptr
    PyMem_FREE(ts->field_buffer);
    ts->field_buffer = nullptr;

    // 释放 fields 指针指向的内存，并将其置为 nullptr，并重置 fields_size 为 0
    PyMem_FREE(ts->fields);
    ts->fields = nullptr;
    ts->fields_size = 0;
}


/*
 * 初始化 tokenizer_state 结构体，可能会将所有重要的配置变量复制到 tokenizer_state 中，
 * 这样在进行 tokenizing 过程中可以提高缓存的局部性。
 */
NPY_NO_EXPORT int
npy_tokenizer_init(tokenizer_state *ts, parser_config *config)
{
    /* 如果我们按行处理，state 和 buf_state 可能会移到 tokenize 函数中 */
    
    // 初始化 buf_state 为 BUFFER_MAY_CONTAIN_NEWLINE
    ts->buf_state = BUFFER_MAY_CONTAIN_NEWLINE;

    // 初始化 state 为 TOKENIZE_INIT
    ts->state = TOKENIZE_INIT;

    // 根据配置设置 unquoted_state，如果 delimiter_is_whitespace 为真，则设置为 TOKENIZE_UNQUOTED_WHITESPACE，否则设置为 TOKENIZE_UNQUOTED
    if (config->delimiter_is_whitespace) {
        ts->unquoted_state = TOKENIZE_UNQUOTED_WHITESPACE;
    }
    else {
        ts->unquoted_state = TOKENIZE_UNQUOTED;
    }

    // 初始化 num_fields 为 0
    ts->num_fields = 0;

    // 将 buf_state 重置为 0
    ts->buf_state = 0;

    // 初始化 pos 和 end 为 nullptr
    ts->pos = nullptr;
    ts->end = nullptr;

    // 分配并初始化 field_buffer，长度为 32 个 Py_UCS4 字符
    ts->field_buffer = (Py_UCS4 *)PyMem_Malloc(32 * sizeof(Py_UCS4));
    if (ts->field_buffer == nullptr) {
        PyErr_NoMemory();
        return -1;
    }
    ts->field_buffer_length = 32;

    // 分配并初始化 fields 数组，大小为 4 个 field_info 结构体大小
    ts->fields = (field_info *)PyMem_Malloc(4 * sizeof(*ts->fields));
    if (ts->fields == nullptr) {
        // 分配失败时释放已分配的内存，返回内存错误
        PyMem_Free(ts->field_buffer);
        ts->field_buffer = nullptr;
        PyErr_NoMemory();
        return -1;
    }
    ts->fields_size = 4;

    // 初始化成功，返回 0
    return 0;
}
```