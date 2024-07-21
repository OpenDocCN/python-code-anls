# `.\pytorch\caffe2\perfkernels\hp_emblookup_codegen.py`

```
# mypy: allow-untyped-defs

# 导入必要的模块
import argparse  # 导入命令行参数解析模块
import sys  # 导入系统模块

# 定义不同数据类型的大小
sizeof = {"float": 4, "at::Half": 2, "at::BFloat16": 2, "uint8_t": 1}

# 定义函数 unroll，用于展开循环计算
def unroll(uf, IndexType, InType, OutType, use_weights, isa, fused, use_offsets):
    # 定义内部函数 compute，用于生成特定类型数据的计算代码
    def compute(regid, InType, use_weights, isa, prefetch):
        code = []

        # 根据不同的输入数据类型生成不同的计算指令
        if InType == "float":
            code.append(
                "        vop%d = _mm256_fmadd_ps(vwgt, _mm256_loadu_ps(ip + (%d)), vop%d);"  # noqa
                % (regid, regid, regid)
            )
        elif InType == "at::Half":
            code.append(
                "        vop%d = _mm256_fmadd_ps(\n"
                "            vwgt,\n"
                "            _mm256_cvtph_ps(\n"
                "                _mm_loadu_si128(reinterpret_cast<const __m128i*>(ip + (%d)))),\n"  # noqa
                "            vop%d);" % (regid, regid, regid)
            )
        elif InType == "at::BFloat16":
            code.append(
                "        vop%d = _mm256_fmadd_ps(\n"
                "            vwgt,\n"
                "            _mm256_castsi256_ps(_mm256_slli_epi32(\n"
                "                _mm256_cvtepu16_epi32(_mm_loadu_si128(\n"
                "                    reinterpret_cast<const __m128i*>(ip + (%d)))),\n"
                "                16)),\n"  # noqa
                "            vop%d);" % (regid, regid, regid)
            )
        elif InType == "uint8_t":
            code.append(
                "        vop%d = _mm256_fmadd_ps(\n"
                "            vwgt,\n"
                "            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(\n"
                "                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ip + (%d))))),\n"  # noqa
                "            _mm256_add_ps(vop%d, vbio));" % (regid, regid, regid)
            )
        else:
            assert False

        # 如果需要预取数据，则添加预取指令
        if prefetch:
            code.append(
                "        _mm_prefetch(\n"
                "            reinterpret_cast<const char*>(&ip_next_T0[%d]), _MM_HINT_T0);"
                % (regid)
            )
        else:
            code.append(
                "        // 跳过不必要的预取 (&ip_next_T0[%d])" % (regid)
            )

        return code

    code = []
    code.append("    // 展开循环 " + str(uf) + " 次")

    # 根据 use_offsets 参数决定是否使用偏移量
    if use_offsets:
        code.append(
            "    for ("
            + IndexType
            + " rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {"
        )
    else:
        code.append(
            "    for ("
            + IndexType
            + " rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {"
        )

    # 设置输出数组指针 op
    code.append("      " + OutType + "* op = &out[rangeIndex * block_size];")

    # 初始化 SIMD 向量寄存器 vop
    for i in range(0, uf):
        j = 8 * i
        code.append("      __m256 vop" + str(j) + " = _mm256_setzero_ps();")

    # 内部循环代码未完整提供，需要继续补充
    // 如果使用偏移量
    if use_offsets:
        // 如果数据索引不等于当前范围的起始偏移量减去整体起始偏移量，则返回false
        code.append(
            "      if (dataInd != offsets[rangeIndex] - offsets[0]) {\n"
            + "        return false;\n"
            + "      }"
        )
        // 计算当前范围的结束偏移量和长度
        code.append("""\
      int64_t end_offset = offsets[rangeIndex + 1];
      int64_t length = end_offset - offsets[rangeIndex];""")
        // 循环遍历当前范围内的数据
        code.append(
            "      for ("
            + "int64_t"
            + " start = dataInd; dataInd < end_offset - offsets[0];\n           ++dataInd) {"  # noqa
        )
    else:
        // 如果数据索引加上当前范围的长度超过索引的大小，则返回false
        code.append(
            "      if (dataInd + lengths[rangeIndex] > index_size) {\n"
            + "        return false;\n"
            + "      }"
        )
        // 循环遍历当前范围内的数据
        code.append(
            "      for ("
            + IndexType
            + " start = dataInd; dataInd < start + lengths[rangeIndex];\n           ++dataInd) {"  # noqa
        )
    // 获取当前数据索引处的索引值
    code.append("        const " + IndexType + " idx = indices[dataInd];")
    // 如果索引值小于0或者大于等于数据大小，则返回false
    code.append(
        "        if (idx < 0 || idx >= data_size) {\n"
        + "          return false;\n"
        + "        }"
    )

    // 如果输入类型为 uint8_t
    if InType == "uint8_t":
        // 设置权重初始值为1.0
        code.append("        " + OutType + " wgt = 1.f;")
        // 定义一个输出类型的变量 bio，并注释NOLINTNEXTLINE以忽略特定类型的警告
        code.append("        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)")
        code.append("        " + OutType + " bio;")
        // 如果有权重数组
        code.append("        if (weights) {")
        // 根据权重数组和位置判断是否按位置获取权重值
        code.append(
            "          wgt = weights[IS_WEIGHT_POSITIONAL ? (dataInd - start) : dataInd];"  # noqa
        )
        code.append("        }")
        // 如果进行了融合操作
        if fused:
            // 将输入转换为float指针，获取偏移量，计算bio和wgt
            code.append(
                "        const float* scale_bias = reinterpret_cast<const float*>(\n"
                "            &input[idx * fused_block_size + block_size]);"
            )
            code.append("        bio = wgt * scale_bias[1];")
            code.append("        wgt = wgt * scale_bias[0];")
        else:
            // 计算bio和wgt
            code.append("        bio = wgt * scale_bias[2 * idx + 1];")
            code.append("        wgt = wgt * scale_bias[2 * idx];")
        // 使用bio设置一个256位的向量
        code.append("        __m256 vbio = _mm256_set1_ps(bio);")
    else:
        // 设置权重初始值为1.0
        code.append("        " + OutType + " wgt = 1.f;")
        // 如果有权重数组，根据权重数组和位置判断是否按位置获取权重值
        code.append("        if (weights) {")
        code.append(
            "          wgt = weights[IS_WEIGHT_POSITIONAL ? (dataInd - start) : dataInd];"  # noqa
        )
        code.append("        }")
    // 使用wgt设置一个256位的向量
    code.append("        __m256 vwgt = _mm256_set1_ps(wgt);")

    // 定义输入指针ip，并注释NOLINTNEXTLINE以忽略特定类型的警告
    code.append("        const {}* ip = &input[idx * fused_block_size];".format(InType))
    // 根据下一个T0的位置设置下一个索引值，考虑到T0距离边界的情况，并注释NOLINTNEXTLINE以忽略特定类型的警告
    code.append(
        "        const {} next_T0 = (dataInd < index_size - prefdist_T0)\n"
        "            // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)\n"
        "            ? (dataInd + prefdist_T0)\n"
        "            // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)\n"
        "            : dataInd;".format(
            IndexType
        )
    )
    code.append("        const " + IndexType + " idx_pref_T0 = indices[next_T0];")
    # 将索引值从indices数组中取出，赋给idx_pref_T0变量
    code.append(
        "        if (idx_pref_T0 < 0 || idx_pref_T0 >= data_size) {\n"
        + "          return false;\n"
        + "        }"
    )
    # 检查idx_pref_T0的取值范围，如果超出data_size的范围，则返回false

    code.append(
        "        const {}* ip_next_T0 = "
        "&input[idx_pref_T0 * fused_block_size];".format(InType)
    )
    # 计算ip_next_T0指针，指向input数组中对应的数据块，根据idx_pref_T0和fused_block_size计算偏移量

    for i in range(0, uf):
        j = 8 * i
        cachelinesize = 64
        byteoffset = sizeof[InType] * j
        prefetch = (byteoffset % cachelinesize) == 0
        code.extend(compute(j, InType, use_weights, isa, prefetch))
    # 对于每个i在[0, uf)范围内，根据i计算出j，然后根据InType和其他参数调用compute函数生成相关代码，将其添加到code数组中

    code.append("      }")
    # 添加一个代码块结束符号

    if use_offsets:
        code.append("      if (!normalize_by_lengths || length == 0) {")
    else:
        code.append("      if (!normalize_by_lengths || lengths[rangeIndex] == 0) {")
    # 根据use_offsets的值，选择不同的条件语句进行判断，检查是否需要进行长度归一化操作

    for i in range(0, uf):
        j = 8 * i
        code.append("        _mm256_storeu_ps(&op[" + str(j) + "], vop" + str(j) + ");")
    # 将vop数组中的数据存储到op数组中，使用AVX指令集对齐存储

    code.append("      } else {")
    # 添加一个条件语句的else分支开始

    if use_offsets:
        code.append("        __m256 vlen_inv = _mm256_set1_ps(1.0f / length);")
    else:
        code.append("        __m256 vlen_inv = _mm256_set1_ps(1.0f / lengths[rangeIndex]);")
    # 根据use_offsets的值，创建长度倒数的AVX向量vlen_inv

    for i in range(0, uf):
        j = 8 * i
        code.append(
            "        _mm256_storeu_ps(&op["
            + str(j)
            + "], _mm256_mul_ps("
            + "vop"
            + str(j)
            + ", vlen_inv));"
        )
    # 对于每个i在[0, uf)范围内，计算倒数后的数据，并使用AVX指令集存储到op数组中

    code.append("      }")
    # 添加一个条件语句的结束标志

    code.append("    }")
    # 添加一个代码块结束标志
    return code
    # 返回生成的代码数组
# 定义一个函数 generic，接受多个参数，用于生成特定类型的计算代码块
def generic(IndexType, InType, OutType, use_weights, isa, fused, use_offsets):
    # 定义一个内部函数 compute，根据不同的输入类型生成特定的计算代码
    def compute(InType, use_weights, isa):
        # 初始化一个空的代码列表
        code = []
        # 根据输入类型不同生成不同的计算代码
        if InType == "float":
            code.append(
                "          _mm256_storeu_ps(\n"
                "              &op[j],\n"
                "              _mm256_fmadd_ps(\n"
                "                  vwgt, _mm256_loadu_ps(&ip[j]), _mm256_loadu_ps(&op[j])));"  # noqa
            )
        elif InType == "at::Half":
            code.append(
                "          _mm256_storeu_ps(\n"
                "              &op[j],\n"
                "              _mm256_fmadd_ps(\n"
                "                  vwgt,\n"
                "                  _mm256_cvtph_ps(_mm_loadu_si128(\n"
                "                      reinterpret_cast<const __m128i*>(&ip[j]))),\n"
                "                  _mm256_loadu_ps(&op[j])));"
            )
        elif InType == "at::BFloat16":
            code.append(
                "          _mm256_storeu_ps(\n"
                "              &op[j],\n"
                "              _mm256_fmadd_ps(\n"
                "                  vwgt,\n"
                "                  _mm256_castsi256_ps(_mm256_slli_epi32(\n"
                "                      _mm256_cvtepu16_epi32(_mm_loadu_si128(\n"
                "                          reinterpret_cast<const __m128i*>(&ip[j]))),\n"
                "                      16)),\n"
                "                  _mm256_loadu_ps(&op[j])));"
            )
        elif InType == "uint8_t":
            code.append(
                "          _mm256_storeu_ps(\n"
                "              &op[j],\n"
                "              _mm256_fmadd_ps(\n"
                "                  vwgt,\n"
                "                  _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(\n"  # noqa
                "                      reinterpret_cast<const __m128i*>(&ip[j])))),\n"
                "                  _mm256_add_ps(_mm256_loadu_ps(&op[j]), vbio)));"
            )
        else:
            assert False  # 如果未知的输入类型，则断言错误

        # 添加数据预取指令到代码列表
        code.append(
            "          _mm_prefetch(\n"
            "              reinterpret_cast<const char*>(&ip_next_T0[j]), _MM_HINT_T0);"
        )

        return code  # 返回生成的代码列表

    code = []  # 初始化一个代码列表
    # 根据输入类型是否为 "at::Half"，生成特定的对齐声明代码
    if InType == "at::Half":
        code.append("    alignas(64) at::Half vtmp1[8] = {0};")
    # 根据输入类型是否为 "at::BFloat16"，生成特定的对齐声明代码
    if InType == "at::BFloat16":
        code.append("    alignas(64) at::BFloat16 vtmp1[8] = {0};")

    # 根据 use_offsets 的值决定生成不同的循环代码
    if use_offsets:
        code.append(
            "    for ("
            + IndexType
            + " rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {"
        )
    else:
        code.append(
            "    for ("
            + IndexType
            + " rangeIndex = 0; rangeIndex < output_size; ++rangeIndex) {"
        )

    code.append("      " + OutType + "* op = &out[rangeIndex * block_size];")  # 生成输出数据指针的定义代码
    code.append("      int64_t j = 0;")  # 初始化 j 变量为 0
    # 将一个用于 j 的循环添加到代码列表中，每次增加 8，直到 j + 8 大于块大小
    code.append("      for (; j + 8 <= block_size; j += 8) {")
    # 使用 AVX2 指令集将全零向量存储到 op + j 处
    code.append("        _mm256_storeu_ps(op + j, _mm256_setzero_ps());")
    # 完成 j 循环，处理剩余的元素直到块大小
    code.append("      }")
    # 处理剩余的 j 元素，将它们设置为 0.0f
    code.append("      for (; j < block_size; j++) {")
    code.append("        op[j] = 0.0f;")
    code.append("      }")

    # 内部循环
    if use_offsets:
        # 如果使用偏移量，检查 dataInd 是否等于当前范围的偏移量减去第一个偏移量
        code.append(
            "      if (dataInd != offsets[rangeIndex] - offsets[0]) {\n"
            + "        return false;\n"
            + "      }"
        )
        # 计算结束偏移量和长度
        code.append("""\
      int64_t end_offset = offsets[rangeIndex + 1];
      int64_t length = end_offset - offsets[rangeIndex];""")
        # 开始 dataInd 循环，直到达到当前范围的结束偏移量减去第一个偏移量
        code.append(
            "      for ("
            + "int64_t"
            + " start = dataInd; dataInd < end_offset - offsets[0];\n           ++dataInd) {"  # noqa
        )
    else:
        # 如果不使用偏移量，检查 dataInd + lengths[rangeIndex] 是否大于索引大小
        code.append(
            "      if (dataInd + lengths[rangeIndex] > index_size) {\n"
            + "        return false;\n"
            + "      }"
        )
        # 开始 dataInd 循环，直到达到当前范围的起始位置加上长度
        code.append(
            "      for ("
            + IndexType
            + " start = dataInd; dataInd < start + lengths[rangeIndex];\n           ++dataInd) {"  # noqa
        )
    # 获取当前索引值
    code.append("        const " + IndexType + " idx = indices[dataInd];")
    # 检查索引值是否有效
    code.append(
        "        if (idx < 0 || idx >= data_size) {\n"
        + "          return false;\n"
        + "        }"
    )

    # 如果输入类型为 uint8_t，设置权重 wgt 为 1.0f
    code.append("        " + OutType + " wgt = 1.f;")
    # NOLINTNEXTLINE(cppcoreguidelines-init-variables)：忽略初始化变量警告
    code.append("        " + OutType + " bio;")
    # 如果提供了权重，根据是否是位置权重，设置 wgt 的值
    code.append("        if (weights) {")
    code.append(
        "          wgt = weights[IS_WEIGHT_POSITIONAL ? (dataInd - start) : dataInd];"
    )
    code.append("        }")
    if fused:
        # 如果进行了融合操作，获取 scale_bias 数组的指针，并计算 bio 和 wgt
        code.append(
            "        const float* scale_bias = reinterpret_cast<const float*>(\n"
            "            &input[idx * fused_block_size + block_size]);"
        )
        code.append("        bio = wgt * scale_bias[1];")
        code.append("        wgt = wgt * scale_bias[0];")
    else:
        # 如果没有融合操作，根据 idx 获取 scale_bias 数组的值，并计算 bio 和 wgt
        code.append("        bio = wgt * scale_bias[2 * idx + 1];")
        code.append("        wgt = wgt * scale_bias[2 * idx];")
    # 使用 AVX2 指令集，设置 vbio 为包含 bio 值的向量
    code.append("        __m256 vbio = _mm256_set1_ps(bio);")

    # 如果输入类型不是 uint8_t，根据是否提供权重，设置 wgt 的值
    else:
        code.append("        if (weights) {")
        code.append(
            "          wgt = weights[IS_WEIGHT_POSITIONAL ? (dataInd - start) : dataInd];"
        )
        code.append("        }")
    # 使用 AVX2 指令集，设置 vwgt 为包含 wgt 值的向量
    code.append("        __m256 vwgt = _mm256_set1_ps(wgt);")

    # 设置 ip 为指向输入数组中对应索引的指针
    code.append("        const {}* ip = &input[idx * fused_block_size];".format(InType))
    # 将一段代码添加到列表`code`中，用于计算`next_T0`的值
    code.append(
        "        const {} next_T0 = (dataInd < index_size - prefdist_T0)\n"
        "            // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)\n"
        "            ? (dataInd + prefdist_T0)\n"
        "            // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)\n"
        "            : dataInd;".format(
            IndexType
        )
    )
    # 添加一行代码到`code`列表，定义`idx_pref_T0`为`indices[next_T0]`
    code.append("        const " + IndexType + " idx_pref_T0 = indices[next_T0];")
    # 添加一段代码到`code`列表，检查`idx_pref_T0`的有效性
    code.append(
        "        if (idx_pref_T0 < 0 || idx_pref_T0 >= data_size) {\n"
        + "          return false;\n"
        + "        }"
    )
    # 添加一行代码到`code`列表，指定`ip_next_T0`指向`input`数组中的特定位置
    code.append(
        "        const {}* ip_next_T0 = "
        "&input[idx_pref_T0 * fused_block_size];".format(InType)
    )

    # 初始化循环计数器`j`为0
    code.append("        j = 0;")
    # 添加循环开始标记，对`block_size`进行主循环计算
    code.append("        for (; j + 8 <= block_size; j += 8) {")
    # 调用`compute`函数并将结果添加到`code`列表中
    code.extend(compute(InType, use_weights, isa))
    # 添加主循环结束标记
    code.append("        }")
    # 处理剩余的数据（不足8个元素）
    code.append("        for (; j < block_size; j++) {")
    # 根据`InType`的不同类型执行不同的操作
    if InType == "float":
        code.append("          op[j] = std::fma(wgt, ip[j], op[j]);")
    elif InType == "at::Half":
        code.append("          vtmp1[0] = ip[j];")
        code.append(
            "          __m256 vtmp2 =\n"
            "              _mm256_cvtph_ps(*(reinterpret_cast<const __m128i*>(vtmp1)));"
        )
        code.append("          op[j] = std::fma(wgt, ((float*)(&vtmp2))[0], op[j]);")
    elif InType == "at::BFloat16":
        code.append("          vtmp1[0] = ip[j];")
        code.append(
            "          __m256 vtmp2 = _mm256_castsi256_ps(_mm256_slli_epi32(\n"
            "              _mm256_cvtepu16_epi32(*(reinterpret_cast<const __m128i*>(vtmp1))),\n"
            "              16));"
        )
        code.append("          op[j] = std::fma(wgt, ((float*)(&vtmp2))[0], op[j]);")
    elif InType == "uint8_t":
        code.append("          op[j] = std::fma(wgt, (float)ip[j], bio + op[j]);")
    else:
        assert False  # 如果出现未知的`InType`类型，断言失败

    code.append("        }")

    code.append("      }")

    # 如果使用偏移量，检查是否需要对长度进行归一化处理
    if use_offsets:
        code.append("      if (normalize_by_lengths && length) {")
        code.append("        float len_inv = 1.0f / length;")
    else:
        code.append("      if (normalize_by_lengths && lengths[rangeIndex]) {")
        code.append("        float len_inv = 1.0f / lengths[rangeIndex];")
    # 将`len_inv`值转换为`__m256`类型的向量
    code.append("        __m256 vlen_inv = _mm256_set1_ps(len_inv);")
    # 初始化循环计数器`j`为0
    code.append("        j = 0;")
    # 添加循环开始标记，对`block_size`进行长度归一化处理
    code.append("        for (; j + 8 <= block_size; j += 8) {")
    # 执行长度归一化处理并将结果存储回`op`数组
    code.append(
        "          _mm256_storeu_ps(\n"
        "              &op[j], _mm256_mul_ps(_mm256_loadu_ps(&op[j]), vlen_inv));"
    )
    # 循环处理剩余的数据（不足8个元素）
    code.append("        }")
    code.append("        for (; j < block_size; j++) {")
    # 对剩余元素执行长度归一化处理
    code.append("          op[j] = len_inv * op[j];")
    code.append("        }")

    code.append("      }")

    code.append("    }")
    # 返回当前函数的执行结果
    return code
// start main code
// 创建一个参数解析器对象
parser = argparse.ArgumentParser()
// 添加一个命令行参数，用于指定文件名
parser.add_argument("-f", "--filename", help="file name")
// 添加一个布尔类型的命令行参数，用于标记是否使用偏移量
parser.add_argument("--fused", action="store_true")
// 添加一个布尔类型的命令行参数，用于标记是否使用偏移量
parser.add_argument("--use-offsets", action="store_true")
// 解析命令行参数
opts = parser.parse_args()

// 根据命令行参数设置文件名变量
if opts.filename:
    filename = opts.filename
elif opts.fused:
    // 如果使用了 --fused 参数
    if opts.use_offsets:
        filename = "embedding_lookup_fused_8bit_rowwise_idx_avx2.cc"
    else:
        filename = "embedding_lookup_fused_8bit_rowwise_avx2.cc"
else:
    // 如果未使用 --filename 参数且未使用 --fused 参数
    if opts.use_offsets:
        filename = "embedding_lookup_idx_avx2.cc"
    else:
        filename = "embedding_lookup_avx2.cc"

// 定义选项列表，包含多个长度为6的字符串列表，表示不同的类型组合
options = [
    ["int32_t", "int", "float", "float", "float", "float"],
    ["int64_t", "int64_t", "float", "float", "float", "float"],
    ["int32_t", "int", "half", "at::Half", "float", "float"],
    ["int64_t", "int64_t", "half", "at::Half", "float", "float"],
    ["int32_t", "int", "bfloat16", "at::BFloat16", "float", "float"],
    ["int64_t", "int64_t", "bfloat16", "at::BFloat16", "float", "float"],
    ["int32_t", "int", "uint8_t", "uint8_t", "float", "float"],
    ["int64_t", "int64_t", "uint8_t", "uint8_t", "float", "float"],
]

// 初始化空的代码列表
code = []

// 添加注释头部
code.append("//// --------------------------")
code.append("//// ATTENTION:")
code.append("//// THIS CODE IS AUTOGENERATED")
code.append("//// BY {}".format(sys.argv[0]))
code.append("//// DO NOT MODIFY!!!")
code.append("//// --------------------------\n")

// 添加包含文件
code.append("#include <c10/util/Half.h>")
code.append("#include <c10/util/BFloat16.h>")
code.append("#include <immintrin.h>")

// 进入命名空间 caffe2
code.append("namespace caffe2 {\n")

// 遍历选项列表中的每个选项
for o in options:
    // 解构当前选项的六个元素
    [IndexTypeName, IndexType, InTypeName, InType, OutTypeName, OutType] = o

    // 根据 --fused 参数确定前缀
    prefix = "Fused8BitRowwise" if opts.fused else ""

    // 根据 --use-offsets 参数确定函数基础名称
    if opts.use_offsets:
        fn_base = "{}EmbeddingLookupIdx_{}_{}_{}".format(
            prefix, IndexTypeName, InTypeName, OutTypeName
        )
    else:
        fn_base = "{}EmbeddingLookup_{}_{}_{}".format(
            prefix, IndexTypeName, InTypeName, OutTypeName
        )
    suffix = "__avx2_fma"
    // 定义函数名称
    fn = "static bool " + fn_base + suffix
    code.append(fn + "(")

    // 构造函数参数列表
    args = []
    args.append("    const int64_t block_size,")
    args.append("    const int64_t output_size,")
    args.append("    const int64_t index_size,")
    args.append("    const int64_t data_size,")
    args.append("    const " + InType + "* input,")
    args.append("    const " + IndexType + "* indices,")
    // 根据 --use-offsets 参数添加不同的参数
    if opts.use_offsets:
        args.append("    const " + IndexType + "* offsets,")
    else:
        args.append("    const int* lengths,")
    args.append("    const float* weights,")
    // 如果未使用 --fused 参数，添加额外的参数
    if not opts.fused:
        args.append("    const float* scale_bias,")
    args.append("    bool normalize_by_lengths,")
    args.append("    " + OutType + "* out) {")
    // 添加参数列表到代码
    code += args

    // 添加函数内部的常量定义
    code.append("  const " + IndexType + " prefdist_T0 = 16;")
    # 将一个字符串追加到代码列表中，包含编译器指令禁止特定的静态代码分析工具警告
    code.append("  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)")

    # 设置偏移量，根据是否融合操作确定，block_size是元素数量，fused_block_size包括了整行的大小，包括了缩放和偏置
    # 如果opts.fused为真，则偏移量为8除以InType的字节大小，否则偏移量为0
    offset = (8 // sizeof[InType]) if opts.fused else 0
    code.append(
        "  const {} fused_block_size = block_size + {};".format(IndexType, offset)
    )

    # 如果使用偏移量，则初始化dataInd为int64_t类型，否则为IndexType类型
    if opts.use_offsets:
        code.append("  int64_t dataInd = 0;")
    else:
        code.append("  " + IndexType + " dataInd = 0;")

    # 执行特定块大小的条件语句，根据block_size的大小选择执行相应的向量化代码展开操作
    code.append("  if (block_size == 128) {")
    code += unroll(16, IndexType, InType, OutType, True, "AVX2", opts.fused, opts.use_offsets)
    code.append("  } else if (block_size == 64) {")
    code += unroll(8, IndexType, InType, OutType, True, "AVX2", opts.fused, opts.use_offsets)
    code.append("  } else if (block_size == 32) {")
    code += unroll(4, IndexType, InType, OutType, True, "AVX2", opts.fused, opts.use_offsets)
    code.append("  } else if (block_size == 16) {")
    code += unroll(2, IndexType, InType, OutType, True, "AVX2", opts.fused, opts.use_offsets)
    code.append("  } else {")
    code.append("    // 通用代码")
    code.append("    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-c-arrays)")
    code += generic(IndexType, InType, OutType, True, "AVX2", opts.fused, opts.use_offsets)
    code.append("  }")

    # 返回一个布尔值，表示dataInd是否等于index_size
    code.append("  return dataInd == index_size;")
    code.append("}")

    # 为两种权重位置情况（false和true）生成函数定义，并处理参数列表
    for is_weight_positional in ["false", "true"]:
        code.append("bool " + fn_base + "_" + is_weight_positional + suffix + "(")
        code += args

        # 为了解决Lint警告，限制一行的长度为80个字符
        extra_space = "\n      "
        ret_string = "  return " + fn_base + suffix + "<" + is_weight_positional + ">("

        # 根据长度判断是否需要换行
        if len(ret_string) <= 80:
            code.append(ret_string)
        else:
            code.append("  return " + fn_base + suffix + "<" + extra_space + is_weight_positional + ">(")

        # 逐行添加函数参数
        code.append("      block_size,")
        code.append("      output_size,")
        code.append("      index_size,")
        code.append("      data_size,")
        code.append("      input,")
        code.append("      indices,")

        # 根据是否使用offsets选择添加offsets或lengths
        if opts.use_offsets:
            code.append("      offsets,")
        else:
            code.append("      lengths,")
        
        # 添加权重参数，如果未融合操作则还添加scale_bias参数
        code.append("      weights,")
        if not opts.fused:
            code.append("      scale_bias,")
        code.append("      normalize_by_lengths,")
        code.append("      out);")
        code.append("}")
# 在代码列表末尾添加字符串 "} // namespace caffe2"
code.append("} // namespace caffe2")

# 打开文件 `filename` 用于写入操作，使用 `fout` 作为文件对象
with open(filename, "w") as fout:
    # 遍历代码列表中的每个元素 `c`
    for c in code:
        # 将字符串 `c` 写入文件 `fout`，并添加换行符
        fout.write(c + "\n")

# 打印消息，指示已创建特定名称的文件
print("Created " + filename)
```