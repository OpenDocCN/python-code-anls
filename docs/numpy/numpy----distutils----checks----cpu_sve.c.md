# `.\numpy\numpy\distutils\checks\cpu_sve.c`

```
# 包含 ARM SVE 指令集的头文件

# 定义一个函数，接收两个 svint64_t 类型的参数并返回一个整数
int accumulate(svint64_t a, svint64_t b) {
    # 使用全局有效标志位创建一个布尔类型向量
    svbool_t p = svptrue_b64();
    # 返回 a*a+b 的和
    return svaddv(p, svmla_z(p, a, a, b));
}

# 主函数
int main(void)
{
    # 使用全局有效标志位创建一个布尔类型向量
    svbool_t p = svptrue_b64();
    # 使用 svdup_s64 创建一个包含相同值的向量
    svint64_t a = svdup_s64(1);
    svint64_t b = svdup_s64(2);
    # 调用 accumulate 函数并返回结果
    return accumulate(a, b);
}
```