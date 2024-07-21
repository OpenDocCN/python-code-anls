# `.\pytorch\aten\src\ATen\cpu\FlushDenormal.cpp`

```py
namespace at::cpu {

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
// 如果支持 SSE（GCC）、x86-64（MSVC）、或带有 SSE 的 x86（MSVC），则定义 DENORMALS_ZERO 和 FLUSH_ZERO
static constexpr unsigned int DENORMALS_ZERO = 0x0040;
static constexpr unsigned int FLUSH_ZERO = 0x8000;

// 设置是否将 denormals 转换为零
bool set_flush_denormal(bool on) {
  // 如果 CPU 支持 x86 的 denormals-are-zero (DAZ)
  if (cpuinfo_has_x86_daz()) {
    // 获取当前的控制状态寄存器值
    unsigned int csr = _mm_getcsr();
    // 清除 DENORMALS_ZERO 和 FLUSH_ZERO 标志位
    csr &= ~DENORMALS_ZERO;
    csr &= ~FLUSH_ZERO;
    // 如果需要打开 denormals-are-zero 和 flush-to-zero
    if (on) {
      csr |= DENORMALS_ZERO;
      csr |= FLUSH_ZERO;
    }
    // 设置控制状态寄存器的值
    _mm_setcsr(csr);
    return true;
  }
  return false;
}
#elif defined(__ARM_FP) && (__ARM_FP > 0)
// 如果是 ARM 平台且浮点指令集大于 0，则定义 ARM_FPCR_FZ
#define ARM_FPCR_FZ   (1 << 24)

// 设置 ARM 浮点控制寄存器的值
static inline void ArmSetFloatingPointControlRegister(uint32_t fpcr) {
#if defined(__aarch64__)
  // 如果是 AArch64 架构，使用 msr 指令设置
  __asm__ __volatile__("msr fpcr, %[fpcr]"
                       :
                       : [fpcr] "r"(static_cast<uint64_t>(fpcr)));
#else
  // 否则，使用 vmsr 指令设置
  __asm__ __volatile__("vmsr fpscr, %[fpcr]" : : [fpcr] "r"(fpcr));
#endif
}

// 获取 ARM 浮点控制寄存器的值
static inline uint32_t ArmGetFloatingPointControlRegister() {
  uint32_t fpcr;
#if defined(__aarch64__)
  uint64_t fpcr64;
  // 如果是 AArch64 架构，使用 mrs 指令读取
  __asm__ __volatile__("mrs %[fpcr], fpcr" : [fpcr] "=r"(fpcr64));
  fpcr = static_cast<uint32_t>(fpcr64);
#else
  // 否则，使用 vmrs 指令读取
  __asm__ __volatile__("vmrs %[fpcr], fpscr" : [fpcr] "=r"(fpcr));
#endif
  return fpcr;
}

// 设置是否将 denormals 转换为零（针对 ARM 平台）
bool set_flush_denormal(bool on) {
    // 获取当前 ARM 浮点控制寄存器的值
    uint32_t fpcr = ArmGetFloatingPointControlRegister();
    // 根据传入的参数 on，设置 ARM_FPCR_FZ 标志位
    if (on) {
      fpcr |= ARM_FPCR_FZ;
    } else {
      fpcr &= ~ ARM_FPCR_FZ;
    }
    // 设置 ARM 浮点控制寄存器的新值
    ArmSetFloatingPointControlRegister(fpcr);
    return true;
}
#else
// 如果既不是 x86 平台也不是 ARM 平台，则返回 false
bool set_flush_denormal(bool on) {
  return false;
}
#endif

}  // namespace at::cpu
```