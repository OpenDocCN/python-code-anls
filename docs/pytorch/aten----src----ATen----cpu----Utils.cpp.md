# `.\pytorch\aten\src\ATen\cpu\Utils.cpp`

```
namespace at::cpu {
    // 检查当前 CPU 是否支持 AVX2 指令集
    bool is_cpu_support_avx2() {
        // 如果不是 s390x 或 powerpc 架构，则初始化 CPU 信息并检查是否支持 AVX2
#if !defined(__s390x__) && !defined(__powerpc__)
        return cpuinfo_initialize() && cpuinfo_has_x86_avx2();
#else
        return false;  // 否则返回 false
#endif
    }

    // 检查当前 CPU 是否支持 AVX-512 指令集
    bool is_cpu_support_avx512() {
        // 如果不是 s390x 或 powerpc 架构，则初始化 CPU 信息并检查是否支持 AVX-512 的各个扩展
#if !defined(__s390x__) && !defined(__powerpc__)
        return cpuinfo_initialize() && cpuinfo_has_x86_avx512f() &&
               cpuinfo_has_x86_avx512vl() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq();
#else
        return false;  // 否则返回 false
#endif
    }

    // 检查当前 CPU 是否支持 AVX-512 VNNI 指令集
    bool is_cpu_support_avx512_vnni() {
        // 如果不是 s390x 或 powerpc 架构，则初始化 CPU 信息并检查是否支持 AVX-512 VNNI
#if !defined(__s390x__) && !defined(__powerpc__)
        return cpuinfo_initialize() && cpuinfo_has_x86_avx512vnni();
#else
        return false;  // 否则返回 false
#endif
    }

    // 检查当前 CPU 是否支持 AMX TILE 指令集
    bool is_cpu_support_amx_tile() {
        // 如果不是 s390x 或 powerpc 架构，则初始化 CPU 信息并检查是否支持 AMX TILE
#if !defined(__s390x__) && !defined(__powerpc__)
        return cpuinfo_initialize() && cpuinfo_has_x86_amx_tile();
#else
        return false;  // 否则返回 false
#endif
    }

    // 初始化 AMX 指令集，检查并请求系统对 AMX 指令的支持
    bool init_amx() {
        // 如果当前 CPU 不支持 AMX TILE 指令集，则返回 false
        if (!is_cpu_support_amx_tile()) {
            return false;
        }

        // 如果是 Linux 平台且非 Android，且是 x86_64 架构，则进行以下操作
#if defined(__linux__) && !defined(__ANDROID__) && defined(__x86_64__)
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

        unsigned long bitmask = 0;
        // 请求使用 AMX 指令的权限
        long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
        if (rc) {
            return false;  // 请求失败则返回 false
        }
        // 检查系统是否支持 AMX 指令
        rc = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
        if (rc) {
            return false;  // 获取权限信息失败则返回 false
        }
        // 如果系统支持 AMX TILE 指令集，则返回 true
        if (bitmask & XFEATURE_MASK_XTILE) {
            return true;
        }
        return false;  // 否则返回 false
#else
        return true;  // 其它平台默认返回 true
#endif
    }

} // namespace at::cpu
```