# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\init.c`

```py
/*
 * 包含必要的头文件和库
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _MSC_VER
#include <windows.h>
#else
#include <pthread.h>
#endif

/*
 * 包含 QNNPACK 和 CPUINFO 库的头文件
 */
#include <cpuinfo.h>
#include <pytorch_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/params.h>
#include <qnnpack/q8avgpool.h>
#include <qnnpack/q8conv.h>
#include <qnnpack/q8dwconv.h>
#include <qnnpack/q8gavgpool.h>
#include <qnnpack/q8gemm.h>
#include <qnnpack/q8gemm_sparse.h>
#include <qnnpack/q8vadd.h>
#include <qnnpack/u8clamp.h>
#include <qnnpack/u8lut32norm.h>
#include <qnnpack/u8maxpool.h>
#include <qnnpack/u8rmax.h>
#include <qnnpack/x8lut.h>
#include <qnnpack/x8zip.h>

#ifdef _MSC_VER
/*
 * 在 Windows 上使用 INIT_ONCE 机制进行初始化保护
 */
static INIT_ONCE init_guard;
BOOL CALLBACK pytorch_qnnp_init_win(PINIT_ONCE InitOnce, PVOID Parameter, PVOID* lpContex);
#else
/*
 * 在 POSIX 系统上使用 pthread_once_t 进行初始化保护
 */
static pthread_once_t init_guard = PTHREAD_ONCE_INIT;
#endif

/*
 * 定义 QNNPACK 参数结构体，并初始化为未初始化状态
 */
struct pytorch_qnnp_parameters pytorch_qnnp_params = {.initialized = false};

/*
 * 初始化函数，根据平台和硬件支持情况设置 QNNPACK 参数
 */
static void init(void) {
    /*
     * 如果不支持 ARM NEON，打印错误信息并返回
     */
#if CPUINFO_ARCH_ARM
    if (!cpuinfo_has_arm_neon()) {
        pytorch_qnnp_log_error(
            "QNNPACK initialization failed: NEON is not supported");
        return;
    }
    /*
     * 设置 Q8Conv 相关参数，包括 gemm 和 conv 的 ukernel
     */
    pytorch_qnnp_params.q8conv = (struct pytorch_q8conv_parameters){
        .gemm = pytorch_q8gemm_ukernel_4x8__aarch32_neon,
        .conv = pytorch_q8conv_ukernel_4x8__aarch32_neon,
        .gemm_dq = pytorch_q8gemm_dq_ukernel_4x8__aarch32_neon,
        .mr = 4,
        .nr = 8,
        .kr = 1,
    };
    /*
     * 设置 Q8GemmSparse 的 c1x4 参数，包括 sparse gemm 的相关函数和参数
     */
    pytorch_qnnp_params.q8gemm_sparse_c1x4 = (struct pytorch_q8gemm_sparse_parameters){
        .gemm_dq = NULL,
        .packedA_w32_gemm_dq = pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w32__aarch32_neon,
        .packedA_w16_gemm_dq = pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w16__aarch32_neon,
        .packedA_w8_gemm_dq = pytorch_q8gemm_dq_sparse_1x4_ukernel_4x8_packedA_w8__aarch32_neon,
        .packA = pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
        .mr = 4,
        .nr = 8,
        .kr = 4,
        .log2_mr = 2,
        .log2_row_block_size = 0,
        .row_block_size = 1,
        .col_block_size = 4,
    };
    /*
     * 设置 Q8GemmSparse 的 c8x1 参数，包括 sparse gemm 的相关函数和参数
     */
    pytorch_qnnp_params.q8gemm_sparse_c8x1 = (struct pytorch_q8gemm_sparse_parameters){
        .gemm_dq = NULL,
        .packedA_w32_gemm_dq = pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w32__aarch32_neon,
        .packedA_w16_gemm_dq = pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w16__aarch32_neon,
        .packedA_w8_gemm_dq = pytorch_q8gemm_dq_sparse_8x1_ukernel_4x8_packedA_w8__aarch32_neon,
        .packA = pytorch_q8gemm_sparse_packA_ukernel_4x4__aarch32_neon,
        .mr = 4,
        .nr = 8,
        .kr = 4, // kr 实际上是 1，但我们设置为 4 是因为重用了 4x4 的预打包内核
        .log2_mr = 2,
        .log2_row_block_size = 3,
        .row_block_size = 8,
        .col_block_size = 1,
    };
#if !PYTORCH_QNNPACK_RUNTIME_QUANTIZATION
  # 如果不是运行时量化的情况下，设置 QNNPACK 参数中的 q8conv_xzp 结构体
  pytorch_qnnp_params.q8conv_xzp = (struct pytorch_q8conv_xzp_parameters){
      .gemm = pytorch_q8gemm_xzp_ukernel_4x8c2__aarch32_neon,  // 设置 gemm 函数指针
      .mr = 4,  // 设置 gemm 的行块大小
      .nr = 8,  // 设置 gemm 的列块大小
      .kr = 2,  // 设置 gemm 的内核行数
      .kc = 8,  // 设置 gemm 的内核列数
      .kthreshold = SIZE_MAX,  // 设置初始的 kthreshold 为 SIZE_MAX
  };
  /* setup xzp threshold based on measurements */
  // 基于测量结果设置 xzp 阈值
  switch (cpuinfo_get_core(0)->uarch) {
    case cpuinfo_uarch_cortex_a72:
      pytorch_qnnp_params.q8conv_xzp.kthreshold = 64;  // 根据 Cortex A72 设置 kthreshold
      break;
    case cpuinfo_uarch_cortex_a73:
      pytorch_qnnp_params.q8conv_xzp.kthreshold = 256;  // 根据 Cortex A73 设置 kthreshold
      break;
    case cpuinfo_uarch_cortex_a75:
      pytorch_qnnp_params.q8conv_xzp.kthreshold = 32;  // 根据 Cortex A75 设置 kthreshold
      break;
    case cpuinfo_uarch_cortex_a76:
      pytorch_qnnp_params.q8conv_xzp.kthreshold = 16;  // 根据 Cortex A76 设置 kthreshold
      break;
    default:
      break;
  }
#else
  // 如果是运行时量化，则将 q8conv_xzp 结构体中的 kthreshold 设置为 SIZE_MAX
  pytorch_qnnp_params.q8conv_xzp = (struct pytorch_q8conv_xzp_parameters){
      .kthreshold = SIZE_MAX,
  };
#endif
  // 设置 QNNPACK 中的 Q8DW9 参数，包括卷积核函数和通道相关的函数
  pytorch_qnnp_params.q8dw9 = (struct pytorch_q8dwconv2d_up_parameters){
      .updw = pytorch_q8dwconv_ukernel_up8x9__aarch32_neon,
      .updw_per_channel =
          pytorch_q8dwconv_ukernel_up8x9_per_channel__aarch32_neon,
      .cr = 8,
  };
  // 设置 QNNPACK 中的 Q8DW25 参数，包括卷积核函数和通道相关的函数
  pytorch_qnnp_params.q8dw25 = (struct pytorch_q8dwconv2d_mp_parameters){
      .mpdw = pytorch_q8dwconv_ukernel_mp8x25__neon,
      .mpdw_per_channel = pytorch_q8dwconv_ukernel_mp8x25_per_channel__neon,
      .cr = 8,
  };
  // 设置 QNNPACK 中的 Q8DW27 参数，包括卷积核函数和通道相关的函数
  pytorch_qnnp_params.q8dw27 = (struct pytorch_q8dwconv3d_mp_parameters){
      .mpdw = pytorch_q8dwconv_ukernel_mp8x27__neon,
      .cr = 8,
  };
  // 设置 QNNPACK 中的 Q8SUM_ROWS 参数，包括行求和函数和相关参数
  pytorch_qnnp_params.q8sum_rows = (struct pytorch_q8sum_rows_parameters){
      .sum_rows = pytorch_q8sumrows_ukernel_4x__neon,
      .m = 4,
  };
  // 设置 QNNPACK 中的 Q8VADD 参数，包括向量加法函数
  pytorch_qnnp_params.q8vadd = pytorch_q8vadd_ukernel__neon;
  // 设置 QNNPACK 中的 Q8GAVGPOOL 参数，包括全局平均池化函数和相关参数
  pytorch_qnnp_params.q8gavgpool = (struct pytorch_q8gavgpool_parameters){
      .ltnr = pytorch_q8gavgpool_ukernel_up8xm__neon,
      .genr_lemr = pytorch_q8gavgpool_ukernel_up8x7__neon,
      .genr_gtmr = pytorch_q8gavgpool_ukernel_mp8x7p7q__neon,
      .mr = 7,
      .nr = 8,
  };
  // 设置 QNNPACK 中的 Q8AVGPOOL 参数，包括平均池化函数和相关参数
  pytorch_qnnp_params.q8avgpool = (struct pytorch_q8avgpool_parameters){
      .ltkr = pytorch_q8avgpool_ukernel_up8xm__neon,
      .gekr_lemr = pytorch_q8avgpool_ukernel_up8x9__neon,
      .gekr_gtmr = pytorch_q8avgpool_ukernel_mp8x9p8q__neon,
      .mr = 9,
      .qr = 8,
      .kr = 8,
  };
  // 设置 QNNPACK 中的 U8MAXPOOL 参数，包括最大池化函数和相关参数
  pytorch_qnnp_params.u8maxpool = (struct pytorch_u8maxpool_parameters){
      .ltkr = pytorch_u8maxpool_ukernel_sub16__neon,
      .gekr = pytorch_u8maxpool_ukernel_16x9p8q__neon,
      .mr = 9,
      .qr = 8,
      .kr = 16,
  };
  // 设置 QNNPACK 中的 X8ZIP 参数，包括数据压缩函数和相关参数
  pytorch_qnnp_params.x8zip = (struct pytorch_x8zip_parameters){
      .x2 = pytorch_qnnp_x8zip_x2__neon,
      .x3 = pytorch_qnnp_x8zip_x3__neon,
      .x4 = pytorch_qnnp_x8zip_x4__neon,
      .xm = pytorch_qnnp_x8zip_xm__neon,
  };
  // 设置 QNNPACK 中的 U8CLAMP 参数，包括数据截断函数
  pytorch_qnnp_params.u8clamp = pytorch_u8clamp_ukernel__neon;
  // 设置 QNNPACK 中的 U8RMAX 参数，包括最大值计算函数
  pytorch_qnnp_params.u8rmax = pytorch_u8rmax_ukernel__neon;
  // 设置 QNNPACK 中的 U8LUT32NORM 参数，包括 LUT 正规化函数
  pytorch_qnnp_params.u8lut32norm = pytorch_u8lut32norm_ukernel__scalar;
  // 设置 QNNPACK 中的 X8LUT 参数，包括 LUT 查询函数
  pytorch_qnnp_params.x8lut = pytorch_x8lut_ukernel__scalar;
#elif CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  // 如果不支持 SSE2 指令集，打印错误信息并中止初始化
  if (!cpuinfo_has_x86_sse2()) {
    pytorch_qnnp_log_error(
        "QNNPACK initialization failed: SSE2 is not supported");
#else
  // 若不支持的体系结构，则抛出错误
#error "Unsupported architecture"
#endif
  // 标记 QNNPACK 已初始化
  pytorch_qnnp_params.initialized = true;
}

// 初始化 QNNPACK 库
enum pytorch_qnnp_status pytorch_qnnp_initialize(void) {
  // 初始化 CPU 信息库，检查是否成功
  if (!cpuinfo_initialize()) {
    return pytorch_qnnp_status_out_of_memory;
  }
#ifdef _MSC_VER
  // 如果是 Microsoft Visual Studio 编译，使用 InitOnceExecuteOnce 函数初始化
  InitOnceExecuteOnce(&init_guard, pytorch_qnnp_init_win, NULL, NULL);
#else
  // 使用 pthread_once 函数初始化
  pthread_once(&init_guard, &init);
#endif
  // 若 QNNPACK 已初始化，返回成功状态
  if (pytorch_qnnp_params.initialized) {
    return pytorch_qnnp_status_success;
  } else {
    // 否则返回硬件不支持的状态
    return pytorch_qnnp_status_unsupported_hardware;
  }
}

// 反初始化 QNNPACK 库
enum pytorch_qnnp_status pytorch_qnnp_deinitialize(void) {
  // 反初始化 CPU 信息库
  cpuinfo_deinitialize();
  return pytorch_qnnp_status_success;
}

#ifdef _MSC_VER
// 使用 PyTorch QNNPACK 库初始化回调函数，仅在初始化一次时调用
BOOL CALLBACK pytorch_qnnp_init_win(PINIT_ONCE InitOnce, PVOID Parameter, PVOID* lpContex) {
  // 调用初始化函数，准备 PyTorch QNNPACK 库
  init();
  // 返回 TRUE 表示初始化成功
  return TRUE;
}
// #endif 是预处理器指令的结束标记，用于条件编译，与上文代码相关，需保留
#endif
```