# `.\pytorch\test\run_test.py`

```
#!/usr/bin/env python3
# 指定使用Python 3解释器运行脚本

import argparse
import copy
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, cast, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import pkg_resources

import torch
import torch.distributed as dist
from torch.multiprocessing import current_process, get_context
from torch.testing._internal.common_utils import (
    get_report_path,
    IS_CI,
    IS_MACOS,
    IS_WINDOWS,
    retry_shell,
    set_cwd,
    shell,
    TEST_CUDA,
    TEST_WITH_ASAN,
    TEST_WITH_CROSSREF,
    TEST_WITH_ROCM,
    TEST_WITH_SLOW_GRADCHECK,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
# 获取当前脚本所在目录的上级目录作为代码库的根目录

# using tools/ to optimize test run.
# 将代码库的根目录添加到系统路径中，以便优化测试运行
sys.path.insert(0, str(REPO_ROOT))
from tools.stats.import_test_stats import (
    ADDITIONAL_CI_FILES_FOLDER,
    TEST_CLASS_TIMES_FILE,
    TEST_TIMES_FILE,
)
from tools.stats.upload_metrics import add_global_metric, emit_metric
from tools.testing.discover_tests import (
    CPP_TEST_PATH,
    CPP_TEST_PREFIX,
    CPP_TESTS_DIR,
    parse_test_module,
    TESTS,
)
from tools.testing.do_target_determination_for_s3 import import_results
from tools.testing.target_determination.gen_artifact import gen_ci_artifact
from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
    gen_additional_test_failures_file,
)
from tools.testing.target_determination.heuristics.utils import get_pr_number
from tools.testing.test_run import TestRun
from tools.testing.test_selections import (
    calculate_shards,
    get_test_case_configs,
    NUM_PROCS,
    ShardedTest,
    THRESHOLD,
)

HAVE_TEST_SELECTION_TOOLS = True
# 标记为True，表示已经导入了测试选择工具

# Make sure to remove REPO_ROOT after import is done
# 导入完成后，确保从系统路径中移除代码库的根目录
sys.path.remove(str(REPO_ROOT))

TEST_CONFIG = os.getenv("TEST_CONFIG", "")
# 获取环境变量TEST_CONFIG的值，用于配置测试行为
BUILD_ENVIRONMENT = os.getenv("BUILD_ENVIRONMENT", "")
# 获取环境变量BUILD_ENVIRONMENT的值，用于配置构建环境
RERUN_DISABLED_TESTS = os.getenv("PYTORCH_TEST_RERUN_DISABLED_TESTS", "0") == "1"
# 检查环境变量PYTORCH_TEST_RERUN_DISABLED_TESTS是否为"1"，表示是否重新运行禁用的测试
DISTRIBUTED_TEST_PREFIX = "distributed"
INDUCTOR_TEST_PREFIX = "inductor"
IS_SLOW = "slow" in TEST_CONFIG or "slow" in BUILD_ENVIRONMENT
# 检查TEST_CONFIG或BUILD_ENVIRONMENT中是否包含"slow"，用于标记测试是否为慢速测试

# Note [ROCm parallel CI testing]
# https://github.com/pytorch/pytorch/pull/85770 added file-granularity parallel testing.
# In .ci/pytorch/test.sh, TEST_CONFIG == "default", CUDA and HIP_VISIBLE_DEVICES is set to 0.
# This results in multiple test files sharing the same GPU.
# This should be a supported use case for ROCm, but it exposed issues in the kernel driver resulting in hangs.
# See https://github.com/pytorch/pytorch/issues/90940.
#
# Further, ROCm self-hosted runners have up to 4 GPUs.
# Device visibility was set to 0 to match CUDA test behavior, but this was wasting available GPU resources.
# Assigning each Pool worker their own dedicated GPU avoids the ROCm oversubscription issues.
# 说明ROCm并行CI测试的注意事项，以及与CUDA行为的比较和硬件资源的优化问题
# This function sets environment variables to control GPU visibility for ROCm-based systems.
def maybe_set_hip_visible_devies():
    # Check if the torch version supports ROCm
    if torch.version.hip:
        # Get the current process
        p = current_process()
        # Check if the process is not the main process
        if p.name != "MainProcess":
            # This is a parallel process, set HIP_VISIBLE_DEVICES based on process identity
            os.environ["HIP_VISIBLE_DEVICES"] = str(p._identity[0] % NUM_PROCS)


# Convert a string to a boolean value, considering common falsey values.
def strtobool(s):
    if s.lower() in ["", "0", "false", "off"]:
        return False
    return True


# Custom list subclass for storing test choices with additional parsing.
class TestChoices(list):
    def __init__(self, *args, **kwargs):
        super().__init__(args[0])

    def __contains__(self, item):
        # Parse the test module name before checking if it's in the list
        return list.__contains__(self, parse_test_module(item))


# Filter out tests related to FSDP from the main TESTS list.
FSDP_TEST = [test for test in TESTS if test.startswith("distributed/fsdp")]

# List of Windows-specific tests to exclude from execution.
WINDOWS_BLOCKLIST = [
    "distributed/nn/jit/test_instantiator",
    "distributed/rpc/test_faulty_agent",
    "distributed/rpc/test_tensorpipe_agent",
    "distributed/rpc/test_share_memory",
    "distributed/rpc/cuda/test_tensorpipe_agent",
    "distributed/pipeline/sync/skip/test_api",
    "distributed/pipeline/sync/skip/test_gpipe",
    "distributed/pipeline/sync/skip/test_inspect_skip_layout",
    "distributed/pipeline/sync/skip/test_leak",
    "distributed/pipeline/sync/skip/test_portal",
    "distributed/pipeline/sync/skip/test_stash_pop",
    "distributed/pipeline/sync/skip/test_tracker",
    "distributed/pipeline/sync/skip/test_verify_skippables",
    "distributed/pipeline/sync/test_balance",
    "distributed/pipeline/sync/test_bugs",
    "distributed/pipeline/sync/test_checkpoint",
    "distributed/pipeline/sync/test_copy",
    "distributed/pipeline/sync/test_deferred_batch_norm",
    "distributed/pipeline/sync/test_dependency",
    "distributed/pipeline/sync/test_inplace",
    "distributed/pipeline/sync/test_microbatch",
    "distributed/pipeline/sync/test_phony",
    "distributed/pipeline/sync/test_pipe",
    "distributed/pipeline/sync/test_pipeline",
    "distributed/pipeline/sync/test_stream",
    "distributed/pipeline/sync/test_transparency",
    "distributed/pipeline/sync/test_worker",
    "distributed/elastic/agent/server/test/api_test",
    "distributed/elastic/multiprocessing/api_test",
    "distributed/_shard/checkpoint/test_checkpoint",  # Exclude checkpoint related tests
    "distributed/_shard/checkpoint/test_file_system_checkpoint",  # Exclude file system checkpoint tests
    "distributed/_shard/sharding_spec/test_sharding_spec",
    "distributed/_shard/sharding_plan/test_sharding_plan",
    "distributed/_shard/sharded_tensor/test_sharded_tensor",
    "distributed/_shard/sharded_tensor/test_sharded_tensor_reshard",
    "distributed/_shard/sharded_tensor/ops/test_embedding",
    "distributed/_shard/sharded_tensor/ops/test_embedding_bag",
    "distributed/_shard/sharded_tensor/ops/test_binary_cmp",
    "distributed/_shard/sharded_tensor/ops/test_init",
    "distributed/_shard/sharded_optim/test_sharded_optim",
] + FSDP_TEST

# ROCm-specific tests blocklist, initially empty.
ROCM_BLOCKLIST = []
    # 定义一个包含多个字符串的列表，这些字符串表示不同的测试文件路径
    tests = [
        "distributed/rpc/test_faulty_agent",
        "distributed/rpc/test_tensorpipe_agent",
        "distributed/rpc/test_share_memory",
        "distributed/rpc/cuda/test_tensorpipe_agent",
        "distributed/_shard/checkpoint/test_checkpoint"
        "distributed/_shard/checkpoint/test_file_system_checkpoint"
        "distributed/_shard/sharding_spec/test_sharding_spec",
        "distributed/_shard/sharding_plan/test_sharding_plan",
        "distributed/_shard/sharded_tensor/test_sharded_tensor",
        "distributed/_shard/sharded_tensor/test_sharded_tensor_reshard",
        "distributed/_shard/sharded_tensor/ops/test_embedding",
        "distributed/_shard/sharded_tensor/ops/test_embedding_bag",
        "distributed/_shard/sharded_tensor/ops/test_binary_cmp",
        "distributed/_shard/sharded_tensor/ops/test_init",
        "distributed/_shard/sharded_optim/test_sharded_optim",
        "test_determination",
        "test_jit_legacy",
        "test_cuda_nvml_based_avail",
        "test_jit_cuda_fuser",
        "distributed/_tensor/test_attention",
    ]
# 不应该与其他测试同时并行运行的测试文件列表
RUN_PARALLEL_BLOCKLIST = [
    "test_cpp_extensions_jit",  # JIT 扩展测试，不应并行运行
    "test_cpp_extensions_open_device_registration",  # 开放设备注册测试，不应并行运行
    "test_cpp_extensions_stream_and_event",  # 流和事件测试，不应并行运行
    "test_cpp_extensions_mtia_backend",  # MTIA 后端测试，不应并行运行
    "test_jit_disabled",  # 禁用 JIT 测试，不应并行运行
    "test_mobile_optimizer",  # 移动优化器测试，不应并行运行
    "test_multiprocessing",  # 多进程测试，不应并行运行
    "test_multiprocessing_spawn",  # 使用 spawn 方式的多进程测试，不应并行运行
    "test_namedtuple_return_api",  # 命名元组返回 API 测试，不应并行运行
    "test_overrides",  # 覆盖测试，不应并行运行
    "test_show_pickle",  # 显示 pickle 测试，不应并行运行
    "test_tensorexpr",  # 张量表达式测试，不应并行运行
    "test_cuda_primary_ctx",  # CUDA 主上下文测试，不应并行运行
    "test_cuda_trace",  # CUDA 跟踪测试，不应并行运行
    "inductor/test_benchmark_fusion",  # 基准融合测试，不应并行运行
    "test_cuda_nvml_based_avail",  # 基于 CUDA NVML 可用性测试，不应并行运行
    "test_autograd_fallback",  # 自动求导回退测试，不应并行运行，临时设置全局配置
] + FSDP_TEST

# 总是应该与其他测试文件串行运行的测试文件列表，但其中的测试可以并行运行
CI_SERIAL_LIST = [
    "test_nn",  # NN 测试，可以与其它测试文件并行运行
    "test_fake_tensor",  # 假张量测试，可以与其它测试文件并行运行
    "test_cpp_api_parity",  # C++ API 对等性测试，可以与其它测试文件并行运行
    "test_reductions",  # 缩减测试，可以与其它测试文件并行运行
    "test_fx_backends",  # FX 后端测试，可以与其它测试文件并行运行
    "test_cpp_extensions_jit",  # JIT 扩展测试，虽然不应并行运行，但与其它测试文件串行运行
    "test_torch",  # Torch 测试，可以与其它测试文件并行运行
    "test_tensor_creation_ops",  # 张量创建操作测试，可以与其它测试文件并行运行
    "test_dispatch",  # 分发测试，可以与其它测试文件并行运行
    "test_python_dispatch",  # Python 分发测试，必须串行运行以便 torch.library 的创建和删除
    "test_spectral_ops",  # 光谱操作测试，由于 CUDA 非法内存访问，必须串行运行
    "nn/test_pooling",  # NN 池化测试，可以与其它测试文件并行运行
    "nn/test_convolution",  # NN 卷积测试，由于不遵守 set_per_process_memory_fraction，导致其他测试在慢速 gradcheck 中 OOM
    "distributions/test_distributions",  # 分布测试，可以与其它测试文件并行运行
    "test_fx",  # FX 测试，因为获得 SIGKILL 信号，必须串行运行
    "functorch/test_memory_efficient_fusion",  # 内存高效融合测试，由于 CUDA OOM（ROCm），必须串行运行
    "test_utils",  # 实用工具测试，由于 OOM，必须串行运行
    "test_sort_and_select",  # 排序和选择测试，由于 OOM，必须串行运行
    "test_backward_compatible_arguments",  # 向后兼容参数测试，由于 OOM，必须串行运行
    "test_autocast",  # 自动类型转换测试，由于 OOM，必须串行运行
    "test_native_mha",  # 原生 MHA 测试，由于 OOM，必须串行运行
    "test_module_hooks",  # 模块钩子测试，由于 OOM，必须串行运行
    "inductor/test_max_autotune",  # 最大自动调谐测试，可以与其它测试文件并行运行
    "inductor/test_cutlass_backend",  # Cutlass 后端测试，由于 nvcc 编译步骤多，速度慢，可以与其它测试文件并行运行
    "inductor/test_flex_attention",  # 灵活注意力测试，由于 OOM，必须串行运行
]

# 由于内存使用高，不能并行运行的 ONNX 测试文件子集
ONNX_SERIAL_LIST = [
    "onnx/test_models",  # ONNX 模型测试，由于内存使用高，不能并行运行
    "onnx/test_models_quantized_onnxruntime",  # 量化 ONNX 运行时模型测试，由于内存使用高，不能并行运行
    "onnx/test_models_onnxruntime",  # ONNX 运行时模型测试，由于内存使用高，不能并行运行
    "onnx/test_custom_ops",  # 自定义操作测试，由于内存使用高，不能并行运行
    "onnx/test_utility_funs",  # 实用函数测试，由于内存使用高，不能并行运行
]

# 用于验证 PyTorch 的操作、模块和自动求导功能的核心测试列表
CORE_TEST_LIST = [
    "test_autograd",  # 自动求导测试
    "test_autograd_fallback",  # 自动求导回退测试
    "test_modules",  # 模块测试
    "test_nn",  # NN 测试
    "test_ops",  # 操作测试
    "test_ops_gradients",  # 操作梯度测试
    "test_ops_fwd_gradients",  # 操作前向梯度测试
    "test_ops_jit",  # JIT 操作测试
    "test_torch",  # Torch 测试
]

# 如果测试文件运行时间超过 5 分钟，将其添加到 TARGET_DET_LIST 中
SLOW_TEST_THRESHOLD = 300

# 分布式测试的配置，如果支持分布式，则设置 WORLD_SIZE 为 1
DISTRIBUTED_TESTS_CONFIG = {}
if dist.is_available():
    DISTRIBUTED_TESTS_CONFIG["test"] = {"WORLD_SIZE": "1"}
    # 如果不是使用 ROCm 进行测试并且 MPI 可用，则配置 MPI 测试
    if not TEST_WITH_ROCM and dist.is_mpi_available():
        # 设置 MPI 的分布式测试配置
        DISTRIBUTED_TESTS_CONFIG["mpi"] = {
            "WORLD_SIZE": "3",  # 设置世界大小为 3
            "TEST_REPORT_SOURCE_OVERRIDE": "dist-mpi",  # 设置测试报告来源为 dist-mpi
        }
    
    # 如果 NCCL 可用，则配置 NCCL 测试
    if dist.is_nccl_available():
        # 设置 NCCL 的分布式测试配置
        DISTRIBUTED_TESTS_CONFIG["nccl"] = {
            "WORLD_SIZE": "2" if torch.cuda.device_count() == 2 else "3",  # 根据 GPU 数量设置世界大小为 2 或 3
            "TEST_REPORT_SOURCE_OVERRIDE": "dist-nccl",  # 设置测试报告来源为 dist-nccl
        }
    
    # 如果 Gloo 可用，则配置 Gloo 测试
    if dist.is_gloo_available():
        # 设置 Gloo 的分布式测试配置
        DISTRIBUTED_TESTS_CONFIG["gloo"] = {
            "WORLD_SIZE": "2" if torch.cuda.device_count() == 2 else "3",  # 根据 GPU 数量设置世界大小为 2 或 3
            "TEST_REPORT_SOURCE_OVERRIDE": "dist-gloo",  # 设置测试报告来源为 dist-gloo
        }
    
    # 如果 UCC 可用，则配置 UCC 测试
    if dist.is_ucc_available():
        # 设置 UCC 的分布式测试配置
        DISTRIBUTED_TESTS_CONFIG["ucc"] = {
            "WORLD_SIZE": "2" if torch.cuda.device_count() == 2 else "3",  # 根据 GPU 数量设置世界大小为 2 或 3
            "TEST_REPORT_SOURCE_OVERRIDE": "dist-ucc",  # 设置测试报告来源为 dist-ucc
            "UCX_TLS": "tcp,cuda",  # 设置 UCX 支持的传输层
            "UCC_TLS": "nccl,ucp,cuda",  # 设置 UCC 支持的传输层
            "UCC_TL_UCP_TUNE": "cuda:0",  # 设置 UCC 使用 CUDA 0 号设备的传输层优化
            "UCC_EC_CUDA_USE_COOPERATIVE_LAUNCH": "n",  # 如果开启，CI 节点（M60）会失败，因此关闭协作启动
        }
# 创建一个字典，将信号编号映射为其名称，只包括以"SIG"开头且不包含"_"的属性
SIGNALS_TO_NAMES_DICT = {
    getattr(signal, n): n for n in dir(signal) if n.startswith("SIG") and "_" not in n
}

# 如果缺少 Ninja（用于一些 C++ 扩展测试），则显示的错误消息
CPP_EXTENSIONS_ERROR = """
Ninja (https://ninja-build.org) is required for some of the C++ extensions
tests, but it could not be found. Install ninja with `pip install ninja`
or `conda install ninja`. Alternatively, disable said tests with
`run_test.py --exclude test_cpp_extensions_aot_ninja test_cpp_extensions_jit`.
"""

# 检查环境变量中是否设置了 PYTORCH_COLLECT_COVERAGE，并将其转换为布尔值
PYTORCH_COLLECT_COVERAGE = bool(os.environ.get("PYTORCH_COLLECT_COVERAGE"))

# 包含要执行的 JIT 执行器测试的列表
JIT_EXECUTOR_TESTS = [
    "test_jit_profiling",
    "test_jit_legacy",
    "test_jit_fuser_legacy",
]

# 根据给定前缀（INDUCTOR_TEST_PREFIX）过滤出测试名称列表
INDUCTOR_TESTS = [test for test in TESTS if test.startswith(INDUCTOR_TEST_PREFIX)]
# 根据给定前缀（DISTRIBUTED_TEST_PREFIX）过滤出测试名称列表
DISTRIBUTED_TESTS = [test for test in TESTS if test.startswith(DISTRIBUTED_TEST_PREFIX)]
# 根据前缀 "export" 过滤出测试名称列表
TORCH_EXPORT_TESTS = [test for test in TESTS if test.startswith("export")]
# 根据前缀 "functorch" 过滤出测试名称列表
FUNCTORCH_TESTS = [test for test in TESTS if test.startswith("functorch")]
# 根据前缀 "onnx" 过滤出测试名称列表
ONNX_TESTS = [test for test in TESTS if test.startswith("onnx")]
# 根据给定前缀（CPP_TEST_PREFIX）过滤出测试名称列表
CPP_TESTS = [test for test in TESTS if test.startswith(CPP_TEST_PREFIX)]

# 要求 LAPACK 的测试列表
TESTS_REQUIRING_LAPACK = [
    "distributions/test_constraints",
    "distributions/test_distributions",
]

# 不使用 gradcheck 的测试列表（非详尽）
TESTS_NOT_USING_GRADCHECK = [
    # 如果不希望跳过文件中的所有测试，可以使用 skipIfSlowGradcheckEnv，例如 test_mps
    "doctests",
    "test_meta",
    "test_hub",
    "test_fx",
    "test_decomp",
    "test_cpp_extensions_jit",
    "test_jit",
    "test_ops",
    "test_ops_jit",
    "dynamo/test_recompile_ux",
    "inductor/test_smoke",
    "test_quantization",
]


def print_to_stderr(message):
    # 将消息输出到标准错误流
    print(message, file=sys.stderr)


def get_executable_command(options, disable_coverage=False, is_cpp_test=False):
    # 根据选项和条件生成可执行命令列表
    if options.coverage and not disable_coverage:
        if not is_cpp_test:
            # 使用 coverage 运行测试，覆盖率模式下并行执行，源文件为 torch
            executable = ["coverage", "run", "--parallel-mode", "--source=torch"]
        else:
            # 对于 C++ 测试，暂时不支持覆盖率检测
            # TODO: C++ with coverage is not yet supported
            executable = []
    else:
        if not is_cpp_test:
            # 直接使用 Python 解释器运行测试，设置断言（assert）的行为
            executable = [sys.executable, "-bb"]
        else:
            # 对于 C++ 测试，使用 pytest 运行
            executable = ["pytest"]

    return executable


def run_test(
    test_module: ShardedTest,
    test_directory,
    options,
    launcher_cmd=None,
    extra_unittest_args=None,
    env=None,
    print_log=True,
) -> int:
    # 拷贝当前环境变量，若未指定则使用空字典
    env = env or os.environ.copy()
    # 可能设置 HIP_VISIBLE_DEVICES 环境变量
    maybe_set_hip_visible_devies()
    # 复制额外的单元测试参数列表
    unittest_args = options.additional_args.copy()
    # 获取测试文件名
    test_file = test_module.name
    # 设置当前步骤的键值
    stepcurrent_key = test_file

    # 检查是否是分布式测试
    is_distributed_test = test_file.startswith(DISTRIBUTED_TEST_PREFIX)
    # 检查是否是 C++ 测试
    is_cpp_test = test_file.startswith(CPP_TEST_PREFIX)
    # 注意事项：重新运行禁用的测试依赖于 pytest-flakefinder，目前与 pytest-cpp 不兼容，且尚不支持禁用 C++ 测试，因此这是可以接受的
    # NB: Rerun disabled tests depends on pytest-flakefinder and it doesn't work with
    # pytest-cpp atm. We also don't have support to disable C++ test yet, so it's ok
    # 如果是 C++ 测试并且禁用了重新运行测试选项，则直接返回成功
    if is_cpp_test and RERUN_DISABLED_TESTS:
        # 打印消息说明跳过 C++ 测试，因为处于禁用重新运行测试模式下
        print_to_stderr(
            "Skipping C++ tests when running under RERUN_DISABLED_TESTS mode"
        )
        return 0

    # 如果是 C++ 测试，生成当前步骤的唯一键值，包括测试文件名和随机生成的十六进制字符串
    if is_cpp_test:
        stepcurrent_key = f"{test_file}_{os.urandom(8).hex()}"
    else:
        # 如果不是 C++ 测试，将单元测试参数扩展，包括分片 ID 和总分片数
        unittest_args.extend(
            [
                f"--shard-id={test_module.shard}",
                f"--num-shards={test_module.num_shards}",
            ]
        )
        # 生成当前步骤的唯一键值，包括测试文件名、分片 ID 和随机生成的十六进制字符串
        stepcurrent_key = f"{test_file}_{test_module.shard}_{os.urandom(8).hex()}"

    # 如果选项中开启了详细输出模式，为单元测试参数添加相应的详细输出标志（例如对于 pytest）
    if options.verbose:
        unittest_args.append(f'-{"v" * options.verbose}')

    # 如果当前测试文件在运行并行测试的黑名单中，则从单元测试参数中移除相关的并行运行标志
    if test_file in RUN_PARALLEL_BLOCKLIST:
        unittest_args = [
            arg for arg in unittest_args if not arg.startswith("--run-parallel")
        ]

    # 如果存在额外的单元测试参数，将其添加到单元测试参数列表中
    if extra_unittest_args:
        assert isinstance(extra_unittest_args, list)
        unittest_args.extend(extra_unittest_args)

    # 如果使用 pytest，将 -f 参数替换为等效的 -x 参数
    if options.pytest:
        # 获取 pytest 相关的单元测试参数，并将其扩展到单元测试参数列表中
        unittest_args.extend(
            get_pytest_args(
                options,
                is_cpp_test=is_cpp_test,
                is_distributed_test=is_distributed_test,
            )
        )
        # 获取测试模块的 pytest 参数，并将其扩展到单元测试参数列表中
        unittest_args.extend(test_module.get_pytest_args())
        # 替换单元测试参数列表中的 -f 参数为 -x 参数（仅适用于 pytest）
        unittest_args = [arg if arg != "-f" else "-x" for arg in unittest_args]

    # 注意：这些功能对于 C++ 测试不可用，但因为我们从未见过有关 C++ 测试的不稳定报告，所以实现这些功能的动机不大。
    # 如果处于 CI 环境并且当前不是 C++ 测试，则添加 CI 相关的参数，包括导入慢速测试和禁用的测试
    if IS_CI and not is_cpp_test:
        ci_args = ["--import-slow-tests", "--import-disabled-tests"]
        if RERUN_DISABLED_TESTS:
            ci_args.append("--rerun-disabled-tests")
        # 将 CI 相关的参数扩展到单元测试参数列表中
        unittest_args.extend(ci_args)

    # 如果当前测试文件在 PYTEST_SKIP_RETRIES 中，且未使用 pytest，则抛出运行时错误，因为不支持跳过重试
    if test_file in PYTEST_SKIP_RETRIES:
        if not options.pytest:
            raise RuntimeError(
                "A test running without pytest cannot skip retries using "
                "the PYTEST_SKIP_RETRIES set."
            )
        # 从单元测试参数列表中移除带有 "--reruns" 参数的选项，以便禁用重试
        unittest_args = [arg for arg in unittest_args if "--reruns" not in arg]

    # 如果未能获取可执行命令，例如对于 C++ 测试覆盖率等不支持的情况，则直接返回成功
    executable = get_executable_command(options, is_cpp_test=is_cpp_test)
    if not executable:
        # 如果没有符合条件的可执行命令，则返回成功
        return 0
    # 如果测试文件以CPP_TEST_PREFIX开头
    if test_file.startswith(CPP_TEST_PREFIX):
        # C++ 测试不在常规测试目录中
        if CPP_TESTS_DIR:
            # 构建 C++ 测试文件的路径
            cpp_test = os.path.join(
                CPP_TESTS_DIR,
                test_file.replace(f"{CPP_TEST_PREFIX}/", ""),
            )
        else:
            # 使用默认路径构建 C++ 测试文件的路径
            cpp_test = os.path.join(
                Path(test_directory).parent,
                CPP_TEST_PATH,
                test_file.replace(f"{CPP_TEST_PREFIX}/", ""),
            )

        # 准备执行 C++ 测试的命令参数列表
        argv = [
            cpp_test if sys.platform != "win32" else cpp_test + ".exe"
        ] + unittest_args
    else:
        # 这里无法调用 `python -m unittest test_*`，因为它不会运行在 `if __name__ == '__main__':` 中的代码。
        # 因此，改为调用 `python test_*.py`。
        argv = [test_file + ".py"] + unittest_args

    # 创建存放测试报告的目录，如果目录已存在则不会重新创建
    os.makedirs(REPO_ROOT / "test" / "test-reports", exist_ok=True)

    # 如果设置了选项 `pipe_logs`
    if options.pipe_logs:
        # 创建临时文件来存储日志
        log_fd, log_path = tempfile.mkstemp(
            dir=REPO_ROOT / "test" / "test-reports",
            prefix=f"{sanitize_file_name(str(test_module))}_",
            suffix="_toprint.log",
        )
        os.close(log_fd)

    # 构建完整的测试命令
    command = (launcher_cmd or []) + executable + argv

    # 判断是否应该重试测试
    should_retry = (
        "--subprocess" not in command
        and not RERUN_DISABLED_TESTS
        and not is_cpp_test
        and "-n" not in command
    )

    # 设置超时时间
    timeout = (
        None
        if not options.enable_timeout
        else THRESHOLD * 6
        if IS_SLOW
        else THRESHOLD * 3
        if should_retry
        and isinstance(test_module, ShardedTest)
        and test_module.time is not None
        else THRESHOLD * 3
        if is_cpp_test
        else None
    )

    # 输出执行命令的相关信息到标准错误输出
    print_to_stderr(f"Executing {command} ... [{datetime.now()}]")
    # 使用 ExitStack 确保上下文管理的资源正确释放
    with ExitStack() as stack:
        # 初始化 output 变量为 None
        output = None
        # 如果选择了将日志输出到管道中
        if options.pipe_logs:
            # 打开日志文件以供写入，并将其加入上下文管理器中
            output = stack.enter_context(open(log_path, "w"))

        # 如果需要重试测试
        if should_retry:
            # 调用 run_test_retries 函数执行测试重试，并获取返回码和是否重新运行的信息
            ret_code, was_rerun = run_test_retries(
                command,
                test_directory,
                env,
                timeout,
                stepcurrent_key,
                output,
                options.continue_through_error,
            )
        else:
            # 否则，添加特定参数到命令列表中以执行测试
            command.extend([f"--sc={stepcurrent_key}", "--print-items"])
            # 调用 retry_shell 函数执行命令，并获取返回码和是否重新运行的信息
            ret_code, was_rerun = retry_shell(
                command,
                test_directory,
                stdout=output,
                stderr=output,
                env=env,
                timeout=timeout,
                retries=0,
            )

            # 如果返回码是 5，表示未收集到任何测试。将返回码设为 0
            # 返回码为 4 时，说明二进制文件不是 C++ 测试可执行文件，但也可能
            # 在运行任何测试之前失败。在撰写本注释时，所有不是 C++ 测试的二进制文件
            # 在 build/bin 目录下均已排除，新文件应添加到 tools/testing/discover_tests.py 的排除列表中
            ret_code = 0 if ret_code == 5 else ret_code

    # 如果选择将日志输出到管道并且需要打印日志，则处理日志文件
    if options.pipe_logs and print_log:
        handle_log_file(
            test_module, log_path, failed=(ret_code != 0), was_rerun=was_rerun
        )
    # 返回测试执行的最终返回码
    return ret_code
# 定义一个内部函数print_to_file，用于将字符串s输出到指定的输出流output中
def run_test_retries(
    command,
    test_directory,
    env,
    timeout,
    stepcurrent_key,
    output,
    continue_through_error,
):
    # 执行测试命令，并使用'-x'选项在首次失败时停止。如果失败，重试跳过已运行的测试，直到某个测试连续失败3次（通常是我们允许的次数）。
    #
    # 如果continue_through_error未设置，则快速失败。
    #
    # 如果设置了continue_through_error，则跳过该测试并继续执行。如果同一个测试连续失败3次，则在下一次运行时跳过该测试，但最终仍然失败。利用stepcurrent_key保存的值来跟踪最近运行的测试（即如果有失败，则是失败的测试）。

    def print_to_file(s):
        print(s, file=output, flush=True)

    # 使用defaultdict来计算每个失败测试的次数
    num_failures = defaultdict(int)

    # 初始化打印选项，包含"--print-items"
    print_items = ["--print-items"]
    sc_command = f"--sc={stepcurrent_key}"
    while True:
        # 调用retry_shell函数执行命令，并传入所需参数，包括指定的选项和环境变量
        ret_code, _ = retry_shell(
            command + [sc_command] + print_items,
            test_directory,
            stdout=output,
            stderr=output,
            env=env,
            timeout=timeout,
            retries=0,  # 这里不进行重试，因为我们自己处理，这是因为它能很好地处理超时异常
        )
        # 如果ret_code为5，将其转换为0，表示成功执行到测试套件的末尾
        ret_code = 0 if ret_code == 5 else ret_code
        if ret_code == 0:
            break  # 成功执行到测试套件的末尾
        # 如果ret_code小于0，则转换为信号名称，例如(-SIGTERM)，否则为空字符串
        signal_name = f" ({SIGNALS_TO_NAMES_DICT[-ret_code]})" if ret_code < 0 else ""
        # 将退出码及其信号名称（如果有）写入到输出流中
        print_to_file(f"Got exit code {ret_code}{signal_name}")

        # 读取刚刚失败的测试步骤
        try:
            with open(
                REPO_ROOT / ".pytest_cache/v/cache/stepcurrent" / stepcurrent_key
            ) as f:
                current_failure = f.read()
        except FileNotFoundError:
            # 如果找不到stepcurrent文件，输出相应信息
            print_to_file(
                "No stepcurrent file found. Either pytest didn't get to run (e.g. import error)"
                + " or file got deleted (contact dev infra)"
            )
            break

        # 增加当前失败测试步骤的失败次数计数
        num_failures[current_failure] += 1
        # 如果当前失败测试步骤的失败次数达到3次以上
        if num_failures[current_failure] >= 3:
            # 如果不允许继续执行错误，则输出信息并跳出循环
            if not continue_through_error:
                print_to_file("Stopping at first consistent failure")
                break
            # 否则，设置sc_command以跳过该测试步骤，并继续执行
            sc_command = f"--scs={stepcurrent_key}"
        else:
            # 如果失败次数未达到3次，则设置sc_command以重新运行该测试步骤
            sc_command = f"--sc={stepcurrent_key}"
        # 输出重试信息
        print_to_file("Retrying...")
        
        # 打印完整的C++堆栈跟踪信息以便调试
        # 对于macOS的inductor测试，不要打印堆栈跟踪，因为它可能导致段错误
        if not (
            IS_MACOS
            and len(command) >= 2
            and command[2].startswith(INDUCTOR_TEST_PREFIX)
        ):
            # 如果env未定义，则初始化为空字典
            env = env or {}
            # 设置环境变量TORCH_SHOW_CPP_STACKTRACES为1，以显示C++堆栈跟踪
            env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
        # 不再继续打印任何项目，避免浪费大量空间
        print_items = []  # 不再继续打印项目，这样会浪费大量空间
    # 从 num_failures 字典中选取出现次数大于等于 3 次的失败测试名称（去掉首尾字符），存入列表
    consistent_failures = [x[1:-1] for x in num_failures.keys() if num_failures[x] >= 3]
    # 从 num_failures 字典中选取出现次数大于 0 且小于 3 次的失败测试名称（去掉首尾字符），存入列表
    flaky_failures = [x[1:-1] for x in num_failures.keys() if 0 < num_failures[x] < 3]
    # 如果存在 flaky_failures 列表中的元素，将其打印到文件中，指示这些测试在新进程中有时失败有时成功
    if len(flaky_failures) > 0:
        print_to_file(
            "The following tests failed and then succeeded when run in a new process"
            + f"{flaky_failures}",
        )
    # 如果存在 consistent_failures 列表中的元素，将其打印到文件中，并返回状态码 1 和 True，表示存在一致失败的测试
    if len(consistent_failures) > 0:
        print_to_file(f"The following tests failed consistently: {consistent_failures}")
        return 1, True
    # 否则返回原始的 ret_code，以及判断 num_failures 值中是否有大于 0 的项的逻辑值的组合结果
    return ret_code, any(x > 0 for x in num_failures.values())
# 使用 subprocess 运行测试模块，增加额外的单元测试参数 "--subprocess"
def run_test_with_subprocess(test_module, test_directory, options):
    return run_test(
        test_module, test_directory, options, extra_unittest_args=["--subprocess"]
    )


# 对 C++ 扩展进行 Ahead-Of-Time (AOT) 编译测试
def _test_cpp_extensions_aot(test_directory, options, use_ninja):
    # 如果使用 Ninja 构建系统，则检查 Ninja 的可用性
    if use_ninja:
        try:
            from torch.utils import cpp_extension

            cpp_extension.verify_ninja_availability()
        except RuntimeError:
            # 如果运行时出错，打印 C++ 扩展错误信息并返回错误码 1
            print_to_stderr(CPP_EXTENSIONS_ERROR)
            return 1

    # 确定 C++ 扩展测试目录和构建目录
    cpp_extensions_test_dir = os.path.join(test_directory, "cpp_extensions")
    cpp_extensions_test_build_dir = os.path.join(cpp_extensions_test_dir, "build")
    
    # 如果构建目录已存在，则递归删除它
    if os.path.exists(cpp_extensions_test_build_dir):
        shutil.rmtree(cpp_extensions_test_build_dir)

    # 准备执行安装脚本的环境变量
    shell_env = os.environ.copy()
    shell_env["USE_NINJA"] = str(1 if use_ninja else 0)
    
    # 准备执行的命令，安装 C++ 扩展模块到指定路径
    cmd = [sys.executable, "setup.py", "install", "--root", "./install"]
    return_code = shell(cmd, cwd=cpp_extensions_test_dir, env=shell_env)
    
    # 如果安装过程返回非零错误码，则直接返回该错误码
    if return_code != 0:
        return return_code
    
    # 在非 Windows 平台上，再次执行安装命令，用于测试不带 Python ABI 后缀的情况
    if sys.platform != "win32":
        return_code = shell(
            cmd,
            cwd=os.path.join(cpp_extensions_test_dir, "no_python_abi_suffix_test"),
            env=shell_env,
        )
        if return_code != 0:
            return return_code

    # 安装测试模块并运行测试
    python_path = os.environ.get("PYTHONPATH", "")
    from shutil import copyfile

    # 设置运行测试的环境变量 USE_NINJA，并复制测试模块文件
    os.environ["USE_NINJA"] = shell_env["USE_NINJA"]
    test_module = "test_cpp_extensions_aot" + ("_ninja" if use_ninja else "_no_ninja")
    copyfile(
        test_directory + "/test_cpp_extensions_aot.py",
        test_directory + "/" + test_module + ".py",
    )
    try:
        # 查找安装目录，通常为带有 "-packages" 后缀的目录
        cpp_extensions = os.path.join(test_directory, "cpp_extensions")
        install_directory = ""
        for root, directories, _ in os.walk(os.path.join(cpp_extensions, "install")):
            for directory in directories:
                if "-packages" in directory:
                    install_directory = os.path.join(root, directory)

        # 确保安装目录不为空
        assert install_directory, "install_directory must not be empty"

        # 设置 PYTHONPATH 包含安装目录，并运行测试
        os.environ["PYTHONPATH"] = os.pathsep.join([install_directory, python_path])
        return run_test(ShardedTest(test_module, 1, 1), test_directory, options)
    finally:
        # 恢复 PYTHONPATH 环境变量，并清理临时复制的测试模块文件
        os.environ["PYTHONPATH"] = python_path
        if os.path.exists(test_directory + "/" + test_module + ".py"):
            os.remove(test_directory + "/" + test_module + ".py")
        # 移除环境变量 USE_NINJA
        os.environ.pop("USE_NINJA")


# 使用 Ninja 构建系统运行 Ahead-Of-Time (AOT) 编译的测试
def test_cpp_extensions_aot_ninja(test_module, test_directory, options):
    return _test_cpp_extensions_aot(test_directory, options, use_ninja=True)


# 使用非 Ninja 构建系统运行 Ahead-Of-Time (AOT) 编译的测试
def test_cpp_extensions_aot_no_ninja(test_module, test_directory, options):
    return _test_cpp_extensions_aot(test_directory, options, use_ninja=False)
def test_distributed(test_module, test_directory, options):
    # 检查当前环境是否支持 MPI，并且 Python 版本低于 3.9
    mpi_available = subprocess.call(
        "command -v mpiexec", shell=True
    ) == 0 and sys.version_info < (3, 9)
    # 如果设置了详细输出选项，并且 MPI 不可用，则输出警告信息
    if options.verbose and not mpi_available:
        print_to_stderr("MPI not available -- MPI backend tests will be skipped")

    # 使用预定义的 DISTRIBUTED_TESTS_CONFIG 配置
    config = DISTRIBUTED_TESTS_CONFIG
    # 遍历配置字典中的每个后端及其对应的环境变量
    for backend, env_vars in config.items():
        # 如果操作系统是 Windows 并且后端不是 "gloo"，则跳过当前后端的测试
        if sys.platform == "win32" and backend != "gloo":
            continue
        # 如果后端是 "mpi" 但当前系统不支持 MPI，则跳过 MPI 后端的测试
        if backend == "mpi" and not mpi_available:
            continue
        # 对每个后端，分别进行带和不带 init 文件的测试
        for with_init_file in {True, False}:
            # 如果操作系统是 Windows 但当前测试不带 init 文件，则跳过
            if sys.platform == "win32" and not with_init_file:
                continue
            # 创建临时目录用于测试
            tmp_dir = tempfile.mkdtemp()
            # 如果启用了详细输出选项，打印当前正在运行的分布式测试信息
            if options.verbose:
                init_str = "with {} init_method"
                with_init = init_str.format("file" if with_init_file else "env")
                print_to_stderr(
                    f"Running distributed tests for the {backend} backend {with_init}"
                )
            # 备份当前环境变量
            old_environ = dict(os.environ)
            # 设置测试需要的临时环境变量
            os.environ["TEMP_DIR"] = tmp_dir
            os.environ["BACKEND"] = backend
            os.environ.update(env_vars)
            try:
                # 在临时目录下创建必要的测试文件夹
                os.mkdir(os.path.join(tmp_dir, "barrier"))
                os.mkdir(os.path.join(tmp_dir, "test_dir"))
                # 如果当前后端是 "mpi"，则进行 mpiexec 的特定选项测试
                if backend == "mpi":
                    # 测试 mpiexec 命令的 --noprefix 选项
                    with open(os.devnull, "w") as devnull:
                        allowrunasroot_opt = (
                            "--allow-run-as-root"
                            if subprocess.call(
                                'mpiexec --allow-run-as-root -n 1 bash -c ""',
                                shell=True,
                                stdout=devnull,
                                stderr=subprocess.STDOUT,
                            )
                            == 0
                            else ""
                        )
                        noprefix_opt = (
                            "--noprefix"
                            if subprocess.call(
                                f'mpiexec {allowrunasroot_opt} -n 1 --noprefix bash -c ""',
                                shell=True,
                                stdout=devnull,
                                stderr=subprocess.STDOUT,
                            )
                            == 0
                            else ""
                        )
                    # 构建 mpiexec 命令的参数列表
                    mpiexec = ["mpiexec", "-n", "3", noprefix_opt, allowrunasroot_opt]
                    # 运行测试，并返回测试结果代码
                    return_code = run_test(
                        test_module, test_directory, options, launcher_cmd=mpiexec
                    )
                else:
                    # 对于非 MPI 后端，运行测试，并返回测试结果代码
                    return_code = run_test(
                        test_module,
                        test_directory,
                        options,
                        extra_unittest_args=["--subprocess"],
                    )
                # 如果测试结果代码不为 0，则返回该代码
                if return_code != 0:
                    return return_code
            finally:
                # 清理临时目录及恢复原始环境变量
                shutil.rmtree(tmp_dir)
                os.environ.clear()
                os.environ.update(old_environ)
    # 所有测试运行完成后，返回代码 0 表示成功
    return 0
# 运行 doctest 测试，使用指定的模块和目录，及选项
def run_doctests(test_module, test_directory, options):
    """
    Assumes the incoming test module is called doctest, and simply executes the
    xdoctest runner on the torch library itself.
    """
    # 导入 xdoctest 模块
    import xdoctest

    # 获取 torch 库的路径
    pkgpath = Path(torch.__file__).parent

    # 排除的模块列表
    exclude_module_list = ["torch._vendor.*"]

    # 各个功能测试的启用情况
    enabled = {
        # TODO: 将这些选项暴露给用户
        # 暂时禁用所有基于特性条件的测试
        # 'lapack': 'auto',
        # 'cuda': 'auto',
        # 'cuda1': 'auto',
        # 'qengine': 'auto',
        "lapack": 0,
        "cuda": 0,
        "cuda1": 0,
        "qengine": 0,
        "autograd_profiler": 0,
        "cpp_ext": 0,
        "monitor": 0,
        "onnx": "auto",
    }

    # 根据条件设置 CUDA 功能测试的启用状态
    if enabled["cuda"] == "auto" and torch.cuda.is_available():
        enabled["cuda"] = True

    # 根据条件设置多 CUDA 设备功能测试的启用状态
    if (
        enabled["cuda1"] == "auto"
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    ):
        enabled["cuda1"] = True

    # 根据条件设置 LAPACK 功能测试的启用状态
    if enabled["lapack"] == "auto" and torch._C.has_lapack:
        enabled["lapack"] = True

    # 根据条件设置量化引擎功能测试的启用状态
    if enabled["qengine"] == "auto":
        try:
            # 检查是否导入了量化模块
            import torch.ao.nn.quantized as nnq  # NOQA: F401

            # 设置量化后端引擎为 "qnnpack" 或 "fbgemm"
            torch.backends.quantized.engine = "qnnpack"
            torch.backends.quantized.engine = "fbgemm"
        except (ImportError, RuntimeError):
            ...
        else:
            enabled["qengine"] = True

    # 根据条件设置 ONNX 功能测试的启用状态
    if enabled["onnx"] == "auto":
        try:
            import onnx  # NOQA: F401
            import onnxruntime  # NOQA: F401
            import onnxscript  # NOQA: F401
        except ImportError:
            # 如果导入失败，将 ONNX 相关模块加入排除列表，并禁用 ONNX 功能测试
            exclude_module_list.append("torch.onnx.*")
            enabled["onnx"] = False
        else:
            enabled["onnx"] = True

    # 设置 doctest 的环境变量

    if enabled["cuda"]:
        os.environ["TORCH_DOCTEST_CUDA"] = "1"

    if enabled["cuda1"]:
        os.environ["TORCH_DOCTEST_CUDA1"] = "1"

    if enabled["lapack"]:
        os.environ["TORCH_DOCTEST_LAPACK"] = "1"

    if enabled["qengine"]:
        os.environ["TORCH_DOCTEST_QENGINE"] = "1"

    if enabled["autograd_profiler"]:
        os.environ["TORCH_DOCTEST_AUTOGRAD_PROFILER"] = "1"

    if enabled["cpp_ext"]:
        os.environ["TORCH_DOCTEST_CPP_EXT"] = "1"

    if enabled["monitor"]:
        os.environ["TORCH_DOCTEST_MONITOR"] = "1"

    if enabled["onnx"]:
        os.environ["TORCH_DOCTEST_ONNX"] = "1"

    # 可能的未来测试项，当前未启用
    if 0:
        # TODO: 可尝试启用其中一些
        os.environ["TORCH_DOCTEST_QUANTIZED_DYNAMIC"] = "1"
        os.environ["TORCH_DOCTEST_ANOMALY"] = "1"
        os.environ["TORCH_DOCTEST_AUTOGRAD"] = "1"
        os.environ["TORCH_DOCTEST_HUB"] = "1"
        os.environ["TORCH_DOCTEST_DATALOADER"] = "1"
        os.environ["TORCH_DOCTEST_FUTURES"] = "1"

    # 获取 torch 库的路径
    pkgpath = os.path.dirname(torch.__file__)
    # 定义 xdoctest_config 字典，配置 xdoctest 的全局执行环境和样式
    xdoctest_config = {
        "global_exec": r"\n".join(
            [
                "from torch import nn",
                "import torch.nn.functional as F",
                "import torch",
            ]
        ),
        "analysis": "static",  # 分析模式设为静态，可以设为 "auto" 以测试编译模块中的文档测试
        "style": "google",  # 设定文档测试的样式为 Google 风格
        "options": "+IGNORE_WHITESPACE",  # 设置文档测试选项，忽略空白
    }
    # 确定详细输出级别，至少为 1（取 options.verbose 和 1 的最大值）
    xdoctest_verbose = max(1, options.verbose)
    # 运行 xdoctest 的模块测试，并获取运行结果摘要
    run_summary = xdoctest.runner.doctest_module(
        os.fspath(pkgpath),  # 指定要测试的模块路径
        config=xdoctest_config,  # 使用上面定义的配置
        verbose=xdoctest_verbose,  # 设置详细输出级别
        command=options.xdoctest_command,  # 指定 xdoctest 命令
        argv=[],  # 传递空的命令行参数列表
        exclude=exclude_module_list,  # 排除的模块列表
    )
    # 如果测试结果中存在失败的测试用例，则返回 1；否则返回 0
    result = 1 if run_summary.get("n_failed", 0) else 0
    # 返回测试结果
    return result
# 将文件名中的 '\\' 替换为 '.'，'/' 替换为 '.'，空格替换为 '_'
def sanitize_file_name(file: str):
    return file.replace("\\", ".").replace("/", ".").replace(" ", "_")


# 处理日志文件，根据测试情况进行重命名和打印日志内容
def handle_log_file(
    test: ShardedTest,  # 测试对象
    file_path: str,      # 日志文件路径
    failed: bool,        # 测试是否失败的标志
    was_rerun: bool      # 是否重新运行过测试的标志
) -> None:
    test = str(test)  # 将测试对象转换为字符串形式
    with open(file_path, errors="ignore") as f:
        full_text = f.read()  # 读取整个日志文件内容

    # 构造新的文件名路径，格式为 'test/test-reports/test_{随机16进制串}_.log'
    new_file = "test/test-reports/" + sanitize_file_name(
        f"{test}_{os.urandom(8).hex()}_.log"
    )
    # 重命名文件，移动到新路径下
    os.rename(file_path, REPO_ROOT / new_file)

    # 如果测试成功且未重新运行且日志文件中不包含 '=== RERUNS ===' 字符串
    if not failed and not was_rerun and "=== RERUNS ===" not in full_text:
        # 打印成功信息和日志文件路径
        print_to_stderr(
            f"\n{test} was successful, full logs can be found in artifacts with path {new_file}"
        )
        # 遍历日志文件内容的每一行，打印包含 'Running ... items in this shard:' 的行
        for line in full_text.splitlines():
            if re.search("Running .* items in this shard:", line):
                print_to_stderr(line.rstrip())
        print_to_stderr("")  # 打印空行
        return

    # 否则：打印整个日志文件内容
    print_to_stderr(f"\nPRINTING LOG FILE of {test} ({new_file})")
    print_to_stderr(full_text)
    print_to_stderr(f"FINISHED PRINTING LOG FILE of {test} ({new_file})\n")


# 获取 pytest 的运行参数列表
def get_pytest_args(options, is_cpp_test=False, is_distributed_test=False):
    if RERUN_DISABLED_TESTS:
        # 如果禁用了重新运行测试，根据测试类型设置不同的重新运行选项
        # 分布式测试或 ASAN 测试，重新运行次数为 15 次，否则为 50 次
        count = 15 if is_distributed_test or TEST_WITH_ASAN else 50
        rerun_options = ["--flake-finder", f"--flake-runs={count}"]
    else:
        # 正常模式下，如果测试失败，最多重新运行 2 次，-x 表示第一次失败后停止
        rerun_options = ["-x", "--reruns=2"]

    # 默认的 pytest 参数列表
    pytest_args = [
        "-vv",   # 输出详细信息
        "-rfEX", # 显示出错信息、退出信息和调试信息
    ]
    if not is_cpp_test:
        # 非 C++ 测试，不使用 xdist 插件，而是使用 pytest 插件
        pytest_args.extend(["-p", "no:xdist", "--use-pytest"])
    else:
        # 使用 pytext-dist 运行 C++ 测试，通过多进程加速运行
        pytest_args.extend(["-n", str(NUM_PROCS)])

        if IS_CI:
            # 在 CI 环境下生成 XML 测试报告
            test_report_path = get_report_path(pytest=True)
            pytest_args.extend(["--junit-xml-reruns", test_report_path])

    if options.pytest_k_expr:
        # 如果指定了 pytest 的 -k 表达式，添加到参数列表中
        pytest_args.extend(["-k", options.pytest_k_expr])
    # 将 rerun_options 列表中的元素逐一添加到 pytest_args 列表中
    pytest_args.extend(rerun_options)
    # 返回更新后的 pytest_args 列表作为函数结果
    return pytest_args
# 定义函数，用于运行 CI 稳定性检查测试
def run_ci_sanity_check(test: ShardedTest, test_directory, options):
    # 断言检查测试名称是否为 "test_ci_sanity_check_fail"
    assert (
        test.name == "test_ci_sanity_check_fail"
    ), f"This handler only works for test_ci_sanity_check_fail, got {test.name}"
    
    # 调用运行测试的函数，设置打印日志为 False，获取返回码
    ret_code = run_test(test, test_directory, options, print_log=False)
    
    # 检查测试是否失败，应该返回 1
    if ret_code != 1:
        return 1
    
    # 指定测试报告存储目录
    test_reports_dir = str(REPO_ROOT / "test/test-reports")
    
    # 删除测试生成的日志文件和 XML 文件
    for file in glob.glob(f"{test_reports_dir}/{test.name}*.log"):
        os.remove(file)
    
    # 删除测试生成的目录和子目录
    for dirname in glob.glob(f"{test_reports_dir}/**/{test.name}"):
        shutil.rmtree(dirname)
    
    # 返回成功状态码
    return 0
    parser.add_argument(
        "--functorch",
        "--functorch",
        action="store_true",
        help=(
            "If this flag is present, we will only run functorch tests. "
            "If this flag is not present, we will run all tests "
            "(including functorch tests)."
        ),
    )
    # 添加命令行参数 --functorch，当存在时仅运行 functorch 测试，否则运行所有测试（包括 functorch 测试）

    parser.add_argument(
        "--mps",
        "--mps",
        action="store_true",
        help=("If this flag is present, we will only run test_mps and test_metal"),
    )
    # 添加命令行参数 --mps，当存在时仅运行 test_mps 和 test_metal 测试

    parser.add_argument(
        "--xpu",
        "--xpu",
        action="store_true",
        help=("If this flag is present, we will run xpu tests except XPU_BLOCK_LIST"),
    )
    # 添加命令行参数 --xpu，当存在时运行 xpu 测试，但排除 XPU_BLOCK_LIST 中指定的测试

    parser.add_argument(
        "--cpp",
        "--cpp",
        action="store_true",
        help=("If this flag is present, we will only run C++ tests"),
    )
    # 添加命令行参数 --cpp，当存在时仅运行 C++ 测试

    parser.add_argument(
        "-core",
        "--core",
        action="store_true",
        help="Only run core tests, or tests that validate PyTorch's ops, modules,"
        "and autograd. They are defined by CORE_TEST_LIST.",
    )
    # 添加命令行参数 --core，仅运行核心测试，即验证 PyTorch 的操作、模块和自动求导功能，这些测试由 CORE_TEST_LIST 定义

    parser.add_argument(
        "--onnx",
        "--onnx",
        action="store_true",
        help=(
            "Only run ONNX tests, or tests that validate PyTorch's ONNX export. "
            "If this flag is not present, we will exclude ONNX tests."
        ),
    )
    # 添加命令行参数 --onnx，仅运行 ONNX 测试，即验证 PyTorch 的 ONNX 导出，如果不存在此标志，则排除 ONNX 测试

    parser.add_argument(
        "-k",
        "--pytest-k-expr",
        default="",
        help="Pass to pytest as its -k expr argument",
    )
    # 添加命令行参数 -k 或 --pytest-k-expr，将其作为 pytest 的 -k 表达式参数传递

    parser.add_argument(
        "-c",
        "--coverage",
        action="store_true",
        help="enable coverage",
        default=PYTORCH_COLLECT_COVERAGE,
    )
    # 添加命令行参数 -c 或 --coverage，启用测试覆盖率，默认使用 PYTORCH_COLLECT_COVERAGE 的值

    parser.add_argument(
        "-i",
        "--include",
        nargs="+",
        choices=TestChoices(TESTS),
        default=TESTS,
        metavar="TESTS",
        help="select a set of tests to include (defaults to ALL tests)."
        " tests must be a part of the TESTS list defined in run_test.py",
    )
    # 添加命令行参数 -i 或 --include，选择要包含的一组测试（默认为所有测试），必须是 run_test.py 中 TESTS 列表的一部分

    parser.add_argument(
        "-x",
        "--exclude",
        nargs="+",
        choices=TESTS,
        metavar="TESTS",
        default=[],
        help="select a set of tests to exclude",
    )
    # 添加命令行参数 -x 或 --exclude，选择要排除的一组测试

    parser.add_argument(
        "--ignore-win-blocklist",
        action="store_true",
        help="always run blocklisted windows tests",
    )
    # 添加命令行参数 --ignore-win-blocklist，始终运行 Windows 的 blocklisted 测试

    # NS: Disable target determination until it can be made more reliable
    # parser.add_argument(
    #     "--determine-from",
    #     help="File of affected source filenames to determine which tests to run.",
    # )
    # 禁用目标确定功能，直到其更加可靠

    parser.add_argument(
        "--continue-through-error",
        "--keep-going",
        action="store_true",
        help="Runs the full test suite despite one of the tests failing",
        default=strtobool(os.environ.get("CONTINUE_THROUGH_ERROR", "False")),
    )
    # 添加命令行参数 --continue-through-error 或 --keep-going，即使有一个测试失败也继续运行完整的测试套件
    # 添加一个命令行参数，用于在运行测试时将日志输出到文件中。如果在 CI 环境并且环境变量 VERBOSE_TEST_LOGS 未设置为 True，则默认为 True。
    parser.add_argument(
        "--pipe-logs",
        action="store_true",
        help="Print logs to output file while running tests.  True if in CI and env var is not set",
        default=IS_CI and not strtobool(os.environ.get("VERBOSE_TEST_LOGS", "False")),
    )

    # 添加一个命令行参数，用于根据测试时间的 JSON 文件设置超时。仅在有可用的测试时间时有效。默认情况下，在 CI 环境中并且环境变量 NO_TEST_TIMEOUT 未设置为 True 时为 True。
    parser.add_argument(
        "--enable-timeout",
        action="store_true",
        help="Set a timeout based on the test times json file.  Only works if there are test times available",
        default=IS_CI and not strtobool(os.environ.get("NO_TEST_TIMEOUT", "False")),
    )

    # 添加一个命令行参数，用于基于特定的 TD 移除测试。条件包括在 CI 环境中，同时满足多个配置条件和环境变量设置。
    parser.add_argument(
        "--enable-td",
        action="store_true",
        help="Enables removing tests based on TD",
        default=IS_CI
        and (
            TEST_WITH_CROSSREF
            or TEST_WITH_ASAN
            or (TEST_CONFIG == "distributed" and TEST_CUDA)
            or (IS_WINDOWS and not TEST_CUDA)
            or TEST_CONFIG == "nogpu_AVX512"
            or TEST_CONFIG == "nogpu_NO_AVX2"
            or TEST_CONFIG == "default"
        )
        and get_pr_number() is not None
        and not strtobool(os.environ.get("NO_TD", "False"))
        and not TEST_WITH_ROCM
        and not IS_MACOS
        and "xpu" not in BUILD_ENVIRONMENT
        and "onnx" not in BUILD_ENVIRONMENT
        and os.environ.get("GITHUB_WORKFLOW", "slow") in ("trunk", "pull"),
    )

    # 添加一个命令行参数，允许指定测试的分片运行。参数类型为两个整数，表示将选定的测试分成几个片段并运行其中一个片段。
    parser.add_argument(
        "--shard",
        nargs=2,
        type=int,
        help="runs a shard of the tests (taking into account other selections), e.g., "
        "--shard 2 3 will break up the selected tests into 3 shards and run the tests "
        "in the 2nd shard (the first number should not exceed the second)",
    )

    # 添加一个命令行参数，排除特定 JIT 配置的测试。
    parser.add_argument(
        "--exclude-jit-executor",
        action="store_true",
        help="exclude tests that are run for a specific jit config",
    )

    # 添加一个命令行参数，排除 Torch 导出测试。
    parser.add_argument(
        "--exclude-torch-export-tests",
        action="store_true",
        help="exclude torch export tests",
    )

    # 添加一个命令行参数，排除分布式测试。
    parser.add_argument(
        "--exclude-distributed-tests",
        action="store_true",
        help="exclude distributed tests",
    )

    # 添加一个命令行参数，排除感应器测试。
    parser.add_argument(
        "--exclude-inductor-tests",
        action="store_true",
        help="exclude inductor tests",
    )

    # 添加一个命令行参数，仅列出将要运行的测试，而不实际运行。
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the test that will run.",
    )

    # 添加一个命令行参数，控制特定的 doctest 操作。
    parser.add_argument(
        "--xdoctest-command",
        default="all",
        help=(
            "Control the specific doctest action. "
            "Use 'list' to simply parse doctests and check syntax. "
            "Use 'all' to execute all doctests or specify a specific "
            "doctest to run"
        ),
    )

    # 添加一个命令行参数，运行测试时不进行翻译验证。
    parser.add_argument(
        "--no-translation-validation",
        action="store_false",
        help="Run tests without translation validation.",
    )

    # 添加一个互斥组，用于处理互斥的命令行参数选择。
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dynamo",
        action="store_true",
        help="Run tests with TorchDynamo+EagerBackend turned on",
    )
    # 添加命令行参数 "--dynamo"，当存在时设置为 True，用于启用 TorchDynamo+EagerBackend

    group.add_argument(
        "--inductor",
        action="store_true",
        help="Run tests with TorchInductor turned on",
    )
    # 添加命令行参数 "--inductor"，当存在时设置为 True，用于启用 TorchInductor

    args, extra = parser.parse_known_args()
    # 解析命令行参数，并获取额外的未知参数

    if "--" in extra:
        extra.remove("--")
    # 如果额外的参数列表中包含 "--"，则移除该项

    args.additional_args = extra
    # 将未知参数列表赋值给 args 对象的 additional_args 属性

    return args
    # 返回解析后的命令行参数对象 args
# 从选定的测试中排除特定的测试用例
def exclude_tests(
    exclude_list, selected_tests, exclude_message=None, exact_match=False
):
    # 遍历要排除的测试用例列表
    for exclude_test in exclude_list:
        # 复制一份选定的测试用例列表
        tests_copy = selected_tests[:]
        # 遍历复制的测试用例列表
        for test in tests_copy:
            # 如果不需要精确匹配且测试用例以排除的测试用例开头，或者测试用例正好等于排除的测试用例
            if (
                not exact_match and test.startswith(exclude_test)
            ) or test == exclude_test:
                # 如果有排除消息，打印到标准错误输出
                if exclude_message is not None:
                    print_to_stderr(f"Excluding {test} {exclude_message}")
                # 从选定的测试用例列表中移除该测试用例
                selected_tests.remove(test)
    # 返回更新后的选定的测试用例列表
    return selected_tests


# 判断给定文件是否必须串行执行测试
def must_serial(file: Union[str, ShardedTest]) -> bool:
    # 如果文件是ShardedTest实例，则取其名称
    if isinstance(file, ShardedTest):
        file = file.name
    # 判断是否设置了环境变量PYTORCH_TEST_RUN_EVERYTHING_IN_SERIAL为"1"
    # 或者测试配置中包含分布式测试的前缀，或者文件名包含分布式测试的前缀，
    # 或者文件名在自定义处理程序列表中，或者在并行运行黑名单中，或者在CI串行列表中，
    # 或者在JIT执行器测试列表中，或者在ONNX串行列表中，或者进程数为1
    return (
        os.getenv("PYTORCH_TEST_RUN_EVERYTHING_IN_SERIAL", "0") == "1"
        or DISTRIBUTED_TEST_PREFIX in os.getenv("TEST_CONFIG", "")
        or DISTRIBUTED_TEST_PREFIX in file
        or file in CUSTOM_HANDLERS
        or file in RUN_PARALLEL_BLOCKLIST
        or file in CI_SERIAL_LIST
        or file in JIT_EXECUTOR_TESTS
        or file in ONNX_SERIAL_LIST
        or NUM_PROCS == 1
    )


# 判断是否可以在pytest中运行测试
def can_run_in_pytest(test):
    # 判断是否设置了环境变量PYTORCH_TEST_DO_NOT_USE_PYTEST为"0"
    return os.getenv("PYTORCH_TEST_DO_NOT_USE_PYTEST", "0") == "0"


# 根据选项获取选定的测试用例列表
def get_selected_tests(options) -> List[str]:
    # 获取包含的测试用例列表
    selected_tests = options.include

    # 如果设置了--jit选项，则只保留包含"jit"的测试用例
    if options.jit:
        selected_tests = list(
            filter(lambda test_name: "jit" in test_name, selected_tests)
        )

    # 如果设置了--distributed_tests选项，则只保留包含在DISTRIBUTED_TESTS中的测试用例
    if options.distributed_tests:
        selected_tests = list(
            filter(lambda test_name: test_name in DISTRIBUTED_TESTS, selected_tests)
        )

    # 如果设置了--core选项，则只保留包含在CORE_TEST_LIST中的测试用例
    if options.core:
        selected_tests = list(
            filter(lambda test_name: test_name in CORE_TEST_LIST, selected_tests)
        )

    # 如果设置了--functorch选项，则只保留在FUNCTORCH_TESTS中的测试用例
    if options.functorch:
        selected_tests = [tname for tname in selected_tests if tname in FUNCTORCH_TESTS]

    # 如果设置了--cpp选项，则只保留在CPP_TESTS中的测试用例，否则将所有C++测试用例加入排除列表
    if options.cpp:
        selected_tests = [tname for tname in selected_tests if tname in CPP_TESTS]
    else:
        options.exclude.extend(CPP_TESTS)

    # 如果设置了--mps选项，则只保留特定的mps测试用例，否则将所有mps测试用例加入排除列表
    if options.mps:
        selected_tests = ["test_mps", "test_metal", "test_modules"]
    else:
        options.exclude.extend(["test_mps", "test_metal"])

    # 如果设置了--xpu选项，则排除XPU_BLOCKLIST中的测试用例，否则将所有XPU测试用例加入排除列表
    if options.xpu:
        selected_tests = exclude_tests(XPU_BLOCKLIST, selected_tests, "on XPU")
    else:
        options.exclude.extend(XPU_TEST)

    # 如果设置了--onnx选项，则只保留在ONNX_TESTS中的测试用例，否则将所有ONNX测试用例加入排除列表
    onnx_tests = [tname for tname in selected_tests if tname in ONNX_TESTS]
    if options.onnx:
        selected_tests = onnx_tests
    else:
        options.exclude.extend(onnx_tests)
    # 处理排除选项
    # 如果设置了排除 JIT 执行器测试的选项，则将其加入排除列表中
    if options.exclude_jit_executor:
        options.exclude.extend(JIT_EXECUTOR_TESTS)

    # 如果设置了排除分布式测试的选项，则将其加入排除列表中
    if options.exclude_distributed_tests:
        options.exclude.extend(DISTRIBUTED_TESTS)

    # 如果设置了排除感应器测试的选项，则将其加入排除列表中
    if options.exclude_inductor_tests:
        options.exclude.extend(INDUCTOR_TESTS)

    # 如果设置了排除 Torch 导出测试的选项，则将其加入排除列表中
    if options.exclude_torch_export_tests:
        options.exclude.extend(TORCH_EXPORT_TESTS)

    # 在 CUDA 11.6 中这些测试失败了，临时禁用。问题链接：https://github.com/pytorch/pytorch/issues/75375
    # 如果当前使用的是 CUDA，则将特定测试加入排除列表中
    if torch.version.cuda is not None:
        options.exclude.extend(["distributions/test_constraints"])

    # 在 Python 3.12 中这些测试失败了，临时禁用
    # 如果 Python 版本大于等于 3.12，则将指定的测试加入排除列表中
    if sys.version_info >= (3, 12):
        options.exclude.extend(
            [
                "functorch/test_dims",
                "functorch/test_rearrange",
                "functorch/test_parsing",
                "functorch/test_memory_efficient_fusion",
                "torch_np/numpy_tests/core/test_multiarray",
            ]
        )

    # 根据排除列表中的测试，选择性地排除测试用例
    selected_tests = exclude_tests(options.exclude, selected_tests)

    # 如果运行环境是 win32 并且未忽略 Windows 阻止列表，则进行以下处理
    if sys.platform == "win32" and not options.ignore_win_blocklist:
        # 获取目标架构信息
        target_arch = os.environ.get("VSCMD_ARG_TGT_ARCH")
        # 如果目标架构不是 x64，则将特定测试加入 Windows 阻止列表中
        if target_arch != "x64":
            WINDOWS_BLOCKLIST.append("cpp_extensions_aot_no_ninja")
            WINDOWS_BLOCKLIST.append("cpp_extensions_aot_ninja")
            WINDOWS_BLOCKLIST.append("cpp_extensions_jit")
            WINDOWS_BLOCKLIST.append("jit")
            WINDOWS_BLOCKLIST.append("jit_fuser")

        # 根据 Windows 阻止列表排除相应的测试
        selected_tests = exclude_tests(WINDOWS_BLOCKLIST, selected_tests, "on Windows")

    # 如果测试使用 ROCm，则根据 ROCm 阻止列表排除相应的测试
    elif TEST_WITH_ROCM:
        selected_tests = exclude_tests(ROCM_BLOCKLIST, selected_tests, "on ROCm")

    # 如果分布式包不可用，则跳过所有分布式测试
    if not dist.is_available():
        selected_tests = exclude_tests(
            DISTRIBUTED_TESTS,
            selected_tests,
            "PyTorch is built without distributed support.",
        )

    # 如果 PyTorch 没有安装 LAPACK，则跳过需要 LAPACK 的测试
    if not torch._C.has_lapack:
        selected_tests = exclude_tests(
            TESTS_REQUIRING_LAPACK,
            selected_tests,
            "PyTorch is built without LAPACK support.",
        )

    # 如果处于慢速 gradcheck 模式，则跳过不使用 gradcheck 的测试
    if TEST_WITH_SLOW_GRADCHECK:
        selected_tests = exclude_tests(
            TESTS_NOT_USING_GRADCHECK,
            selected_tests,
            "Running in slow gradcheck mode, skipping tests "
            "that don't use gradcheck.",
            exact_match=True,
        )

    # 解析选定测试模块，返回测试列表
    selected_tests = [parse_test_module(x) for x in selected_tests]
    return selected_tests
# 从文件中加载测试时间信息，用于分片决策
def load_test_times_from_file(
    file: str,
) -> Dict[str, Any]:
    # 构建文件路径，以确保在操作系统上的正确性
    path = os.path.join(str(REPO_ROOT), file)
    # 如果文件不存在，则打印警告信息并返回空字典
    if not os.path.exists(path):
        print_to_stderr(
            f"::warning:: Failed to find test times file `{path}`. Using round robin sharding."
        )
        return {}

    # 打开文件并加载 JSON 数据，转换为字典类型
    with open(path) as f:
        # 使用类型注释确保加载的数据结构符合预期
        test_times_file = cast(Dict[str, Any], json.load(f))
    
    # 获取当前构建环境的环境变量
    build_environment = os.environ.get("BUILD_ENVIRONMENT")
    # 获取测试配置的环境变量
    test_config = os.environ.get("TEST_CONFIG")
    
    # 如果测试配置存在于构建环境对应的测试时间文件中，则返回相关测试时间信息
    if test_config in test_times_file.get(build_environment, {}):
        print_to_stderr("Found test times from artifacts")
        return test_times_file[build_environment][test_config]
    # 否则，如果存在于默认测试时间文件中，则返回相关测试时间信息
    elif test_config in test_times_file["default"]:
        print_to_stderr(
            f"::warning:: Gathered no stats from artifacts for {build_environment} build env"
            f" and {test_config} test config. Using default build env and {test_config} test config instead."
        )
        return test_times_file["default"][test_config]
    # 如果都不存在，则返回默认测试时间信息
    else:
        print_to_stderr(
            f"::warning:: Gathered no stats from artifacts for build env {build_environment} build env"
            f" and {test_config} test config. Using default build env and default test config instead."
        )
        return test_times_file["default"]["default"]


# 加载额外的测试文件时间信息，默认为指定文件夹下的特定文件
def load_test_file_times(
    file: str = ADDITIONAL_CI_FILES_FOLDER / TEST_TIMES_FILE,
) -> Dict[str, float]:
    return cast(Dict[str, float], load_test_times_from_file(file))


# 加载测试类别时间信息，默认为指定文件夹下的特定文件
def load_test_class_times(
    file: str = ADDITIONAL_CI_FILES_FOLDER / TEST_CLASS_TIMES_FILE,
) -> Dict[str, Dict[str, float]]:
    return cast(Dict[str, Dict[str, float]], load_test_times_from_file(file))


# 获取分片选项，根据用户提供的选项参数确定分片的具体设置
def get_sharding_opts(options) -> Tuple[int, int]:
    which_shard, num_shards = 1, 1
    # 如果选项中包含分片信息，则进行验证和设置
    if options.shard:
        assert len(options.shard) == 2, "Unexpected shard format"
        assert min(options.shard) > 0, "Shards must be positive numbers"
        which_shard, num_shards = options.shard
        assert (
            which_shard <= num_shards
        ), "Selected shard must be less than or equal to total number of shards"

    return (which_shard, num_shards)


# 进行测试分片，基于给定的选项和测试信息进行分片计算
def do_sharding(
    options,
    selected_tests: Sequence[TestRun],
    test_file_times: Dict[str, float],
    test_class_times: Dict[str, Dict[str, float]],
    sort_by_time: bool = True,
) -> Tuple[float, List[ShardedTest]]:
    # 获取分片选项
    which_shard, num_shards = get_sharding_opts(options)

    # 计算分片
    shards = calculate_shards(
        num_shards,
        selected_tests,
        test_file_times,
        test_class_times=test_class_times,
        must_serial=must_serial,
        sort_by_time=sort_by_time,
    )
    # 返回选定的分片
    return shards[which_shard - 1]


# 表示测试失败的命名元组，包含测试信息和失败消息
class TestFailure(NamedTuple):
    test: TestRun
    message: str


# 运行测试模块，返回可能的测试失败信息
def run_test_module(
    test: ShardedTest, test_directory: str, options
) -> Optional[TestFailure]:
    # 实际运行测试模块的函数体，未提供具体代码
    try:
        maybe_set_hip_visible_devies()
        # 调用可能存在的函数设置环境变量或状态

        test_name = test.name
        # 获取测试对象的名称

        # 打印当前测试的运行时间，用于诊断慢速测试
        print_to_stderr(f"Running {str(test)} ... [{datetime.now()}]")

        handler = CUSTOM_HANDLERS.get(test_name, run_test)
        # 根据测试名称从自定义处理器字典中获取处理函数，如果不存在则使用默认的运行测试函数

        return_code = handler(test, test_directory, options)
        # 调用处理函数来运行测试，获取返回的状态码

        assert isinstance(return_code, int) and not isinstance(
            return_code, bool
        ), f"While running {str(test)} got non integer return code {return_code}"
        # 断言返回码是整数且不是布尔类型，如果不符合条件则抛出异常

        if return_code == 0:
            return None
        # 如果测试返回码为0，表示测试成功，返回None

        message = f"{str(test)} failed!"
        # 准备失败信息，包含测试对象的名称

        if return_code < 0:
            # 如果返回码小于0，说明测试过程中收到了信号
            # subprocess.Popen返回子进程的退出信号作为负数的返回码，其中N是信号编号
            signal_name = SIGNALS_TO_NAMES_DICT[-return_code]
            message += f" Received signal: {signal_name}"
            # 添加接收到的信号信息到失败消息中

        return TestFailure(test.test, message)
        # 返回一个TestFailure对象，表示测试失败，包含测试对象和失败消息

    except Exception as e:
        # 捕获所有异常
        return TestFailure(test.test, f"{str(test)} failed! {e}")
        # 返回一个TestFailure对象，表示测试失败，包含测试对象和异常信息
# 运行测试函数，接受选定的测试集合、测试目录、选项和失败列表作为参数，无返回值
def run_tests(
    selected_tests: List[ShardedTest],
    test_directory: str,
    options,
    failures: List[TestFailure],
) -> None:
    # 如果选定的测试集合为空，直接返回，不执行后续操作
    if len(selected_tests) == 0:
        return

    # 根据条件筛选出可以并行执行的测试列表
    selected_tests_parallel = [x for x in selected_tests if not must_serial(x)]
    # 根据条件筛选出需要串行执行的测试列表
    selected_tests_serial = [
        x for x in selected_tests if x not in selected_tests_parallel
    ]

    # 根据不同的条件创建并行处理池，这里使用了多进程池
    # 参考文档 [ROCm parallel CI testing]
    pool = get_context("spawn").Pool(
        NUM_PROCS, maxtasksperchild=None if torch.version.hip else 1
    )

    # 注意：这是一个hack，用于将conftest.py及其依赖文件复制到CPP_TESTS_DIR中
    # 以确保在CPP_TESTS_DIR中可以访问这些文件
    conftest_files = [
        "conftest.py",
        "pytest_shard_custom.py",
    ]
    for conftest_file in conftest_files:
        cpp_file = os.path.join(CPP_TESTS_DIR, conftest_file)
        # 如果开启了cpp选项，并且CPP_TESTS_DIR存在且是一个目录，并且cpp_file不存在
        # 则将测试目录中的conftest_file复制到CPP_TESTS_DIR中
        if (
            options.cpp
            and os.path.exists(CPP_TESTS_DIR)
            and os.path.isdir(CPP_TESTS_DIR)
            and not os.path.exists(cpp_file)
        ):
            shutil.copy(os.path.join(test_directory, conftest_file), cpp_file)

    # 定义处理错误消息的函数，将失败的测试记录到failures列表中，并打印错误消息到标准错误输出
    def handle_error_messages(failure: Optional[TestFailure]):
        if failure is None:
            return False
        failures.append(failure)
        print_to_stderr(failure.message)
        return True

    # 定义并行测试完成的回调函数，处理测试失败情况
    def parallel_test_completion_callback(failure):
        test_failed = handle_error_messages(failure)
        # 如果测试失败且未指定继续运行选项，并且未禁用重试失败的测试，则终止进程池
        if (
            test_failed
            and not options.continue_through_error
            and not RERUN_DISABLED_TESTS
        ):
            pool.terminate()

    # 提示消息，说明可以通过传递--keep-going参数来在失败后继续运行测试
    # 如果在CI环境中运行，建议将'keep-going'标签添加到PR并重新运行作业
    keep_going_message = (
        "\n\nTip: You can keep running tests even on failure by passing --keep-going to run_test.py.\n"
        "If running on CI, add the 'keep-going' label to your PR and rerun your jobs."
    )
    try:
        # 对于每个被选中的串行测试执行以下操作
        for test in selected_tests_serial:
            # 深拷贝选项，以便在每次循环中修改而不影响原始选项
            options_clone = copy.deepcopy(options)
            # 如果可以在 pytest 中运行测试，则设置选项中的 pytest 标志为 True
            if can_run_in_pytest(test):
                options_clone.pytest = True
            # 运行测试模块，并返回可能的失败情况
            failure = run_test_module(test, test_directory, options_clone)
            # 处理可能的错误消息，并返回测试是否失败
            test_failed = handle_error_messages(failure)
            # 如果测试失败并且不允许继续执行错误以及未禁用重新运行已禁用的测试
            if (
                test_failed
                and not options.continue_through_error
                and not RERUN_DISABLED_TESTS
            ):
                # 抛出运行时错误，包含失败消息和继续运行的提示消息
                raise RuntimeError(failure.message + keep_going_message)

        # 先运行标记为串行的测试
        for test in selected_tests_parallel:
            # 深拷贝选项，以便在每次循环中修改而不影响原始选项
            options_clone = copy.deepcopy(options)
            # 如果可以在 pytest 中运行测试，则设置选项中的 pytest 标志为 True
            if can_run_in_pytest(test):
                options_clone.pytest = True
            # 添加额外的参数以标记这些测试是串行执行的
            options_clone.additional_args.extend(["-m", "serial"])
            # 运行测试模块，并返回可能的失败情况
            failure = run_test_module(test, test_directory, options_clone)
            # 处理可能的错误消息，并返回测试是否失败
            test_failed = handle_error_messages(failure)
            # 如果测试失败并且不允许继续执行错误以及未禁用重新运行已禁用的测试
            if (
                test_failed
                and not options.continue_through_error
                and not RERUN_DISABLED_TESTS
            ):
                # 抛出运行时错误，包含失败消息和继续运行的提示消息
                raise RuntimeError(failure.message + keep_going_message)

        # 设置环境变量 NUM_PARALLEL_PROCS 为并行处理的进程数
        os.environ["NUM_PARALLEL_PROCS"] = str(NUM_PROCS)
        # 对于每个被选中的并行测试执行以下操作
        for test in selected_tests_parallel:
            # 深拷贝选项，以便在每次循环中修改而不影响原始选项
            options_clone = copy.deepcopy(options)
            # 如果可以在 pytest 中运行测试，则设置选项中的 pytest 标志为 True
            if can_run_in_pytest(test):
                options_clone.pytest = True
            # 添加额外的参数以标记这些测试不是串行执行的
            options_clone.additional_args.extend(["-m", "not serial"])
            # 异步地在进程池中运行测试模块，提供测试、测试目录和选项
            pool.apply_async(
                run_test_module,
                args=(test, test_directory, options_clone),
                callback=parallel_test_completion_callback,
            )
        # 关闭进程池，等待所有进程结束
        pool.close()
        pool.join()
        # 删除 NUM_PARALLEL_PROCS 环境变量
        del os.environ["NUM_PARALLEL_PROCS"]

    finally:
        # 终止进程池中所有未完成的进程
        pool.terminate()
        # 等待所有进程池中的进程结束
        pool.join()

    # 函数执行完毕，无返回值
    return
def check_pip_packages() -> None:
    # 定义需要检查的依赖包列表
    packages = [
        "pytest-rerunfailures",
        "pytest-flakefinder",
        "pytest-xdist",
    ]
    # 获取当前已安装的所有包的名称列表
    installed_packages = [i.key for i in pkg_resources.working_set]
    # 遍历需要检查的包列表
    for package in packages:
        # 如果某个包未安装，则打印错误信息并退出程序
        if package not in installed_packages:
            print_to_stderr(
                f"Missing pip dependency: {package}, please run `pip install -r .ci/docker/requirements-ci.txt`"
            )
            sys.exit(1)


def main():
    # 检查所需的 pip 包是否已安装
    check_pip_packages()

    # 解析命令行参数
    options = parse_args()

    # 获取分片选项信息
    which_shard, num_shards = get_sharding_opts(options)
    # 将分片信息作为全局度量标准加入到度量中
    add_global_metric("shard", which_shard)
    add_global_metric("num_shards", num_shards)

    # 设置测试目录路径
    test_directory = str(REPO_ROOT / "test")
    # 获取选定的测试集合
    selected_tests = get_selected_tests(options)

    # 导入测试优先级结果
    test_prioritizations = import_results()
    # 更新测试优先级结果中的选定测试集合
    test_prioritizations.amend_tests(selected_tests)

    # 确保测试报告目录存在，若不存在则创建
    os.makedirs(REPO_ROOT / "test" / "test-reports", exist_ok=True)

    # 如果开启了覆盖率测试且未启用 PyTorch 覆盖率收集，执行覆盖率擦除操作
    if options.coverage and not PYTORCH_COLLECT_COVERAGE:
        shell(["coverage", "erase"])

    # 如果处于持续集成环境
    if IS_CI:
        # 下载测试用例配置到本地环境
        get_test_case_configs(dirpath=test_directory)

    # 加载测试文件执行时间数据
    test_file_times_dict = load_test_file_times()
    # 加载测试类执行时间数据
    test_class_times_dict = load_test_class_times()

    class TestBatch:
        """定义一组具有相似优先级的测试，应在当前分片上一起运行"""

        name: str
        sharded_tests: List[ShardedTest]
        failures: List[TestFailure]

        def __init__(
            self, name: str, raw_tests: Sequence[TestRun], should_sort_shard: bool
        ):
            self.name = name
            self.failures = []
            # 进行测试分片，计算预计时间并获取分片后的测试列表
            self.time, self.sharded_tests = do_sharding(
                options,
                raw_tests,
                test_file_times_dict,
                test_class_times_dict,
                sort_by_time=should_sort_shard,
            )

        def __str__(self):
            # 生成测试批次的字符串表示
            s = f"Name: {self.name} (est. time: {round(self.time / 60, 2)}min)\n"
            # 按照是否必须串行运行划分测试列表
            serial = [test for test in self.sharded_tests if must_serial(test)]
            parallel = [test for test in self.sharded_tests if not must_serial(test)]
            s += f"  Serial tests ({len(serial)}):\n"
            s += "".join(f"    {test}\n" for test in serial)
            s += f"  Parallel tests ({len(parallel)}):\n"
            s += "".join(f"    {test}\n" for test in parallel)
            return s.strip()

    # 根据是否开启测试动态化调整需要运行的测试百分比
    percent_to_run = 25 if options.enable_td else 100
    print_to_stderr(
        f"Running {percent_to_run}% of tests based on TD"
        if options.enable_td
        else "Running all tests"
    )
    # 获取优先级结果中按百分比选择的测试集合和排除的测试集合
    include, exclude = test_prioritizations.get_top_per_tests(percent_to_run)

    # 创建需要运行的测试批次
    test_batch = TestBatch("tests to run", include, False)
    # 创建被排除的测试批次
    test_batch_exclude = TestBatch("excluded", exclude, True)
    # 如果在 CI 环境下：
    if IS_CI:
        # 生成包含包括和排除的对象列表的 CI 构件
        gen_ci_artifact([x.to_json() for x in include], [x.to_json() for x in exclude])

    # 输出并行测试正在运行的信息，使用的进程数为 NUM_PROCS
    print_to_stderr(f"Running parallel tests on {NUM_PROCS} processes")
    # 输出测试批次的信息
    print_to_stderr(test_batch)
    # 输出排除的测试批次的信息
    print_to_stderr(test_batch_exclude)

    # 如果设置了 dry_run 选项，则直接返回，不执行后续操作
    if options.dry_run:
        return

    # 如果设置了 dynamo 选项：
    if options.dynamo:
        # 设置环境变量 PYTORCH_TEST_WITH_DYNAMO 为 "1"
        os.environ["PYTORCH_TEST_WITH_DYNAMO"] = "1"

    # 如果设置了 inductor 选项：
    elif options.inductor:
        # 设置环境变量 PYTORCH_TEST_WITH_INDUCTOR 为 "1"
        os.environ["PYTORCH_TEST_WITH_INDUCTOR"] = "1"

    # 如果未设置 no_translation_validation 选项：
    if not options.no_translation_validation:
        # 设置环境变量 PYTORCH_TEST_WITH_TV 为 "1"
        os.environ["PYTORCH_TEST_WITH_TV"] = "1"

    try:
        # 实际执行测试
        start_time = time.time()
        elapsed_time = time.time() - start_time
        # 输出测试批次的开始信息和经过的时间
        print_to_stderr(
            f"Starting test batch '{test_batch.name}' {round(elapsed_time, 2)} seconds after initiating testing"
        )
        # 运行测试，包括分片测试，测试目录，选项和失败的测试列表
        run_tests(
            test_batch.sharded_tests, test_directory, options, test_batch.failures
        )

    finally:
        # 如果设置了 coverage 选项：
        if options.coverage:
            # 导入 coverage 模块
            from coverage import Coverage

            # 在测试目录下设置当前工作目录
            with set_cwd(test_directory):
                cov = Coverage()
                # 如果允许收集 coverage 信息：
                if PYTORCH_COLLECT_COVERAGE:
                    # 加载现有的 coverage 数据
                    cov.load()
                # 合并 coverage 数据，不执行严格模式
                cov.combine(strict=False)
                # 保存 coverage 数据
                cov.save()
                # 如果不收集 coverage 数据，则生成 HTML 报告
                if not PYTORCH_COLLECT_COVERAGE:
                    cov.html_report()

        # 获取所有失败的测试列表
        all_failures = test_batch.failures

        # 如果在 CI 环境下：
        if IS_CI:
            # 对于所有失败的测试，获取测试的统计信息
            for test, _ in all_failures:
                test_stats = test_prioritizations.get_test_stats(test)
                # 输出测试失败统计信息正在发射
                print_to_stderr("Emiting td_test_failure_stats_v2")
                # 发射指标，包括选择的测试，失败的测试名称以及测试统计信息
                emit_metric(
                    "td_test_failure_stats_v2",
                    {
                        "selected_tests": selected_tests,
                        "failure": str(test),
                        **test_stats,
                    },
                )
            # 生成附加的测试失败文件，包括所有失败测试的测试文件名
            gen_additional_test_failures_file(
                [test.test_file for test, _ in all_failures]
            )

    # 如果存在失败的测试：
    if len(all_failures):
        # 对于所有失败的测试，输出错误信息
        for _, err in all_failures:
            print_to_stderr(err)

        # 如果不重新运行禁用的测试，则退出程序并返回非零状态码
        if not RERUN_DISABLED_TESTS:
            sys.exit(1)
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行 main() 函数
if __name__ == "__main__":
    main()
```