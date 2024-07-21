# `.\pytorch\torch\backends\xeon\run_cpu.py`

```py
# mypy: allow-untyped-defs
"""
This is a script for launching PyTorch inference on Intel(R) Xeon(R) Scalable Processors with optimal configurations.

Single instance inference, multi-instance inference are enabled.

Note: term "instance" here doesn't refer to a cloud instance. This script is executed as a single process. It invokes
multiple "instances" which are formed from multiple threads for each. "instance" is kind of group of threads in this
context.

Illustrated as below:

::

    +-----------------------------+----------------------+-------+
    |            process          |        thread        | core  |
    +=============================+======================+=======+
    | torch.backends.xeon.run_cpu | instance 0: thread 0 |   0   |
    |                             |             thread 1 |   1   |
    |                             +----------------------+-------+
    |                             | instance 1: thread 0 |   2   |
    |                             |             thread 1 |   3   |
    |                             +----------------------+-------+
    |                             | ...                  |  ...  |
    |                             +----------------------+-------+
    |                             | instance N: thread 0 |   M   |
    |                             |             thread 1 |  M+1  |
    +-----------------------------+----------------------+-------+

To get the peak performance on Intel(R) Xeon(R) Scalable Processors, the script optimizes the configuration of thread and memory
management. For thread management, the script configures thread affinity and the preload of Intel OMP library.
For memory management, it configures NUMA binding and preload optimized memory allocation library (e.g. tcmalloc, jemalloc).

Environment variables that will be set by this script:

+------------------+-------------------------------------------------------------------------------------------------+
| Environ Variable |                                             Value                                               |
+==================+=================================================================================================+
|    LD_PRELOAD    | Depending on knobs you set, <lib>/libiomp5.so, <lib>/libjemalloc.so, <lib>/libtcmalloc.so might |
|                  | be appended to LD_PRELOAD.                                                                      |
+------------------+-------------------------------------------------------------------------------------------------+
|   KMP_AFFINITY   | If libiomp5.so is preloaded, KMP_AFFINITY could be set to "granularity=fine,compact,1,0".       |
+------------------+-------------------------------------------------------------------------------------------------+
|   KMP_BLOCKTIME  | If libiomp5.so is preloaded, KMP_BLOCKTIME is set to "1".                                       |
"""

# 标记脚本允许使用未类型化的函数定义
# 导入所需模块：glob、logging、os、platform、re
import glob     # 文件路径名模式匹配
import logging  # 日志记录模块
import os       # 提供了访问操作系统服务的功能
import platform # 用于访问平台相关属性和操作系统的功能
import re       # 提供了正则表达式操作

# OMP_NUM_THREADS 环境变量用于设置每个实例的线程数
# MALLOC_CONF 环境变量用于设置内存分配器的配置，如果预加载了 libjemalloc.so，则会设置特定的参数
# 脚本尊重预先设置的环境变量。即，如果在运行脚本之前设置了上述环境变量，则脚本不会覆盖脚本中的值。

# 如何使用这个模块：
# 单实例推理
# 第一种情况：在单节点上使用所有 CPU 核心进行单实例推理
# 使用 torch.backends.xeon.run_cpu 模块执行脚本，并通过 --throughput-mode 选项设置为 script.py
# args 是额外的参数
# 第二种情况：在单个 CPU 节点上进行单实例推理
# 使用 torch.backends.xeon.run_cpu 模块执行脚本，并通过 --node-id 选项设置节点 ID 为 1
# script.py args 是额外的参数

# 多实例推理
# 第一种情况：多实例推理，默认每个节点运行一个进程。可以设置实例数和每个实例的核心数
# 使用 torch.backends.xeon.run_cpu 模块执行脚本，通过 --ninstances 和 --ncores-per-instance 选项设置实例数和每个实例的核心数
# python_script args 是额外的参数
# 例如，在拥有 14 个实例、每个实例 4 个核心的 Intel(R) Xeon(R) Scalable 处理器上运行

# 第二种情况：在多个实例中运行单实例推理。默认情况下，运行所有实例。如果要单独运行一个实例，需要指定排名（rank）
# 使用 torch.backends.xeon.run_cpu 模块执行脚本，通过 --ninstances 和 --rank 选项设置实例数和实例的排名
# python_script args 是额外的参数
# 例如，在拥有 2 个实例的 Intel(R) Xeon(R) Scalable 处理器上运行第 0 个实例

# 第三种情况：在多个实例中运行单实例推理，设置每个实例的核心数和核心列表
# 使用 torch.backends.xeon.run_cpu 模块执行脚本，通过 --ninstances、--ncores-per-instance、--core-list 和 --rank 选项设置
# python_script args 是额外的参数
# 例如，在拥有 2 个实例、每个实例 2 个核心的 Intel(R) Xeon(R) Scalable 处理器上运行第 0 个实例的前四个核心

# 查看此模块提供的可选参数
# 使用 torch.backends.xeon.run_cpu 模块执行脚本，并通过 --help 选项查看可选参数

# 内存分配器
# 可以使用 "--enable-tcmalloc" 和 "--enable-jemalloc" 来启用不同的内存分配器
# 导入 subprocess 模块，用于执行外部命令
import subprocess
# 导入 sys 模块，用于访问与 Python 解释器交互的变量和函数
import sys
# 从 argparse 模块导入 ArgumentParser 类、RawTextHelpFormatter 类和 REMAINDER 常量
from argparse import ArgumentParser, RawTextHelpFormatter, REMAINDER
# 从 os.path 模块导入 expanduser 函数，用于展开用户路径
from os.path import expanduser
# 从 typing 模块导入 Dict 和 List 类型，用于类型提示
from typing import Dict, List

# 从 torch.distributed.elastic.multiprocessing 模块导入 start_processes 函数和 Std 枚举
from torch.distributed.elastic.multiprocessing import (
    DefaultLogsSpecs as _DefaultLogsSpecs,
    start_processes,
    Std,
)

# 定义日志格式字符串
format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# 配置 logging 模块，设置日志级别和格式
logging.basicConfig(level=logging.INFO, format=format_str)
# 获取当前模块的 logger 对象
logger = logging.getLogger(__name__)

# 定义 _CPUinfo 类，用于获取 CPU 相关信息，如核心列表和 NUMA 信息
class _CPUinfo:
    """Get CPU information, such as cores list and NUMA information."""
    # 初始化方法，用于初始化对象，可以接受一个测试输入参数
    def __init__(self, test_input=""):
        # 初始化一个空的 CPU 信息列表
        self.cpuinfo = []
        
        # 检查操作系统类型，如果是 Windows 或者 Darwin，抛出运行时错误
        if platform.system() in ["Windows", "Darwin"]:
            raise RuntimeError(f"{platform.system()} is not supported!!!")
        elif platform.system() == "Linux":
            # 如果没有提供测试输入，执行 lscpu 命令获取 CPU 信息
            if test_input == "":
                lscpu_cmd = ["lscpu", "--parse=CPU,Core,Socket,Node"]
                lscpu_info = subprocess.check_output(
                    lscpu_cmd, universal_newlines=True
                ).split("\n")
            else:
                # 否则，使用提供的测试输入作为 lscpu 的输出
                lscpu_info = test_input.split("\n")

            # 遍历 lscpu 命令输出的每一行
            for line in lscpu_info:
                pattern = r"^([\d]+,[\d]+,[\d]+,[\d]?)"
                regex_out = re.search(pattern, line)
                # 如果找到匹配的模式
                if regex_out:
                    # 将匹配到的 CPU 信息添加到 self.cpuinfo 中
                    self.cpuinfo.append(regex_out.group(1).strip().split(","))

            # 计算节点数目，即 self.cpuinfo 中 Node 列的最大值加一
            self.node_nums = int(max(line[3] for line in self.cpuinfo)) + 1
            
            # 初始化节点物理核心和逻辑核心的列表
            self.node_physical_cores: List[List[int]] = []  # 每个节点的物理核心列表，以节点 ID 为索引
            self.node_logical_cores: List[List[int]] = []   # 每个节点的逻辑核心列表，以节点 ID 为索引
            
            # 物理核心到 NUMA 节点 ID 的映射
            self.physical_core_node_map = {}  
            # 逻辑核心到 NUMA 节点 ID 的映射
            self.logical_core_node_map = {}   

            # 遍历每个节点
            for node_id in range(self.node_nums):
                cur_node_physical_core = []   # 当前节点的物理核心列表
                cur_node_logical_core = []    # 当前节点的逻辑核心列表
                
                # 遍历 CPU 信息列表
                for cpuinfo in self.cpuinfo:
                    nid = cpuinfo[3] if cpuinfo[3] != "" else "0"
                    # 如果当前 CPU 信息的 Node ID 等于当前节点 ID
                    if node_id == int(nid):
                        # 如果当前物理核心不在当前节点的物理核心列表中，则添加
                        if int(cpuinfo[1]) not in cur_node_physical_core:
                            cur_node_physical_core.append(int(cpuinfo[1]))
                            # 记录物理核心到 NUMA 节点 ID 的映射关系
                            self.physical_core_node_map[int(cpuinfo[1])] = int(node_id)
                        
                        # 添加当前逻辑核心到当前节点的逻辑核心列表中
                        cur_node_logical_core.append(int(cpuinfo[0]))
                        # 记录逻辑核心到 NUMA 节点 ID 的映射关系
                        self.logical_core_node_map[int(cpuinfo[0])] = int(node_id)
                
                # 将当前节点的物理核心列表添加到 self.node_physical_cores 中
                self.node_physical_cores.append(cur_node_physical_core)
                # 将当前节点的逻辑核心列表添加到 self.node_logical_cores 中
                self.node_logical_cores.append(cur_node_logical_core)

    # 返回物理核心总数，即节点数乘以每个节点的物理核心数
    def _physical_core_nums(self):
        return len(self.node_physical_cores) * len(self.node_physical_cores[0])

    # 返回逻辑核心总数，即节点数乘以每个节点的逻辑核心数
    def _logical_core_nums(self):
        return len(self.node_logical_cores) * len(self.node_logical_cores[0])
    def get_node_physical_cores(self, node_id):
        # 检查节点 ID 是否有效，如果无效则抛出 ValueError 异常
        if node_id < 0 or node_id > self.node_nums - 1:
            raise ValueError(
                f"Invalid node id: {node_id}. Valid node ids: {list(range(len(self.node_physical_cores)))}"
            )
        # 返回指定节点的物理核心列表
        return self.node_physical_cores[node_id]

    def get_node_logical_cores(self, node_id):
        # 检查节点 ID 是否有效，如果无效则抛出 ValueError 异常
        if node_id < 0 or node_id > self.node_nums - 1:
            raise ValueError(
                f"Invalid node id: {node_id}. Valid node ids: {list(range(len(self.node_physical_cores)))}"
            )
        # 返回指定节点的逻辑核心列表
        return self.node_logical_cores[node_id]

    def get_all_physical_cores(self):
        # 获取所有节点的物理核心列表，并将其扁平化成一个列表返回
        all_cores = []
        for cores in self.node_physical_cores:
            all_cores.extend(cores)
        return all_cores

    def get_all_logical_cores(self):
        # 获取所有节点的逻辑核心列表，并将其扁平化成一个列表返回
        all_cores = []
        for cores in self.node_logical_cores:
            all_cores.extend(cores)
        return all_cores

    def numa_aware_check(self, core_list):
        """
        Check whether all cores in core_list are in the same NUMA node.

        Cross NUMA will reduce performance.
        We strongly advice to not use cores on different nodes.
        """
        # 获取逻辑核心到 NUMA 节点的映射关系
        cores_numa_map = self.logical_core_node_map
        # 存储核心所属的 NUMA 节点 ID
        numa_ids = []
        # 遍历传入的核心列表，确定其所属的 NUMA 节点 ID
        for core in core_list:
            numa_id = cores_numa_map[core]
            if numa_id not in numa_ids:
                numa_ids.append(numa_id)
        # 如果核心分布在多个 NUMA 节点上，则记录警告日志
        if len(numa_ids) > 1:
            logger.warning(
                "Numa Aware: cores:%s on different NUMA nodes:%s. To avoid \
                performance degradation, use cores from the same NUMA node.",
                core_list, numa_ids
            )
def set_memory_allocator(
    self, enable_tcmalloc=True, enable_jemalloc=False, use_default_allocator=False
):
    # 设置内存分配器，可以选择启用 TCMalloc、JeMalloc 或默认分配器
    if use_default_allocator:
        # 如果选择使用默认分配器，直接返回 False，表示未设置任何特定的分配器
        return False
    
    # 初始化一个 _CPUinfo 对象，用于获取关于 CPU 的信息
    self.cpuinfo = _CPUinfo()

    # 添加要预加载的动态链接库路径
    def add_lib_preload(self, lib_type):
        """Enable TCMalloc/JeMalloc/intel OpenMP."""
        library_paths = []
        if "CONDA_PREFIX" in os.environ:
            library_paths.append(f"{os.environ['CONDA_PREFIX']}/lib")
        if "VIRTUAL_ENV" in os.environ:
            library_paths.append(f"{os.environ['VIRTUAL_ENV']}/lib")

        library_paths += [
            f"{expanduser('~')}/.local/lib",
            "/usr/local/lib",
            "/usr/local/lib64",
            "/usr/lib",
            "/usr/lib64",
        ]

        lib_find = False
        lib_set = False
        for item in os.getenv("LD_PRELOAD", "").split(":"):
            if item.endswith(f"lib{lib_type}.so"):
                lib_set = True
                break
        if not lib_set:
            # 遍历库路径，查找指定类型的库文件
            for lib_path in library_paths:
                library_file = os.path.join(lib_path, f"lib{lib_type}.so")
                matches = glob.glob(library_file)
                if len(matches) > 0:
                    # 如果找到匹配的库文件，设置 LD_PRELOAD 环境变量
                    ld_preloads = [f"{matches[0]}", os.getenv("LD_PRELOAD", "")]
                    os.environ["LD_PRELOAD"] = os.pathsep.join(
                        [p.strip(os.pathsep) for p in ld_preloads if p]
                    )
                    lib_find = True
                    break
        # 返回是否成功设置 LD_PRELOAD 环境变量的标志
        return lib_set or lib_find

    # 检查是否可用 numactl 命令
    def is_numactl_available(self):
        numactl_available = False
        try:
            cmd = ["numactl", "-C", "0", "-m", "0", "hostname"]
            r = subprocess.run(
                cmd,
                env=os.environ,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            # 如果命令成功执行，返回 True，表示可用
            if r.returncode == 0:
                numactl_available = True
        except Exception:
            pass
        # 返回 numactl 是否可用的标志
        return numactl_available
    ):
        """
        Enable TCMalloc/JeMalloc with LD_PRELOAD and set configuration for JeMalloc.

        By default, PTMalloc will be used for PyTorch, but TCMalloc and JeMalloc can get better
        memory reuse and reduce page fault to improve performance.
        """
        # 如果同时启用了 TCMalloc 和 JeMalloc，则抛出运行时错误
        if enable_tcmalloc and enable_jemalloc:
            raise RuntimeError(
                "Unable to enable TCMalloc and JEMalloc at the same time."
            )

        # 如果启用了 TCMalloc
        if enable_tcmalloc:
            # 尝试添加 TCMalloc 的动态链接库预加载
            find_tc = self.add_lib_preload(lib_type="tcmalloc")
            # 如果未找到 TCMalloc，则发出警告信息
            if not find_tc:
                msg = f'{self.msg_lib_notfound} you can use "conda install -c conda-forge gperftools" to install {{0}}'
                logger.warning(msg.format("TCmalloc", "tcmalloc"))  # noqa: G001
            else:
                logger.info("Use TCMalloc memory allocator")

        # 如果启用了 JeMalloc
        elif enable_jemalloc:
            # 尝试添加 JeMalloc 的动态链接库预加载
            find_je = self.add_lib_preload(lib_type="jemalloc")
            # 如果未找到 JeMalloc，则发出警告信息
            if not find_je:
                msg = f'{self.msg_lib_notfound} you can use "conda install -c conda-forge jemalloc" to install {{0}}'
                logger.warning(msg.format("Jemalloc", "jemalloc"))  # noqa: G001
            else:
                logger.info("Use JeMalloc memory allocator")
                # 设置 JeMalloc 的环境变量配置
                self.set_env(
                    "MALLOC_CONF",
                    "oversize_threshold:1,background_thread:true,metadata_thp:auto",
                )

        # 如果使用默认分配器
        elif use_default_allocator:
            pass

        # 如果既未启用 TCMalloc 也未启用 JeMalloc
        else:
            # 尝试添加 TCMalloc 的动态链接库预加载，并立即返回
            find_tc = self.add_lib_preload(lib_type="tcmalloc")
            if find_tc:
                logger.info("Use TCMalloc memory allocator")
                return
            # 尝试添加 JeMalloc 的动态链接库预加载，并立即返回
            find_je = self.add_lib_preload(lib_type="jemalloc")
            if find_je:
                logger.info("Use JeMalloc memory allocator")
                return
            # 如果在指定路径下都未找到 TCMalloc 和 JeMalloc，则发出警告信息
            logger.warning(
                """Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib
                            or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or
                           %s/.local/lib/ so the LD_PRELOAD environment variable will not be set.
                           This may drop the performance""",
                expanduser("~"),
            )

    def log_env_var(self, env_var_name=""):
        # 如果指定的环境变量名称存在于当前环境变量中，则记录其值到日志中
        if env_var_name in os.environ:
            logger.info("%s=%s", env_var_name, os.environ[env_var_name])

    def set_env(self, env_name, env_value):
        # 如果环境变量值为空，则发出警告信息
        if not env_value:
            logger.warning("%s is None", env_name)
        # 如果环境变量名称不在当前环境变量中，则设置其值
        if env_name not in os.environ:
            os.environ[env_name] = env_value
        # 如果环境变量名称已存在于当前环境变量中，并且其值不同于待设置的值，则发出警告信息
        elif os.environ[env_name] != env_value:
            logger.warning(
                "Overriding value with the one set in environment variable: %s. \
    r"""
     Launcher for single instance and multi-instance
     """



    def launch(self, args):
        # 初始化一个空列表用来存储核心数
        cores = []
        # 默认设置 KMP_AFFINITY 为 True
        set_kmp_affinity = True
        # 默认禁用 taskset
        enable_taskset = False
        # 如果用户指定了核心列表，则解析并存储到 cores 列表中
        if args.core_list:  # user specify what cores will be used by params
            cores = [int(x) for x in args.core_list.split(",")]
            # 如果未指定每个实例的核心数，则抛出运行时错误
            if args.ncores_per_instance == -1:
                raise RuntimeError(
                    'please specify the "--ncores-per-instance" if you have pass the --core-list params'
                )
            # 如果实例数大于1，并且总核心数少于用户指定的核心列表数，则发出警告
            elif (
                args.ninstances > 1
                and args.ncores_per_instance * args.ninstances < len(cores)
            ):
                logger.warning(
                    "only first %s cores will be used, \
        # 如果未指定实例数量和每个实例的核心数，则根据核心列表长度确定实例数量
        if (
            not args.multi_instance
            and args.ninstances == -1
            and args.ncores_per_instance == -1
        ):
            args.ninstances = 1
            args.ncores_per_instance = len(cores)
        
        # 如果启用了逻辑核心，并且指定了节点 ID，则获取该节点的逻辑核心列表
        elif args.use_logical_core:
            if args.node_id != -1:
                cores = self.cpuinfo.get_node_logical_cores(args.node_id)
            else:
                cores = self.cpuinfo.get_all_logical_cores()
                # 当使用所有节点的所有逻辑核心时，设置 KMP_AFFINITY 会禁用逻辑核心，因此不应设置 KMP_AFFINITY
                set_kmp_affinity = False
        
        # 如果未启用逻辑核心，并且指定了节点 ID，则获取该节点的物理核心列表
        else:
            if args.node_id != -1:
                cores = self.cpuinfo.get_node_physical_cores(args.node_id)
            else:
                cores = self.cpuinfo.get_all_physical_cores()
            
            # 如果是单实例模式且未指定实例数量和每个实例的核心数，则将实例数量设置为 1，每个实例的核心数设置为所有核心数
            if (
                not args.multi_instance
                and args.ninstances == -1
                and args.ncores_per_instance == -1
            ):
                args.ninstances = 1
                args.ncores_per_instance = len(cores)
            
            # 如果是多实例模式且未指定实例数量和每个实例的核心数，则设置 throughput_mode 为 True
            elif (
                args.multi_instance
                and args.ninstances == -1
                and args.ncores_per_instance == -1
            ):
                args.throughput_mode = True
            
            # 如果未指定每个实例的核心数但指定了实例数量，则根据实例数量计算每个实例的核心数
            elif args.ncores_per_instance == -1 and args.ninstances != -1:
                if args.ninstances > len(cores):
                    # 如果指定的实例数量大于总核心数，则抛出运行时错误
                    raise RuntimeError(
                        f"there are {len(cores)} total cores but you specify {args.ninstances} ninstances; \
please make sure ninstances <= total_cores)"
                    )
                else:
                    args.ncores_per_instance = len(cores) // args.ninstances
            
            # 如果指定了每个实例的核心数但未指定实例数量，则根据核心数计算实例数量
            elif args.ncores_per_instance != -1 and args.ninstances == -1:
                if not args.skip_cross_node_cores:
                    args.ninstances = len(cores) // args.ncores_per_instance
                else:
                    # 计算每个节点的核心数和剩余核心数
                    ncore_per_node = len(self.cpuinfo.node_physical_cores[0])
                    num_leftover_cores = ncore_per_node % args.ncores_per_instance
                    
                    if args.ncores_per_instance > ncore_per_node:
                        # 如果指定的每个实例核心数大于每个节点的核心数，则发出警告
                        logger.warning(
                            "there are %s core(s) per socket, but you specify %s ncores_per_instance and \
skip_cross_node_cores. Please make sure --ncores-per-instance < core(s) per socket."
                            % (ncore_per_node, args.ncores_per_instance)
                        )
def _add_memory_allocator_params(parser):
    # 添加一个参数组，用于内存分配器的参数配置
    group = parser.add_argument_group("Memory Allocator Parameters")
    
    # 添加参数选项 --enable-tcmalloc 和 --enable_tcmalloc
    # 当选中时，启用 tcmalloc 分配器
    group.add_argument(
        "--enable-tcmalloc",
        "--enable_tcmalloc",
        action="store_true",
        default=False,
        help="Enable tcmalloc allocator",
    )
    
    # 添加参数选项 --enable-jemalloc 和 --enable_jemalloc
    # 当选中时，启用 jemalloc 分配器
    group.add_argument(
        "--enable-jemalloc",
        "--enable_jemalloc",
        action="store_true",
        default=False,
        help="Enable jemalloc allocator",
    )
    # 向命令行参数组(group)添加一个参数选项
    group.add_argument(
        "--use-default-allocator",  # 参数选项名称，用双破折号表示
        "--use_default_allocator",  # 参数选项的别名，同样作用
        action="store_true",  # 当命令行中存在该选项时，将其值设为True
        default=False,  # 默认情况下，该选项的值为False
        help="Use default memory allocator",  # 参数选项的帮助信息，显示在命令行帮助文档中
    )
def _add_multi_instance_params(parser):
    # 创建一个参数组，用于存放多实例参数
    group = parser.add_argument_group("Multi-instance Parameters")
    # 添加参数：每个实例的核心数
    group.add_argument(
        "--ncores-per-instance",
        "--ncores_per_instance",
        metavar="\b",
        default=-1,
        type=int,
        help="Cores per instance",
    )
    # 添加参数：实例数量
    group.add_argument(
        "--ninstances",
        metavar="\b",
        default=-1,
        type=int,
        help="For multi-instance, you should give the cores number you used for per instance.",
    )
    # 添加参数：跳过跨节点核心
    group.add_argument(
        "--skip-cross-node-cores",
        "--skip_cross_node_cores",
        action="store_true",
        default=False,
        help="If specified --ncores-per-instance, skips cross-node cores.",
    )
    # 添加参数：指定实例索引以为 rank 分配 ncores_per_instance；否则 ncores_per_instance 将按顺序分配给 ninstances
    group.add_argument(
        "--rank",
        metavar="\b",
        default="-1",
        type=int,
        help="Specify instance index to assign ncores_per_instance for rank; otherwise ncores_per_instance will be assigned sequentially to ninstances. Please refer to https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/performance_tuning/launch_script.md",
    )
    # 添加参数：延迟模式
    group.add_argument(
        "--latency-mode",
        "--latency_mode",
        action="store_true",
        default=False,
        help="By default 4 core per instance and use all physical cores",
    )
    # 添加参数：吞吐量模式
    group.add_argument(
        "--throughput-mode",
        "--throughput_mode",
        action="store_true",
        default=False,
        help="By default one instance per node and use all physical cores",
    )
    # 添加参数：节点 ID
    group.add_argument(
        "--node-id",
        "--node_id",
        metavar="\b",
        default=-1,
        type=int,
        help="node id for multi-instance, by default all nodes will be used",
    )
    # 添加参数：是否只使用逻辑核心
    group.add_argument(
        "--use-logical-core",
        "--use_logical_core",
        action="store_true",
        default=False,
        help="Whether only use physical cores",
    )
    # 添加参数：禁用 numactl
    group.add_argument(
        "--disable-numactl",
        "--disable_numactl",
        action="store_true",
        default=False,
        help="Disable numactl",
    )
    # 添加参数：禁用 taskset
    group.add_argument(
        "--disable-taskset",
        "--disable_taskset",
        action="store_true",
        default=False,
        help="Disable taskset",
    )
    # 添加参数：核心列表
    group.add_argument(
        "--core-list",
        "--core_list",
        metavar="\b",
        default=None,
        type=str,
        help='Specify the core list as "core_id, core_id, ....", otherwise, all the cores will be used.',
    )
    # 添加参数：日志路径
    group.add_argument(
        "--log-path",
        "--log_path",
        metavar="\b",
        default="",
        type=str,
        help="The log file directory. Default path is , which means disable logging to files.",
    )
    # 添加参数：日志文件前缀
    group.add_argument(
        "--log-file-prefix",
        "--log_file_prefix",
        metavar="\b",
        default="run",
        type=str,
        help="log file prefix",
    # 定义一个名为 add_numbers 的函数，接受两个参数 num1 和 num2
    def add_numbers(num1, num2):
        # 将 num1 和 num2 相加得到结果，返回给调用者
        return num1 + num2
# 将 IOMP 参数添加到命令行解析器中
def _add_kmp_iomp_params(parser):
    # 创建一个参数组，用于存放 IOMP 参数
    group = parser.add_argument_group("IOMP Parameters")
    # 添加一个布尔类型的参数，用于禁用 Intel OpenMP
    group.add_argument(
        "--disable-iomp",
        "--disable_iomp",
        action="store_true",
        default=False,
        help="By default, we use Intel OpenMP and libiomp5.so will be add to LD_PRELOAD",
    )


def create_args(parser=None):
    """
    Parse the command line options.

    @retval ArgumentParser
    """
    # 添加一个布尔类型的参数，用于启用多实例模式，默认为单节点一个实例
    parser.add_argument(
        "--multi-instance",
        "--multi_instance",
        action="store_true",
        default=False,
        help="Enable multi-instance, by default one instance per node",
    )

    # 添加一个布尔类型的参数，用于将每个进程作为 Python 模块进行解释
    parser.add_argument(
        "-m",
        "--module",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script "
        "as a python module, executing with the same behavior as"
        '"python -m".',
    )

    # 添加一个布尔类型的参数，用于在执行程序时不使用 Python 解释器
    parser.add_argument(
        "--no-python",
        "--no_python",
        default=False,
        action="store_true",
        help='Do not prepend the --program script with "python" - just exec '
        "it directly. Useful when the script is not a Python script.",
    )

    # 添加内存分配器相关的参数
    _add_memory_allocator_params(parser)
    # 添加 IOMP 参数到命令行解析器中
    _add_kmp_iomp_params(parser)
    # 添加多实例相关的参数
    _add_multi_instance_params(parser)
    
    # 添加一个位置参数，表示要启动的程序或脚本的完整路径
    parser.add_argument(
        "program",
        type=str,
        help="The full path to the program/script to be launched. "
        "followed by all the arguments for the script",
    )

    # 添加从训练程序中获取的其余位置参数
    parser.add_argument("program_args", nargs=REMAINDER)


def main(args):
    # 记录程序运行前的环境变量集合
    env_before = set(os.environ.keys())
    
    # 如果运行平台为 Windows 或 Darwin，则抛出运行时错误
    if platform.system() in ["Windows", "Darwin"]:
        raise RuntimeError(f"{platform.system()} is not supported!!!")

    # 如果指定了日志路径，则确保路径存在；否则将日志路径设置为 os.devnull
    if args.log_path:
        os.makedirs(args.log_path, exist_ok=True)
    else:
        args.log_path = os.devnull

    # 如果同时指定了延迟模式和吞吐量模式，则抛出运行时错误
    if args.latency_mode and args.throughput_mode:
        raise RuntimeError(
            "Either args.latency_mode or args.throughput_mode should be set"
        )

    # 如果未禁用 Python 并且程序不是以 .py 结尾，则抛出运行时错误
    if not args.no_python and not args.program.endswith(".py"):
        raise RuntimeError(
            'For non Python script, you should use "--no-python" parameter.'
        )

    # 验证 LD_PRELOAD 环境变量
    if "LD_PRELOAD" in os.environ:
        lst_valid = []
        tmp_ldpreload = os.environ["LD_PRELOAD"]
        for item in tmp_ldpreload.split(":"):
            matches = glob.glob(item)
            if len(matches) > 0:
                lst_valid.append(item)
            else:
                logger.warning("%s doesn't exist. Removing it from LD_PRELOAD.", item)
        # 如果有有效的 LD_PRELOAD 项，则重新设置 LD_PRELOAD 环境变量
        if len(lst_valid) > 0:
            os.environ["LD_PRELOAD"] = ":".join(lst_valid)
        else:
            os.environ["LD_PRELOAD"] = ""

    # 创建 Launcher 实例并启动程序
    launcher = _Launcher()
    launcher.launch(args)
    
    # 输出运行后新增的环境变量
    for x in sorted(set(os.environ.keys()) - env_before):
        logger.debug("%s=%s", x, os.environ[x])


if __name__ == "__main__":
    # 创建参数解析器对象，并设置脚本的描述信息
    parser = ArgumentParser(
        description="This is a script for launching PyTorch inference on Intel(R) Xeon(R) Scalable "
        "Processors with optimal configurations. Single instance inference, "
        "multi-instance inference are enable. To get the peak performance on Intel(R) "
        "Xeon(R) Scalable Processors, the script optimizes the configuration "
        "of thread and memory management. For thread management, the script configures thread "
        "affinity and the preload of Intel OMP library. For memory management, it configures "
        "NUMA binding and preload optimized memory allocation library (e.g. tcmalloc, jemalloc) "
        "\n################################# Basic usage ############################# \n"
        "\n 1. single instance\n"
        "\n   >>> python -m torch.backends.xeon.run_cpu python_script args \n"
        "\n2. multi-instance \n"
        "\n   >>> python -m torch.backends.xeon.run_cpu --ninstances xxx "
        "--ncores-per-instance xx python_script args\n"
        "\n############################################################################# \n",
        formatter_class=RawTextHelpFormatter,
    )
    # 调用create_args函数，向参数解析器添加自定义参数
    create_args(parser)
    # 解析命令行参数，并将其存储在args变量中
    args = parser.parse_args()
    # 调用main函数，并传入解析后的参数args
    main(args)
```