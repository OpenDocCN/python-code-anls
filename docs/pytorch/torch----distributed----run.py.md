# `.\pytorch\torch\distributed\run.py`

```py
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Superset of ``torch.distributed.launch``.

``torchrun`` provides a superset of the functionality as ``torch.distributed.launch``
with the following additional functionalities:

1. Worker failures are handled gracefully by restarting all workers.

2. Worker ``RANK`` and ``WORLD_SIZE`` are assigned automatically.

3. Number of nodes is allowed to change between minimum and maximum sizes (elasticity).

.. note:: ``torchrun`` is a python
          `console script <https://packaging.python.org/en/latest/specifications/entry-points/#use-for-scripts>`_
          to the main module
          `torch.distributed.run <https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py>`_
          declared in the ``entry_points`` configuration in
          `setup.py <https://github.com/pytorch/pytorch/blob/master/setup.py>`_.
          It is equivalent to invoking ``python -m torch.distributed.run``.


Transitioning from torch.distributed.launch to torchrun
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``torchrun`` supports the same arguments as ``torch.distributed.launch`` **except**
for ``--use-env`` which is now deprecated. To migrate from ``torch.distributed.launch``
to ``torchrun`` follow these steps:

1.  If your training script is already reading ``local_rank`` from the ``LOCAL_RANK`` environment variable.
    Then you need simply omit the ``--use-env`` flag, e.g.:

    +--------------------------------------------------------------------+--------------------------------------------+
    |         ``torch.distributed.launch``                               |                ``torchrun``                |
    +====================================================================+============================================+
    |                                                                    |                                            |
    | .. code-block:: shell-session                                      | .. code-block:: shell-session              |
    |                                                                    |                                            |
    |    $ python -m torch.distributed.launch --use-env train_script.py  |    $ torchrun train_script.py              |
    |                                                                    |                                            |
    +--------------------------------------------------------------------+--------------------------------------------+

2.  If your training script reads local rank from a ``--local-rank`` cmd argument.
    Change your training script to read from the ``LOCAL_RANK`` environment variable as
    demonstrated by the following code snippet:
"""

# 从 argparse 导入 ArgumentParser 类
from argparse import ArgumentParser

# 从 os 模块导入 getenv 函数
from os import getenv

# 从 subprocess 模块导入 Popen, PIPE 类
from subprocess import Popen, PIPE

# 从 sys 模块导入 exit 函数
from sys import exit

# 从 time 模块导入 sleep 函数
from time import sleep

# 从 typing 模块导入 Dict 类
from typing import Dict
    import argparse                                    # 导入 argparse 模块，用于解析命令行参数
    parser = argparse.ArgumentParser()                 # 创建 ArgumentParser 对象，用于解析命令行参数
    parser.add_argument("--local-rank", type=int)      # 添加命令行参数选项 "--local-rank"，指定参数类型为整数
    args = parser.parse_args()                         # 解析命令行参数，并将结果存储在 args 变量中
    
    local_rank = args.local_rank                       # 从 args 中获取解析后的 "--local-rank" 参数值，赋给 local_rank
# 从版本2.0.0开始，启动器将传递“--local-rank=<rank>”参数给你的脚本。
# 在PyTorch 2.0.0及以后版本中，推荐使用破折号“--local-rank”，而不是之前使用的下划线“--local_rank”。

# 对于向后兼容性，用户在参数解析代码中可能需要处理两种情况。
# 这意味着在参数解析器中包含“--local-rank”和“--local_rank”两者。
# 如果只提供“--local_rank”，启动器将触发错误：“error: unrecognized arguments: --local-rank=<rank>”。
# 对于仅支持PyTorch 2.0.0+的训练代码，包含“--local-rank”应该就足够了。

# 示例代码片段如下：

>>> # xdoctest: +SKIP
>>> import argparse
>>> parser = argparse.ArgumentParser()
>>> parser.add_argument("--local-rank", "--local_rank", type=int)
>>> args = parser.parse_args()

# 上述变更足以从“torch.distributed.launch”迁移到“torchrun”。
# 要利用“torchrun”的新功能，如弹性、容错和错误报告，请参考下列内容：

# 若要了解更多关于编写符合“torchrun”标准的训练脚本的信息，请参阅：
# :ref:`elastic_train_script`
# 本页面的其余部分介绍了“torchrun”的更多功能。

# 使用示例
# --------

# 单节点多工作进程
# ++++++++++++++++++++++++++++

torchrun
    --standalone
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

# 堆叠的单节点多工作进程
# ++++++++++++++++++++++++++++++++++

# 若要在同一主机上运行单节点多工作进程的多个实例（独立作业），
# 我们需要确保每个实例（作业）在不同的端口上设置，以避免端口冲突（或更糟糕的情况，两个作业合并为一个作业）。
# 为此，需要使用“--rdzv-backend=c10d”并通过“--rdzv-endpoint=localhost:$PORT_k”设置不同的端口。
# 对于“--nodes=1”，通常方便让“torchrun”自动选择一个空闲的随机端口，而不是手动为每次运行分配不同的端口。

torchrun
    --rdzv-backend=c10d
    --rdzv-endpoint=localhost:0
    --nnodes=1
    --nproc-per-node=$NUM_TRAINERS
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

# 容错（固定大小的工作进程数，无弹性，容忍3次失败）
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

torchrun
    --nnodes=$NUM_NODES
    --nproc-per-node=$NUM_TRAINERS
    --max-restarts=3
    --rdzv-id=$JOB_ID
    --rdzv-backend=c10d
    --rdzv-endpoint=$HOST_NODE_ADDR
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

# “HOST_NODE_ADDR”以<host>[:<port>]的形式（例如node1.example.com:29400）指定了节点和端口。
# 设置 C10d 会合后端应该实例化和托管的端口号。它可以是训练集群中的任何节点，但最好选择带有高带宽的节点。

.. note::
   如果未指定端口号，``HOST_NODE_ADDR`` 默认为 29400。

Elastic (``min=1``, ``max=4``, 容忍最多 3 次成员变更或失败)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

    torchrun
        --nnodes=1:4
        --nproc-per-node=$NUM_TRAINERS
        --max-restarts=3
        --rdzv-id=$JOB_ID
        --rdzv-backend=c10d
        --rdzv-endpoint=$HOST_NODE_ADDR
        YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)

``HOST_NODE_ADDR``，格式为 <host>[:<port>]（例如 node1.example.com:29400），指定了应该实例化和托管 C10d 会合后端的节点和端口号。它可以是训练集群中的任何节点，但最好选择带有高带宽的节点。

Note on rendezvous backend
------------------------------

对于多节点训练，您需要指定：

1. ``--rdzv-id``: 唯一的作业 ID（所有参与作业的节点共享）
2. ``--rdzv-backend``: :py:class:`torch.distributed.elastic.rendezvous.RendezvousHandler` 的实现
3. ``--rdzv-endpoint``: 会合后端运行的端点；通常以 ``host:port`` 形式。

目前支持直接使用的会合后端包括 ``c10d``（推荐使用）、``etcd-v2`` 和 ``etcd``（旧版）。要使用 ``etcd-v2`` 或 ``etcd``，请设置启用了 ``v2`` API 的 etcd 服务器（例如 ``--enable-v2``）。

.. warning::
   ``etcd-v2`` 和 ``etcd`` 会合后端使用 etcd API v2。您必须在 etcd 服务器上启用 v2 API。我们的测试使用 etcd v3.4.3。

.. warning::
   对于基于 etcd 的会合，我们建议优先使用 ``etcd-v2`` 而不是 ``etcd``，尽管它们在功能上是等效的，但 ``etcd`` 正在维护模式中，并且将在将来的某个版本中移除。

Definitions
--------------

1. ``Node`` - 物理实例或容器；映射到作业管理器处理的单位。

2. ``Worker`` - 分布式训练上下文中的工作进程。

3. ``WorkerGroup`` - 执行相同功能的一组工作进程（例如训练器）。

4. ``LocalWorkerGroup`` - 在同一节点上运行的工作进程组的子集。

5. ``RANK`` - 工作进程在工作进程组中的排名。

6. ``WORLD_SIZE`` - 工作进程组中的总工作进程数。

7. ``LOCAL_RANK`` - 工作进程在本地工作进程组中的排名。

8. ``LOCAL_WORLD_SIZE`` - 本地工作进程组的大小。

9. ``rdzv_id`` - 用于唯一标识作业的工作进程组的用户定义 ID。每个节点使用此 ID 加入特定的工作进程组。
# rdzv_backend - 会议点（rendezvous）的后端（例如c10d）。通常是一个强一致性的键值存储。
# rdzv_endpoint - 会议点后端的端点；通常是以 `<host>:<port>` 的形式。

A ``Node`` runs ``LOCAL_WORLD_SIZE`` workers which comprise a ``LocalWorkerGroup``. The union of
all ``LocalWorkerGroups`` in the nodes in the job comprise the ``WorkerGroup``.

Environment Variables
----------------------

The following environment variables are made available to you in your script:

1. ``LOCAL_RANK`` -  The local rank.

2. ``RANK`` -  The global rank.

3. ``GROUP_RANK`` - The rank of the worker group. A number between 0 and ``max_nnodes``. When
   running a single worker group per node, this is the rank of the node.

4. ``ROLE_RANK`` -  The rank of the worker across all the workers that have the same role. The role
   of the worker is specified in the ``WorkerSpec``.

5. ``LOCAL_WORLD_SIZE`` - The local world size (e.g. number of workers running locally); equals to
   ``--nproc-per-node`` specified on ``torchrun``.

6. ``WORLD_SIZE`` - The world size (total number of workers in the job).

7. ``ROLE_WORLD_SIZE`` - The total number of workers that was launched with the same role specified
   in ``WorkerSpec``.

8. ``MASTER_ADDR`` - The FQDN of the host that is running worker with rank 0; used to initialize
   the Torch Distributed backend.

9. ``MASTER_PORT`` - The port on the ``MASTER_ADDR`` that can be used to host the C10d TCP store.

10. ``TORCHELASTIC_RESTART_COUNT`` - The number of worker group restarts so far.

11. ``TORCHELASTIC_MAX_RESTARTS`` - The configured maximum number of restarts.

12. ``TORCHELASTIC_RUN_ID`` - Equal to the rendezvous ``run_id`` (e.g. unique job id).

13. ``PYTHON_EXEC`` - System executable override. If provided, the python user script will
    use the value of ``PYTHON_EXEC`` as executable. The `sys.executable` is used by default.

Deployment
------------

1. (Not needed for the C10d backend) Start the rendezvous backend server and get the endpoint (to be
   passed as ``--rdzv-endpoint`` to the launcher script)

2. Single-node multi-worker: Start the launcher on the host to start the agent process which
   creates and monitors a local worker group.

3. Multi-node multi-worker: Start the launcher with the same arguments on all the nodes
   participating in training.

When using a job/cluster manager the entry point command to the multi-node job should be this
launcher.

Failure Modes
---------------

1. Worker failure: For a training job with ``n`` workers, if ``k<=n`` workers fail all workers
   are stopped and restarted up to ``max_restarts``.

2. Agent failure: An agent failure results in a local worker group failure. It is up to the job
   manager to fail the entire job (gang semantics) or attempt to replace the node. Both behaviors
   are supported by the agent.

3. Node failure: Same as agent failure.

Membership Changes
--------------------
# Node departure (scale-down): 当节点缩减时，通知代理节点的离开，停止所有现有的工作进程，
# 创建一个新的“WorkerGroup”，并且使用新的“RANK”和“WORLD_SIZE”启动所有工作进程。

# Node arrival (scale-up): 当新节点加入时，允许新节点加入作业，停止所有现有的工作进程，
# 创建一个新的“WorkerGroup”，并且使用新的“RANK”和“WORLD_SIZE”启动所有工作进程。

# Important Notices:
# This utility and multi-process distributed (single-node or multi-node) GPU training
# currently only achieves the best performance using the NCCL distributed backend.
# Thus NCCL backend is the recommended backend to use for GPU training.

# The environment variables necessary to initialize a Torch process group are provided
# by this module, eliminating the need to manually pass "RANK". To initialize a process
# group in your training script, simply run:

# xdoctest: +SKIP("stub")
# import torch.distributed as dist
# dist.init_process_group(backend="gloo|nccl")

# In your training program, you can either use regular distributed functions
# or use :func:`torch.nn.parallel.DistributedDataParallel` module. If your
# training program uses GPUs for training and you would like to use
# :func:`torch.nn.parallel.DistributedDataParallel` module,
# here is how to configure it.

# local_rank = int(os.environ["LOCAL_RANK"])
# model = torch.nn.parallel.DistributedDataParallel(model,
#                                                  device_ids=[local_rank],
#                                                  output_device=local_rank)

# Please ensure that `device_ids` argument is set to be the only GPU device id
# that your code will be operating on. This is generally the local rank of the
# process. In other words, the `device_ids` needs to be `[int(os.environ("LOCAL_RANK"))]`,
# and `output_device` needs to be `int(os.environ("LOCAL_RANK"))` in order to use this utility

# On failures or membership changes ALL surviving workers are killed immediately.
# Make sure to checkpoint your progress. The frequency of checkpoints should depend
# on your job's tolerance for lost work.

# This module only supports homogeneous `LOCAL_WORLD_SIZE`. That is, it is assumed
# that all nodes run the same number of local workers (per role).

# `RANK` is NOT stable. Between restarts, the local workers on a node can be assigned
# a different range of ranks than before. NEVER hard code any assumptions about the
# stable-ness of ranks or some correlation between `RANK` and `LOCAL_RANK`.

# When using elasticity (`min_size!=max_size`) DO NOT hard code assumptions about
# `WORLD_SIZE` as the world size can change as nodes are allowed to leave and join.

# It is recommended for your script to have the following structure:

# def main():
#   load_checkpoint(checkpoint_path)
#   initialize()
#   train()

# def train():
    # 迭代数据集中的每个批次
    for batch in iter(dataset):
        # 对当前批次执行训练步骤
        train_step(batch)

        # 检查是否应该创建检查点（保存模型状态）
        if should_checkpoint:
            # 保存当前模型检查点到指定路径
            save_checkpoint(checkpoint_path)
# 引入记录错误的模块
from torch.distributed.elastic.multiprocessing.errors import record

# 装饰主函数以记录错误详细信息，包括时间、排名、主机、进程号、回溯信息等
@record
def main():
    # 在这里执行训练过程
    pass

# 如果当前脚本作为主程序执行，则调用主函数进行处理
if __name__ == "__main__":
    main()
    # 添加一个名为 "--rdzv-conf" 的命令行参数，用于指定额外的会合配置，格式为 "<key1>=<value1>,<key2>=<value2>,..."
    parser.add_argument(
        "--rdzv-conf",
        "--rdzv_conf",
        action=env,
        type=str,
        default="",
        help="Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).",
    )
    
    # 添加一个名为 "--standalone" 的命令行参数，启动本地独立的会合后端，该后端由一个C10d TCP存储表示，并监听一个空闲端口。
    # 在启动单节点多工作进程任务时很有用。如果指定了此参数，则 "--rdzv-backend", "--rdzv-endpoint", "--rdzv-id" 会自动分配，
    # 并忽略任何显式设置的值。
    parser.add_argument(
        "--standalone",
        action=check_env,
        help="Start a local standalone rendezvous backend that is represented by a C10d TCP store "
        "on a free port. Useful when launching single-node, multi-worker job. If specified "
        "--rdzv-backend, --rdzv-endpoint, --rdzv-id are auto-assigned and any explicitly set values "
        "are ignored.",
    )

    #
    # User-code launch related arguments.
    #

    # 添加一个名为 "--max-restarts" 的命令行参数，设置工作组重启的最大次数，超过此次数则任务失败。
    parser.add_argument(
        "--max-restarts",
        "--max_restarts",
        action=env,
        type=int,
        default=0,
        help="Maximum number of worker group restarts before failing.",
    )
    
    # 添加一个名为 "--monitor-interval" 的命令行参数，设置监视工作状态的时间间隔，单位为秒。
    parser.add_argument(
        "--monitor-interval",
        "--monitor_interval",
        action=env,
        type=float,
        default=0.1,
        help="Interval, in seconds, to monitor the state of workers.",
    )
    
    # 添加一个名为 "--start-method" 的命令行参数，设置多进程启动时的启动方法，可选值有 "spawn", "fork", "forkserver"。
    parser.add_argument(
        "--start-method",
        "--start_method",
        action=env,
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method to use when creating workers.",
    )
    
    # 添加一个名为 "--role" 的命令行参数，设置工作进程的用户定义角色。
    parser.add_argument(
        "--role",
        action=env,
        type=str,
        default="default",
        help="User-defined role for the workers.",
    )
    
    # 添加一个名为 "-m" 或 "--module" 的命令行参数，使每个进程将启动脚本解释为 Python 模块，类似于 'python -m' 的行为。
    parser.add_argument(
        "-m",
        "--module",
        action=check_env,
        help="Change each process to interpret the launch script as a Python module, executing "
        "with the same behavior as 'python -m'.",
    )
    
    # 添加一个名为 "--no-python" 的命令行参数，跳过在训练脚本前面加上 'python' 的步骤，直接执行脚本。
    # 在脚本不是 Python 脚本时特别有用。
    parser.add_argument(
        "--no-python",
        "--no_python",
        action=check_env,
        help="Skip prepending the training script with 'python' - just execute it directly. Useful "
        "when the script is not a Python script.",
    )

    # 添加一个名为 "--run-path" 的命令行参数，使用 runpy.run_path 在相同解释器中运行训练脚本。
    # 脚本必须以绝对路径提供（例如 /abs/path/script.py）。此参数优先于 "--no-python"。
    parser.add_argument(
        "--run-path",
        "--run_path",
        action=check_env,
        help="Run the training script with runpy.run_path in the same interpreter."
        " Script must be provided as an abs path (e.g. /abs/path/script.py)."
        " Takes precedence over --no-python.",
    )
    
    # 添加一个名为 "--log-dir" 的命令行参数，设置日志文件的基础目录。
    # 同一个目录会被多次运行复用（使用 rdzv_id 作为前缀创建唯一的作业级子目录）。
    parser.add_argument(
        "--log-dir",
        "--log_dir",
        action=env,
        type=str,
        default=None,
        help="Base directory to use for log files (e.g. /var/log/torch/elastic). The same "
        "directory is re-used for multiple runs (a unique job-level sub-directory is created with "
        "rdzv_id as the prefix).",
    )
    parser.add_argument(
        "-r",
        "--redirects",
        action=env,
        type=str,
        default="0",
        help="Redirect std streams into a log file in the log directory (e.g. [-r 3] redirects "
        "both stdout+stderr for all workers, [-r 0:1,1:2] redirects stdout for local rank 0 and "
        "stderr for local rank 1).",
    )
    # 添加命令行参数选项 '-r' 或 '--redirects'，使用环境变量控制其行为，参数类型为字符串，默认为"0"，
    # 用于将标准输出流重定向到日志文件中，日志保存在指定的日志目录中。

    parser.add_argument(
        "-t",
        "--tee",
        action=env,
        type=str,
        default="0",
        help="Tee std streams into a log file and also to console (see --redirects for format).",
    )
    # 添加命令行参数选项 '-t' 或 '--tee'，使用环境变量控制其行为，参数类型为字符串，默认为"0"，
    # 将标准输出流同时输出到日志文件和控制台，具体格式参见 '--redirects' 参数说明。

    parser.add_argument(
        "--local-ranks-filter",
        "--local_ranks_filter",
        action=env,
        type=str,
        default="",
        help="Only show logs from specified ranks in console (e.g. [--local_ranks_filter=0,1,2] will "
        "only show logs from rank 0, 1 and 2). This will only apply to stdout and stderr, not to"
        "log files saved via --redirect or --tee",
    )
    # 添加命令行参数选项 '--local-ranks-filter' 或 '--local_ranks_filter'，使用环境变量控制其行为，
    # 参数类型为字符串，默认为空字符串，用于在控制台上仅显示特定排名的日志信息，例如 [--local_ranks_filter=0,1,2]
    # 只会显示来自排名 0、1 和 2 的日志信息。这仅适用于标准输出和标准错误，不适用于通过 '--redirect' 或 '--tee' 保存的日志文件。

    #
    # Backwards compatible parameters with caffe2.distributed.launch.
    #

    parser.add_argument(
        "--node-rank",
        "--node_rank",
        type=int,
        action=env,
        default=0,
        help="Rank of the node for multi-node distributed training.",
    )
    # 添加命令行参数选项 '--node-rank' 或 '--node_rank'，使用环境变量控制其行为，参数类型为整数，默认为0，
    # 用于指定节点在多节点分布式训练中的排名。

    parser.add_argument(
        "--master-addr",
        "--master_addr",
        default="127.0.0.1",
        type=str,
        action=env,
        help="Address of the master node (rank 0) that only used for static rendezvous. It should "
        "be either the IP address or the hostname of rank 0. For single node multi-proc training "
        "the --master-addr can simply be 127.0.0.1; IPv6 should have the pattern "
        "`[0:0:0:0:0:0:0:1]`.",
    )
    # 添加命令行参数选项 '--master-addr' 或 '--master_addr'，使用环境变量控制其行为，参数类型为字符串，
    # 默认为"127.0.0.1"，用于指定主节点（排名为0）的地址，仅用于静态会合。可以是排名为0的IP地址或主机名。
    # 对于单节点多进程训练，'--master-addr' 可以简单地设置为 127.0.0.1；IPv6 地址应具有模式 `[0:0:0:0:0:0:0:1]`。

    parser.add_argument(
        "--master-port",
        "--master_port",
        default=29500,
        type=int,
        action=env,
        help="Port on the master node (rank 0) to be used for communication during distributed "
        "training. It is only used for static rendezvous.",
    )
    # 添加命令行参数选项 '--master-port' 或 '--master_port'，使用环境变量控制其行为，参数类型为整数，
    # 默认为 29500，用于指定主节点（排名为0）在分布式训练期间用于通信的端口。仅用于静态会合。

    parser.add_argument(
        "--local-addr",
        "--local_addr",
        default=None,
        type=str,
        action=env,
        help="Address of the local node. If specified, will use the given address for connection. "
        "Else, will look up the local node address instead. Else, it will be default to local "
        "machine's FQDN.",
    )
    # 添加命令行参数选项 '--local-addr' 或 '--local_addr'，使用环境变量控制其行为，参数类型为字符串，
    # 默认为 None，用于指定本地节点的地址。如果指定了地址，将使用给定地址进行连接；否则，将查找本地节点地址；
    # 否则，默认为本地机器的完全限定域名（FQDN）。

    parser.add_argument(
        "--logs-specs",
        "--logs_specs",
        default=None,
        type=str,
        help="torchrun.logs_specs group entrypoint name, value must be type of LogsSpecs. "
        "Can be used to override custom logging behavior.",
    )
    # 添加命令行参数选项 '--logs-specs' 或 '--logs_specs'，参数类型为字符串，默认为 None，
    # 用于指定 torchrun.logs_specs 分组入口名称，其值必须是 LogsSpecs 类型，可以用于覆盖自定义的日志记录行为。

    #
    # Positional arguments.
    #

    parser.add_argument(
        "training_script",
        type=str,
        help="Full path to the (single GPU) training program/script to be launched in parallel, "
        "followed by all the arguments for the training script.",
    )
    # 添加位置参数 'training_script'，参数类型为字符串，用于指定要并行启动的（单 GPU）训练程序或脚本的完整路径，
    # 后跟训练脚本的所有参数。

    # Rest from the training program.
    # 添加一个位置参数"training_script_args"到参数解析器(parser)中，并且允许它获取剩余的所有参数
    parser.add_argument("training_script_args", nargs=REMAINDER)
    
    # 返回更新后的参数解析器(parser)对象
    return parser
# 解析命令行参数并返回解析结果
def parse_args(args):
    # 获取命令行参数解析器对象
    parser = get_args_parser()
    # 解析给定的参数列表并返回解析结果
    return parser.parse_args(args)


# 解析最小节点数和最大节点数
def parse_min_max_nnodes(nnodes: str):
    # 使用冒号分割节点数字符串
    arr = nnodes.split(":")

    # 根据分割后的数组长度确定最小节点数和最大节点数
    if len(arr) == 1:
        min_nodes = max_nodes = int(arr[0])
    elif len(arr) == 2:
        min_nodes = int(arr[0])
        max_nodes = int(arr[1])
    else:
        # 如果格式不正确，抛出异常
        raise RuntimeError(f'nnodes={nnodes} is not in "MIN:MAX" format')  # noqa: E231

    # 返回解析后的最小节点数和最大节点数
    return min_nodes, max_nodes


# 确定本地的进程数或设备数
def determine_local_world_size(nproc_per_node: str):
    try:
        # 记录信息，指定使用的每节点进程数或设备数
        logging.info("Using nproc_per_node=%s.", nproc_per_node)
        # 尝试将节点数转换为整数并返回
        return int(nproc_per_node)
    except ValueError as e:
        # 处理值错误，根据不同的nproc_per_node值选择对应的处理方式
        if nproc_per_node == "cpu":
            # 如果是cpu，获取系统CPU数量作为进程数，设备类型为cpu
            num_proc = os.cpu_count()
            device_type = "cpu"
        elif nproc_per_node == "gpu":
            # 如果是gpu，检查CUDA是否可用，获取GPU数量作为进程数，设备类型为gpu
            if not torch.cuda.is_available():
                raise ValueError("Cuda is not available.") from e
            device_type = "gpu"
            num_proc = torch.cuda.device_count()
        elif nproc_per_node == torch._C._get_privateuse1_backend_name():
            # 如果是私有后端名称，检查自定义模块是否可用，获取设备数量，设备类型使用自定义的后端名称
            if not _get_custom_mod_func("is_available")():
                raise ValueError(f"{nproc_per_node} is not available.") from e
            device_type = nproc_per_node
            num_proc = _get_custom_mod_func("device_count")()
        elif nproc_per_node == "auto":
            # 如果是自动模式，优先选择GPU，其次选择私有后端，最后选择CPU
            if torch.cuda.is_available():
                num_proc = torch.cuda.device_count()
                device_type = "gpu"
            elif (
                hasattr(torch, torch._C._get_privateuse1_backend_name())
                and _get_custom_mod_func("is_available")()
            ):
                num_proc = _get_custom_mod_func("device_count")()
                device_type = torch._C._get_privateuse1_backend_name()
            else:
                num_proc = os.cpu_count()
                device_type = "cpu"
        else:
            # 不支持的nproc_per_node值，抛出异常
            raise ValueError(
                f"Unsupported nproc_per_node value: {nproc_per_node}"
            ) from e

        # 记录信息，根据实例配置设置使用的每节点进程数或设备数
        logger.info(
            "Using nproc_per_node=%s," " setting to %s since the instance " "has %s %s",
            nproc_per_node,
            num_proc,
            os.cpu_count(),
            device_type,
        )
        # 返回确定的进程数或设备数
        return num_proc


# 获取分布式训练的rendezvous（RDZV）端点
def get_rdzv_endpoint(args):
    if args.rdzv_backend == "static" and not args.rdzv_endpoint:
        # 如果RDZV后端是静态且没有指定RDZV端点，返回主节点地址和端口号的组合
        return f"{args.master_addr}:{args.master_port}"  # noqa: E231
    # 否则，返回指定的RDZV端点
    return args.rdzv_endpoint


# 获取是否使用环境变量的标志
def get_use_env(args) -> bool:
    """
    Retrieve ``use_env`` from the args.

    ``use_env`` is a legacy argument, if ``use_env`` is False, the
    ``--node-rank`` argument will be transferred to all worker processes.
    ``use_env`` is only used by the ``torch.distributed.launch`` and will
    be deprecated in future releases.
    """
    # 如果参数对象没有use_env属性，默认返回True
    if not hasattr(args, "use_env"):
        return True
    # 否则，返回use_env的当前值
    return args.use_env


# 获取日志规格类的名称
def _get_logs_specs_class(logs_specs_name: Optional[str]) -> Type[LogsSpecs]:
    """
    Retrieve the LogsSpecs class based on the logs_specs_name.
    Placeholder for actual implementation.
    """
    # 这里是一个占位符，用于实际的实现
    # 目前并没有实现内容，因此没有具体的注释
    pass
    """
    Attempts to load `torchrun.logs_spec` entrypoint with the key of `logs_specs_name` param.
    Provides a plugin mechanism to provide a custom implementation of LogsSpecs.

    Returns `DefaultLogsSpecs` when logs_specs_name is None.
    Raises ValueError when the entrypoint for `logs_specs_name` can't be found in entrypoints.
    """
    # 初始化 logs_specs_cls 变量为 None
    logs_specs_cls = None
    # 如果 logs_specs_name 不为 None，则开始查找对应的 entrypoint
    if logs_specs_name is not None:
        # 获取所有的 entrypoints
        eps = metadata.entry_points()
        # Python 版本 >= 3.10，使用新的 entry_points 方法
        if hasattr(eps, "select"):
            # 选择 group 为 "torchrun.logs_specs" 的 entrypoints
            group = eps.select(group="torchrun.logs_specs")
            # 如果找到指定名称的 entrypoint，则加载对应的类
            if group.select(name=logs_specs_name):
                logs_specs_cls = group[logs_specs_name].load()

        # Python 版本 < 3.10，使用旧的 entry_points 方法
        elif specs := eps.get("torchrun.logs_specs"):
            # 查找名称匹配 logs_specs_name 的 entrypoint
            if entrypoint_list := [ep for ep in specs if ep.name == logs_specs_name]:
                logs_specs_cls = entrypoint_list[0].load()

        # 如果未找到对应的 logs_specs_cls，则抛出 ValueError 异常
        if logs_specs_cls is None:
            raise ValueError(
                f"Could not find entrypoint under 'torchrun.logs_specs[{logs_specs_name}]' key"
            )

        # 记录日志，指示正在使用哪个 logs_spec，并且显示其对应的类
        logging.info(
            "Using logs_spec '%s' mapped to %s", logs_specs_name, str(logs_specs_cls)
        )
    else:
        # 当 logs_specs_name 为 None 时，使用默认的 DefaultLogsSpecs
        logs_specs_cls = DefaultLogsSpecs

    # 返回确定的 logs_specs_cls 类
    return logs_specs_cls
# 从命令行参数 ``args`` 中获取配置，返回一个包含 LaunchConfig 对象、命令或字符串以及字符串列表的元组
def config_from_args(args) -> Tuple[LaunchConfig, Union[Callable, str], List[str]]:
    # 解析最小和最大节点数，并确保最小节点数大于0且小于等于最大节点数
    min_nodes, max_nodes = parse_min_max_nnodes(args.nnodes)
    assert 0 < min_nodes <= max_nodes
    # 确保 args.max_restarts 大于等于0
    assert args.max_restarts >= 0

    # 如果 args 中包含 "master_addr" 属性，并且 rdzv_backend 不是 "static"，并且 rdzv_endpoint 为空
    if (
        hasattr(args, "master_addr")
        and args.rdzv_backend != "static"
        and not args.rdzv_endpoint
    ):
        # 输出警告日志，提醒 master_addr 仅在 static rdzv_backend 和未指定 rdzv_endpoint 时使用
        logger.warning(
            "master_addr is only used for static rdzv_backend and when rdzv_endpoint "
            "is not specified."
        )

    # 确定每个节点的本地进程数量
    nproc_per_node = determine_local_world_size(args.nproc_per_node)
    # 如果环境变量中不存在 "OMP_NUM_THREADS"，并且 nproc_per_node 大于1
    if "OMP_NUM_THREADS" not in os.environ and nproc_per_node > 1:
        # 设置 OMP_NUM_THREADS 环境变量为 1，以避免系统超载，并输出警告日志
        omp_num_threads = 1
        logger.warning(
            "\n*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process to be "
            "%s in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************",
            omp_num_threads,
        )
        # 将该环境变量传递给子进程
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    # 从环境变量中获取 TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE
    log_line_prefix_template = os.getenv("TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE")

    # 解析 rdzv_conf 参数的配置
    rdzv_configs = _parse_rendezvous_config(args.rdzv_conf)

    # 如果 rdzv_backend 为 "static"，设置 rdzv_configs 中的 "rank" 属性为 args.node_rank
    if args.rdzv_backend == "static":
        rdzv_configs["rank"] = args.node_rank

    # 获取 rdzv_endpoint
    rdzv_endpoint = get_rdzv_endpoint(args)

    # 初始化 ranks 变量为 None
    ranks: Optional[Set[int]] = None
    # 如果 args.local_ranks_filter 存在
    if args.local_ranks_filter:
        try:
            # 将 args.local_ranks_filter 解析为整数集合，并确保至少有一个元素
            ranks = set(map(int, args.local_ranks_filter.split(",")))
            assert ranks
        except Exception as e:
            # 如果解析失败，抛出 ValueError 异常
            raise ValueError(
                "--local_ranks_filter must be a comma-separated list of integers e.g. --local_ranks_filter=0,1,2"
            ) from e

    # 根据 args.logs_specs 获取 LogsSpecs 类
    logs_specs_cls: Type[LogsSpecs] = _get_logs_specs_class(args.logs_specs)
    # 初始化 logs_specs 对象
    logs_specs = logs_specs_cls(
        log_dir=args.log_dir,
        redirects=Std.from_str(args.redirects),
        tee=Std.from_str(args.tee),
        local_ranks_filter=ranks,
    )

    # 创建 LaunchConfig 对象，用于配置启动参数
    config = LaunchConfig(
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        nproc_per_node=nproc_per_node,
        run_id=args.rdzv_id,
        role=args.role,
        rdzv_endpoint=rdzv_endpoint,
        rdzv_backend=args.rdzv_backend,
        rdzv_configs=rdzv_configs,
        max_restarts=args.max_restarts,
        monitor_interval=args.monitor_interval,
        start_method=args.start_method,
        log_line_prefix_template=log_line_prefix_template,
        local_addr=args.local_addr,
        logs_specs=logs_specs,
    )

    # 是否包含 Python 环境，即非 args.no_python
    with_python = not args.no_python
    # 初始化 cmd 变量，类型为 Union[Callable, str]
    cmd: Union[Callable, str]
    # 初始化 cmd_args 列表
    cmd_args = []
    # 获取 use_env 参数
    use_env = get_use_env(args)
    # 如果指定了 args.run_path
    if args.run_path:
        # 将 cmd 设置为 run_script_path 函数
        cmd = run_script_path
        # 添加 args.training_script 到 cmd_args 列表中
        cmd_args.append(args.training_script)
    # 如果不使用环境变量，向命令参数列表中添加本地排名信息
    if not use_env:
        cmd_args.append(f"--local-rank={macros.local_rank}")
    # 将训练脚本的额外参数扩展到命令参数列表中
    cmd_args.extend(args.training_script_args)

    # 返回配置信息、命令路径和命令参数列表作为结果
    return config, cmd, cmd_args
# 定义一个函数 `run_script_path`，用于在当前解释器中运行指定的训练脚本。
# `training_script` 是训练脚本的绝对路径，`training_script_args` 是可变长度的参数列表。
def run_script_path(training_script: str, *training_script_args: str):
    """
    Run the provided `training_script` from within this interpreter.

    Usage: `script_as_function("/abs/path/to/script.py", "--arg1", "val1")`
    """
    # 导入必要的模块
    import runpy
    import sys

    # 将命令行参数设置为指定的脚本路径和参数列表
    sys.argv = [training_script] + [*training_script_args]
    # 使用 `runpy` 模块执行指定路径的脚本文件，作为主程序执行
    runpy.run_path(sys.argv[0], run_name="__main__")


# 定义一个函数 `run`，用于启动训练过程
def run(args):
    # 设置 Torch 的多进程线程名为 "pt_elastic"
    torch.multiprocessing._set_thread_name("pt_elastic")

    # 如果参数中指定了 standalone 模式
    if args.standalone:
        # 设置分布式后端为 "c10d"
        args.rdzv_backend = "c10d"
        # 设置分布式通信的端点为本地主机的随机端口
        args.rdzv_endpoint = "localhost:0"
        # 为当前会话生成一个唯一的分布式 ID
        args.rdzv_id = str(uuid.uuid4())
        # 记录分布式配置信息到日志
        logger.info(
            "\n**************************************\n"
            "Rendezvous info:\n"
            "--rdzv-backend=%s "
            "--rdzv-endpoint=%s "
            "--rdzv-id=%s\n"
            "**************************************\n",
            args.rdzv_backend,
            args.rdzv_endpoint,
            args.rdzv_id,
        )

    # 从命令行参数中获取配置信息、命令和命令参数
    config, cmd, cmd_args = config_from_args(args)
    # 启动弹性训练任务，传入配置、入口命令和命令参数
    elastic_launch(
        config=config,
        entrypoint=cmd,
    )(*cmd_args)


# 使用装饰器 `record` 包装的主函数 `main`
@record
# 主函数 `main`，用于解析命令行参数并启动运行
def main(args=None):
    # 解析命令行参数
    args = parse_args(args)
    # 调用 `run` 函数，传入解析后的参数对象
    run(args)


# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 调用主函数 `main`，开始执行主程序逻辑
    main()
```