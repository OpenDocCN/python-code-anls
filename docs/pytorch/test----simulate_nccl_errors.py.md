# `.\pytorch\test\simulate_nccl_errors.py`

```
# 导入必要的库
import argparse  # 用于解析命令行参数
import logging   # 用于日志记录
import os        # 用于与操作系统交互

import torch               # PyTorch深度学习框架
import torch.distributed as c10d  # PyTorch的分布式包，包括NCCL通信

# 配置日志格式和级别
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# 主程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="Simple script to simulate NCCL errors. The script is "
        "supposed to be run on multiple different nodes simultaneously with "
        "appropriate rank and world_size. The script run an allreduce() on "
        "the rank 0 node and aborts all the other nodes to simulate an error "
        "in NCCL"
    )
    # 添加命令行参数
    parser.add_argument("addr", help="address of the master node to connect to.")
    parser.add_argument("port", help="port of the master node to connect to.")
    parser.add_argument("rank", help="rank of this node")
    parser.add_argument("world_size", help="number of nodes in process group")
    # 解析命令行参数
    args = parser.parse_args()
    
    # 将解析得到的参数转换成整数
    rank = int(args.rank)
    world_size = int(args.world_size)
    port = int(args.port)

    # 创建TCPStore对象，用于节点间通信
    store = c10d.TCPStore(args.addr, port, world_size, rank == 0)
    # 创建NCCL进程组对象
    process_group = c10d.ProcessGroupNCCL(store, rank, world_size)

    # 打印日志，表示开始进行第一次全reduce操作
    logging.info("Running first allreduce")
    # 执行第一次全reduce操作，并等待其完成
    process_group.allreduce(torch.rand(10).cuda(rank)).wait()

    # 如果当前节点是rank为0的节点
    if rank == 0:
        # 打印日志，表示开始进行第二次全reduce操作（仅在rank为0的节点上执行）
        logging.info("Running second allreduce only on rank 0")
        # 执行第二次全reduce操作，并获得对应的work对象
        work = process_group.allreduce(torch.rand(10).cuda(rank))
        # 打印日志，表示正在等待全reduce操作完成
        logging.info("Waiting for allreduce to complete...")
        work.wait()
        # 打印日志，表示第二次全reduce操作成功完成
        logging.info("Second allreduce successful: %s", work.is_success())
    else:
        # 如果当前节点不是rank为0的节点，打印日志并中止当前进程
        logging.info("Aborting all other ranks.")
        os.abort()
```