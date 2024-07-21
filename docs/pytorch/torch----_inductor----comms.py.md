# `.\pytorch\torch\_inductor\comms.py`

```py
# 声明代码静态分析工具类型检查为不进行类型定义
# 启用严格类型检查
from __future__ import annotations

# 导入默认字典集合模块
from collections import defaultdict
# 导入类型提示相关模块：字典、列表、集合、元组和类型检查
from typing import Dict, List, Set, Tuple, TYPE_CHECKING

# 导入 PyTorch 库
import torch

# 导入当前目录下的模块：config 和 ir
from . import config, ir
# 导入弱依赖模块
from .dependencies import WeakDep
# 导入工具函数：is_collective、is_wait 和 tuple_sorted
from .utils import is_collective, is_wait, tuple_sorted

# 使用 PyTorch 提供的日志功能创建日志记录器
overlap_log = torch._logging.getArtifactLogger(__name__, "overlap")

# 如果是类型检查阶段，则导入 BaseSchedulerNode 类
if TYPE_CHECKING:
    from .scheduler import BaseSchedulerNode


def sink_waits(
    snodes: List[BaseSchedulerNode],
    node_users: Dict[BaseSchedulerNode, Set[BaseSchedulerNode]],
    inverse_users: Dict[BaseSchedulerNode, Set[BaseSchedulerNode]],
) -> List[BaseSchedulerNode]:
    """
    尽可能晚地（直到到达使用点）贪婪地移动等待节点。在通信重叠方面是最优的。
    """
    # 存储重新排序后的节点列表
    new_order = []
    # 当前等待节点集合
    cur_waits = set()
    # 遍历调度节点列表
    for snode in snodes:
        # 如果节点是等待节点
        if is_wait(snode.node):
            # 将其添加到当前等待节点集合中
            cur_waits.add(snode)
        else:
            # 对当前等待节点集合按照特定顺序排序后遍历
            for wait in tuple_sorted(cur_waits):
                # 如果当前节点在等待节点的用户集合中
                if snode in node_users[wait]:
                    # 将等待节点添加到新顺序列表中
                    new_order.append(wait)
                    # 从当前等待节点集合中移除该节点
                    cur_waits.remove(wait)
            # 将当前节点添加到新顺序列表中
            new_order.append(snode)
    # 将剩余的等待节点按照特定顺序添加到新顺序列表末尾
    new_order.extend(tuple_sorted(cur_waits))
    # 返回最终的节点顺序列表
    return new_order


def raise_comms(
    snodes: List[BaseSchedulerNode],
    node_users: Dict[BaseSchedulerNode, Set[BaseSchedulerNode]],
    inverse_users: Dict[BaseSchedulerNode, Set[BaseSchedulerNode]],
) -> List[BaseSchedulerNode]:
    """
    尽可能早地（直到到达输入点）贪婪地移动通信节点。在通信重叠方面是最优的。

    TODO: 可能需要在将来调整以考虑内存限制。
    例如，当我们编译 FSDP 时，这种启发式方法将导致所有聚集操作尽可能早地预取，即在前向传递的开始处。
    我们可能需要为 FSDP 做特殊处理，或者重新考虑此次通信节点移动以处理通用的内存考虑。
    """
    # 存储逆序重新排序后的节点列表
    new_order_reversed: List[BaseSchedulerNode] = []
    # 当前通信节点列表
    cur_comms: List[BaseSchedulerNode] = []
    # 逆序遍历调度节点列表
    for snode in reversed(snodes):
        # 如果节点是集体通信节点
        if is_collective(snode.node):
            # 将其添加到当前通信节点列表中
            cur_comms.append(snode)
        else:
            # 确保每个通信节点的逆向用户集合长度大于零
            for comm in cur_comms:
                assert len(inverse_users[comm]) > 0
            # 当当前通信节点列表不为空，并且存在任意节点在当前节点的逆向用户集合中时
            while len(cur_comms) > 0 and any(
                snode in inverse_users[comm] for comm in cur_comms
            ):
                # 弹出当前通信节点列表的第一个节点
                comm = cur_comms.pop(0)
                # 将该通信节点添加到逆序重新排序列表中
                new_order_reversed.append(comm)
            # 将当前节点添加到逆序重新排序列表中
            new_order_reversed.append(snode)
    # 确保当前通信节点列表长度小于等于1
    assert len(cur_comms) <= 1
    # 将剩余的通信节点按照特定顺序添加到逆序重新排序列表末尾并反转返回
    new_order_reversed.extend(tuple_sorted(cur_comms))
    return new_order_reversed[::-1]


def get_ancestors(node, inverse_users):
    ancestors = set()
    cur_nodes = [node]
    # 当当前节点列表还有节点时，继续执行循环
    while len(cur_nodes) > 0:
        # 新节点列表初始化为空
        new_nodes = []
        # 遍历当前节点列表中的每个节点
        for node in cur_nodes:
            # 遍历节点的逆向用户列表中的每个输入
            for inp in inverse_users[node]:
                # 如果输入不在祖先集合中，则将其添加到祖先集合和新节点列表中
                if inp not in ancestors:
                    ancestors.add(inp)
                    new_nodes.append(inp)
        # 更新当前节点列表为新节点列表，以便下一轮循环处理
        cur_nodes = new_nodes
    # 循环结束后返回祖先集合
    return ancestors
    # 定义函数，用于获取给定节点的所有后代节点
    def get_descendants(node, node_users):
        # 初始化一个空集合来存储后代节点
        descendants = set()
        # 初始将给定节点作为当前节点列表的唯一成员
        cur_nodes = [node]
        # 当当前节点列表非空时循环
        while len(cur_nodes) > 0:
            # 初始化一个空列表，用于存储下一轮迭代中的新节点
            new_nodes = []
            # 遍历当前节点列表中的每个节点
            for node in cur_nodes:
                # 遍历每个节点的使用者（依赖节点）
                for inp in node_users[node]:
                    # 如果依赖节点不在已知的后代节点集合中，则添加到后代节点集合和新节点列表中
                    if inp not in descendants:
                        descendants.add(inp)
                        new_nodes.append(inp)
            # 更新当前节点列表为新节点列表，进行下一轮迭代
            cur_nodes = new_nodes
        # 返回所有后代节点的集合
        return descendants


    # 定义函数，决定通信节点的全局顺序
    def decide_global_ordering_of_comms(nodes: List[BaseSchedulerNode]):
        """
        Decide global ordering of comms, by just enforcing the ordering that's in the input graph
        (might not be the same ordering as the eager mode program).
        TODO: Come up with a better approach
        """
        # 筛选出所有的通信节点
        comm_nodes = [n for n in nodes if is_collective(n.node)]
        # 遍历除第一个外的每个通信节点
        for i in range(1, len(comm_nodes)):
            # 将前一个通信节点作为当前通信节点的弱依赖添加进去
            comm_nodes[i].add_fake_dep(WeakDep(comm_nodes[i - 1].get_name()))


    # 定义函数，检查是否存在通信节点
    def assert_no_comm_nodes(snodes: List[BaseSchedulerNode]) -> None:
        # 断言不存在任何通信节点
        assert not any(is_collective(snode.node) for snode in snodes)


    # 定义函数，估计操作节点的运行时间
    def estimate_op_runtime(snode: BaseSchedulerNode) -> float:
        """
        Returns estimated op runtime in nanoseconds (ns)
        """
        # 根据配置确定如何估计操作节点的运行时间
        if config.estimate_op_runtime == "default":
            runtime = snode.get_estimated_runtime()
        else:
            assert callable(config.estimate_op_runtime)
            runtime = config.estimate_op_runtime(snode)
        # 返回估计的运行时间（单位：纳秒）
        return runtime


    # 定义函数，计算节点的使用者和被使用者
    def compute_node_users(
        snodes: List[BaseSchedulerNode],
    ) -> Tuple[
        Dict[BaseSchedulerNode, Set[BaseSchedulerNode]],
        Dict[BaseSchedulerNode, Set[BaseSchedulerNode]],
    ]:
        from .scheduler import FusedSchedulerNode

        # 设置缓冲名称到（融合）节点的映射
        buf_to_snode: Dict[str, BaseSchedulerNode] = {}
        # 遍历所有调度节点
        for node in snodes:
            # 如果节点是融合调度节点，则遍历其子节点，并建立映射
            if isinstance(node, FusedSchedulerNode):
                for x in node.snodes:
                    buf_to_snode[x.get_name()] = node
            # 建立缓冲名称到节点本身的映射
            buf_to_snode[node.get_name()] = node

        # 计算反向使用者（节点被哪些节点使用）
        inverse_users = {
            node: {buf_to_snode[dep.name] for dep in node.unmet_dependencies}
            for node in snodes
        }

        # 计算节点使用者（节点使用哪些节点）
        # TODO: 理想情况下，应该合并 users 和 node_users，但目前 users 包含难以提取的额外信息。
        node_users: Dict[BaseSchedulerNode, Set[BaseSchedulerNode]] = defaultdict(set)
        for node, node_inverse_users in inverse_users.items():
            for inverse_user in node_inverse_users:
                node_users[inverse_user].add(node)

        # 返回节点使用者和反向使用者的字典
        return inverse_users, node_users


    # 定义函数，重新排序以实现计算重叠
    def reorder_compute_for_overlap(
        snodes: List[BaseSchedulerNode],
        node_users: Dict[BaseSchedulerNode, Set[BaseSchedulerNode]],
        inverse_users: Dict[BaseSchedulerNode, Set[BaseSchedulerNode]],
    ) -> List[BaseSchedulerNode]:
        """
        Decides a global ordering of all compute and communication nodes,
        ```
        # 此处是函数的主体部分，实现了对所有计算和通信节点的全局排序，
        #```py
    assuming that we already have a global ordering of communication nodes.

    Overall scheduling procedure is:
        Step 1: Given that we've currently scheduled comm N, we now schedule all compute nodes
            that are required for comm N + 1 but do not depend on comm N, to run at the same time with comm N.
        Step 2: If all those compute nodes are sufficient to overlap comm N, we're done.
            Otherwise, we now need to look elsewhere to find compute that overlaps with comm N.
            We prioritize compute nodes that are needed sooner.
        Step 3: We schedule the compute nodes dependent on comm N and required for comm N + 1.
        Step 4: We schedule comm N + 1.
        Repeat this for subsequent comm nodes.
    """
    # 初始化最终调度顺序为空列表
    final_order = []

    # 根据是否为集体通信节点筛选出通信节点列表
    comm_nodes = []
    for snode in snodes:
        if is_collective(snode.node):
            comm_nodes.append(snode)
    if len(comm_nodes) == 0:
        # 如果没有通信节点，直接返回当前顺序
        return snodes

    # 获取每个通信节点的祖先节点
    comm_ancestors = {node: get_ancestors(node, inverse_users) for node in comm_nodes}
    # 获取每个通信节点的后代节点
    comm_descendants = {node: get_descendants(node, node_users) for node in comm_nodes}

    # 初始化节点入度字典，所有节点入度为0
    indeg = dict.fromkeys(snodes, 0)
    # 计算每个节点的入度
    for snode in snodes:
        for user in node_users[snode]:
            if user in indeg:
                indeg[user] += 1

    # 初始化可调度的节点集合为所有节点
    ready_to_schedule_nodes = {node for node in snodes if indeg[node] == 0}

    # 初始化未调度节点集合为所有节点
    unscheduled_nodes = set(snodes)

    def schedule_node(snode):
        """
        Schedule a single node.
        """
        # 断言节点在未调度节点集合和可调度节点集合中
        assert snode in unscheduled_nodes
        assert snode in ready_to_schedule_nodes
        # 从可调度节点集合移除当前节点
        ready_to_schedule_nodes.remove(snode)
        # 从未调度节点集合移除当前节点
        unscheduled_nodes.remove(snode)
        # 将当前节点加入最终调度顺序中
        final_order.append(snode)
        # 遍历当前节点的每个后继节点
        for user in tuple_sorted(node_users[snode]):
            if user in indeg:
                # 减少后继节点的入度
                indeg[user] -= 1
                # 如果后继节点的入度为0，则加入可调度节点集合
                if indeg[user] == 0:
                    ready_to_schedule_nodes.add(user)
    # 定义函数，用于调度给定节点集合`snodes`中的所有节点，按照任意合法的拓扑顺序进行调度
    def schedule_nodes(snodes):
        """
        Schedules all nodes in `snodes` in an arbitrary topologically valid order.
        """
        # 获取所有未调度节点的集合
        all_nodes = set(snodes)
        # 断言所有节点都在未调度节点集合中
        assert all(node in unscheduled_nodes for node in all_nodes)
        # 当仍有未调度节点时循环进行调度
        while len(all_nodes) > 0:
            # 注意：由于模型图始终为有向无环图（DAG），内部不存在循环依赖，
            # 必定存在至少一个“自由节点”（即入度为0的节点），
            # 因此不可能进入无限循环。但这里进行检查是为了安全起见。
            progress = False
            # 对所有节点按照排序后的元组进行遍历
            for node in tuple_sorted(all_nodes):
                # 如果节点在可以调度的节点集合中
                if node in ready_to_schedule_nodes:
                    # 调度该节点
                    schedule_node(node)
                    # 从未调度节点集合中移除该节点
                    all_nodes.remove(node)
                    # 标记进展为True
                    progress = True
            # 如果没有进展（即没有找到可调度的节点），抛出断言错误
            if not progress:
                raise AssertionError(
                    "Unable to find a free node (indeg == 0). This is an impossible state to reach. "
                    "Please report a bug to PyTorch."
                )

    # 首先，调度所有计算节点，这些节点是第一个通信节点所需的先驱节点，以及第一个通信节点本身。
    assert len(comm_nodes) > 0
    # 调度节点集合为第一个通信节点的所有祖先节点加上第一个通信节点本身
    schedule_nodes(
        list(comm_ancestors[comm_nodes[0]]) + [comm_nodes[0]],
    )

    # 初始化已调度计算成本的累计值为0
    rolled_over_compute_cost = 0
    # 调度所有未调度节点
    schedule_nodes(unscheduled_nodes)
    # 返回最终的节点调度顺序
    return final_order
# 生成节点摘要信息，包括节点类型、外部内核输出、张量信息和节点名称
def node_summary(snode):
    detail = ""
    # 检查节点是否是外部内核输出，若是则附加内核名称到摘要详情中
    if isinstance(snode.node, ir.ExternKernelOut):
        detail = f" ({snode.node.python_kernel_name})"
    out_tensor_info = ""
    # 检查节点是否有布局信息及其大小和步长，若有则附加到张量信息中
    if (
        hasattr(snode.node, "layout")
        and hasattr(snode.node.layout, "size")
        and hasattr(snode.node.layout, "stride")
    ):
        out_tensor_info = (
            f" (size={snode.node.layout.size}, stride={snode.node.layout.stride})"
        )
    node_name = ""
    # 检查节点是否有名称，若有则附加到节点名称中
    if hasattr(snode.node, "name"):
        node_name = snode.node.name
    # 构建节点摘要字符串并返回
    return f"{snode.node.__class__.__name__}{detail}{out_tensor_info} ({node_name})"


# 可视化节点执行顺序和重叠情况
def visualize_overlap(order):
    total_est_runtime: float = 0.0
    cur_comm_node = None
    # 遍历调度顺序中的每个节点
    for snode in order:
        # 如果当前通信节点为空
        if cur_comm_node is None:
            # 如果当前节点是集合操作，则估计其运行时间并累加到总估计运行时间中
            if is_collective(snode.node):
                total_est_runtime += estimate_op_runtime(snode)
                cur_comm_node = snode.node
            # 如果当前节点是等待操作，抛出断言错误，因为没有集合操作正在运行时不应有等待操作
            elif is_wait(snode.node):
                raise AssertionError(
                    "Wait is not expected when there is no collective running"
                )
            else:  # 暴露的计算操作
                # 估计暴露的计算操作的运行时间并累加到总估计运行时间中
                total_est_runtime += estimate_op_runtime(snode)
            # 记录节点摘要信息到日志中
            overlap_log.debug(f"{node_summary(snode)}")  # noqa: G004
        else:  # 当前通信节点不为空
            # 如果当前节点是集合操作，则抛出断言错误，因为不应同时存在两个集合操作
            if is_collective(snode.node):
                raise AssertionError(
                    "Found two collectives running at the same time. "
                    "`visualize_overlap` needs to be updated to handle this case"
                )
            # 如果当前节点是等待操作，则表示当前通信操作结束
            elif is_wait(snode.node):
                # 记录节点摘要信息到日志中
                overlap_log.debug(f"{node_summary(snode)}")  # noqa: G004
                cur_comm_node = None
            else:  # 重叠的计算操作
                # 记录节点摘要信息到日志中，标记为重叠的计算操作
                overlap_log.debug(f"| {node_summary(snode)}")  # noqa: G004
    # 记录总估计运行时间到日志中
    overlap_log.debug(
        f"Est. runtime (ms): {total_est_runtime / 1000 / 1000}"  # noqa: G004
    )


# 重新排序计算和通信节点，以处理它们的重叠
def reorder_compute_and_comm_for_overlap(
    snodes: List[BaseSchedulerNode],
) -> List[BaseSchedulerNode]:
    order = snodes
    # 计算节点的逆用户和节点用户关系
    inverse_users, node_users = compute_node_users(snodes)
    # 返回重新排序后的节点顺序
    return order
    # 遍历配置中指定的重排序计算通信重叠的各个步骤
    for p in config.reorder_for_compute_comm_overlap_passes:
        # 如果当前步骤是字符串并且在全局变量中存在对应的变量
        if isinstance(p, str) and p in globals():
            # 将步骤替换为全局变量中的实际对象，表明这是一个内置的步骤
            p = globals()[p]  # it is a builtin pass

        # 如果当前进程的分布式计算中的主节点
        if torch.distributed.get_rank() == 0:
            # 记录重排序前的通信重叠可视化信息，包括当前步骤的名称
            overlap_log.debug(
                f"==== Visualize overlap before reordering pass {p} ===="  # noqa: G004
            )
            try:
                # 尝试可视化当前排序的通信重叠
                visualize_overlap(order)
            except Exception as e:
                # 记录可视化过程中的异常信息
                overlap_log.debug(str(e))

        # 执行当前步骤的重排序操作，更新排序结果
        order = p(order, node_users, inverse_users)  # type: ignore[operator]

        # 如果当前进程的分布式计算中的主节点
        if torch.distributed.get_rank() == 0:
            # 记录重排序后的通信重叠可视化信息，包括当前步骤的名称
            overlap_log.debug(
                f"==== Visualize overlap after reordering pass {p} ===="  # noqa: G004
            )
            try:
                # 尝试可视化重排序后的通信重叠
                visualize_overlap(order)
            except Exception as e:
                # 记录可视化过程中的异常信息
                overlap_log.debug(str(e))

    # 返回最终的排序结果
    return order
```