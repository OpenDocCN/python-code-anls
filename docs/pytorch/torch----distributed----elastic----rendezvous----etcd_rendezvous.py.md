# `.\pytorch\torch\distributed\elastic\rendezvous\etcd_rendezvous.py`

```py
# 指定 Python 解释器路径，以及声明允许未类型化的函数定义
#!/usr/bin/env python3
# mypy: allow-untyped-defs

# 导入必要的模块
import json  # 导入处理 JSON 的模块
import logging  # 导入日志记录模块
import sys  # 导入系统相关的功能
import threading  # 导入线程支持模块
import time  # 导入时间相关的功能
from typing import Optional  # 引入类型提示中的 Optional 类型

import etcd  # type: ignore[import]  # 导入 etcd 客户端库，忽略类型检查

# 导入 Elastic 分布式框架的相关模块和类
from torch.distributed.elastic.rendezvous import (
    RendezvousClosedError,
    RendezvousError,
    RendezvousHandler,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStoreInfo,
    RendezvousTimeoutError,
)

# 导入本地的 etcd 存储和解析器
from .etcd_store import cas_delay, EtcdStore
from .utils import parse_rendezvous_endpoint

# 导出的模块列表
__all__ = [
    "EtcdRendezvousRetryableFailure",
    "EtcdRendezvousRetryImmediately",
    "EtcdRendezvousHandler",
    "EtcdRendezvous",
    "create_rdzv_handler",
]

# 设置日志的格式
_log_fmt = logging.Formatter("%(levelname)s %(asctime)s %(message)s")
_log_handler = logging.StreamHandler(sys.stderr)
_log_handler.setFormatter(_log_fmt)

# 创建日志记录器对象
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.INFO)
logger.addHandler(_log_handler)

# 自定义异常类：EtcdRendezvousRetryableFailure
# 表示由于某些原因无法进行状态转换，建议稍作延迟后重试
class EtcdRendezvousRetryableFailure(Exception):
    pass

# 自定义异常类：EtcdRendezvousRetryImmediately
# 表示可以立即重试而无需等待“安全延迟”
class EtcdRendezvousRetryImmediately(Exception):
    pass

# 默认的会合超时时间，单位：秒
_DEFAULT_TIMEOUT: int = 600  # 10 minutes

# 在达到最小节点数后的额外等待时间，单位：秒
_DEFAULT_LAST_CALL_TIMEOUT: int = 30  # 30 seconds

# EtcdRendezvous 内部使用的各种常量
CONST_ETCD_SETUP_TTL = 5  # Etcd 设置 TTL 时间，单位：秒
CONST_ETCD_FROZEN_TTL = 10  # 冻结状态 TTL 时间，单位：秒
CONST_ETCD_JOINABLE_EPHEMERAL_TTL = 10  # 可连接的临时 TTL 时间，单位：秒

# 工作节点保持活跃的临时节点 TTL 时间，单位：秒
CONST_WORKER_KEEPALIVE_TTL = 10

# 特定 run_id 的临时目录 TTL 时间，单位：秒
# 用于清理旧运行的会合数据，应大于预期工作进程的任何超时时间
CONST_RUNID_SUBROOT_TTL = 7200  # 2 hours

# 实现了 RendezvousHandler 接口的 EtcdRendezvousHandler 类
# 作为 EtcdRendezvous 的后端
class EtcdRendezvousHandler(RendezvousHandler):
    """
    Implements a
    :py:class:`torch.distributed.elastic.rendezvous.RendezvousHandler` interface
    backed by
    :py:class:`torch.distributed.elastic.rendezvous.etcd_rendezvous.EtcdRendezvous`.
    ``EtcdRendezvousHandler`` uses a URL to configure the type of rendezvous to
    use and to pass implementation specific configurations to the rendezvous
    """
    # The basic etcd rendezvous configuration URL looks like the following:
    #
    # etcd://<etcd_address>:<port>/<job_id>?min_workers=<min_workers>&max_workers=<max_workers>
    # noqa: W605
    #
    # -- example --
    # etcd://localhost:2379/1234?min_workers=1&max_workers=3
    #
    # The URL above is interpreted as follows:
    #
    # 1. Use the rendezvous handler that is registered with the ``etcd`` scheme
    # 2. The ``etcd`` endpoint to use is ``localhost:2379``
    # 3. ``job_id == 1234`` is used as the prefix in etcd (this allows one to
    #    share a common etcd server for multiple jobs so long as the
    #    ``job_ids`` are guaranteed to be unique). Note that the job id can be
    #    any string (e.g. does not need to be a number) as long as it is
    #    unique.
    # 4. ``min_workers=1`` and ``max_workers=3`` specifies a range for
    #    membership size - Torch Distributed Elastic starts running the job as
    #    long as the cluster size is greater than or equal to ``min_workers``
    #    and admits up to ``max_workers`` into the cluster.
    #
    # Below are a full list of the parameters that can be passed to etcd
    # rendezvous:
    #
    # +--------------------------------------------+--------------------------+
    # | Parameter                                  | Description              |
    # +============================================+==========================+
    # | min_workers                                | minimum number of        |
    # |                                            | workers for the          |
    # |                                            | rendezvous to be valid   |
    # +--------------------------------------------+--------------------------+
    # | max_workers                                | maximum number of        |
    # |                                            | workers to admit         |
    # +--------------------------------------------+--------------------------+
    # | timeout                                    | total timeout within     |
    # |                                            | which next_rendezvous is |
    # |                                            | expected to succeed      |
    # |                                            | (default 600s)           |
    # +--------------------------------------------+--------------------------+
    # | last_call_timeout                          | additional wait amount   |
    # |                                            | ("last call") after min  |
    # |                                            | number of workers has    |
    # |                                            | been reached (defaults   |
    # |                                            | to 30s)                  |
    # +--------------------------------------------+--------------------------+
    # | etcd_prefix                                | path prefix (from etcd   |
    # |                                            | root), inside which all  |
    # ```
    def __init__(self, rdzv_impl):
        self._rdzv_impl = rdzv_impl


# 初始化方法，用给定的 rdzv_impl 参数来设置 _rdzv_impl 实例变量
def __init__(self, rdzv_impl):
    self._rdzv_impl = rdzv_impl



    def __del__(self):
        # TODO: look into using weakref here instead.
        del self._rdzv_impl


# 析构方法，用于清理资源，删除 _rdzv_impl 实例变量
# TODO: 考虑使用 weakref 来替代当前的实现
def __del__(self):
    del self._rdzv_impl



    def get_backend(self) -> str:
        return "etcd"


# 返回当前后端存储的名称，固定返回 "etcd"
def get_backend(self) -> str:
    return "etcd"



    def next_rendezvous(self):
        rdzv_version, rank, world_size = self._rdzv_impl.rendezvous_barrier()

        logger.info("Creating EtcdStore as the c10d::Store implementation")
        store = self._rdzv_impl.setup_kv_store(rdzv_version)

        bootstrap_store_info = RendezvousStoreInfo.build(rank, store)
        return RendezvousInfo(store, rank, world_size, bootstrap_store_info)


# 执行下一个会面操作，获取当前版本、进程排名和总进程数
# 设置 EtcdStore 作为 c10d::Store 实现，并返回会面信息对象
def next_rendezvous(self):
    rdzv_version, rank, world_size = self._rdzv_impl.rendezvous_barrier()

    logger.info("Creating EtcdStore as the c10d::Store implementation")
    store = self._rdzv_impl.setup_kv_store(rdzv_version)

    bootstrap_store_info = RendezvousStoreInfo.build(rank, store)
    return RendezvousInfo(store, rank, world_size, bootstrap_store_info)



    def is_closed(self):
        try:
            _, state = self._rdzv_impl.get_rdzv_state()
            return state["status"] == "closed"
        except etcd.EtcdKeyNotFound:
            # No rendezvous state, so it cannot be closed.
            return False


# 检查会面是否已关闭
def is_closed(self):
    try:
        _, state = self._rdzv_impl.get_rdzv_state()
        return state["status"] == "closed"
    except etcd.EtcdKeyNotFound:
        # 如果未找到会面状态键，则表示会面未关闭
        return False



    def set_closed(self):
        self._rdzv_impl.set_closed()


# 设置会面为已关闭状态
def set_closed(self):
    self._rdzv_impl.set_closed()



    def num_nodes_waiting(self):
        try:
            _, state = self._rdzv_impl.get_rdzv_state()
            if state["status"] == "final":
                return state["num_workers_waiting"]
        except etcd.EtcdKeyNotFound:
            pass
        return 0


# 返回正在等待的节点数目
def num_nodes_waiting(self):
    try:
        _, state = self._rdzv_impl.get_rdzv_state()
        if state["status"] == "final":
            return state["num_workers_waiting"]
    except etcd.EtcdKeyNotFound:
        pass
    return 0



    def get_run_id(self) -> str:
        return self._rdzv_impl._run_id


# 返回当前运行的会话 ID
def get_run_id(self) -> str:
    return self._rdzv_impl._run_id



    def shutdown(self) -> bool:
        try:
            self.set_closed()
            return True
        except BaseException as e:
            logger.warning("Shutdown failed. Error occurred: %s", str(e))
            return False


# 关闭当前会话
def shutdown(self) -> bool:
    try:
        self.set_closed()
        return True
    except BaseException as e:
        # 如果关闭失败，则记录警告信息并返回 False
        logger.warning("Shutdown failed. Error occurred: %s", str(e))
        return False
# TODO: 我们可能需要处理一些额外的错误，比如 EtcdLeaderElectionInProgress 和 EtcdWatcherCleared。
# 这些只适用于多节点 Etcd 集群。简单的重试会起作用，但在每处都添加会显得冗长。
# 考虑为这些错误包装客户端调用的自动重试？

class EtcdRendezvous:
    """一个使用 `etcd <https://etcd.io/>`__ 作为后端存储的会合实现。"""

    def __init__(
        self,
        client,
        prefix,
        run_id,
        num_min_workers,
        num_max_workers,
        timeout,
        last_call_timeout,
    ):
        self.client = client  # 设置 Etcd 客户端

        # 输出 Etcd 机器列表到日志
        logger.info("Etcd machines: %s", self.client.machines)

        # 初始化实例变量
        self._prefix = prefix
        self._run_id = run_id
        self._num_min_workers = num_min_workers
        self._num_max_workers = num_max_workers
        self._timeout = timeout
        self._last_call_timeout = last_call_timeout

        # 用于清理 TTL 刷新线程（用于临时键）
        self._lease_run_id_stop = None
        self._lease_this_rank_stop = None

        # 如果前缀不以斜杠结尾，则添加斜杠
        if not self._prefix.endswith("/"):
            self._prefix += "/"

        # 如果前缀不是根目录，则尝试创建路径
        if self._prefix != "/":
            self.create_path_if_not_exists(self._prefix)

        # 为此作业实例（run_id）租用一个“子根”节点
        self.create_path_if_not_exists(self.get_path(""), ttl=CONST_RUNID_SUBROOT_TTL)
        self._lease_run_id_stop = self.setup_lease_renewal(
            self.get_path(""), ttl=CONST_RUNID_SUBROOT_TTL
        )

        # 创建所有会合工作的子目录
        self.create_path_if_not_exists(self.get_path("/rdzv"))

        # 创建会合版本计数器，如果不存在的话
        try:
            self.client.write(
                key=self.get_path("/rdzv/version_counter"), value="0", prevExist=False
            )
        except etcd.EtcdAlreadyExist:
            pass

    def __del__(self):
        # TODO: 考虑在这里使用弱引用（weakref）。
        # 如果存在 run_id 租约停止对象，则设置它
        if self._lease_run_id_stop is not None:
            self._lease_run_id_stop.set()

        # 如果存在 this_rank 租约停止对象，则设置它
        if self._lease_this_rank_stop is not None:
            self._lease_this_rank_stop.set()
    def rendezvous_barrier(self):
        """
        Main entry point for next rendezvous.

        This method is blocking until rendezvous succeeds or a timeout occurs.

        Returns:
             ``(rdzv_version, rank, world_size)``

        Raises:
            RendezvousTimeoutError - timeout waiting for rendezvous
            RendezvousClosedError - rendezvous is or was closed while waiting
            RendezvousError - other persistent errors that
             render the rendezvous non-retryable
        """
        # 设置等待超时的截止时间
        self._rendezvous_deadline = time.time() + self._timeout
        while True:
            # 检查是否超过了等待截止时间
            if time.time() > self._rendezvous_deadline:
                raise RendezvousTimeoutError

            # 记录尝试加入下一个 rendezvous 的日志信息
            logger.info("Attempting to join next rendezvous")
            try:
                # 如果存在之前 rendezvous 的租约，取消当前进程的租约
                if self._lease_this_rank_stop is not None:
                    self._lease_this_rank_stop.set()

                # 初始化当前阶段的操作，并返回结果
                return self.init_phase()

            except EtcdRendezvousRetryImmediately:
                # 如果是 EtcdRendezvousRetryImmediately 异常，表示可以立即重试
                pass

            except EtcdRendezvousRetryableFailure:
                # 如果是可重试的失败，等待一小段时间后再重试，避免频繁访问 etcd
                time.sleep(1)

            except RendezvousTimeoutError:
                # 如果是超时异常，则记录日志并重新抛出异常
                logger.info("Rendezvous timeout occurred in EtcdRendezvousHandler")
                raise

            except RendezvousClosedError:
                # 如果是 rendezvous 已关闭异常，则记录日志并重新抛出异常
                logger.info(
                    "Rendezvous for run_id=%s was observed to be closed", self._run_id
                )
                raise

            except RendezvousError:
                # 如果是其他 rendezvous 错误，则重新抛出异常
                raise

            except Exception as e:
                # 对于一般性异常，记录日志，并等待一小段时间后再重试，避免频繁访问 etcd
                logger.info("Rendezvous attempt failed, will retry. Reason: %s", e)
                time.sleep(1)
    def init_phase(self):
        """
        Initially, the rendezvous state is expected to be one of:

        1. empty (non-existent) - in this case we try to create a new one.
        2. joinable - we try to join it.
        3. final - we announce ourselves as waiting, and go into monitoring mode

        Any other state is considered transitional, and will be retried after
        a short delay.

        Returns:
            ``(rdzv_version, rank, world_size)``

        Raises:
            RendezvousClosedError - current rendezvous was/is closed
            EtcdRendezvousRetryableFailure - observed some intermediate
             state, which is best handled by retrying later
        """
        try:
            # 尝试创建新的会合点并获取当前活跃版本
            active_version = self.try_create_rendezvous()
            # 解析当前活跃版本的状态信息
            state = json.loads(active_version.value)
            logger.info("New rendezvous state created: %s", state)
        except etcd.EtcdAlreadyExist:
            # 如果会合点已经存在，则获取其状态信息
            active_version, state = self.get_rdzv_state()
            # 注意：上述查询可能失败 (etcd.EtcdKeyNotFound)，但这对我们来说没关系，意味着我们会重新开始。
            logger.info("Observed existing rendezvous state: %s", state)

        # 如果会合点状态为 "closed"，抛出异常表示会合点已关闭
        if state["status"] == "closed":
            raise RendezvousClosedError

        # 如果会合点状态为 "joinable"，执行加入阶段操作
        if state["status"] == "joinable":
            return self.join_phase(state["version"])

        # 如果会合点状态为 "final"，处理已存在的会合点状态
        if state["status"] == "final":
            self.handle_existing_rendezvous(state["version"])
            # 立即触发重试异常
            raise EtcdRendezvousRetryImmediately

        # 尝试等待状态变更，并抛出可重试异常
        self.try_wait_for_state_change(etcd_index=active_version.etcd_index + 1)
        raise EtcdRendezvousRetryableFailure
    def join_phase(self, expected_version):
        """
        We observed a rendezvous state in 'joinable' state, and attempt to join this
        particular version, and then wait for all other peers to join.
        """
        # Failure to join will propagate an exception, causing a re-entry.
        # 获取当前活跃的版本号和当前的rank
        active_version, this_rank = self.join_rendezvous(expected_version)
        # 解析活跃版本的状态信息
        state = json.loads(active_version.value)
        logger.info(
            "Joined rendezvous version %s as rank %s. Full state: %s",
            state["version"],
            this_rank,
            state,
        )

        # If this worker was first to reach num_min_workers requirement,
        # and rendezvous is still joinable (therefore it is elastic),
        # then this worker will be responsible for waiting out the "last call"
        # timeout and closing (i.e. transitioning to 'frozen') the rendezvous
        # afterwards.
        # As a safety against a potential failure of this worker (during the
        # last call timeout), the rendezvous state is made ephemeral
        # when min_num_workers is reached.
        # 如果当前worker是第一个达到num_min_workers要求的，并且rendezvous仍然是joinable状态
        # 则这个worker将负责等待“最终呼叫”超时并关闭（即转变为'frozen'）rendezvous
        # 作为对这个worker潜在故障的一种保护（在“最终呼叫”超时期间）
        # 当达到min_num_workers时，rendezvous状态会变为临时的
        if this_rank == self._num_min_workers - 1 and state["status"] == "joinable":
            logger.info("Rank %s is responsible for join last call.", this_rank)
            last_call_deadline = time.time() + self._last_call_timeout
            self.handle_join_last_call(expected_version, last_call_deadline)
            logger.info("Rank %s finished join last call.", this_rank)

        # Wait for rendezvous state to be frozen, which means a fixed set of peers
        # 等待rendezvous状态转变为frozen，这意味着一个固定的peer集合
        logger.info("Waiting for remaining peers.")
        active_version = self.wait_for_peers(expected_version)
        state = json.loads(active_version.value)

        assert (
            state["version"] == expected_version
        ), "Logic error: failed to observe version mismatch"

        return self.confirm_phase(expected_version, this_rank)

    def confirm_phase(self, expected_version, this_rank):
        """
        Once the rendezvous state transitions from 'joinable' to 'frozen',
        we have every participant confirm their membership and setup per-member
        keep-alive TTL keys, and then wait for all other participants to confirm,
        which would then successfully conclude this rendezvous.
        """
        # 所有peer都到达了，确认会员资格
        logger.info("All peers arrived. Confirming membership.")
        self.confirm_membership(expected_version, this_rank)

        # 等待所有peer确认
        logger.info("Waiting for confirmations from all peers.")
        active_version = self.wait_for_final(expected_version)
        state = json.loads(active_version.value)

        logger.info(
            "Rendezvous version %s is complete. Final state: %s",
            state["version"],
            state,
        )

        # 返回rendezvous版本号、当前rank以及参与者数量
        return state["version"], this_rank, len(state["participants"])
    def handle_existing_rendezvous(self, expected_version):
        """
        Handle the case when there's an existing (state 'final) rendezvous already
        in place, and we have to announce ourselves waiting, and wait until
        the next rendezvous opportunity.
        """
        # 如果状态为'final'，增加等待的工作节点数目
        active_state = self.announce_self_waiting(expected_version)
        logger.info(
            "Added self to waiting list. Rendezvous full state: %s", active_state.value
        )

        # 等待现有的会合状态改变，重新尝试加入
        self.wait_for_rendezvous_to_free(expected_version)
        logger.info(
            "Previously existing rendezvous state changed. Will re-try joining."
        )

    def try_create_rendezvous(self):
        """
        Create new rendezvous state or raise an exception that indicates an unexpected state (e.g. already exists).

        Raises:
             RendezvousError - on unexpected state
        """
        # 初始时活动版本是短暂的，用于处理可能无法完成设置事务的情况
        active_version = self.client.write(
            key=self.get_path("/rdzv/active_version"),
            value=json.dumps({"status": "setup"}),
            prevExist=False,
            ttl=CONST_ETCD_SETUP_TTL,
        )

        try:
            # 获取会合版本计数器并增加计数
            version_counter = self.client.get(self.get_path("/rdzv/version_counter"))
            version_counter.value = str(int(version_counter.value) + 1)
            self.client.update(version_counter)
        except (etcd.EtcdKeyNotFound, etcd.EtcdCompareFailed) as e:
            # 如果未找到计数器或比较失败，抛出异常
            raise RendezvousError(
                "Unexpected state of EtcdRendezvousHandler, worker needs to die."
            ) from e

        # 创建参与者数据的目录节点
        self.client.write(
            key=self.get_path(f"/rdzv/v_{version_counter.value}"),
            value=None,
            dir=True,
            prevExist=False,
        )

        # 发布会合版本并标记为可以加入状态
        # 如果在此之前会合被关闭，将会进行重试，处理关闭状态
        return self.client.test_and_set(
            key=self.get_path("/rdzv/active_version"),
            value=json.dumps(
                {
                    "status": "joinable",
                    "version": version_counter.value,
                    "participants": [],
                }
            ),
            prev_value=active_version.value,
        )
    def join_rendezvous(self, expected_version):
        """Helper method for the join phase."""
        # 使用比较和交换（CAS）操作将自身添加到会面状态中：
        while True:
            cas_delay()  # 调用CAS延迟函数，可能是用来模拟延迟的函数

            # 获取当前会面状态的版本号和状态信息
            active_version, state = self.get_rdzv_state()

            # 如果会面状态不可加入，则抛出可重试的失败异常
            if state["status"] != "joinable":
                raise EtcdRendezvousRetryableFailure(
                    "Rendezvous state became non-joinable before we could join. "
                    "Must join next one."
                )

            # 如果会面状态版本号不符合预期，则抛出立即重试异常
            if state["version"] != expected_version:
                raise EtcdRendezvousRetryImmediately(
                    "Rendezvous version changed. Must try join the new one."
                )

            # 断言：加入会面的参与者数量应小于最大工作进程数
            assert (
                len(state["participants"]) < self._num_max_workers
            ), "Logic error: joinable rendezvous should always have space left"

            # 计算当前加入会面的进程排名
            this_rank = len(state["participants"])
            state["participants"].append(this_rank)

            # 当达到最小工作进程数或状态变为冻结时，设置活动版本节点为临时的
            set_ttl: Optional[int] = None
            if len(state["participants"]) == self._num_max_workers:
                state["status"] = "frozen"
                state["keep_alives"] = []
                set_ttl = CONST_ETCD_FROZEN_TTL
            elif len(state["participants"]) >= self._num_min_workers:
                set_ttl = CONST_ETCD_JOINABLE_EPHEMERAL_TTL

            try:
                # 比较和交换操作
                active_version = self.client.test_and_set(
                    key=self.get_path("/rdzv/active_version"),
                    value=json.dumps(state),
                    prev_value=active_version.value,
                    ttl=set_ttl,
                )
                # 成功加入会面，返回活动版本和当前进程排名
                return active_version, this_rank

            except etcd.EtcdCompareFailed:
                logger.info("Join rendezvous CAS unsuccessful, retrying")

    def wait_for_peers(self, expected_version):
        """Helper method for the join phase."""
        # 获取当前会面状态的版本号和状态信息
        active_version, state = self.get_rdzv_state()
        while True:
            # 如果会面状态为冻结且版本号与预期相符，则成功，所有对等方已到达
            if state["status"] == "frozen" and state["version"] == expected_version:
                return active_version  # 返回活动版本

            # 如果会面状态为可加入且版本号与预期相符，则继续等待状态变化
            elif state["status"] == "joinable" and state["version"] == expected_version:
                active_version, state = self.try_wait_for_state_change(
                    etcd_index=active_version.etcd_index + 1
                )

            else:
                # 在此点无法进行有效的状态转换，抛出可重试的失败异常
                raise EtcdRendezvousRetryableFailure(
                    "Rendezvous state transition no longer possible. Must re-enter."
                )
    def confirm_membership(self, expected_version, this_rank):
        """Helper method for the confirm phase."""
        # Compare-and-swap loop
        while True:
            cas_delay()  # 调用延迟函数，用于比较和交换操作之间的延迟

            # 获取当前活跃版本和状态
            active_version, state = self.get_rdzv_state()

            # 如果状态不是 "frozen"，则抛出异常，要求立即重试
            if state["status"] != "frozen":
                raise EtcdRendezvousRetryImmediately(
                    "Rendezvous no longer frozen, before we confirmed. "
                    "Must join next one"
                )

            # 如果版本号与预期版本号不一致，抛出异常，要求立即重试
            if state["version"] != expected_version:
                raise EtcdRendezvousRetryImmediately(
                    "Rendezvous version changed. Must try join the new one."
                )

            # 获取当前节点的租约键路径
            this_lease_key = self.get_path(
                f"/rdzv/v_{expected_version}/rank_{this_rank}"
            )

            # 将当前节点的租约信息写入 Etcd，并设置 TTL
            self.client.set(this_lease_key, value=None, ttl=CONST_WORKER_KEEPALIVE_TTL)

            # 将当前租约键添加到状态中的保持活跃列表中
            state["keep_alives"].append(this_lease_key)

            # 如果所有参与者都确认了，当前节点是最后一个确认的
            if len(state["keep_alives"]) == len(state["participants"]):
                state["status"] = "final"  # 更新状态为 "final"
                state["num_workers_waiting"] = 0  # 重置等待的工作节点数量
                finalize = True  # 标记为最终确认
            else:
                finalize = False  # 否则标记为未最终确认

            try:
                # 比较并交换操作。如果新状态仍然是 "frozen"，则保持为临时状态
                active_version = self.client.test_and_set(
                    key=self.get_path("/rdzv/active_version"),
                    value=json.dumps(state),
                    prev_value=active_version.value,
                    ttl=None if finalize else CONST_ETCD_FROZEN_TTL,
                )

                # 设置当前租约键的续约机制
                self._lease_this_rank_stop = self.setup_lease_renewal(
                    this_lease_key, ttl=CONST_WORKER_KEEPALIVE_TTL
                )
                return active_version  # 返回更新后的活跃版本信息

            except etcd.EtcdCompareFailed:
                logger.info("Confirm membership CAS unsuccessful, retrying")

    def wait_for_final(self, expected_version):
        """Helper method for the confirm phase."""
        # 获取当前活跃版本和状态
        active_version, state = self.get_rdzv_state()

        while True:
            # 如果状态为 "final" 且版本号与预期版本号一致，则表示成功，返回活跃版本信息
            if state["status"] == "final" and state["version"] == expected_version:
                return active_version

            # 如果状态为 "frozen" 且版本号与预期版本号一致，则继续等待状态变更
            elif state["status"] == "frozen" and state["version"] == expected_version:
                # 尝试等待状态变更，增加 Etcd 索引以检测状态变更
                active_version, state = self.try_wait_for_state_change(
                    etcd_index=active_version.etcd_index + 1
                )

            else:
                # 在此点上不可能进行有效的状态转换，抛出可重试的失败异常
                raise EtcdRendezvousRetryableFailure(
                    "Rendezvous state transition no longer possible. Must re-enter."
                )
    def announce_self_waiting(self, expected_version):
        """
        Announce this worker is waiting (via num_workers_waiting counter) to join next
        rendezvous, but only if state and version match.
        """
        # 无限循环，等待状态与版本匹配后进行下一步操作
        while True:
            # 调用 cas_delay() 函数引入延迟以减少竞争
            cas_delay()
            # 获取当前的 rendezvous 状态和活跃版本号
            active_version, state = self.get_rdzv_state()

            # 如果状态不是 "final" 或者版本号不等于期望版本号，则立即抛出重试异常
            if state["status"] != "final" or state["version"] != expected_version:
                raise EtcdRendezvousRetryImmediately

            # 增加等待中的工作进程计数器
            state["num_workers_waiting"] += 1

            try:
                # 使用 test_and_set 方法尝试更新 etcd 中的活跃版本信息
                active_version = self.client.test_and_set(
                    key=self.get_path("/rdzv/active_version"),
                    value=json.dumps(state),
                    prev_value=active_version.value,
                )
                return active_version

            except etcd.EtcdCompareFailed:
                logger.info("Announce self as waiting CAS unsuccessful, retrying")

    def set_closed(self):
        """
        Mark rendezvous 'closed' for current run_id, which is used to signal other
        participants to not attempt to perform (re-)rendezvous. This is useful
        when one of the workers decides the job is complete.
        """
        # 无限循环，直到成功将状态标记为 "closed"
        while True:
            # 获取当前的 rendezvous 状态和活跃版本号
            active_version, state = self.get_rdzv_state()

            # 如果状态已经是 "closed"，则说明已经被其他进程关闭，直接返回
            if state["status"] == "closed":
                # 已被其他进程标记为关闭状态，直接返回
                return

            # 将状态标记为 "closed"
            state["status"] = "closed"
            try:
                # 使用 test_and_set 方法尝试更新 etcd 中的活跃版本信息
                self.client.test_and_set(
                    key=self.get_path("/rdzv/active_version"),
                    value=json.dumps(state),
                    prev_value=active_version.value,
                )
                return

            except etcd.EtcdCompareFailed:
                logger.info("Set closed CAS unsuccessful, retrying")
                cas_delay()

    def get_rdzv_state(self):
        # 获取当前在 etcd 中存储的活跃版本信息
        active_version = self.client.get(key=self.get_path("/rdzv/active_version"))
        # 解析活跃版本信息的值，返回活跃版本对象和其对应的 JSON 解析结果
        return active_version, json.loads(active_version.value)

    def try_wait_for_state_change(self, etcd_index, timeout=None):
        # 不要超过整体截止时间（至少再多 1 秒）
        overall_timeout = max(self._rendezvous_deadline - time.time(), 0.0) + 1.0
        timeout = overall_timeout if timeout is None else min(timeout, overall_timeout)

        try:
            # 使用 watch 方法监视 etcd 中活跃版本的变化，基于指定的 etcd 索引和超时时间
            self.client.watch(
                self.get_path("/rdzv/active_version"), index=etcd_index, timeout=timeout
            )
        except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
            # 如果监视期间发生 EtcdEventIndexCleared 或 EtcdWatchTimedOut 异常，则捕获并忽略
            pass

        # 如果当前时间超过了 rendezvous 截止时间，则抛出超时异常
        if time.time() > self._rendezvous_deadline:
            raise RendezvousTimeoutError

        # 不幸的是，我们必须再次获取以获取最后的 etcd 索引
        return self.get_rdzv_state()
    # 构建完整的路径，格式为：self._prefix + "run_" + self._run_id + path
    def get_path(self, path):
        if not path.startswith("/"):
            path = "/" + path

        return f"{self._prefix}run_{self._run_id}{path}"

    # 如果路径不存在，则创建该路径，并设置TTL（Time-To-Live，生存时间）
    def create_path_if_not_exists(self, full_path, ttl=None):
        try:
            # 使用etcd客户端写入操作，在etcd中创建指定路径的目录节点
            self.client.write(
                key=full_path, value=None, dir=True, prevExist=False, ttl=ttl
            )
        except etcd.EtcdAlreadyExist:
            # 如果路径已经存在（并且设置了prevExist=False），则捕获已存在异常并忽略
            pass

    # 设置租约续约机制，用于维护临时键的TTL（Time-To-Live，生存时间）
    def setup_lease_renewal(self, full_path, ttl):
        # 内部函数定义，用于在后台线程中定期刷新租约
        def lease_worker(client, path, ttl, stop_event):
            while True:
                try:
                    # 使用etcd客户端刷新指定路径的租约，延长其生存时间
                    client.refresh(path, ttl=ttl)
                except etcd.EtcdKeyNotFound:
                    # 如果路径不存在，则退出循环
                    break
                except ConnectionRefusedError:
                    # 当连接被拒绝时，通常发生在测试期间服务器已终止但Python垃圾收集器尚未调用__del__方法时，也退出循环
                    break

                # 等待一段时间后继续刷新租约，时间间隔为TTL的一半
                if stop_event.wait(timeout=ttl / 2):
                    break

        # 创建一个线程停止事件
        lease_stop_event = threading.Event()
        # 创建后台线程，用于执行租约刷新工作
        lease_thread = threading.Thread(
            target=lease_worker, args=(self.client, full_path, ttl, lease_stop_event)
        )

        # 将线程设置为守护线程，以确保在主线程退出时自动退出
        lease_thread.daemon = True
        # 启动后台线程
        lease_thread.start()

        # 返回租约停止事件对象，以便需要时可以停止后台线程
        return lease_stop_event

    # 存储额外的数据到指定的路径中
    def store_extra_data(self, rdzv_version, key, value):
        # 获取完整的存储路径
        node = self.get_path(f"/rdzv/v_{rdzv_version}/extra_data")
        try:
            # 尝试写入额外数据到指定路径，如果路径不存在则创建
            extra_data = self.client.write(
                key=node, value=json.dumps({key: value}), prevExist=False
            )
            return
        except etcd.EtcdAlreadyExist:
            # 如果路径已经存在，捕获异常并忽略
            pass

        # CAS（比较并设置）循环，确保不会丢失并发存储
        while True:
            # 获取当前存储路径的数据
            extra_data = self.client.get(node)

            # 解析JSON格式的数据
            new_extra_data_value = json.loads(extra_data.value)
            # 更新数据中的指定键值对
            new_extra_data_value[key] = value

            try:
                # 使用CAS操作更新数据，如果前值匹配则更新成功，否则继续重试
                extra_data = self.client.test_and_set(
                    key=node,
                    value=json.dumps(new_extra_data_value),
                    prev_value=extra_data.value,
                )
                return
            except etcd.EtcdCompareFailed:
                # 如果CAS操作失败，则记录日志并等待一段时间后重试
                logger.info("Store extra_data CAS unsuccessful, retrying")
                time.sleep(0.1)
    # 加载额外数据，根据给定的rdzv版本和key，可选超时参数
    def load_extra_data(self, rdzv_version, key, timeout=None):
        # 获取 'extra_data' 节点本身，以及它所在的目录：
        node = self.get_path(f"/rdzv/v_{rdzv_version}/extra_data")
        node_dir = self.get_path(f"/rdzv/v_{rdzv_version}")

        # TODO: implement timeout
        # https://github.com/pytorch/elastic/issues/12
        # 循环直到操作完成
        while True:
            # 组合等待节点本身和其中的键。
            root = self.client.get(node_dir)

            # 查找 extra_data 节点，如果存在的话
            extra_data = [n for n in root.children if n.key == node]
            assert len(extra_data) <= 1

            # 如果找到了 extra_data 节点，检查其中的指定键。
            if len(extra_data) == 1:
                extra_data_dict = json.loads(extra_data[0].value)
                if key in extra_data_dict:
                    return extra_data_dict[key]

            # 'extra_data' 节点不存在，或者键尚未发布。
            # 等待 'extra_data' 节点上的有趣事件并重试。
            try:
                self.client.watch(node, index=root.etcd_index + 1)
            except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
                pass

    # 设置键值存储，使用给定的 rdzv 版本
    def setup_kv_store(self, rdzv_version):
        store_path = self.get_path(f"/rdzv/v_{rdzv_version}/kv")
        self.create_path_if_not_exists(store_path)
        return EtcdStore(etcd_client=self.client, etcd_store_prefix=store_path)
# 从给定的 RendezvousParameters 创建一个新的 etcd.Client 对象
def _create_etcd_client(params: RendezvousParameters) -> etcd.Client:
    # 解析 RendezvousParameters 中的 hostname 和 port
    hostname, port = parse_rendezvous_endpoint(params.endpoint, 2379)

    # 获取通信协议，默认为 "http"，如果未指定则使用默认值
    protocol = params.config.get("protocol")
    if protocol is None:
        protocol = "http"
    else:
        # 检查协议是否为 "http" 或 "https"，否则抛出异常
        if protocol != "http" and protocol != "https":
            raise ValueError("The etcd protocol must be HTTP or HTTPS.")

    # 获取 SSL 客户端证书，如果提供了证书，则检查是否也提供了证书密钥
    ssl_cert = params.config.get("cert")
    if ssl_cert is not None:
        cert_key = params.config.get("key")
        if cert_key is not None:
            # 将证书和密钥封装为元组，符合 etcd 客户端的要求
            ssl_cert = (ssl_cert, cert_key)

    # 获取根证书路径
    ca_cert = params.config.get("cacert")

    # 创建 etcd.Client 对象并返回
    return etcd.Client(
        hostname,
        port,
        protocol=protocol,
        cert=ssl_cert,
        ca_cert=ca_cert,
        allow_reconnect=True,
    )


# 处理 torch.distributed 的 "static" 注册的方法
def create_rdzv_handler(params: RendezvousParameters) -> RendezvousHandler:
    """
    使用示例:

    ::

        rdzv_params = RendezvousParameters(
                            backend="etcd",
                            endpoint="192.168.0.42:2379",
                            run_id="123",
                            min_nodes=4,
                            max_nodes=8,
                            timeout=300,
                            last_call_timeout=30,
                            etcd_prefix="custom_prefix",
                            protocol="https",
                            cacert="/etc/kubernetes/certs/ca.crt",
                            cert="/etc/kubernetes/certs/client.crt",
                            key="/etc/kubernetes/certs/client.key")
        # -- 或 --
        rdzv_params = RendezvousParameters(
                            backend="etcd",
                            endpoint="192.168.0.42:2379",
                            run_id="123",
                            min_nodes=4,
                            max_nodes=8)

        etcd_rdzv_handler = create_etcd_rendezvous_handler(rdzv_params)
    """
    # 此函数主要用于创建基于 etcd 的 rendezvous handler，返回相应的 handler 对象
    pass
    # 创建一个 etcd 客户端，使用给定的参数初始化
    client = _create_etcd_client(params)

    # 从参数中获取 etcd 节点的路径前缀，默认为 "/torchelastic/p2p"
    etcd_prefix = params.get("etcd_prefix", "/torchelastic/p2p")

    # 使用 EtcdRendezvous 类创建一个 rendezvous 对象
    rdzv = EtcdRendezvous(
        client=client,                                 # 使用上面创建的 etcd 客户端
        prefix=etcd_prefix,                            # 使用上面获取的 etcd 前缀路径
        run_id=params.run_id,                          # 使用给定的 run_id
        num_min_workers=params.min_nodes,               # 使用给定的 min_nodes 作为最小工作节点数
        num_max_workers=params.max_nodes,               # 使用给定的 max_nodes 作为最大工作节点数
        timeout=params.get_as_int("timeout", _DEFAULT_TIMEOUT),  # 获取或使用默认的超时时间
        last_call_timeout=params.get_as_int(
            "last_call_timeout", _DEFAULT_LAST_CALL_TIMEOUT    # 获取或使用默认的最后调用超时时间
        ),
    )

    # 返回一个 EtcdRendezvousHandler 对象，其包装了上面创建的 rendezvous 对象
    return EtcdRendezvousHandler(rdzv_impl=rdzv)
```