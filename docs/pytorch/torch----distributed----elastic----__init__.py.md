# `.\pytorch\torch\distributed\elastic\__init__.py`

```py
#!/usr/bin/env/python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""

Torchelastic agent and user worker failover contract:

**TL;DR;**:

* TE(torchelastic) expects user workers to finish with the 5 minutes drift
* It is better to design DDP app to fail for all workers, rather than a single one.
* TE does not synchronize number of restarts between agents
* TE re-rendezvous does not trigger restart decrease
* When a single agent finishes its job(successfully or not), it will close rendezvous.
  If other agents still have workers in progress, they will be terminated.
* Based on above, scale down does not work if at least single agent finishes the job.
* When Scale up is detected by agents, it will not decrease ``max_restarts``


In general TE(torchelastic) can launch arbitrary user code, but there is some
clarifications need to be done around what failover mechanism torchelastic
provides and what failover mechanism it expects from user workers.

Torchelastic currently supports DDP style applications.  That means that
TE expects *ALL* workers finish approximately at the same time. In practice,
it is nearly to impossible to guarantee that all workers in arbitrary
DDP application finish at the time, so TE provides a finalization barrier
that waits for TIMEOUT(5 minutes) for worker finalization.

**Worker Failure**

When worker fails, TE will check the number of restarts
available, if there is more than 0 restarts, TE will start a new rendezvous
round and restart the worker process. New rendezvous round will other
TE agents to terminate their workers.

.. note:: The TE agent does not synchronize restarts between themselves.
          When a single agent performs restart, it will trigger a local ``max_restarts``
          decrease, other agent will not decrease their ``max_restarts``.

# 如果某个单一代理完成其工作（无论成功与否），它将关闭集会。如果其他代理仍有正在进行的工作，它们将被终止。
# 基于以上情况，如果至少有一个代理完成了工作，缩减规模将不起作用。

A single worker failure can cause the whole cluster to fail:
If a single worker is constantly failing, it will cause the TE agent
``max_restarts``  to go to zero. This will cause an agent to finish its
work and close rendezvous. If there are any other workers on different
agents, they will be terminated.


**Re-Rendezvous**

Re-rendezvous occurs when TE agents detect a new node
trying to joint a cluster. TE will not decrease ``max_restarts``. TE agents
will terminate its workers and start a new rendezvous round.

Note about DynamicRendezvous(etcd-v2, c10d-experimental): If the rendezvous
has already max_nodes, the new node won't be added to the wait list right
away since there is no need to tear down a rendezvous that is already fully
utilized. The new node will wait until its timeout (600 secs by default)
and periodically check the number of participants. If the number becomes
"""
*Scale up event*. 当发生扩展事件时，torchelastic会检测到有新节点试图加入。Torchelastic代理将停止所有工作进程并进行重新集合。注意：当发生扩展事件时，*``max_restarts``* 不会减少。

*Scale down event*. 当发生缩减事件时，集合不会通知torchelastic代理。如果TE代理使用 ``max_restarts=0`` 启动，它依赖底层调度程序处理作业重启。如果 ``max_restarts>0`` ，TE代理将终止工作进程并启动新的集合轮次，这是一个*扩展事件*。
"""
```