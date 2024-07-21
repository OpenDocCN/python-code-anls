# `.\pytorch\third_party\valgrind-headers\callgrind.h`

```py
/*
   ----------------------------------------------------------------

   Notice that the following BSD-style license applies to this one
   file (callgrind.h) only.  The rest of Valgrind is licensed under the
   terms of the GNU General Public License, version 2, unless
   otherwise indicated.  See the COPYING file in the source
   distribution for details.

   ----------------------------------------------------------------

   This file is part of callgrind, a valgrind tool for cache simulation
   and call tree tracing.

   Copyright (C) 2003-2017 Josef Weidendorfer.  All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. The origin of this software must not be misrepresented; you must
      not claim that you wrote the original software.  If you use this
      software in a product, an acknowledgment in the product
      documentation would be appreciated but is not required.

   3. Altered source versions must be plainly marked as such, and must
      not be misrepresented as being the original software.

   4. The name of the author may not be used to endorse or promote
      products derived from this software without specific prior written
      permission.

   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
   OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
   DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
   GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   ----------------------------------------------------------------

   Notice that the above BSD-style license applies to this one file
   (callgrind.h) only.  The entire rest of Valgrind is licensed under
   the terms of the GNU General Public License, version 2.  See the
   COPYING file in the source distribution for details.

   ----------------------------------------------------------------
*/

#ifndef __CALLGRIND_H
#define __CALLGRIND_H

#include "valgrind.h"
/* !! ABIWARNING !! ABIWARNING !! ABIWARNING !! ABIWARNING !!
   This enum comprises an ABI exported by Valgrind to programs
   which use client requests.  DO NOT CHANGE THE ORDER OF THESE
   ENTRIES, NOR DELETE ANY -- add new ones at the end.

   The identification ('C','T') for Callgrind has historical
   reasons: it was called "Calltree" before. Besides, ('C','G') would
   clash with cachegrind.
 */

/* 定义 Valgrind 导出给使用客户端请求的程序的 ABI。不要改变这些条目的顺序，也不要删除任何条目 —— 新增条目应添加到末尾。

   'C','T' 的标识用于 Callgrind，有历史原因：之前称为 "Calltree"。此外，'C','G' 会与 cachegrind 冲突。
 */
typedef
   enum {
      VG_USERREQ__DUMP_STATS = VG_USERREQ_TOOL_BASE('C','T'),
      VG_USERREQ__ZERO_STATS,
      VG_USERREQ__TOGGLE_COLLECT,
      VG_USERREQ__DUMP_STATS_AT,
      VG_USERREQ__START_INSTRUMENTATION,
      VG_USERREQ__STOP_INSTRUMENTATION
   } Vg_CallgrindClientRequest;

/* Dump current state of cost centers, and zero them afterwards */
/* 转储当前成本中心的状态，并在此后将它们清零 */
#define CALLGRIND_DUMP_STATS                                    \
  VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__DUMP_STATS,       \
                                  0, 0, 0, 0, 0)

/* Dump current state of cost centers, and zero them afterwards.
   The argument is appended to a string stating the reason which triggered
   the dump. This string is written as a description field into the
   profile data dump. */
/* 转储当前成本中心的状态，并在此后将它们清零。
   参数将附加到一个字符串上，说明触发转储的原因。这个字符串将写入配置数据转储的描述字段中。 */
#define CALLGRIND_DUMP_STATS_AT(pos_str)                        \
  VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__DUMP_STATS_AT,    \
                                  pos_str, 0, 0, 0, 0)

/* Zero cost centers */
/* 将成本中心清零 */
#define CALLGRIND_ZERO_STATS                                    \
  VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__ZERO_STATS,       \
                                  0, 0, 0, 0, 0)

/* Toggles collection state.
   The collection state specifies whether the happening of events
   should be noted or if they are to be ignored. Events are noted
   by increment of counters in a cost center */
/* 切换收集状态。
   收集状态指定是否应记录事件的发生，或者它们是否应被忽略。事件通过在成本中心的计数器增加来记录 */
#define CALLGRIND_TOGGLE_COLLECT                                \
  VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__TOGGLE_COLLECT,   \
                                  0, 0, 0, 0, 0)

/* Start full callgrind instrumentation if not already switched on.
   When cache simulation is done, it will flush the simulated cache;
   this will lead to an artificial cache warmup phase afterwards with
   cache misses which would not have happened in reality. */
/* 如果尚未开启，启动完整的 Callgrind 仪器。
   当完成缓存模拟时，它将刷新模拟的缓存；这将导致在之后的人工缓存预热阶段中出现缓存未命中，这在现实中不会发生 */
#define CALLGRIND_START_INSTRUMENTATION                              \
  VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__START_INSTRUMENTATION, \
                                  0, 0, 0, 0, 0)

/* Stop full callgrind instrumentation if not already switched off.
   This flushes Valgrind's translation cache, and does no additional
   instrumentation afterwards, which effectively will run at the same
   speed as the "none" tool (i.e., at minimal slowdown).
   Use this to bypass Callgrind aggregation for uninteresting code parts.
   To start Callgrind in this mode to ignore the setup phase, use
   the option "--instr-atstart=no". */
/* 如果尚未关闭，停止完整的 Callgrind 仪器。
   这会刷新 Valgrind 的翻译缓存，并且之后不会进行任何额外的仪器操作，这实际上将以与 "none" 工具相同的速度运行（即最小减速）。
   使用此选项可以跳过 Callgrind 对于无趣代码部分的聚合。
   要以此模式启动 Callgrind 以忽略设置阶段，请使用选项 "--instr-atstart=no"。
 */
# 定义一个宏，用于停止 Callgrind 工具的代码插装
#define CALLGRIND_STOP_INSTRUMENTATION                               \
  VALGRIND_DO_CLIENT_REQUEST_STMT(VG_USERREQ__STOP_INSTRUMENTATION,  \
                                  0, 0, 0, 0, 0)
# 结束宏的定义
#endif /* __CALLGRIND_H */
```