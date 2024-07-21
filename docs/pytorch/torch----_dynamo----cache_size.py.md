# `.\pytorch\torch\_dynamo\cache_size.py`

```py
# 引入必要的模块和库
# mypy: allow-untyped-defs
import logging  # 导入日志记录模块
import types  # 导入类型模块
import weakref  # 导入弱引用模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Tuple  # 导入类型提示中的元组类型

from . import config  # 导入当前目录下的config模块

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

"""
[Note on cache size limit]

Background - TorchDynamo cache is a linked list. Each cache entry is a
(check_fn, out_code, next pointer). These are stored on the f_code's co_extra
scratch space. When a frame is invoked, we walk this linked list and run
check_fn in each cache_entry to decide if the frame needs recompilation. If none
of the check_fn's returns True, we recompile and add a new entry. To ensure we
don't end up recompiling infinitely, we put limits on the cache size.

There are two limits
1) cache_size_limit
2) accumulated_cache_size_limit

Earlier we used to have only limit - maximum number of entries in 1 cache line
(which is now represented by (2) above). So, why do we need two limits? Lets try
to understand that.

In general, we want our cache limit value to be a small number (e.g. 8 or even
lower). This ensures that for frames that cause too many recompilation fall to
eager quickly. However, there is another problem that prevents us from lowering
the value of cache_size_limit. This is due to ID_MATCH'd guards. Today, we put
ID_MATCH guards on nn module if there is a graph break. This means we will have
many recompilations for the same code object because the ID_MATCH guard fails
for different instances of the nn module. This is a common pattern in how models
are authored. Therefore, this requires us to keep the cache_size_limit high.

We resolve this by introducing these two limits. The first limit (1) limits the
number of cache entries that have an ID_MATCH'd guard for an nn module instance.
And, (2)nd limit becomes a safeguard mechanism to have a maximum compilations
for a code object. One important question is - what is the limit for the code
object that does not have any ID_MATCH guard? For such code objects, we choose
(1) as the cache size limit.

Lets take an example to understand how these limits help. Suppose, we have 16
instances of a nn module and we ID_MATCH on the self object. Further, suppose
the inputs to these functions have varying batch size, leading to one
recompilation. In total, there will be 32 recompilations, and therefore 32 cache
entries on the forward code object. In the older case when we had only 1 limit,
our cache size limit must be >= 32 to capture all these recompilations. Now,
suppose there is a separate function in the same program which is very dynamic
and unsuitable for compilation. Such a function will need to undergo 32
compilations to burst the cache and fallback to eager. These 32 recompilations
are too many and we want to fallback for these compilation-unfriendly functions
sooner.

In the new scenario, we can have (1) cache_size_limit = 2, (2)
accumulated_cache_size_limit = 32. This means that each ID_MATCH'd object can
have maximum of two cache entries, and the maximum number of cache entries
"""
@dataclass
class CacheSizeRelevantForFrame:
    """
    We track the number of cache entries that have same id_match objects as the
    given frame.

    TODO(janimesh) - Consider adding a map from tuple_of_match_ids to count -
    https://github.com/pytorch/pytorch/pull/107496#discussion_r1304564682 - this
    could be useful for debugging as well.
    """

    # Total number of CacheEntry objects in the Dynamo linked list
    num_cache_entries: int = 0

    # Number of CacheEntry objects having same ID_MATCH'd objects as given frame.
    num_cache_entries_with_same_id_matched_objs: int = 0

    def will_compilation_exceed(self, limit: int) -> bool:
        # Checks if a compilation will exceed the given limit (thats why >=).
        return (
            self.will_compilation_exceed_accumulated_limit()
            or self.will_compilation_exceed_specific_limit(limit)
        )

    def will_compilation_exceed_accumulated_limit(self) -> bool:
        return self.num_cache_entries >= config.accumulated_cache_size_limit

    def will_compilation_exceed_specific_limit(self, limit: int) -> bool:
        return self.num_cache_entries_with_same_id_matched_objs >= limit


def _get_weakref_from_f_locals(frame: types.FrameType, local_name: str):
    """
    Get weak reference from the local variable `local_name` in the given frame.

    Returns:
        weakref object or None if the local variable cannot be weakly referenced.
    """
    obj = frame.f_locals.get(local_name, None)
    weak_id = None
    try:
        weak_id = weakref.ref(obj)
    except TypeError:
        pass  # cannot weakref bool object
    return weak_id


def _has_same_id_matched_objs(frame: types.FrameType, cache_entry) -> bool:
    """
    Checks if the ID_MATCH'd objects saved on cache_entry are same as the ones
    in frame.f_locals.

    Returns:
        True if all ID_MATCH objects match; False otherwise.
    """
    if not cache_entry:
        return False

    for (
        local_name,
        weakref_from_cache_entry,
    ) in cache_entry.check_fn.id_matched_objs.items():
        if weakref_from_cache_entry() is not None:
            weakref_from_frame = _get_weakref_from_f_locals(frame, local_name)
            if weakref_from_frame != weakref_from_cache_entry:
                return False

    # Also covers the case where no ID_MATCH objects are saved in frame.f_locals
    return True


def compute_cache_size(
    frame: types.FrameType, cache_entry
) -> CacheSizeRelevantForFrame:
    """
    Compute the cache size relevant for the given frame and cache entry.

    Args:
        frame: The frame for which cache size is computed.
        cache_entry: The cache entry to check against the frame's ID_MATCH'd objects.

    Returns:
        CacheSizeRelevantForFrame object containing relevant cache size information.
    """
    # Walk the linked list to calculate the cache size
    num_cache_entries = 0
    num_cache_entries_with_same_id_matched_objs = 0
    # 循环遍历缓存条目链表，直到遍历完所有条目
    while cache_entry:
        # 计算缓存条目的总数
        num_cache_entries += 1
        
        # 检查当前缓存条目中与 frame.f_locals 中的对象 ID_MATCH 匹配的数量，
        # 以备后续与 cache_size_limit 进行比较
        if _has_same_id_matched_objs(frame, cache_entry):
            # 如果有匹配的对象，则增加匹配计数
            num_cache_entries_with_same_id_matched_objs += 1
        
        # 移动到链表中的下一个缓存条目
        cache_entry = cache_entry.next

    # 返回一个包含统计信息的 CacheSizeRelevantForFrame 对象
    return CacheSizeRelevantForFrame(
        num_cache_entries, num_cache_entries_with_same_id_matched_objs
    )
# 判断是否需要重新编译的函数。根据缓存大小对象cache_size的信息来判断。
def is_recompilation(cache_size: CacheSizeRelevantForFrame) -> bool:
    """
    如果帧（先前由compute_cache_size解析）具有多于1个具有相同ID_MATCH对象的缓存条目，则需要重新编译。
    """
    # 注意，即使缓存中有多个条目，仍然可能不需要重新编译，例如，您可以有64个nn模块实例，每个实例都有一个ID_MATCH保护，
    # 并且每个实例在缓存中只有一个条目。在这种情况下，缓存中可能有64个条目，但由于每个id_matched_obj只有一个条目，所以不需要重新编译。
    return cache_size.will_compilation_exceed(1)


# 检查是否超过了缓存大小限制的函数。
def exceeds_cache_size_limit(cache_size: CacheSizeRelevantForFrame) -> Tuple[bool, str]:
    """
    检查是否超过了缓存大小限制。
    """
    if cache_size.will_compilation_exceed_accumulated_limit():
        return True, "accumulated_cache_size_limit"
    if cache_size.will_compilation_exceed_specific_limit(config.cache_size_limit):
        return True, "cache_size_limit"
    return False, ""
```