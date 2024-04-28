# `.\transformers\generation\beam_constraints.py`

```
# 导入必要的模块
from abc import ABC, abstractmethod
from typing import List, Optional

# 定义一个抽象基类 Constraint
class Constraint(ABC):
    r"""Abstract base class for all constraints that can be applied during generation.
    It must define how the constraint can be satisfied.

    All classes that inherit Constraint must follow the requirement that

    ```py
    completed = False
    while not completed:
        _, completed = constraint.update(constraint.advance())
    ```

    will always terminate (halt).
    """

    # 初始化方法
    def __init__(self):
        # 测试上述条件是否满足
        self.test()

    # 测试方法
    def test(self):
        """
        Tests whether this constraint has been properly defined.
        """
        counter = 0
        completed = False
        while not completed:
            # 如果计数器为1，则重置
            if counter == 1:
                self.reset()
            # 获取下一个步骤的 token
            advance = self.advance()
            # 检查是否进步
            if not self.does_advance(advance):
                raise Exception(
                    "Custom Constraint is not defined correctly. self.does_advance(self.advance()) must be true."
                )

            # 更新约束条件
            stepped, completed, reset = self.update(advance)
            counter += 1

            # 防止无限循环
            if counter > 10000:
                raise Exception("update() does not fulfill the constraint.")

        # 检查是否所有条件都满足
        if self.remaining() != 0:
            raise Exception("Custom Constraint is not defined correctly.")

    # 抽象方法，用于获取下一个 token
    @abstractmethod
    def advance(self):
        """
        When called, returns the token that would take this constraint one step closer to being fulfilled.

        Return:
            token_ids(`torch.tensor`): Must be a tensor of a list of indexable tokens, not some integer.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    # 抽象方法，用于检查是否进步
    @abstractmethod
    def does_advance(self, token_id: int):
        """
        Reads in a token and returns whether it creates progress.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    # 抽象���法
    def update(self, token_id: int):
        """
        Reads in a token and returns booleans that indicate the progress made by it. This function will update the
        state of this object unlikes `does_advance(self, token_id: int)`.

        This isn't to test whether a certain token will advance the progress; it's to update its state as if it has
        been generated. This becomes important if token_id != desired token (refer to else statement in
        PhrasalConstraint)

        Args:
            token_id(`int`):
                The id of a newly generated token in the beam search.
        Return:
            stepped(`bool`):
                Whether this constraint has become one step closer to being fulfuilled.
            completed(`bool`):
                Whether this constraint has been completely fulfilled by this token being generated.
            reset (`bool`):
                Whether this constraint has reset its progress by this token being generated.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def reset(self):
        """
        Resets the state of this constraint to its initialization. We would call this in cases where the fulfillment of
        a constraint is abrupted by an unwanted token.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def remaining(self):
        """
        Returns the number of remaining steps of `advance()` in order to complete this constraint.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def copy(self, stateful=False):
        """
        Creates a new instance of this constraint.

        Args:
            stateful(`bool`): Whether to not only copy the constraint for new instance, but also its state.

        Return:
            constraint(`Constraint`): The same constraint as the one being called from.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )
class PhrasalConstraint(Constraint):
    r"""
    [`Constraint`] enforcing that an ordered sequence of tokens is included in the output.

    Args:
        token_ids (`List[int]`):
            The id of the token that must be generated by the output.
    """

    def __init__(self, token_ids: List[int]):
        super(Constraint, self).__init__()  # 调用父类的构造函数

        # 检查传入的参数是否符合要求
        if not isinstance(token_ids, list) or len(token_ids) == 0:
            raise ValueError(f"`token_ids` has to be a non-empty list, but is {token_ids}.")
        if any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids):
            raise ValueError(f"Each list in `token_ids` has to be a list of positive integers, but is {token_ids}.")

        # 初始化约束对象的属性
        self.token_ids = token_ids
        self.seqlen = len(self.token_ids)
        self.fulfilled_idx = -1  # 当前已满足步骤的索引，初始为-1表示未满足任何步骤
        self.completed = False  # 标记约束是否已完成

    def advance(self):
        # 如果约束已完成，则返回 None
        if self.completed:
            return None
        # 返回下一个要满足的 token id
        return self.token_ids[self.fulfilled_idx + 1]

    def does_advance(self, token_id: int):
        # 检查参数是否符合要求
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        # 如果约束已完成，则不再前进
        if self.completed:
            return False

        # 检查传入的 token_id 是否符合下一个要满足的 token id
        return token_id == self.token_ids[self.fulfilled_idx + 1]

    def update(self, token_id: int):
        # 检查参数是否符合要求
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        # 初始化变量
        stepped = False
        completed = False
        reset = False

        # 如果可以前进
        if self.does_advance(token_id):
            self.fulfilled_idx += 1  # 更新已满足步骤的索引
            stepped = True
            if self.fulfilled_idx == (self.seqlen - 1):  # 如果已满足所有步骤
                completed = True
            self.completed = completed  # 更新约束的完成状态
        else:
            # 无法前进，重置约束
            reset = True
            self.reset()
        return stepped, completed, reset

    def reset(self):
        # 重置约束状态
        self.completed = False
        self.fulfilled_idx = 0

    def remaining(self):
        # 返回剩余未满足步骤的数量
        return self.seqlen - (self.fulfilled_idx + 1)

    def copy(self, stateful=False):
        # 复制约束对象
        new_constraint = PhrasalConstraint(self.token_ids)

        # 如果要求状态保持一致，则复制相关状态
        if stateful:
            new_constraint.seq_len = self.seqlen
            new_constraint.fulfilled_idx = self.fulfilled_idx
            new_constraint.completed = self.completed

        return new_constraint


class DisjunctiveTrie:
    # 初始化函数，构建一个 trie 树，根据嵌套的 token_ids 表示的单词
    class Trie:
        def __init__(self, nested_token_ids: List[List[int]], no_subsets=True):
            # 计算嵌套 token_ids 中最长的列表长度
            self.max_height = max([len(one) for one in nested_token_ids])
    
            # 初始化根节点
            root = {}
            # 遍历每个 token_ids
            for token_ids in nested_token_ids:
                level = root
                # 遍历每个 token_id
                for tidx, token_id in enumerate(token_ids):
                    # 如果当前 token_id 不在当前层级中，则添加
                    if token_id not in level:
                        level[token_id] = {}
    
                    level = level[token_id]
    
            # 如果不允许子集存在，并且存在子集，则抛出异常
            if no_subsets and self.has_subsets(root, nested_token_ids):
                raise ValueError(
                    "Each list in `nested_token_ids` can't be a complete subset of another list, but is"
                    f" {nested_token_ids}."
                )
    
            # 将构建好的 trie 树赋值给实例变量
            self.trie = root
    
        # 返回给定当前序列的下一个可能的 token
        def next_tokens(self, current_seq):
            start = self.trie
    
            for current_token in current_seq:
                start = start[current_token]
    
            next_tokens = list(start.keys())
    
            return next_tokens
    
        # 判断当前序列是否到达叶子节点
        def reached_leaf(self, current_seq):
            next_tokens = self.next_tokens(current_seq)
    
            return len(next_tokens) == 0
    
        # 计算给定根节点下的叶子节点数量
        def count_leaves(self, root):
            next_nodes = list(root.values())
            if len(next_nodes) == 0:
                return 1
            else:
                return sum([self.count_leaves(nn) for nn in next_nodes])
    
        # 判断 trie 树中是否存在子集
        def has_subsets(self, trie, nested_token_ids):
            leaf_count = self.count_leaves(trie)
            return len(nested_token_ids) != leaf_count
class DisjunctiveConstraint(Constraint):
    r"""
    A special [`Constraint`] that is fulfilled by fulfilling just one of several constraints.

    Args:
        nested_token_ids (`List[List[int]]`):
            A list of words, where each word is a list of ids. This constraint is fulfilled by generating just one from
            the list of words.
    """

    def __init__(self, nested_token_ids: List[List[int]]):
        # 调用父类的构造函数初始化对象
        super(Constraint, self).__init__()

        # 检查传入的参数是否合法
        if not isinstance(nested_token_ids, list) or len(nested_token_ids) == 0:
            raise ValueError(f"`nested_token_ids` has to be a non-empty list, but is {nested_token_ids}.")
        if any(not isinstance(token_ids, list) for token_ids in nested_token_ids):
            raise ValueError(f"`nested_token_ids` has to be a list of lists, but is {nested_token_ids}.")
        if any(
            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
            for token_ids in nested_token_ids
        ):
            raise ValueError(
                f"Each list in `nested_token_ids` has to be a list of positive integers, but is {nested_token_ids}."
            )

        # 使用传入的嵌套 token ids 初始化 DisjunctiveTrie 对象
        self.trie = DisjunctiveTrie(nested_token_ids)
        self.token_ids = nested_token_ids

        # 设置当前序列长度和当前序列为空列表
        self.seqlen = self.trie.max_height
        self.current_seq = []
        # 标记当前约束是否已完成
        self.completed = False

    # 返回下一个可选的 token 列表
    def advance(self):
        token_list = self.trie.next_tokens(self.current_seq)

        if len(token_list) == 0:
            return None
        else:
            return token_list

    # 检查给定的 token 是否可以推进当前序列
    def does_advance(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        next_tokens = self.trie.next_tokens(self.current_seq)

        return token_id in next_tokens

    # 更新当前序列，标记是否推进、完成、或重置
    def update(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id):
            self.current_seq.append(token_id)
            stepped = True
        else:
            reset = True
            self.reset()

        completed = self.trie.reached_leaf(self.current_seq)
        self.completed = completed

        return stepped, completed, reset

    # 重置当前序列
    def reset(self):
        self.completed = False
        self.current_seq = []

    # 返回当前序列剩余的 token 数量
    def remaining(self):
        if self.completed:
            # 如果已完成，剩余 token 数量为 0
            return 0
        else:
            # 否则，返回当前序列还需添加的 token 数量
            return self.seqlen - len(self.current_seq)
    # 定义一个复制方法，用于创建当前约束对象的副本
    def copy(self, stateful=False):
        # 创建一个新的不相交约束对象，使用当前约束对象的标识符列表作为参数
        new_constraint = DisjunctiveConstraint(self.token_ids)

        # 如果指定了stateful参数为True，则进行状态的复制
        if stateful:
            # 将新约束对象的序列长度设置为当前约束对象的序列长度
            new_constraint.seq_len = self.seqlen
            # 将新约束对象的当前序列设置为当前约束对象的当前序列
            new_constraint.current_seq = self.current_seq
            # 将新约束对象的完成状态设置为当前约束对象的完成状态
            new_constraint.completed = self.completed

        # 返回新创建的约束对象
        return new_constraint
class ConstraintListState:
    r"""
    A class for beam scorers to track its progress through a list of constraints.

    Args:
        constraints (`List[Constraint]`):
            A list of [`Constraint`] objects that must be fulfilled by the beam scorer.
    """

    def __init__(self, constraints: List[Constraint]):
        # 初始化 ConstraintListState 类的实例
        self.constraints = constraints

        # 计算约束条件中所需的最大步数
        self.max_seqlen = max([c.seqlen for c in constraints])
        # 记录约束条件的数量
        self.n_constraints = len(constraints)
        # 标记约束条件是否已完成
        self.completed = False

        # 初始化状态
        self.init_state()

    def init_state(self):
        # 初始化已完成的约束条件列表
        self.complete_constraints = []
        # 初始化正在进行的约束条件
        self.inprogress_constraint = None
        # 初始化待处理的约束条件列表，每个约束条件都复制一个状态为非 stateful 的副本
        self.pending_constraints = [constraint.copy(stateful=False) for constraint in self.constraints]

    def get_bank(self):
        # 初始化额外得分
        add = 0
        if self.inprogress_constraint:
            # 如果存在正在进行的约束条件，则计算额外得分
            add += self.max_seqlen - self.inprogress_constraint.remaining()

        # 返回银行分数，即已完成约束条件数量乘以最大步数再加上额外得分
        return (len(self.complete_constraints) * self.max_seqlen) + add

    def advance(self):
        """The list of tokens to generate such that we can make progress.
        By "list" we don't mean the list of token that will fully fulfill a constraint.

        Given constraints `c_i = {t_ij | j == # of tokens}`, If we're not in the middle of progressing through a
        specific constraint `c_i`, we return:

        `[t_k1 for k in indices of unfulfilled constraints]`

        If we are in the middle of a constraint, then we return:
            `[t_ij]`, where `i` is the index of the inprogress constraint, `j` is the next step for the constraint.

        Though we don't care which constraint is fulfilled first, if we are in the progress of fulfilling a constraint,
        that's the only one we'll return.
        """
        token_list = []
        if self.inprogress_constraint is None:
            # 如果没有正在进行的约束条件，则遍历待处理的约束条件列表
            for constraint in self.pending_constraints:  # "pending" == "unfulfilled yet"
                # 获取约束条件的下一步生成的 token
                advance = constraint.advance()
                if isinstance(advance, int):
                    token_list.append(advance)
                elif isinstance(advance, list):
                    token_list.extend(advance)
        else:
            # 如果存在正在进行的约束条件，则获取其下一步生成的 token
            advance = self.inprogress_constraint.advance()
            if isinstance(advance, int):
                token_list.append(advance)
            elif isinstance(advance, list):
                token_list.extend(advance)

        # 如果没有生成 token，则返回 None，否则返回生成的 token 列表
        if len(token_list) == 0:
            return None
        else:
            return token_list
    # 重置约束列表的状态，用于重新开始遍历约束条件
    def reset(self, token_ids: Optional[List[int]]):
        """
        token_ids: 已生成的标记，用于重置通过约束条件的进度状态。
        """
        # 初始化约束列表的状态
        self.init_state()

        # 如果存在已生成的标记
        if token_ids is not None:
            # 遍历已生成的标记
            for token in token_ids:
                # 完成或前进一个约束条件
                complete, stepped = self.add(token)

                # 如果所有约束条件都已满足，则退出循环
                if self.completed:
                    break

    # 复制约束列表的状态
    def copy(self, stateful=True):
        # 创建一个新的约束列表状态对象，使用当前约束条件的初始化状态
        new_state = ConstraintListState(self.constraints)  # 实际上我们从未使用过 self.constraints 对象。因此它在初始化状态。

        # 如果需要保留状态信息
        if stateful:
            # 复制已完成的约束条件列表
            new_state.complete_constraints = [
                constraint.copy(stateful=True) for constraint in self.complete_constraints
            ]
            # 复制进行中的约束条件
            if self.inprogress_constraint is not None:
                new_state.inprogress_constraint = self.inprogress_constraint.copy(stateful=True)
            # 复制待处理的约束条件列表
            new_state.pending_constraints = [constraint.copy() for constraint in self.pending_constraints]

        # 返回新的约束列表状态对象
        return new_state
```