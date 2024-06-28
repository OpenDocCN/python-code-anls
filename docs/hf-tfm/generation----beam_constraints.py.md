# `.\generation\beam_constraints.py`

```
# 导入必要的库
from abc import ABC, abstractmethod
from typing import List, Optional

# 定义抽象基类 Constraint
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

    def __init__(self):
        # 调用 test 方法以测试约束条件
        self.test()

    def test(self):
        """
        Tests whether this constraint has been properly defined.
        """
        # 初始化计数器和完成标志
        counter = 0
        completed = False
        # 进入循环，直到约束条件被满足或超过最大尝试次数
        while not completed:
            # 如果计数器为1，调用 reset 方法
            if counter == 1:
                self.reset()
            # 调用 advance 方法获取进展信息
            advance = self.advance()
            # 检查进展是否符合要求
            if not self.does_advance(advance):
                # 若不符合要求则抛出异常
                raise Exception(
                    "Custom Constraint is not defined correctly. self.does_advance(self.advance()) must be true."
                )

            # 调用 update 方法获取更新后的状态
            stepped, completed, reset = self.update(advance)
            counter += 1

            # 如果超过最大尝试次数，抛出异常
            if counter > 10000:
                raise Exception("update() does not fulfill the constraint.")

        # 检查约束是否全部满足
        if self.remaining() != 0:
            raise Exception("Custom Constraint is not defined correctly.")

    @abstractmethod
    def advance(self):
        """
        When called, returns the token that would take this constraint one step closer to being fulfilled.

        Return:
            token_ids(`torch.tensor`): Must be a tensor of a list of indexable tokens, not some integer.
        """
        # 抽象方法，子类必须实现，返回一个可以使约束条件向满足状态推进的 token
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def does_advance(self, token_id: int):
        """
        Reads in a token and returns whether it creates progress.
        """
        # 抽象方法，子类必须实现，读取一个 token 并返回它是否推进了约束条件
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def update(self, token_id: int):
        """
        Given a token, updates the constraint.

        Return:
            stepped(`bool`): Whether the step was successful in moving towards completion.
            completed(`bool`): Whether the constraint is now completed.
            reset(`bool`): Whether the constraint was reset during this update.
        """
        # 抽象方法，子类必须实现，根据给定的 token 更新约束条件
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )
    def update(self, token_id: int):
        """
        Reads in a token and returns booleans that indicate the progress made by it. This function will update the
        state of this object unlike `does_advance(self, token_id: int)`.

        This isn't to test whether a certain token will advance the progress; it's to update its state as if it has
        been generated. This becomes important if token_id != desired token (refer to else statement in
        PhrasalConstraint)

        Args:
            token_id(`int`):
                The id of a newly generated token in the beam search.
        Return:
            stepped(`bool`):
                Whether this constraint has become one step closer to being fulfilled.
            completed(`bool`):
                Whether this constraint has been completely fulfilled by this token being generated.
            reset (`bool`):
                Whether this constraint has reset its progress by this token being generated.
        """
        # 抛出未实现错误，表明这是一个抽象类方法，只能被继承该类的类调用。
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def reset(self):
        """
        Resets the state of this constraint to its initialization. We would call this in cases where the fulfillment of
        a constraint is aborted by an unwanted token.
        """
        # 抛出未实现错误，表明这是一个抽象类方法，只能被继承该类的类调用。
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def remaining(self):
        """
        Returns the number of remaining steps of `advance()` in order to complete this constraint.
        """
        # 抛出未实现错误，表明这是一个抽象类方法，只能被继承该类的类调用。
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
        # 抛出未实现错误，表明这是一个抽象类方法，只能被继承该类的类调用。
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
        super(Constraint, self).__init__()  # 调用父类 Constraint 的构造函数

        if not isinstance(token_ids, list) or len(token_ids) == 0:
            raise ValueError(f"`token_ids` has to be a non-empty list, but is {token_ids}.")
        if any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids):
            raise ValueError(f"Each list in `token_ids` has to be a list of positive integers, but is {token_ids}.")

        self.token_ids = token_ids  # 将参数 token_ids 赋给实例变量 self.token_ids

        self.seqlen = len(self.token_ids)  # 记录 token_ids 的长度，即要求的序列的长度
        self.fulfilled_idx = -1  # 当前已满足的步骤的索引，初始为 -1 表示还未开始
        self.completed = False  # 标志变量，指示约束是否已经完成

    def advance(self):
        if self.completed:
            return None  # 如果约束已完成，则返回 None
        return self.token_ids[self.fulfilled_idx + 1]  # 返回下一个需要满足的 token_id

    def does_advance(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        if self.completed:
            return False  # 如果约束已完成，则返回 False

        return token_id == self.token_ids[self.fulfilled_idx + 1]  # 检查是否可以满足下一个 token_id

    def update(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        stepped = False  # 标志是否成功迈出一步
        completed = False  # 标志是否完成了所有步骤
        reset = False  # 标志是否需要重置状态

        if self.does_advance(token_id):
            self.fulfilled_idx += 1  # 成功满足下一个步骤
            stepped = True
            if self.fulfilled_idx == (self.seqlen - 1):
                completed = True  # 如果已经满足所有步骤，标记为完成
            self.completed = completed
        else:
            # 未能取得进展，需要重置状态
            reset = True
            self.reset()  # 调用 reset 方法重置状态
        return stepped, completed, reset  # 返回操作的结果信息

    def reset(self):
        self.completed = False  # 将完成标志重置为 False
        self.fulfilled_idx = 0  # 将已满足的步骤索引重置为初始状态

    def remaining(self):
        return self.seqlen - (self.fulfilled_idx + 1)  # 返回剩余待满足的步骤数量

    def copy(self, stateful=False):
        new_constraint = PhrasalConstraint(self.token_ids)  # 创建一个新的 PhrasalConstraint 对象

        if stateful:
            new_constraint.seq_len = self.seqlen  # 如果需要复制状态，将状态信息复制到新对象
            new_constraint.fulfilled_idx = self.fulfilled_idx
            new_constraint.completed = self.completed

        return new_constraint  # 返回新创建的约束对象


class DisjunctiveTrie:
    def __init__(self, nested_token_ids: List[List[int]], no_subsets=True):
        r"""
        A helper class that builds a trie with the words represented in `nested_token_ids`.
        """
        # 计算嵌套列表中每个子列表的最大长度，作为树的最大高度
        self.max_height = max([len(one) for one in nested_token_ids])

        # 初始化树的根节点为空字典
        root = {}
        # 遍历嵌套的token_ids列表
        for token_ids in nested_token_ids:
            level = root
            # 遍历每个token_id构建trie
            for tidx, token_id in enumerate(token_ids):
                if token_id not in level:
                    level[token_id] = {}  # 如果token_id不存在当前层级，创建一个空字典

                level = level[token_id]  # 移动到下一个层级

        # 如果指定了不允许子集，并且存在子集关系，则抛出异常
        if no_subsets and self.has_subsets(root, nested_token_ids):
            raise ValueError(
                "Each list in `nested_token_ids` can't be a complete subset of another list, but is"
                f" {nested_token_ids}."
            )

        self.trie = root  # 将构建好的trie作为对象的trie属性保存

    def next_tokens(self, current_seq):
        """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `current_seq`.
        """
        start = self.trie

        # 遍历当前序列中的每个token，向下移动trie
        for current_token in current_seq:
            start = start[current_token]

        # 获取当前trie节点的所有子节点，作为下一个可能的token
        next_tokens = list(start.keys())

        return next_tokens

    def reached_leaf(self, current_seq):
        # 获取当前序列的下一个可能token集合
        next_tokens = self.next_tokens(current_seq)

        # 如果下一个可能token集合为空，表示已经达到叶子节点
        return len(next_tokens) == 0

    def count_leaves(self, root):
        # 获取当前节点的所有子节点
        next_nodes = list(root.values())

        # 如果当前节点没有子节点，返回1，表示当前节点是叶子节点
        if len(next_nodes) == 0:
            return 1
        else:
            # 否则，递归计算所有子节点的叶子节点总数，并返回
            return sum([self.count_leaves(nn) for nn in next_nodes])

    def has_subsets(self, trie, nested_token_ids):
        """
        Returns whether # of leaves == # of words. Otherwise some word is a subset of another.
        """
        # 计算trie中的叶子节点数目
        leaf_count = self.count_leaves(trie)

        # 如果trie中的叶子节点数不等于嵌套token_ids的长度，则存在子集关系
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
        # 调用父类构造函数初始化
        super(Constraint, self).__init__()

        # 检查输入的 nested_token_ids 是否为非空列表
        if not isinstance(nested_token_ids, list) or len(nested_token_ids) == 0:
            raise ValueError(f"`nested_token_ids` has to be a non-empty list, but is {nested_token_ids}.")
        
        # 检查 nested_token_ids 中的每个元素是否为列表
        if any(not isinstance(token_ids, list) for token_ids in nested_token_ids):
            raise ValueError(f"`nested_token_ids` has to be a list of lists, but is {nested_token_ids}.")
        
        # 检查 nested_token_ids 中的每个元素是否为正整数
        if any(
            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
            for token_ids in nested_token_ids
        ):
            raise ValueError(
                f"Each list in `nested_token_ids` has to be a list of positive integers, but is {nested_token_ids}."
            )

        # 使用 nested_token_ids 创建一个 DisjunctiveTrie 对象
        self.trie = DisjunctiveTrie(nested_token_ids)
        self.token_ids = nested_token_ids  # 将 nested_token_ids 存储到实例变量中

        # 计算 trie 的最大高度并存储到实例变量中
        self.seqlen = self.trie.max_height
        self.current_seq = []  # 初始化当前序列为空列表
        self.completed = False  # 标记约束条件是否已完成

    def advance(self):
        # 获取当前序列可以继续的下一个 token 列表
        token_list = self.trie.next_tokens(self.current_seq)

        if len(token_list) == 0:
            return None  # 如果没有可继续的 token，则返回 None
        else:
            return token_list  # 否则返回可继续的 token 列表

    def does_advance(self, token_id: int):
        # 检查给定的 token_id 是否可以在当前序列中继续
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        next_tokens = self.trie.next_tokens(self.current_seq)

        return token_id in next_tokens  # 返回 token_id 是否在可继续的 token 列表中

    def update(self, token_id: int):
        # 更新当前序列，并返回是否有步进、是否完成、是否重置的标志
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id):
            self.current_seq.append(token_id)  # 如果可以继续，则将 token_id 添加到当前序列中
            stepped = True
        else:
            reset = True
            self.reset()  # 否则重置当前序列

        completed = self.trie.reached_leaf(self.current_seq)  # 检查当前序列是否达到叶节点
        self.completed = completed  # 更新约束条件是否已完成的状态

        return stepped, completed, reset  # 返回步进、完成和重置的标志

    def reset(self):
        # 重置当前序列和完成状态
        self.completed = False
        self.current_seq = []

    def remaining(self):
        if self.completed:
            return 0  # 如果约束条件已完成，则剩余长度为 0
        else:
            return self.seqlen - len(self.current_seq)  # 否则返回剩余的最大长度与当前序列长度的差值
    # 定义一个方法 `copy`，用于创建当前对象的副本
    def copy(self, stateful=False):
        # 创建一个新的 DisjunctiveConstraint 对象，使用当前对象的 token_ids 初始化
        new_constraint = DisjunctiveConstraint(self.token_ids)

        # 如果 stateful 参数为 True，则复制当前对象的状态到新对象中
        if stateful:
            new_constraint.seq_len = self.seqlen  # 复制当前对象的 seq_len 属性
            new_constraint.current_seq = self.current_seq  # 复制当前对象的 current_seq 属性
            new_constraint.completed = self.completed  # 复制当前对象的 completed 属性

        # 返回新创建的对象副本
        return new_constraint
class ConstraintListState:
    r"""
    A class for beam scorers to track its progress through a list of constraints.

    Args:
        constraints (`List[Constraint]`):
            A list of [`Constraint`] objects that must be fulfilled by the beam scorer.
    """

    def __init__(self, constraints: List[Constraint]):
        self.constraints = constraints

        # max # of steps required to fulfill a given constraint
        self.max_seqlen = max([c.seqlen for c in constraints])  # 计算所有约束中的最大步数
        self.n_constraints = len(constraints)  # 约束数量
        self.completed = False  # 标志位，表示是否完成

        self.init_state()  # 初始化状态

    def init_state(self):
        self.complete_constraints = []  # 已完成的约束列表
        self.inprogress_constraint = None  # 当前进行中的约束
        self.pending_constraints = [constraint.copy(stateful=False) for constraint in self.constraints]  # 待处理的约束列表，复制并标记为非状态化

    def get_bank(self):
        add = 0
        if self.inprogress_constraint:
            # extra points for having a constraint mid-fulfilled
            add += self.max_seqlen - self.inprogress_constraint.remaining()  # 如果存在进行中的约束，计算其剩余步数对应的额外分数

        return (len(self.complete_constraints) * self.max_seqlen) + add  # 返回当前已完成约束的总步数加上额外分数

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
            for constraint in self.pending_constraints:  # 遍历待处理的约束
                advance = constraint.advance()  # 获取约束的推进状态
                if isinstance(advance, int):
                    token_list.append(advance)  # 如果推进状态是整数，直接添加到 token_list
                elif isinstance(advance, list):
                    token_list.extend(advance)  # 如果推进状态是列表，扩展到 token_list
        else:
            advance = self.inprogress_constraint.advance()  # 获取当前进行中约束的推进状态
            if isinstance(advance, int):
                token_list.append(advance)  # 如果推进状态是整数，直接添加到 token_list
            elif isinstance(advance, list):
                token_list.extend(advance)  # 如果推进状态是列表，扩展到 token_list

        if len(token_list) == 0:
            return None  # 如果 token_list 为空，返回 None
        else:
            return token_list  # 否则返回 token_list
    def reset(self, token_ids: Optional[List[int]]):
        """
        重置对象状态，根据给定的token_ids重新设置约束的进度状态。
        token_ids: 到目前为止生成的令牌，用于重置通过约束的进度状态。
        """
        self.init_state()  # 调用初始化状态方法

        if token_ids is not None:
            for token in token_ids:
                # 添加一个令牌，完成或推进一个约束
                complete, stepped = self.add(token)

                # 如果所有约束已完成，则退出循环
                if self.completed:
                    break

    def copy(self, stateful=True):
        """
        创建并返回一个当前对象的副本，可以选择是否保持状态。
        stateful: 是否保持状态，默认为True。
        """
        new_state = ConstraintListState(self.constraints)  # 使用当前约束列表创建新的状态对象

        if stateful:
            # 复制已完成的约束列表中的每个约束对象
            new_state.complete_constraints = [
                constraint.copy(stateful=True) for constraint in self.complete_constraints
            ]
            # 如果存在正在进行中的约束，则复制该约束对象的状态副本
            if self.inprogress_constraint is not None:
                new_state.inprogress_constraint = self.inprogress_constraint.copy(stateful=True)
            # 复制待处理约束列表中的每个约束对象
            new_state.pending_constraints = [constraint.copy() for constraint in self.pending_constraints]

        return new_state
```