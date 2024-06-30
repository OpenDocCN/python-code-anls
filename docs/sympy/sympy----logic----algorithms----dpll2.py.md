# `D:\src\scipysrc\sympy\sympy\logic\algorithms\dpll2.py`

```
"""Implementation of DPLL algorithm

Features:
  - Clause learning
  - Watch literal scheme
  - VSIDS heuristic

References:
  - https://en.wikipedia.org/wiki/DPLL_algorithm
"""

# 从 collections 模块中导入 defaultdict 类
from collections import defaultdict
# 从 heapq 模块中导入 heappush 和 heappop 函数
from heapq import heappush, heappop

# 从 sympy.core.sorting 模块中导入 ordered 函数
from sympy.core.sorting import ordered
# 从 sympy.assumptions.cnf 模块中导入 EncodedCNF 类
from sympy.assumptions.cnf import EncodedCNF

# 从 sympy.logic.algorithms.lra_theory 模块中导入 LRASolver 类
from sympy.logic.algorithms.lra_theory import LRASolver


def dpll_satisfiable(expr, all_models=False, use_lra_theory=False):
    """
    Check satisfiability of a propositional sentence.
    It returns a model rather than True when it succeeds.
    Returns a generator of all models if all_models is True.

    Examples
    ========

    >>> from sympy.abc import A, B
    >>> from sympy.logic.algorithms.dpll2 import dpll_satisfiable
    >>> dpll_satisfiable(A & ~B)
    {A: True, B: False}
    >>> dpll_satisfiable(A & ~A)
    False

    """
    # 如果 expr 不是 EncodedCNF 的实例，则创建 EncodedCNF 对象并添加表达式
    if not isinstance(expr, EncodedCNF):
        exprs = EncodedCNF()
        exprs.add_prop(expr)
        expr = exprs

    # 如果 CNF 数据中包含 {0}，表示不可满足（UNSAT）
    # 如果 all_models 为 True，则返回一个生成器，生成 False
    if {0} in expr.data:
        if all_models:
            return (f for f in [False])
        return False

    # 如果使用 LRA 理论求解，则使用 LRASolver 解析 CNF 表达式
    if use_lra_theory:
        lra, immediate_conflicts = LRASolver.from_encoded_cnf(expr)
    else:
        lra = None
        immediate_conflicts = []
    
    # 创建 SATSolver 对象，并传入 CNF 数据、变量、空集合、符号及 LRA 理论（如果有）
    solver = SATSolver(expr.data + immediate_conflicts, expr.variables, set(), expr.symbols, lra_theory=lra)
    
    # 调用 SATSolver 的 _find_model 方法获取模型
    models = solver._find_model()

    # 如果 all_models 为 True，则返回所有模型的生成器
    if all_models:
        return _all_models(models)

    # 尝试获取下一个模型，如果 StopIteration 则返回 False
    try:
        return next(models)
    except StopIteration:
        return False

    # Uncomment to confirm the solution is valid (hitting set for the clauses)
    #else:
        #for cls in clauses_int_repr:
            #assert solver.var_settings.intersection(cls)


def _all_models(models):
    # 初始化 satisfiable 为 False
    satisfiable = False
    try:
        # 生成器循环，生成所有模型
        while True:
            yield next(models)
            satisfiable = True
    except StopIteration:
        # 如果没有找到任何模型，则生成 False
        if not satisfiable:
            yield False


class SATSolver:
    """
    Class for representing a SAT solver capable of
     finding a model to a boolean theory in conjunctive
     normal form.
    """
    def __init__(self, clauses, variables, var_settings, symbols=None,
                heuristic='vsids', clause_learning='none', INTERVAL=500,
                lra_theory=None):
        # 初始化函数，设置初始参数和数据结构
        self.var_settings = var_settings
        self.heuristic = heuristic
        self.is_unsatisfied = False
        self._unit_prop_queue = []  # 存储待传播的单元子句
        self.update_functions = []  # 存储更新函数的列表
        self.INTERVAL = INTERVAL  # 间隔参数设置

        if symbols is None:
            self.symbols = list(ordered(variables))  # 若符号未指定，则按变量顺序列出符号
        else:
            self.symbols = symbols  # 使用给定的符号列表

        self._initialize_variables(variables)  # 初始化变量相关数据结构
        self._initialize_clauses(clauses)  # 初始化子句相关数据结构

        if 'vsids' == heuristic:
            self._vsids_init()  # VSIDS启发式初始化
            self.heur_calculate = self._vsids_calculate
            self.heur_lit_assigned = self._vsids_lit_assigned
            self.heur_lit_unset = self._vsids_lit_unset
            self.heur_clause_added = self._vsids_clause_added

            # 注意: 如果启用子句学习，请取消下面一行的注释
            # self.update_functions.append(self._vsids_decay)

        else:
            raise NotImplementedError  # 未实现其他启发式方法的处理

        if 'simple' == clause_learning:
            self.add_learned_clause = self._simple_add_learned_clause
            self.compute_conflict = self._simple_compute_conflict
            self.update_functions.append(self._simple_clean_clauses)
        elif 'none' == clause_learning:
            self.add_learned_clause = lambda x: None  # 不添加学习到的子句
            self.compute_conflict = lambda: None  # 不计算冲突
        else:
            raise NotImplementedError  # 未实现其他子句学习策略的处理

        # 创建基础级别
        self.levels = [Level(0)]
        self._current_level.varsettings = var_settings  # 设置当前级别的变量设置

        # 统计信息
        self.num_decisions = 0  # 决策次数
        self.num_learned_clauses = 0  # 学习到的子句数量
        self.original_num_clauses = len(self.clauses)  # 初始子句数量

        self.lra = lra_theory  # 设置LRA理论

    def _initialize_variables(self, variables):
        """设置所需的变量数据结构。"""
        self.sentinels = defaultdict(set)  # 初始化哨兵数据结构
        self.occurrence_count = defaultdict(int)  # 初始化文档频率计数
        self.variable_set = [False] * (len(variables) + 1)  # 初始化变量集合

    def _initialize_clauses(self, clauses):
        """设置所需的子句数据结构。

        对于每个子句，进行以下更改：
        - 单元子句立即加入传播队列。
        - 非单元子句设置第一个和最后一个文字作为哨兵。
        - 计算每个文字出现在子句中的次数。
        """
        self.clauses = [list(clause) for clause in clauses]

        for i, clause in enumerate(self.clauses):

            # 处理单元子句
            if 1 == len(clause):
                self._unit_prop_queue.append(clause[0])
                continue

            self.sentinels[clause[0]].add(i)
            self.sentinels[clause[-1]].add(i)

            for lit in clause:
                self.occurrence_count[lit] += 1

    ########################
    #    Helper Methods    #
    ########################
    @property
    def _current_level(self):
        """
        The current decision level data structure

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{1}, {2}], {1, 2}, set())
        >>> next(l._find_model())
        {1: True, 2: True}
        >>> l._current_level.decision
        0
        >>> l._current_level.flipped
        False
        >>> l._current_level.var_settings
        {1, 2}

        """
        # 返回当前决策层的数据结构，即最后一个层级的数据
        return self.levels[-1]

    def _clause_sat(self, cls):
        """
        Check if a clause is satisfied by the current variable setting.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{1}, {-1}], {1}, set())
        >>> try:
        ...     next(l._find_model())
        ... except StopIteration:
        ...     pass
        >>> l._clause_sat(0)
        False
        >>> l._clause_sat(1)
        True

        """
        # 遍历指定子句中的所有文字，如果任一文字在变量设置中，则返回True
        for lit in self.clauses[cls]:
            if lit in self.var_settings:
                return True
        return False

    def _is_sentinel(self, lit, cls):
        """
        Check if a literal is a sentinel of a given clause.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> next(l._find_model())
        {1: True, 2: False, 3: False}
        >>> l._is_sentinel(2, 3)
        True
        >>> l._is_sentinel(-3, 1)
        False

        """
        # 检查特定文字是否是给定子句的哨兵文字
        return cls in self.sentinels[lit]
    def _assign_literal(self, lit):
        """Make a literal assignment.

        The literal assignment must be recorded as part of the current
        decision level. Additionally, if the literal is marked as a
        sentinel of any clause, then a new sentinel must be chosen. If
        this is not possible, then unit propagation is triggered and
        another literal is added to the queue to be set in the future.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> next(l._find_model())
        {1: True, 2: False, 3: False}
        >>> l.var_settings
        {-3, -2, 1}

        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> l._assign_literal(-1)
        >>> try:
        ...     next(l._find_model())
        ... except StopIteration:
        ...     pass
        >>> l.var_settings
        {-1}

        """
        # Add the literal to the set of variable settings at the global level
        self.var_settings.add(lit)
        # Add the literal to the set of variable settings at the current decision level
        self._current_level.var_settings.add(lit)
        # Mark the variable corresponding to the literal as assigned (True)
        self.variable_set[abs(lit)] = True
        # Perform heuristic operations related to the assigned literal
        self.heur_lit_assigned(lit)

        # Retrieve the list of clauses where the literal is a sentinel
        sentinel_list = list(self.sentinels[-lit])

        # Iterate over each clause where the literal is a sentinel
        for cls in sentinel_list:
            # Check if the clause is not satisfied under the current variable settings
            if not self._clause_sat(cls):
                other_sentinel = None
                # Try to find another literal in the clause that can be a new sentinel
                for newlit in self.clauses[cls]:
                    if newlit != -lit:
                        # If newlit can be a sentinel and is unassigned
                        if self._is_sentinel(newlit, cls):
                            other_sentinel = newlit
                        elif not self.variable_set[abs(newlit)]:
                            # Move the sentinel status from -lit to newlit
                            self.sentinels[-lit].remove(cls)
                            self.sentinels[newlit].add(cls)
                            other_sentinel = None
                            break

                # If no new sentinel could be chosen, trigger unit propagation
                if other_sentinel:
                    self._unit_prop_queue.append(other_sentinel)

    def _undo(self):
        """
        Undo the changes of the most recent decision level.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> next(l._find_model())
        {1: True, 2: False, 3: False}
        >>> level = l._current_level
        >>> level.decision, level.var_settings, level.flipped
        (-3, {-3, -2}, False)
        >>> l._undo()
        >>> level = l._current_level
        >>> level.decision, level.var_settings, level.flipped
        (0, {1}, False)

        """
        # Undo the variable settings for the current decision level
        for lit in self._current_level.var_settings:
            self.var_settings.remove(lit)
            self.heur_lit_unset(lit)
            self.variable_set[abs(lit)] = False

        # Remove the current decision level from the stack
        self.levels.pop()
    #########################
    """
    Propagation methods should attempt to soundly simplify the boolean
      theory, and return True if any simplification occurred and False
      otherwise.
    """
    def _simplify(self):
        """Iterate over the various forms of propagation to simplify the theory.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> l.variable_set
        [False, False, False, False]
        >>> l.sentinels
        {-3: {0, 2}, -2: {3, 4}, 2: {0, 3}, 3: {2, 4}}

        >>> l._simplify()

        >>> l.variable_set
        [False, True, False, False]
        >>> l.sentinels
        {-3: {0, 2}, -2: {3, 4}, -1: set(), 2: {0, 3},
        ...3: {2, 4}}

        """
        changed = True  # 标志位，表示是否进行了简化操作
        while changed:  # 循环直到没有简化操作为止
            changed = False  # 每次循环开始时先将标志位置为False
            changed |= self._unit_prop()  # 进行单元传播操作，并检查是否发生了变化
            changed |= self._pure_literal()  # 查找并处理纯文字，并检查是否发生了变化

    def _unit_prop(self):
        """Perform unit propagation on the current theory."""
        result = len(self._unit_prop_queue) > 0  # 检查是否有单元传播队列中还有元素
        while self._unit_prop_queue:  # 当单元传播队列不为空时执行循环
            next_lit = self._unit_prop_queue.pop()  # 弹出下一个单元传播文字
            if -next_lit in self.var_settings:  # 如果其否定文字已经被赋值为True
                self.is_unsatisfied = True  # 设置不满足标志为True
                self._unit_prop_queue = []  # 清空单元传播队列
                return False  # 返回没有进行单元传播的标志
            else:
                self._assign_literal(next_lit)  # 将单元传播文字赋值为True

        return result  # 返回是否进行了单元传播的标志

    def _pure_literal(self):
        """Look for pure literals and assign them when found."""
        return False  # 纯文字处理暂时未实现，总是返回False

    #########################
    #      Heuristics       #
    #########################
    def _vsids_init(self):
        """Initialize the data structures needed for the VSIDS heuristic."""
        self.lit_heap = []  # 初始化文字堆为空列表
        self.lit_scores = {}  # 初始化文字得分字典为空字典

        for var in range(1, len(self.variable_set)):  # 遍历变量集合的索引
            self.lit_scores[var] = float(-self.occurrence_count[var])  # 初始化正文字的得分
            self.lit_scores[-var] = float(-self.occurrence_count[-var])  # 初始化负文字的得分
            heappush(self.lit_heap, (self.lit_scores[var], var))  # 将正文字的得分和索引加入堆中
            heappush(self.lit_heap, (self.lit_scores[-var], -var))  # 将负文字的得分和索引加入堆中

    def _vsids_decay(self):
        """Decay the VSIDS scores for every literal.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())

        >>> l.lit_scores
        {-3: -2.0, -2: -2.0, -1: 0.0, 1: 0.0, 2: -2.0, 3: -2.0}

        >>> l._vsids_decay()

        >>> l.lit_scores
        {-3: -1.0, -2: -1.0, -1: 0.0, 1: 0.0, 2: -1.0, 3: -1.0}

        """
        # We divide every literal score by 2 for a decay factor
        #  Note: This doesn't change the heap property
        for lit in self.lit_scores.keys():  # 遍历所有文字的得分
            self.lit_scores[lit] /= 2.0  # 将文字的得分减半，进行衰减
    def _vsids_calculate(self):
        """
        VSIDS Heuristic Calculation

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())

        >>> l.lit_heap
        [(-2.0, -3), (-2.0, 2), (-2.0, -2), (0.0, 1), (-2.0, 3), (0.0, -1)]

        >>> l._vsids_calculate()
        -3

        >>> l.lit_heap
        [(-2.0, -2), (-2.0, 2), (0.0, -1), (0.0, 1), (-2.0, 3)]

        """
        if len(self.lit_heap) == 0:
            return 0

        # Clean out the front of the heap as long the variables are set
        while self.variable_set[abs(self.lit_heap[0][1])]:
            heappop(self.lit_heap)
            if len(self.lit_heap) == 0:
                return 0

        return heappop(self.lit_heap)[1]

    def _vsids_lit_assigned(self, lit):
        """Handle the assignment of a literal for the VSIDS heuristic."""
        pass

    def _vsids_lit_unset(self, lit):
        """Handle the unsetting of a literal for the VSIDS heuristic.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> l.lit_heap
        [(-2.0, -3), (-2.0, 2), (-2.0, -2), (0.0, 1), (-2.0, 3), (0.0, -1)]

        >>> l._vsids_lit_unset(2)

        >>> l.lit_heap
        [(-2.0, -3), (-2.0, -2), (-2.0, -2), (-2.0, 2), (-2.0, 3), (0.0, -1),
        ...(-2.0, 2), (0.0, 1)]

        """
        var = abs(lit)
        heappush(self.lit_heap, (self.lit_scores[var], var))
        heappush(self.lit_heap, (self.lit_scores[-var], -var))

    def _vsids_clause_added(self, cls):
        """Handle the addition of a new clause for the VSIDS heuristic.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())

        >>> l.num_learned_clauses
        0
        >>> l.lit_scores
        {-3: -2.0, -2: -2.0, -1: 0.0, 1: 0.0, 2: -2.0, 3: -2.0}

        >>> l._vsids_clause_added({2, -3})

        >>> l.num_learned_clauses
        1
        >>> l.lit_scores
        {-3: -1.0, -2: -2.0, -1: 0.0, 1: 0.0, 2: -1.0, 3: -2.0}

        """
        self.num_learned_clauses += 1
        for lit in cls:
            self.lit_scores[lit] += 1

    ########################
    #   Clause Learning    #
    ########################


注释：
    def _simple_add_learned_clause(self, cls):
        """Add a new clause to the theory.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())

        >>> l.num_learned_clauses
        0
        >>> l.clauses
        [[2, -3], [1], [3, -3], [2, -2], [3, -2]]
        >>> l.sentinels
        {-3: {0, 2}, -2: {3, 4}, 2: {0, 3}, 3: {2, 4}}

        >>> l._simple_add_learned_clause([3])

        >>> l.clauses
        [[2, -3], [1], [3, -3], [2, -2], [3, -2], [3]]
        >>> l.sentinels
        {-3: {0, 2}, -2: {3, 4}, 2: {0, 3}, 3: {2, 4, 5}}

        """
        cls_num = len(self.clauses)  # 获取当前子句列表的长度，即已有子句数量
        self.clauses.append(cls)  # 将新学习到的子句添加到子句列表末尾

        for lit in cls:
            self.occurrence_count[lit] += 1  # 更新文字的出现次数计数器

        self.sentinels[cls[0]].add(cls_num)  # 更新起始文字的哨兵集合
        self.sentinels[cls[-1]].add(cls_num)  # 更新结束文字的哨兵集合

        self.heur_clause_added(cls)  # 调用启发式函数处理新添加的子句

    def _simple_compute_conflict(self):
        """ Build a clause representing the fact that at least one decision made
        so far is wrong.

        Examples
        ========

        >>> from sympy.logic.algorithms.dpll2 import SATSolver
        >>> l = SATSolver([{2, -3}, {1}, {3, -3}, {2, -2},
        ... {3, -2}], {1, 2, 3}, set())
        >>> next(l._find_model())
        {1: True, 2: False, 3: False}
        >>> l._simple_compute_conflict()
        [3]

        """
        return [-(level.decision) for level in self.levels[1:]]  # 构建表示至少一个错误决策的子句

    def _simple_clean_clauses(self):
        """Clean up learned clauses."""
        pass  # 清理学习到的子句，当前未实现具体功能
class Level:
    """
    Represents a single level in the DPLL algorithm, and contains
    enough information for a sound backtracking procedure.
    """

    # 初始化方法，创建一个新的Level对象
    def __init__(self, decision, flipped=False):
        # 记录该层的决策
        self.decision = decision
        # 记录当前层的变量设置，使用集合来存储，确保每个变量只记录一次
        self.var_settings = set()
        # 记录该层是否被反转（flipped），默认为False
        self.flipped = flipped
```