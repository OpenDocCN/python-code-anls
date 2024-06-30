# `D:\src\scipysrc\sympy\sympy\combinatorics\rewritingsystem.py`

```
# 导入 deque 数据结构用于缓存规则，并导入 StateMachine 类用于构建状态机
from collections import deque
from sympy.combinatorics.rewritingsystem_fsm import StateMachine

class RewritingSystem:
    '''
    A class implementing rewriting systems for `FpGroup`s.

    References
    ==========
    .. [1] Epstein, D., Holt, D. and Rees, S. (1991).
           The use of Knuth-Bendix methods to solve the word problem in automatic groups.
           Journal of Symbolic Computation, 12(4-5), pp.397-414.

    .. [2] GAP's Manual on its KBMAG package
           https://www.gap-system.org/Manuals/pkg/kbmag-1.5.3/doc/manual.pdf

    '''
    
    def __init__(self, group):
        # 初始化重写系统对象，设置群和字母表，初始值为未确定的
        self.group = group
        self.alphabet = group.generators
        self._is_confluent = None

        # these values are taken from [2]
        # 设置最大规则数和规整前的规则数
        self.maxeqns = 32767 # max rules
        self.tidyint = 100 # rules before tidying

        # _max_exceeded is True if maxeqns is exceeded
        # at any point
        # 如果超出最大规则数，设置为 True
        self._max_exceeded = False

        # Reduction automaton
        # 设置规约自动机和新规则的字典
        self.reduction_automaton = None
        self._new_rules = {}

        # dictionary of reductions
        # 规则的字典和规则缓存队列
        self.rules = {}
        self.rules_cache = deque([], 50)
        self._init_rules()


        # All the transition symbols in the automaton
        # 所有状态机中的过渡符号
        generators = list(self.alphabet)
        generators += [gen**-1 for gen in generators]
        # 创建有限状态机作为 StateMachine 对象的实例
        self.reduction_automaton = StateMachine('Reduction automaton for '+ repr(self.group), generators)
        # 构建自动机
        self.construct_automaton()

    def set_max(self, n):
        '''
        Set the maximum number of rules that can be defined

        '''
        # 设置可以定义的最大规则数
        if n > self.maxeqns:
            self._max_exceeded = False
        self.maxeqns = n
        return

    @property
    def is_confluent(self):
        '''
        Return `True` if the system is confluent

        '''
        # 如果未确定是否是可合流的，则进行检查
        if self._is_confluent is None:
            self._is_confluent = self._check_confluence()
        return self._is_confluent

    def _init_rules(self):
        # 初始化规则，添加自由群的恒等元素并添加关系
        identity = self.group.free_group.identity
        for r in self.group.relators:
            self.add_rule(r, identity)
        # 移除冗余规则
        self._remove_redundancies()
        return

    def _add_rule(self, r1, r2):
        '''
        Add the rule r1 -> r2 with no checking or further
        deductions

        '''
        # 添加规则 r1 -> r2，如果超出最大规则数，则引发 RuntimeError
        if len(self.rules) + 1 > self.maxeqns:
            self._is_confluent = self._check_confluence()
            self._max_exceeded = True
            raise RuntimeError("Too many rules were defined.")
        self.rules[r1] = r2
        # 如果存在规约自动机，将新添加的规则添加到 new_rules 字典中
        if self.reduction_automaton:
            self._new_rules[r1] = r2
    def add_rule(self, w1, w2, check=False):
        new_keys = set()  # 创建一个空集合，用于存储新增的关键字

        if w1 == w2:  # 如果 w1 等于 w2，则直接返回空集合
            return new_keys

        if w1 < w2:  # 确保 w1 大于等于 w2，如果不是，则交换它们的位置
            w1, w2 = w2, w1

        if (w1, w2) in self.rules_cache:  # 如果 (w1, w2) 已经在缓存中，则返回空集合
            return new_keys
        self.rules_cache.append((w1, w2))  # 否则将 (w1, w2) 添加到缓存中

        s1, s2 = w1, w2  # 将 w1 和 w2 分别赋值给 s1 和 s2

        # 下面的代码段用于检查 s1 是否与隐式缩减规则 {g*g**-1 -> <identity>} 和 {g**-1*g -> <identity>}
        # 存在重叠，而不安装由于处理重叠而导致的冗余规则。详见 [1]，第 3 节的细节说明。

        if len(s1) - len(s2) < 3:  # 如果 s1 的长度减去 s2 的长度小于 3
            if s1 not in self.rules:  # 如果 s1 不在规则中，则将 s1 添加到 new_keys 中
                new_keys.add(s1)
                if not check:  # 如果不是检查模式，则调用 _add_rule 方法添加规则 (s1, s2)
                    self._add_rule(s1, s2)
            if s2**-1 > s1**-1 and s2**-1 not in self.rules:  # 如果 s2 的逆大于 s1 的逆且不在规则中
                new_keys.add(s2**-1)  # 将 s2 的逆添加到 new_keys 中
                if not check:  # 如果不是检查模式，则调用 _add_rule 方法添加规则 (s2**-1, s1**-1)
                    self._add_rule(s2**-1, s1**-1)

        # 向右的重叠
        while len(s1) - len(s2) > -1:  # 当 s1 的长度减去 s2 的长度大于等于 -1 时循环
            g = s1[len(s1)-1]  # 取出 s1 的最后一个字符赋给 g
            s1 = s1.subword(0, len(s1)-1)  # 截取 s1 的子串，不包括最后一个字符
            s2 = s2*g**-1  # 更新 s2 为 s2 乘以 g 的逆
            if len(s1) - len(s2) < 0:  # 如果 s1 的长度减去 s2 的长度小于 0
                if s2 not in self.rules:  # 如果 s2 不在规则中
                    if not check:  # 如果不是检查模式，则调用 _add_rule 方法添加规则 (s2, s1)
                        self._add_rule(s2, s1)
                    new_keys.add(s2)  # 将 s2 添加到 new_keys 中
            elif len(s1) - len(s2) < 3:  # 如果 s1 的长度减去 s2 的长度小于 3
                new = self.add_rule(s1, s2, check)  # 递归调用 add_rule 处理 s1 和 s2 的规则添加
                new_keys.update(new)  # 将递归调用得到的结果更新到 new_keys 中

        # 向左的重叠
        while len(w1) - len(w2) > -1:  # 当 w1 的长度减去 w2 的长度大于等于 -1 时循环
            g = w1[0]  # 取出 w1 的第一个字符赋给 g
            w1 = w1.subword(1, len(w1))  # 截取 w1 的子串，不包括第一个字符
            w2 = g**-1*w2  # 更新 w2 为 g 的逆乘以 w2
            if len(w1) - len(w2) < 0:  # 如果 w1 的长度减去 w2 的长度小于 0
                if w2 not in self.rules:  # 如果 w2 不在规则中
                    if not check:  # 如果不是检查模式，则调用 _add_rule 方法添加规则 (w2, w1)
                        self._add_rule(w2, w1)
                    new_keys.add(w2)  # 将 w2 添加到 new_keys 中
            elif len(w1) - len(w2) < 3:  # 如果 w1 的长度减去 w2 的长度小于 3
                new = self.add_rule(w1, w2, check)  # 递归调用 add_rule 处理 w1 和 w2 的规则添加
                new_keys.update(new)  # 将递归调用得到的结果更新到 new_keys 中

        return new_keys  # 返回所有新增的关键字集合

    def _remove_redundancies(self, changes=False):
        '''
        Reduce left- and right-hand sides of reduction rules
        and remove redundant equations (i.e. those for which
        lhs == rhs). If `changes` is `True`, return a set
        containing the removed keys and a set containing the
        added keys

        减少缩减规则的左右侧并移除冗余等式（即 lhs == rhs 的情况）。
        如果 `changes` 是 `True`，则返回一个包含被移除的键和被添加的键的集合。
        '''
        removed = set()  # 创建一个空集合，用于存储被移除的键
        added = set()  # 创建一个空集合，用于存储被添加的键
        rules = self.rules.copy()  # 复制当前规则集合到 rules 中
        for r in rules:  # 遍历规则集合中的每一个规则
            v = self.reduce(r, exclude=r)  # 对规则的左侧进行缩减操作，排除自身
            w = self.reduce(rules[r])  # 对规则的右侧进行缩减操作
            if v != r:  # 如果缩减后的左侧不等于原始左侧 r
                del self.rules[r]  # 从规则集合中删除原始左侧 r
                removed.add(r)  # 将原始左侧 r 添加到被移除集合中
                if v > w:  # 如果缩减后的左侧 v 大于右侧 w
                    added.add(v)  # 将 v 添加到被添加集合中
                    self.rules[v] = w  # 更新规则集合，使 v 映射到 w
                elif v < w:  # 如果缩减后的左侧 v 小于右侧 w
                    added.add(w)  # 将 w 添加到被添加集合中
                    self.rules[w] = v  # 更新规则集合，使 w 映射到 v
            else:  # 如果缩减后的左侧等于原始左侧 r
                self.rules[v] = w  # 更新规则集合，使原始左侧 r 映射到右侧 w
        if changes:  # 如果需要返回改变的信息
            return removed, added  # 返回被移除的键集合和被添加的键集合
        return  # 否则返回空
    def _check_confluence(self):
        # 调用 `make_confluent()` 方法，传入 `check=True` 参数，执行检查并返回结果
        return self.make_confluent(check=True)

    def reduce(self, word, exclude=None):
        '''
        Apply reduction rules to `word` excluding the reduction rule
        for the lhs equal to `exclude`

        应用规约规则到 `word` 上，排除左手边等于 `exclude` 的规约规则
        '''
        # 从 self.rules 中选择除了 exclude 外的所有规则，并放入 rules 字典中
        rules = {r: self.rules[r] for r in self.rules if r != exclude}
        
        # 下面的代码本质上是 `FreeGroupElement` 类中的 `eliminate_words()` 代码，
        # 唯一的区别是第一个 "if" 语句
        again = True
        new = word
        while again:
            again = False
            for r in rules:
                prev = new
                # 如果 rules[r]**-1 大于 r**-1，则调用 eliminate_word() 方法
                # 使用 rules[r] 进行词消除，_all=True，inverse=False
                if rules[r]**-1 > r**-1:
                    new = new.eliminate_word(r, rules[r], _all=True, inverse=False)
                else:
                    # 否则调用 eliminate_word() 方法使用 rules[r] 进行词消除，_all=True
                    new = new.eliminate_word(r, rules[r], _all=True)
                # 如果 new 不等于 prev，则设置 again 为 True，继续循环
                if new != prev:
                    again = True
        return new

    def _compute_inverse_rules(self, rules):
        '''
        Compute the inverse rules for a given set of rules.
        The inverse rules are used in the automaton for word reduction.

        计算给定一组规则的逆规则。
        这些逆规则在词规约的自动机中使用。

        Arguments:
            rules (dictionary): 要计算逆规则的规则集合。

        Returns:
            Dictionary of inverse_rules. 返回逆规则的字典。

        '''
        inverse_rules = {}
        for r in rules:
            rule_key_inverse = r**-1
            rule_value_inverse = (rules[r])**-1
            # 如果 rule_value_inverse 小于 rule_key_inverse，则将它们放入 inverse_rules 中
            if (rule_value_inverse < rule_key_inverse):
                inverse_rules[rule_key_inverse] = rule_value_inverse
            else:
                inverse_rules[rule_value_inverse] = rule_key_inverse
        return inverse_rules

    def construct_automaton(self):
        '''
        Construct the automaton based on the set of reduction rules of the system.

        根据系统的规约规则构建自动机。

        Automata Design:
        自动机设计：
        The accept states of the automaton are the proper prefixes of the left hand side of the rules.
        自动机的接受状态是规则左手边的正确前缀。
        The complete left hand side of the rules are the dead states of the automaton.
        规则的完整左手边是自动机的死状态。

        '''
        # 调用 _add_to_automaton() 方法，将当前对象的 rules 添加到自动机中
        self._add_to_automaton(self.rules)
    # 如果存在新规则，则将其添加到自动机中
    if self._new_rules:
        self._add_to_automaton(self._new_rules)
        self._new_rules = {}

    # 标志位，用于控制循环是否继续进行
    flag = 1
    # 循环直到不再需要规约
    while flag:
        flag = 0
        # 初始状态设为自动机的起始状态
        current_state = self.reduction_automaton.states['start']
        # 遍历词的每个字母
        for i, s in enumerate(word.letter_form_elm):
            # 根据当前状态和当前字母找到下一个状态
            next_state_name = current_state.transitions[s]
            next_state = self.reduction_automaton.states[next_state_name]
            # 如果下一个状态是死状态（dead state）
            if next_state.state_type == 'd':
                # 获取替换规则
                subst = next_state.rh_rule
                # 根据规则替换词中的子串
                word = word.substituted_word(i - len(next_state_name) + 1, i+1, subst)
                # 设置标志位为1，表示进行了替换
                flag = 1
                # 跳出当前循环
                break
            # 更新当前状态为下一个状态，继续处理下一个字母
            current_state = next_state
    # 返回规约后的词
    return word
```