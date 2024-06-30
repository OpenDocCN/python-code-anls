# `D:\src\scipysrc\sympy\sympy\core\facts.py`

```
# 用于对 SymPy 的基于规则的推理系统进行编程，此部分主要是规则编译和表格准备，以及运行时推理的实现

# 导入必要的库和模块
from collections import defaultdict  # 导入 defaultdict 类来创建默认字典
from typing import Iterator  # 引入 Iterator 类型提示

from .logic import Logic, And, Or, Not  # 导入逻辑相关的类和函数，包括 And, Or, Not

def _base_fact(atom):
    """返回原子的文字事实。

    实际上，这只是去除原子周围的 Not。
    """
    if isinstance(atom, Not):
        return atom.arg  # 返回 Not 原子的参数
    else:
        return atom  # 返回原子本身

def _as_pair(atom):
    """将原子转换为对（literal, bool）。

    如果是 Not 原子，则布尔值为 False。
    """
    if isinstance(atom, Not):
        return (atom.arg, False)  # 返回 Not 原子参数和 False
    else:
        return (atom, True)  # 返回原子和 True

# XXX 这段代码准备 alpha 网络的前向链接规则
def transitive_closure(implications):
    """
    计算一组蕴含式的传递闭包。

    使用 Warshall 算法，参见 http://www.cs.hope.edu/~cusack/Notes/Notes/DiscreteMath/Warshall.pdf。
    """
    full_implications = set(implications)  # 使用传入的蕴含式创建一个集合
    literals = set().union(*map(set, full_implications))  # 获取所有文字的集合

    # 使用 Warshall 算法计算传递闭包
    for k in literals:
        for i in literals:
            if (i, k) in full_implications:
                for j in literals:
                    if (k, j) in full_implications:
                        full_implications.add((i, j))

    return full_implications  # 返回计算后的完整蕴含式集合

def deduce_alpha_implications(implications):
    """deduce all implications

       Description by example
       ----------------------

       given set of logic rules:

         a -> b
         b -> c

       we deduce all possible rules:

         a -> b, c
         b -> c


       implications: [] of (a,b)
       return:       {} of a -> set([b, c, ...])
    """
    # 将每个 (i, j) 规则转换为 (Not(j), Not(i)) 规则，并添加到 implications 列表中
    implications = implications + [(Not(j), Not(i)) for (i, j) in implications]
    
    # 创建一个默认字典，用于存储每个逻辑命题的推导结果集合
    res = defaultdict(set)
    
    # 计算所有传递闭包后的完整推导结果
    full_implications = transitive_closure(implications)
    
    # 遍历完整的推导结果集合
    for a, b in full_implications:
        if a == b:
            continue    # 跳过 a->a 的循环输入

        # 将推导结果加入到结果字典中
        res[a].add(b)

    # 清理平凡的推导规则并检查一致性
    for a, impl in res.items():
        impl.discard(a)  # 移除自身推导
        na = Not(a)
        if na in impl:
            # 如果发现推导规则不一致，抛出 ValueError 异常
            raise ValueError(
                'implications are inconsistent: %s -> %s %s' % (a, na, impl))

    # 返回推导结果字典
    return res
# 将额外的 beta 规则（并且条件）应用于已构建的 alpha 蕴含表中
def apply_beta_to_alpha_route(alpha_implications, beta_rules):
    """apply additional beta-rules (And conditions) to already-built
    alpha implication tables

       TODO: write about

       - static extension of alpha-chains
       - attaching refs to beta-nodes to alpha chains


       e.g.

       alpha_implications:

       a  ->  [b, !c, d]
       b  ->  [d]
       ...


       beta_rules:

       &(b,d) -> e


       then we'll extend a's rule to the following

       a  ->  [b, !c, d, e]
    """
    # 初始化 alpha 蕴含表的副本
    x_impl = {}
    for x in alpha_implications.keys():
        x_impl[x] = (set(alpha_implications[x]), [])  # 每个 alpha 蕴含项关联一个空的 beta 引用列表

    # 遍历 beta 规则
    for bcond, bimpl in beta_rules:
        for bk in bcond.args:
            if bk in x_impl:
                continue  # 如果 bk 已在 alpha 蕴含表中，跳过
            x_impl[bk] = (set(), [])  # 否则将 bk 加入 alpha 蕴含表，并关联空的 beta 引用列表

    # 静态扩展 alpha 规则：
    # A: x -> a,b   B: &(a,b) -> c  ==>  A: x -> a,b,c
    seen_static_extension = True
    while seen_static_extension:
        seen_static_extension = False

        for bcond, bimpl in beta_rules:
            if not isinstance(bcond, And):
                raise TypeError("Cond is not And")  # 如果条件不是 And 类型，则抛出类型错误异常
            bargs = set(bcond.args)
            for x, (ximpls, bb) in x_impl.items():
                x_all = ximpls | {x}
                # 如果 bimpl 不在 x_all 中并且 bargs 是 x_all 的子集，则将 bimpl 添加到 ximpls 中
                if bimpl not in x_all and bargs.issubset(x_all):
                    ximpls.add(bimpl)

                    # 我们引入了新的蕴含 - 现在我们必须恢复整个集合的完整性。
                    bimpl_impl = x_impl.get(bimpl)
                    if bimpl_impl is not None:
                        ximpls |= bimpl_impl[0]  # 将 bimpl 的蕴含集合合并到 ximpls 中
                    seen_static_extension = True

    # 将可能由 alpha 链触发的 beta 节点附加到它们
    for bidx, (bcond, bimpl) in enumerate(beta_rules):
        bargs = set(bcond.args)
        for x, (ximpls, bb) in x_impl.items():
            x_all = ximpls | {x}
            # 如果 bimpl 在 x_all 中，则跳过
            if bimpl in x_all:
                continue
            # 如果任何 Not(xi) 在 bargs 中或者 Not(xi) 等于 bimpl，则跳过
            if any(Not(xi) in bargs or Not(xi) == bimpl for xi in x_all):
                continue

            if bargs & x_all:
                bb.append(bidx)  # 将 beta 规则索引 bidx 添加到 bb 列表中，表示它可以由当前的 alpha 链触发

    return x_impl  # 返回更新后的 alpha 蕴含表
    """从逻辑规则构建先决条件表

       通过示例描述
       ----------------------

       给定一组逻辑规则：

         a -> b, c
         b -> c

       我们构建先决条件（从哪些点可以推断出某事物）：

         b <- a
         c <- a, b

       规则：{} 的 a -> [b, c, ...]
       返回：{} 的 c <- [a, b, ...]

       注意，这些先决条件可能*不足以*证明一个事实。例如，'a -> b' 规则中，先决条件(a)是b，而先决条件(b)是a。
       这是因为当a=T时 -> b=T，b=F时 -> a=F，但a=F时 -> b=？

       prereq：默认字典，存储先决条件的集合
    """
    prereq = defaultdict(set)
    # 遍历逻辑规则字典中的每对键值对
    for (a, _), impl in rules.items():
        # 如果a是Not类型，则获取其参数
        if isinstance(a, Not):
            a = a.args[0]
        # 遍历impl中的每对(i, _)
        for (i, _) in impl:
            # 如果i是Not类型，则获取其参数
            if isinstance(i, Not):
                i = i.args[0]
            # 将a添加到i的先决条件集合中
            prereq[i].add(a)
    # 返回构建好的先决条件表
    return prereq
################
# RULES PROVER #
################

# 自定义异常类，用于报告检测到的逻辑重言式
class TautologyDetected(Exception):
    """(internal) Prover uses it for reporting detected tautology"""
    pass

# 逻辑规则证明器类
class Prover:
    """ai - prover of logic rules

       given a set of initial rules, Prover tries to prove all possible rules
       which follow from given premises.

       As a result proved_rules are always either in one of two forms: alpha or
       beta:

       Alpha rules
       -----------

       This are rules of the form::

         a -> b & c & d & ...


       Beta rules
       ----------

       This are rules of the form::

         &(a,b,...) -> c & d & ...


       i.e. beta rules are join conditions that say that something follows when
       *several* facts are true at the same time.
    """

    def __init__(self):
        self.proved_rules = []   # 存储已证明的规则列表
        self._rules_seen = set() # 存储已见过的规则的集合

    def split_alpha_beta(self):
        """split proved rules into alpha and beta chains"""
        rules_alpha = []    # 存储形式为 a -> b 的规则
        rules_beta = []     # 存储形式为 &(a,b,...) -> c & d & ... 的规则
        for a, b in self.proved_rules:
            if isinstance(a, And):
                rules_beta.append((a, b))
            else:
                rules_alpha.append((a, b))
        return rules_alpha, rules_beta

    @property
    def rules_alpha(self):
        return self.split_alpha_beta()[0]   # 返回 alpha 类型的规则列表

    @property
    def rules_beta(self):
        return self.split_alpha_beta()[1]   # 返回 beta 类型的规则列表

    def process_rule(self, a, b):
        """process a -> b rule"""   # 处理 a -> b 类型的规则
        if (not a) or isinstance(b, bool):
            return
        if isinstance(a, bool):
            return
        if (a, b) in self._rules_seen:
            return
        else:
            self._rules_seen.add((a, b))

        # this is the core of processing
        try:
            self._process_rule(a, b)   # 调用核心处理函数来处理规则
        except TautologyDetected:
            pass
    # 处理逻辑推理规则的函数，处理规则的右侧部分

    # 如果 b 是 And 类型的逻辑表达式
    if isinstance(b, And):
        # 对 And 表达式的参数按照字符串排序
        sorted_bargs = sorted(b.args, key=str)
        # 遍历排序后的参数列表
        for barg in sorted_bargs:
            # 递归处理规则，将 a -> barg 分解为多个 a -> barg 的规则
            self.process_rule(a, barg)

    # 如果 b 是 Or 类型的逻辑表达式
    elif isinstance(b, Or):
        # 对 Or 表达式的参数按照字符串排序
        sorted_bargs = sorted(b.args, key=str)
        # 检测是否存在重言式，即 a -> a|c|...
        if not isinstance(a, Logic):    # Atom
            if a in sorted_bargs:
                # 如果 a 在 sorted_bargs 中，则抛出重言式检测异常
                raise TautologyDetected(a, b, 'a -> a|c|...')
        # 将 !b & !c -> !a 转换为 !b | !c -> !a 的规则，并处理
        self.process_rule(And(*[Not(barg) for barg in b.args]), Not(a))

        # 遍历排序后的参数列表
        for bidx in range(len(sorted_bargs)):
            barg = sorted_bargs[bidx]
            brest = sorted_bargs[:bidx] + sorted_bargs[bidx + 1:]
            # 将 a & !b -> c 转换为 a & !b -> c 的规则，并处理
            self.process_rule(And(a, Not(barg)), Or(*brest))

    # 处理规则的左侧部分

    # 如果 a 是 And 类型的逻辑表达式
    elif isinstance(a, And):
        # 对 And 表达式的参数按照字符串排序
        sorted_aargs = sorted(a.args, key=str)
        # 如果 b 存在于 sorted_aargs 中，则抛出重言式检测异常
        if b in sorted_aargs:
            raise TautologyDetected(a, b, 'a & b -> a')
        # 将规则 (a, b) 存入 proved_rules 列表中
        self.proved_rules.append((a, b))
        # 注意：目前我们忽略 !c -> !a | !b 的情况

    # 如果 a 是 Or 类型的逻辑表达式
    elif isinstance(a, Or):
        # 对 Or 表达式的参数按照字符串排序
        sorted_aargs = sorted(a.args, key=str)
        # 如果 b 存在于 sorted_aargs 中，则抛出重言式检测异常
        if b in sorted_aargs:
            raise TautologyDetected(a, b, 'a | b -> a')
        # 遍历排序后的参数列表
        for aarg in sorted_aargs:
            # 递归处理规则，将 aarg -> b 分解为多个 aarg -> b 的规则
            self.process_rule(aarg, b)

    else:
        # 当 a 和 b 都是原子命题时，将规则 (a, b) 存入 proved_rules 列表中，以及其否定形式 (!b -> !a)
        self.proved_rules.append((a, b))             # a  -> b
        self.proved_rules.append((Not(b), Not(a)))   # !b -> !a
########################################


class FactRules:
    """Rules that describe how to deduce facts in logic space

       When defined, these rules allow implications to quickly be determined
       for a set of facts. For this precomputed deduction tables are used.
       see `deduce_all_facts`   (forward-chaining)

       Also it is possible to gather prerequisites for a fact, which is tried
       to be proven.    (backward-chaining)


       Definition Syntax
       -----------------

       a -> b       -- a=T -> b=T  (and automatically b=F -> a=F)
       a -> !b      -- a=T -> b=F
       a == b       -- a -> b & b -> a
       a -> b & c   -- a=T -> b=T & c=T
       # TODO b | c


       Internals
       ---------

       .full_implications[k, v]: all the implications of fact k=v
       .beta_triggers[k, v]: beta rules that might be triggered when k=v
       .prereq  -- {} k <- [] of k's prerequisites

       .defined_facts -- set of defined fact names
    """

    def __init__(self, rules):
        """Compile rules into internal lookup tables"""

        if isinstance(rules, str):
            rules = rules.splitlines()

        # --- parse and process rules ---
        # Create a Prover object for managing logical rules
        P = Prover()

        for rule in rules:
            # Split the rule into components: a, op, b
            a, op, b = rule.split(None, 2)

            # Convert a and b into Logic objects
            a = Logic.fromstring(a)
            b = Logic.fromstring(b)

            # Process the rule based on the operator (op)
            if op == '->':
                P.process_rule(a, b)
            elif op == '==':
                P.process_rule(a, b)
                P.process_rule(b, a)
            else:
                raise ValueError('unknown op %r' % op)

        # --- build deduction networks ---
        self.beta_rules = []
        # Extract beta rules from the Prover object
        for bcond, bimpl in P.rules_beta:
            self.beta_rules.append(
                ({_as_pair(a) for a in bcond.args}, _as_pair(bimpl)))

        # Deduce alpha implications from alpha rules
        impl_a = deduce_alpha_implications(P.rules_alpha)

        # Apply beta rules to alpha chains for static extension
        impl_ab = apply_beta_to_alpha_route(impl_a, P.rules_beta)

        # Extract defined fact names from impl_ab keys
        self.defined_facts = {_base_fact(k) for k in impl_ab.keys()}

        # Build forward chains (full implications)
        full_implications = defaultdict(set)
        for k, (impl, betaidxs) in impl_ab.items():
            full_implications[_as_pair(k)] = {_as_pair(i) for i in impl}

        # Assign full implications to the object
        self.full_implications = full_implications

        # Build beta triggers for inference at runtime
        beta_triggers = defaultdict(set)
        for k, (impl, betaidxs) in impl_ab.items():
            beta_triggers[_as_pair(k)] = betaidxs

        # Assign beta triggers to the object
        self.beta_triggers = beta_triggers

        # Build backward chains (prerequisites)
        prereq = defaultdict(set)
        rel_prereq = rules_2prereq(full_implications)
        for k, pitems in rel_prereq.items():
            prereq[k] |= pitems

        # Assign prerequisites to the object
        self.prereq = prereq
    def _to_python(self) -> str:
        """ Generate a string with plain python representation of the instance """
        # 调用实例的print_rules方法，将返回的列表转换成字符串，并用换行符连接起来
        return '\n'.join(self.print_rules())

    @classmethod
    def _from_python(cls, data : dict):
        """ Generate an instance from the plain python representation """
        # 创建一个新的实例self，使用空字符串作为参数
        self = cls('')
        # 对特定键['full_implications', 'beta_triggers', 'prereq']进行迭代
        for key in ['full_implications', 'beta_triggers', 'prereq']:
            # 创建一个默认字典，并用data[key]更新它，然后将其设置为self的属性
            d=defaultdict(set)
            d.update(data[key])
            setattr(self, key, d)
        # 设置self的beta_rules属性为data['beta_rules']
        self.beta_rules = data['beta_rules']
        # 将data['defined_facts']转换成集合并设置为self的defined_facts属性
        self.defined_facts = set(data['defined_facts'])

        # 返回创建的self实例
        return self

    def _defined_facts_lines(self):
        yield 'defined_facts = ['
        # 对self.defined_facts集合进行排序后，逐个生成字符串表示的条目
        for fact in sorted(self.defined_facts):
            yield f'    {fact!r},'
        yield '] # defined_facts'

    def _full_implications_lines(self):
        yield 'full_implications = dict( ['
        # 对self.defined_facts集合进行排序后，以及True和False的值，逐个生成字符串表示的条目
        for fact in sorted(self.defined_facts):
            for value in (True, False):
                yield f'    # Implications of {fact} = {value}:'
                yield f'    (({fact!r}, {value!r}), set( ('
                # 获取self.full_implications[(fact, value)]中的每个implied条目，逐个生成字符串表示的条目
                implications = self.full_implications[(fact, value)]
                for implied in sorted(implications):
                    yield f'        {implied!r},'
                yield '       ) ),'
                yield '     ),'
        yield ' ] ) # full_implications'

    def _prereq_lines(self):
        yield 'prereq = {'
        yield ''
        # 对self.prereq字典的键进行排序后，逐个生成字符串表示的条目
        for fact in sorted(self.prereq):
            yield f'    # facts that could determine the value of {fact}'
            yield f'    {fact!r}: {{'
            # 对self.prereq[fact]集合的每个pfact进行迭代，逐个生成字符串表示的条目
            for pfact in sorted(self.prereq[fact]):
                yield f'        {pfact!r},'
            yield '    },'
            yield ''
        yield '} # prereq'

    def _beta_rules_lines(self):
        reverse_implications = defaultdict(list)
        # 对self.beta_rules中的每个(pre, implied)对进行枚举，将其反向映射到reverse_implications
        for n, (pre, implied) in enumerate(self.beta_rules):
            reverse_implications[implied].append((pre, n))

        yield '# Note: the order of the beta rules is used in the beta_triggers'
        yield 'beta_rules = ['
        yield ''
        m = 0
        indices = {}
        # 对reverse_implications中的implied进行排序后进行迭代，生成字符串表示的条目
        for implied in sorted(reverse_implications):
            fact, value = implied
            yield f'    # Rules implying {fact} = {value}'
            # 对reverse_implications[implied]中的每个(pre, n)对进行迭代，生成字符串表示的条目
            for pre, n in reverse_implications[implied]:
                indices[n] = m
                m += 1
                setstr = ", ".join(map(str, sorted(pre)))
                yield f'    ({{{setstr}}},'
                yield f'        {implied!r}),'
            yield ''
        yield '] # beta_rules'

        yield 'beta_triggers = {'
        # 对self.beta_triggers中的每个query进行排序后进行迭代，生成字符串表示的条目
        for query in sorted(self.beta_triggers):
            fact, value = query
            # 获取self.beta_triggers[query]中的每个n，并生成字符串表示的条目
            triggers = [indices[n] for n in self.beta_triggers[query]]
            yield f'    {query!r}: {triggers!r},'
        yield '} # beta_triggers'
    def print_rules(self) -> Iterator[str]:
        """ 
        返回一个生成器，生成用于表示事实和规则的行
        """
        # 生成已定义事实的行
        yield from self._defined_facts_lines()
        
        # 添加空行
        yield ''
        yield ''
        
        # 生成完整蕴含的行
        yield from self._full_implications_lines()
        
        # 添加空行
        yield ''
        yield ''
        
        # 生成先决条件的行
        yield from self._prereq_lines()
        
        # 添加空行
        yield ''
        yield ''
        
        # 生成 beta 规则的行
        yield from self._beta_rules_lines()
        
        # 添加空行
        yield ''
        yield ''
        
        # 生成一个包含生成的假设的字典
        yield "generated_assumptions = {'defined_facts': defined_facts, 'full_implications': full_implications,"
        yield "               'prereq': prereq, 'beta_rules': beta_rules, 'beta_triggers': beta_triggers}"
class InconsistentAssumptions(ValueError):
    # 自定义异常类，用于表示不一致的假设情况
    def __str__(self):
        # 从异常参数中解包出知识库、事实和值，返回格式化的字符串描述异常信息
        kb, fact, value = self.args
        return "%s, %s=%s" % (kb, fact, value)


class FactKB(dict):
    """
    A simple propositional knowledge base relying on compiled inference rules.
    """
    def __str__(self):
        # 返回当前知识库的字符串表示，格式化显示所有键值对
        return '{\n%s}' % ',\n'.join(
            ["\t%s: %s" % i for i in sorted(self.items())])

    def __init__(self, rules):
        # 初始化函数，接受推理规则作为参数
        self.rules = rules

    def _tell(self, k, v):
        """Add fact k=v to the knowledge base.

        Returns True if the KB has actually been updated, False otherwise.
        """
        # 添加事实 k=v 到知识库中
        if k in self and self[k] is not None:
            # 如果已存在并且不为 None，则检查是否与新值相同
            if self[k] == v:
                return False
            else:
                # 如果不同则抛出不一致假设异常
                raise InconsistentAssumptions(self, k, v)
        else:
            # 否则直接添加新的键值对
            self[k] = v
            return True

    # *********************************************
    # * This is the workhorse, so keep it *fast*. *
    # *********************************************
    def deduce_all_facts(self, facts):
        """
        Update the KB with all the implications of a list of facts.

        Facts can be specified as a dictionary or as a list of (key, value)
        pairs.
        """
        # 将频繁使用的属性保存在本地变量中，以避免额外的属性访问开销
        full_implications = self.rules.full_implications
        beta_triggers = self.rules.beta_triggers
        beta_rules = self.rules.beta_rules

        if isinstance(facts, dict):
            # 如果输入是字典，则转换为键值对列表
            facts = facts.items()

        while facts:
            beta_maytrigger = set()

            # --- alpha chains ---
            # 处理 alpha 链
            for k, v in facts:
                if not self._tell(k, v) or v is None:
                    continue

                # 查找全面推断表中与当前键值对匹配的规则
                for key, value in full_implications[k, v]:
                    self._tell(key, value)

                # 更新 beta 可能触发的集合
                beta_maytrigger.update(beta_triggers[k, v])

            # --- beta chains ---
            # 处理 beta 链
            facts = []
            for bidx in beta_maytrigger:
                bcond, bimpl = beta_rules[bidx]
                # 如果 beta 规则的条件都满足，则添加新的事实
                if all(self.get(k) is v for k, v in bcond):
                    facts.append(bimpl)
```