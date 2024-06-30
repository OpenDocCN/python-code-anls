# `D:\src\scipysrc\sympy\sympy\core\assumptions_generated.py`

```
"""
Do NOT manually edit this file.
Instead, run ./bin/ask_update.py.
"""

# 定义一个包含数学事实的列表
defined_facts = [
    'algebraic',                 # 代数的
    'antihermitian',             # 反厄米的
    'commutative',               # 可交换的
    'complex',                   # 复数的
    'composite',                 # 合成的
    'even',                      # 偶数的
    'extended_negative',         # 扩展负数的
    'extended_nonnegative',      # 扩展非负数的
    'extended_nonpositive',      # 扩展非正数的
    'extended_nonzero',          # 扩展非零的
    'extended_positive',         # 扩展正数的
    'extended_real',             # 扩展实数的
    'finite',                    # 有限的
    'hermitian',                 # 厄米的
    'imaginary',                 # 虚数的
    'infinite',                  # 无限的
    'integer',                   # 整数的
    'irrational',                # 无理数的
    'negative',                  # 负数的
    'noninteger',                # 非整数的
    'nonnegative',               # 非负数的
    'nonpositive',               # 非正数的
    'nonzero',                   # 非零的
    'odd',                       # 奇数的
    'positive',                  # 正数的
    'prime',                     # 素数的
    'rational',                  # 有理数的
    'real',                      # 实数的
    'transcendental',            # 超越的
    'zero',                      # 零的
] # defined_facts


# 定义一个字典，包含每个数学事实的推论
full_implications = dict([
    # 当 algebraic = True 时的推论：
    (('algebraic', True), set([
        ('commutative', True),         # 可交换的为真
        ('complex', True),             # 复数的为真
        ('finite', True),              # 有限的为真
        ('infinite', False),           # 无限的为假
        ('transcendental', False),     # 超越的为假
    ])),
    # 当 algebraic = False 时的推论：
    (('algebraic', False), set([
        ('composite', False),          # 合成的为假
        ('even', False),               # 偶数的为假
        ('integer', False),            # 整数的为假
        ('odd', False),                # 奇数的为假
        ('prime', False),              # 素数的为假
        ('rational', False),           # 有理数的为假
        ('zero', False),               # 零的为假
    ])),
    # 当 antihermitian = True 时的推论（此处为空集）：
    (('antihermitian', True), set([])),
    # 当 antihermitian = False 时的推论：
    (('antihermitian', False), set([
        ('imaginary', False),          # 虚数的为假
    ])),
    # 当 commutative = True 时的推论（此处为空集）：
    (('commutative', True), set([])),
    # 当 commutative = False 时的推论：
    (('commutative', False), set([
        ('algebraic', False),          # 代数的为假
        ('complex', False),            # 复数的为假
        ('composite', False),          # 合成的为假
        ('even', False),               # 偶数的为假
        ('extended_negative', False),  # 扩展负数的为假
        ('extended_nonnegative', False),  # 扩展非负数的为假
        ('extended_nonpositive', False),  # 扩展非正数的为假
        ('extended_nonzero', False),   # 扩展非零的为假
        ('extended_positive', False),  # 扩展正数的为假
        ('extended_real', False),      # 扩展实数的为假
        ('imaginary', False),          # 虚数的为假
        ('integer', False),            # 整数的为假
        ('irrational', False),         # 无理数的为假
        ('negative', False),           # 负数的为假
        ('noninteger', False),         # 非整数的为假
        ('nonnegative', False),        # 非负数的为假
        ('nonpositive', False),        # 非正数的为假
        ('nonzero', False),            # 非零的为假
        ('odd', False),                # 奇数的为假
        ('positive', False),           # 正数的为假
        ('prime', False),              # 素数的为假
        ('rational', False),           # 有理数的为假
        ('real', False),               # 实数的为假
        ('transcendental', False),     # 超越的为假
        ('zero', False),               # 零的为假
    ])),
    # 当 complex = True 时的推论：
    (('complex', True), set([
        ('commutative', True),         # 可交换的为真
        ('finite', True),              # 有限的为真
        ('infinite', False),           # 无限的为假
    ])),
    # 当 complex = False 时的推论：
    # 定义一个复杂的数据结构，包含多个元组，每个元组由一个布尔值和一个集合组成
    (('complex', False), set( (
        # 当 composite = True 时的逻辑推断集合
        ('algebraic', False),
        ('composite', False),
        ('even', False),
        ('imaginary', False),
        ('integer', False),
        ('irrational', False),
        ('negative', False),
        ('nonnegative', False),
        ('nonpositive', False),
        ('nonzero', False),
        ('odd', False),
        ('positive', False),
        ('prime', False),
        ('rational', False),
        ('real', False),
        ('transcendental', False),
        ('zero', False),
       ) ),
     ),
    # composite = True 时的逻辑推断集合
    (('composite', True), set( (
        ('algebraic', True),
        ('commutative', True),
        ('complex', True),
        ('extended_negative', False),
        ('extended_nonnegative', True),
        ('extended_nonpositive', False),
        ('extended_nonzero', True),
        ('extended_positive', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('integer', True),
        ('irrational', False),
        ('negative', False),
        ('noninteger', False),
        ('nonnegative', True),
        ('nonpositive', False),
        ('nonzero', True),
        ('positive', True),
        ('prime', False),
        ('rational', True),
        ('real', True),
        ('transcendental', False),
        ('zero', False),
       ) ),
     ),
    # composite = False 时的逻辑推断集合
    (('composite', False), set( (
       ) ),
     ),
    # even = True 时的逻辑推断集合
    (('even', True), set( (
        ('algebraic', True),
        ('commutative', True),
        ('complex', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('integer', True),
        ('irrational', False),
        ('noninteger', False),
        ('odd', False),
        ('rational', True),
        ('real', True),
        ('transcendental', False),
       ) ),
     ),
    # even = False 时的逻辑推断集合
    (('even', False), set( (
        ('zero', False),
       ) ),
     ),
    # extended_negative = True 时的逻辑推断集合
    (('extended_negative', True), set( (
        ('commutative', True),
        ('composite', False),
        ('extended_nonnegative', False),
        ('extended_nonpositive', True),
        ('extended_nonzero', True),
        ('extended_positive', False),
        ('extended_real', True),
        ('imaginary', False),
        ('nonnegative', False),
        ('positive', False),
        ('prime', False),
        ('zero', False),
       ) ),
     ),
    # extended_negative = False 时的逻辑推断集合
    (('extended_negative', False), set( (
        ('negative', False),
       ) ),
     ),
    # extended_nonnegative = True 时的逻辑推断集合
    # Implications of extended_real = False:
    (('extended_real', False), set( (
        ('imaginary', True),  # When extended_real is False, extended numbers are imaginary
       ) ),
     ),
    # 定义一系列布尔条件和其对应的影响集合
    
    (('extended_real', False), set((
        ('composite', False),                # 如果 extended_real = False，则 composite = False
        ('even', False),                     # 如果 extended_real = False，则 even = False
        ('extended_negative', False),        # 如果 extended_real = False，则 extended_negative = False
        ('extended_nonnegative', False),     # 如果 extended_real = False，则 extended_nonnegative = False
        ('extended_nonpositive', False),     # 如果 extended_real = False，则 extended_nonpositive = False
        ('extended_nonzero', False),         # 如果 extended_real = False，则 extended_nonzero = False
        ('extended_positive', False),        # 如果 extended_real = False，则 extended_positive = False
        ('integer', False),                  # 如果 extended_real = False，则 integer = False
        ('irrational', False),               # 如果 extended_real = False，则 irrational = False
        ('negative', False),                 # 如果 extended_real = False，则 negative = False
        ('noninteger', False),               # 如果 extended_real = False，则 noninteger = False
        ('nonnegative', False),              # 如果 extended_real = False，则 nonnegative = False
        ('nonpositive', False),              # 如果 extended_real = False，则 nonpositive = False
        ('nonzero', False),                  # 如果 extended_real = False，则 nonzero = False
        ('odd', False),                      # 如果 extended_real = False，则 odd = False
        ('positive', False),                 # 如果 extended_real = False，则 positive = False
        ('prime', False),                    # 如果 extended_real = False，则 prime = False
        ('rational', False),                 # 如果 extended_real = False，则 rational = False
        ('real', False),                     # 如果 extended_real = False，则 real = False
        ('zero', False),                     # 如果 extended_real = False，则 zero = False
    ))),
    
    # 当 finite = True 时的影响：
    (('finite', True), set((
        ('infinite', False),                  # 如果 finite = True，则 infinite = False
    ))),
    
    # 当 finite = False 时的影响：
    (('finite', False), set((
        ('algebraic', False),                 # 如果 finite = False，则 algebraic = False
        ('complex', False),                   # 如果 finite = False，则 complex = False
        ('composite', False),                 # 如果 finite = False，则 composite = False
        ('even', False),                      # 如果 finite = False，则 even = False
        ('imaginary', False),                 # 如果 finite = False，则 imaginary = False
        ('infinite', True),                   # 如果 finite = False，则 infinite = True
        ('integer', False),                   # 如果 finite = False，则 integer = False
        ('irrational', False),                # 如果 finite = False，则 irrational = False
        ('negative', False),                  # 如果 finite = False，则 negative = False
        ('nonnegative', False),               # 如果 finite = False，则 nonnegative = False
        ('nonpositive', False),               # 如果 finite = False，则 nonpositive = False
        ('nonzero', False),                   # 如果 finite = False，则 nonzero = False
        ('odd', False),                       # 如果 finite = False，则 odd = False
        ('positive', False),                  # 如果 finite = False，则 positive = False
        ('prime', False),                     # 如果 finite = False，则 prime = False
        ('rational', False),                  # 如果 finite = False，则 rational = False
        ('real', False),                      # 如果 finite = False，则 real = False
        ('transcendental', False),            # 如果 finite = False，则 transcendental = False
        ('zero', False),                      # 如果 finite = False，则 zero = False
    ))),
    
    # 当 hermitian = True 时的影响：
    (('hermitian', True), set((
        # 空集合，没有额外的影响
    ))),
    
    # 当 hermitian = False 时的影响：
    (('hermitian', False), set((
        ('composite', False),                 # 如果 hermitian = False，则 composite = False
        ('even', False),                      # 如果 hermitian = False，则 even = False
        ('integer', False),                   # 如果 hermitian = False，则 integer = False
        ('irrational', False),                # 如果 hermitian = False，则 irrational = False
        ('negative', False),                  # 如果 hermitian = False，则 negative = False
        ('nonnegative', False),               # 如果 hermitian = False，则 nonnegative = False
        ('nonpositive', False),               # 如果 hermitian = False，则 nonpositive = False
        ('nonzero', False),                   # 如果 hermitian = False，则 nonzero = False
        ('odd', False),                       # 如果 hermitian = False，则 odd = False
        ('positive', False),                  # 如果 hermitian = False，则 positive = False
        ('prime', False),                     # 如果 hermitian = False，则 prime = False
        ('rational', False),                  # 如果 hermitian = False，则 rational = False
        ('real', False),                      # 如果 hermitian = False，则 real = False
        ('zero', False),                      # 如果 hermitian = False，则 zero = False
    ))),
    
    # 当 imaginary = True 时的影响：
    (('imaginary', True), set((
        ('antihermitian', True),              # 如果 imaginary = True，则 antihermitian = True
        ('commutative', True),                # 如果 imaginary = True，则 commutative = True
        ('complex', True),                    # 如果 imaginary = True，则 complex = True
        ('composite', False),                 # 如果 imaginary = True，则 composite = False
        ('even', False),                      # 如果 imaginary = True，则 even = False
        ('extended_negative', False),         # 如果 imaginary = True，则 extended_negative = False
        ('extended_nonnegative', False),      # 如果 imaginary = True，则 extended_nonnegative = False
        ('extended_nonpositive', False),      # 如果 imaginary = True，则 extended_nonpositive = False
        ('extended_nonzero', False),          # 如果 imaginary = True，则 extended_nonzero = False
        ('extended_positive', False),         # 如果 imaginary = True，则 extended_positive = False
        ('extended_real', False),             # 如果 imaginary = True，则 extended_real = False
        ('finite', True),                     # 如果 imaginary = True，则 finite = True
        ('infinite', False),                  # 如果 imaginary = True，则 infinite = False
        ('integer', False),                   # 如果 imaginary = True，则 integer = False
        ('irrational', False),                # 如果 imaginary = True，则 irrational = False
        ('negative', False),                  # 如果 imaginary = True，则 negative = False
        ('noninteger', False),                # 如果 imaginary = True，则 noninteger = False
        ('nonnegative', False),               # 如果 imaginary = True，则 nonnegative = False
        ('nonpositive', False),               # 如果 imaginary = True，则 nonpositive = False
        ('nonzero', False),                   # 如果 imaginary = True，则 nonzero = False
        ('odd', False),                       # 如果 imaginary = True，则 odd = False
        ('positive', False),                  # 如果 imaginary = True，则 positive = False
        ('prime', False),                     # 如果 imaginary = True，则 prime = False
        ('rational', False),                  # 如果 imaginary = True，则 rational = False
        ('real', False),                      # 如果 imaginary = True，则 real = False
        ('zero', False),                      # 如果 imaginary = True，则 zero = False
    ))),
    
    # 当 imaginary = False 时的影响：
    (('imaginary', False), set((
        # 空集合，没有额外的影响
    ))),
    # Implications of infinite = True:
    # 当 infinite = True 时的含义：
    (('infinite', True), set( (
        # 设定以下属性为 False：
        ('algebraic', False),
        ('complex', False),
        ('composite', False),
        ('even', False),
        ('finite', False),
        ('imaginary', False),
        ('integer', False),
        ('irrational', False),
        ('negative', False),
        ('nonnegative', False),
        ('nonpositive', False),
        ('nonzero', False),
        ('odd', False),
        ('positive', False),
        ('prime', False),
        ('rational', False),
        ('real', False),
        ('transcendental', False),
        ('zero', False),
       ) ),
     ),
    # Implications of infinite = False:
    # 当 infinite = False 时的含义：
    (('infinite', False), set( (
        # 设定以下属性为 True：
        ('finite', True),
       ) ),
     ),
    # Implications of integer = True:
    # 当 integer = True 时的含义：
    (('integer', True), set( (
        # 设定以下属性为 True：
        ('algebraic', True),
        ('commutative', True),
        ('complex', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('irrational', False),
        ('noninteger', False),
        ('rational', True),
        ('real', True),
        ('transcendental', False),
       ) ),
     ),
    # Implications of integer = False:
    # 当 integer = False 时的含义：
    (('integer', False), set( (
        # 设定以下属性为 False：
        ('composite', False),
        ('even', False),
        ('odd', False),
        ('prime', False),
        ('zero', False),
       ) ),
     ),
    # Implications of irrational = True:
    # 当 irrational = True 时的含义：
    (('irrational', True), set( (
        # 设定以下属性为 True：
        ('commutative', True),
        ('complex', True),
        ('composite', False),
        ('even', False),
        ('extended_nonzero', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('integer', False),
        ('noninteger', True),
        ('nonzero', True),
        ('odd', False),
        ('prime', False),
        ('rational', False),
        ('real', True),
        ('zero', False),
       ) ),
     ),
    # Implications of irrational = False:
    # 当 irrational = False 时的含义：
    (('irrational', False), set( (
        # 没有属性被设定
       ) ),
     ),
    # Implications of negative = True:
    # 当 negative = True 时的含义：
    (('negative', True), set( (
        # 设定以下属性为 True：
        ('commutative', True),
        ('complex', True),
        ('composite', False),
        ('extended_negative', True),
        ('extended_nonnegative', False),
        ('extended_nonpositive', True),
        ('extended_nonzero', True),
        ('extended_positive', False),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('nonnegative', False),
        ('nonpositive', True),
        ('nonzero', True),
        ('positive', False),
        ('prime', False),
        ('real', True),
        ('zero', False),
       ) ),
     ),
    # Implications of negative = False:
    # 当 negative = False 时的含义：
    (('negative', False), set( (
        # 没有属性被设定
       ) ),
     ),
    # Implications of noninteger = True:
    (('noninteger', True), set( (
        ('commutative', True),
        ('composite', False),
        ('even', False),
        ('extended_nonzero', True),
        ('extended_real', True),
        ('imaginary', False),
        ('integer', False),
        ('odd', False),
        ('prime', False),
        ('zero', False),
       ) ),
     ),
    # noninteger = True 时的逻辑推论:
    #   - commutative 是 True
    #   - composite 是 False
    #   - even 是 False
    #   - extended_nonzero 是 True
    #   - extended_real 是 True
    #   - imaginary 是 False
    #   - integer 是 False
    #   - odd 是 False
    #   - prime 是 False
    #   - zero 是 False

    # Implications of noninteger = False:
    (('noninteger', False), set( (
       ) ),
     ),
    # noninteger = False 时的逻辑推论: （此处为空，无逻辑推论）

    # Implications of nonnegative = True:
    (('nonnegative', True), set( (
        ('commutative', True),
        ('complex', True),
        ('extended_negative', False),
        ('extended_nonnegative', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('negative', False),
        ('real', True),
       ) ),
     ),
    # nonnegative = True 时的逻辑推论:
    #   - commutative 是 True
    #   - complex 是 True
    #   - extended_negative 是 False
    #   - extended_nonnegative 是 True
    #   - extended_real 是 True
    #   - finite 是 True
    #   - hermitian 是 True
    #   - imaginary 是 False
    #   - infinite 是 False
    #   - negative 是 False
    #   - real 是 True

    # Implications of nonnegative = False:
    (('nonnegative', False), set( (
        ('composite', False),
        ('positive', False),
        ('prime', False),
        ('zero', False),
       ) ),
     ),
    # nonnegative = False 时的逻辑推论:
    #   - composite 是 False
    #   - positive 是 False
    #   - prime 是 False
    #   - zero 是 False

    # Implications of nonpositive = True:
    (('nonpositive', True), set( (
        ('commutative', True),
        ('complex', True),
        ('composite', False),
        ('extended_nonpositive', True),
        ('extended_positive', False),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('positive', False),
        ('prime', False),
        ('real', True),
       ) ),
     ),
    # nonpositive = True 时的逻辑推论:
    #   - commutative 是 True
    #   - complex 是 True
    #   - composite 是 False
    #   - extended_nonpositive 是 True
    #   - extended_positive 是 False
    #   - extended_real 是 True
    #   - finite 是 True
    #   - hermitian 是 True
    #   - imaginary 是 False
    #   - infinite 是 False
    #   - positive 是 False
    #   - prime 是 False
    #   - real 是 True

    # Implications of nonpositive = False:
    (('nonpositive', False), set( (
        ('negative', False),
        ('zero', False),
       ) ),
     ),
    # nonpositive = False 时的逻辑推论:
    #   - negative 是 False
    #   - zero 是 False

    # Implications of nonzero = True:
    (('nonzero', True), set( (
        ('commutative', True),
        ('complex', True),
        ('extended_nonzero', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('real', True),
        ('zero', False),
       ) ),
     ),
    # nonzero = True 时的逻辑推论:
    #   - commutative 是 True
    #   - complex 是 True
    #   - extended_nonzero 是 True
    #   - extended_real 是 True
    #   - finite 是 True
    #   - hermitian 是 True
    #   - imaginary 是 False
    #   - infinite 是 False
    #   - real 是 True
    #   - zero 是 False

    # Implications of nonzero = False:
    (('nonzero', False), set( (
        ('composite', False),
        ('negative', False),
        ('positive', False),
        ('prime', False),
       ) ),
     ),
    # nonzero = False 时的逻辑推论:
    #   - composite 是 False
    #   - negative 是 False
    #   - positive 是 False
    #   - prime 是 False

    # Implications of odd = True:
    (('odd', True), set( (
        ('algebraic', True),
        ('commutative', True),
        ('complex', True),
        ('even', False),
        ('extended_nonzero', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('integer', True),
        ('irrational', False),
        ('noninteger', False),
        ('nonzero', True),
        ('rational', True),
        ('real', True),
        ('transcendental', False),
        ('zero', False),
       ) ),
     ),
    # odd = True 时的逻辑推论:
    #   - algebraic 是 True
    #   - commutative 是 True
    #   - complex 是 True
    #   - even 是 False
    #   - extended_nonzero 是 True
    #   - extended_real 是 True
    #   - finite 是 True
    #   - hermitian 是 True
    #   - imaginary 是 False
    #   - infinite 是 False
    #   - integer 是 True
    #   - irrational 是 False
    #   - noninteger 是 False
    #   - nonzero 是 True
    #   - rational 是 True
    #   - real 是 True
    #   - transcendental 是 False
    #   - zero 是 False

    # Implications of odd = False:
    (('odd', False), set( (
       ) ),
     ),
    # odd = False 时的逻辑推论: （此处为空，无逻辑推论）
    # Implications of positive = True:
    (('positive', True), set( (
        ('commutative', True),  # If positive, the number is commutative
        ('complex', True),      # If positive, the number is complex
        ('extended_negative', False),   # If positive, it's not extended negative
        ('extended_nonnegative', True), # If positive, it's extended nonnegative
        ('extended_nonpositive', False),    # If positive, it's not extended nonpositive
        ('extended_nonzero', True),     # If positive, it's extended nonzero
        ('extended_positive', True),    # If positive, it's extended positive
        ('extended_real', True),        # If positive, it's extended real
        ('finite', True),       # If positive, the number is finite
        ('hermitian', True),    # If positive, it's hermitian
        ('imaginary', False),   # If positive, it's not imaginary
        ('infinite', False),    # If positive, it's not infinite
        ('negative', False),    # If positive, it's not negative
        ('nonnegative', True),  # If positive, it's nonnegative
        ('nonpositive', False), # If positive, it's not nonpositive
        ('nonzero', True),      # If positive, it's nonzero
        ('real', True),         # If positive, it's real
        ('zero', False),        # If positive, it's not zero
       ) ),
     ),
    
    # Implications of positive = False:
    (('positive', False), set( (
        ('composite', False),   # If not positive, it's not composite
        ('prime', False),       # If not positive, it's not prime
       ) ),
     ),
    
    # Implications of prime = True:
    (('prime', True), set( (
        ('algebraic', True),        # If prime, it's algebraic
        ('commutative', True),      # If prime, it's commutative
        ('complex', True),          # If prime, it's complex
        ('composite', False),       # If prime, it's not composite
        ('extended_negative', False),   # If prime, it's not extended negative
        ('extended_nonnegative', True), # If prime, it's extended nonnegative
        ('extended_nonpositive', False),    # If prime, it's not extended nonpositive
        ('extended_nonzero', True),     # If prime, it's extended nonzero
        ('extended_positive', True),    # If prime, it's extended positive
        ('extended_real', True),        # If prime, it's extended real
        ('finite', True),       # If prime, the number is finite
        ('hermitian', True),    # If prime, it's hermitian
        ('imaginary', False),   # If prime, it's not imaginary
        ('infinite', False),    # If prime, it's not infinite
        ('integer', True),      # If prime, it's an integer
        ('irrational', False),  # If prime, it's not irrational
        ('negative', False),    # If prime, it's not negative
        ('noninteger', False),  # If prime, it's not noninteger
        ('nonnegative', True),  # If prime, it's nonnegative
        ('nonpositive', False), # If prime, it's not nonpositive
        ('nonzero', True),      # If prime, it's nonzero
        ('positive', True),     # If prime, it's positive
        ('rational', True),     # If prime, it's rational
        ('real', True),         # If prime, it's real
        ('transcendental', False),  # If prime, it's not transcendental
        ('zero', False),        # If prime, it's not zero
       ) ),
     ),
    
    # Implications of prime = False:
    (('prime', False), set( (
       ) ),
     ),
    
    # Implications of rational = True:
    (('rational', True), set( (
        ('algebraic', True),    # If rational, it's algebraic
        ('commutative', True),  # If rational, it's commutative
        ('complex', True),      # If rational, it's complex
        ('extended_real', True),    # If rational, it's extended real
        ('finite', True),       # If rational, the number is finite
        ('hermitian', True),    # If rational, it's hermitian
        ('imaginary', False),   # If rational, it's not imaginary
        ('infinite', False),    # If rational, it's not infinite
        ('irrational', False),  # If rational, it's not irrational
        ('real', True),         # If rational, it's real
        ('transcendental', False),  # If rational, it's not transcendental
       ) ),
     ),
    
    # Implications of rational = False:
    (('rational', False), set( (
        ('composite', False),   # If not rational, it's not composite
        ('even', False),        # If not rational, it's not even
        ('integer', False),     # If not rational, it's not an integer
        ('odd', False),         # If not rational, it's not odd
        ('prime', False),       # If not rational, it's not prime
        ('zero', False),        # If not rational, it's not zero
       ) ),
     ),
    
    # Implications of real = True:
    (('real', True), set( (
        ('commutative', True),  # If real, it's commutative
        ('complex', True),      # If real, it's complex
        ('extended_real', True),    # If real, it's extended real
        ('finite', True),       # If real, the number is finite
        ('hermitian', True),    # If real, it's hermitian
        ('imaginary', False),   # If real, it's not imaginary
        ('infinite', False),    # If real, it's not infinite
       ) ),
     ),
    
    # Implications of real = False:
    (('real', False), set( (
        ('composite', False),
        ('even', False),
        ('integer', False),
        ('irrational', False),
        ('negative', False),
        ('nonnegative', False),
        ('nonpositive', False),
        ('nonzero', False),
        ('odd', False),
        ('positive', False),
        ('prime', False),
        ('rational', False),
        ('zero', False),
       ) ),
     ),
    # 当 real = False 时的逻辑推论集合：
    (('transcendental', True), set( (
        ('algebraic', False),
        ('commutative', True),
        ('complex', True),
        ('composite', False),
        ('even', False),
        ('finite', True),
        ('infinite', False),
        ('integer', False),
        ('odd', False),
        ('prime', False),
        ('rational', False),
        ('zero', False),
       ) ),
     ),
    # 当 transcendental = True 时的逻辑推论集合：
    (('transcendental', False), set( (
       ) ),
     ),
    # 当 transcendental = False 时的逻辑推论集合（空集）：
    (('zero', True), set( (
        ('algebraic', True),
        ('commutative', True),
        ('complex', True),
        ('composite', False),
        ('even', True),
        ('extended_negative', False),
        ('extended_nonnegative', True),
        ('extended_nonpositive', True),
        ('extended_nonzero', False),
        ('extended_positive', False),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('integer', True),
        ('irrational', False),
        ('negative', False),
        ('noninteger', False),
        ('nonnegative', True),
        ('nonpositive', True),
        ('nonzero', False),
        ('odd', False),
        ('positive', False),
        ('prime', False),
        ('rational', True),
        ('real', True),
        ('transcendental', False),
       ) ),
     ),
    # 当 zero = True 时的逻辑推论集合：
    (('zero', False), set( (
       ) ),
     ),
    # 当 zero = False 时的逻辑推论集合（空集）：
 ] ) # full_implications
{
    # 用于决定代数性质的事实集合
    'algebraic': {
        'commutative',          # 是否可交换
        'complex',              # 是否复数
        'composite',            # 是否合数
        'even',                 # 是否偶数
        'finite',               # 是否有限的
        'infinite',             # 是否无限的
        'integer',              # 是否整数
        'odd',                  # 是否奇数
        'prime',                # 是否质数
        'rational',             # 是否有理数
        'transcendental',       # 是否超越数
        'zero',                 # 是否为零
    },

    # 用于决定反厄米性质的事实集合
    'antihermitian': {
        'imaginary',            # 是否虚数
    },

    # 用于决定可交换性质的事实集合
    'commutative': {
        'algebraic',            # 是否代数性质
        'complex',              # 是否复数
        'composite',            # 是否合数
        'even',                 # 是否偶数
        'extended_negative',    # 是否扩展负数
        'extended_nonnegative', # 是否扩展非负数
        'extended_nonpositive', # 是否扩展非正数
        'extended_nonzero',     # 是否扩展非零
        'extended_positive',    # 是否扩展正数
        'extended_real',        # 是否扩展实数
        'imaginary',            # 是否虚数
        'integer',              # 是否整数
        'irrational',           # 是否无理数
        'negative',             # 是否负数
        'noninteger',           # 是否非整数
        'nonnegative',          # 是否非负数
        'nonpositive',          # 是否非正数
        'nonzero',              # 是否非零
        'odd',                  # 是否奇数
        'positive',             # 是否正数
        'prime',                # 是否质数
        'rational',             # 是否有理数
        'real',                 # 是否实数
        'transcendental',       # 是否超越数
        'zero',                 # 是否零
    },

    # 用于决定复数性质的事实集合
    'complex': {
        'algebraic',            # 是否代数性质
        'commutative',          # 是否可交换
        'composite',            # 是否合数
        'even',                 # 是否偶数
        'finite',               # 是否有限的
        'imaginary',            # 是否虚数
        'infinite',             # 是否无限的
        'integer',              # 是否整数
        'irrational',           # 是否无理数
        'negative',             # 是否负数
        'nonnegative',          # 是否非负数
        'nonpositive',          # 是否非正数
        'nonzero',              # 是否非零
        'odd',                  # 是否奇数
        'positive',             # 是否正数
        'prime',                # 是否质数
        'rational',             # 是否有理数
        'real',                 # 是否实数
        'transcendental',       # 是否超越数
        'zero',                 # 是否零
    },

    # 用于决定合数性质的事实集合
    'composite': {
        'algebraic',            # 是否代数性质
        'commutative',          # 是否可交换
        'complex',              # 是否复数
        'extended_negative',    # 是否扩展负数
        'extended_nonnegative', # 是否扩展非负数
        'extended_nonpositive', # 是否扩展非正数
        'extended_nonzero',     # 是否扩展非零
        'extended_positive',    # 是否扩展正数
        'extended_real',        # 是否扩展实数
        'finite',               # 是否有限的
        'hermitian',            # 是否厄米数
        'imaginary',            # 是否虚数
        'infinite',             # 是否无限的
        'integer',              # 是否整数
        'irrational',           # 是否无理数
        'negative',             # 是否负数
        'noninteger',           # 是否非整数
        'nonnegative',          # 是否非负数
        'nonpositive',          # 是否非正数
        'nonzero',              # 是否非零
        'positive',             # 是否正数
        'prime',                # 是否质数
        'rational',             # 是否有理数
        'real',                 # 是否实数
        'transcendental',       # 是否超越数
        'zero',                 # 是否零
    },

    # 用于决定偶数性质的事实集合
    'even': {
        'algebraic',            # 是否代数性质
        'commutative',          # 是否可交换
        'complex',              # 是否复数
        'extended_real',        # 是否扩展实数
        'finite',               # 是否有限的
        'hermitian',            # 是否厄米数
        'imaginary',            # 是否虚数
        'infinite',             # 是否无限的
        'integer',              # 是否整数
        'irrational',           # 是否无理数
        'noninteger',           # 是否非整数
        'odd',                  # 是否奇数
        'rational',             # 是否有理数
        'real',                 # 是否实数
        'transcendental',       # 是否超越数
        'zero',                 # 是否零
    },

    # 用于决定扩展负数性质的事实集合
    # 'extended_negative'的定义，包含一组特性，用于描述一个数值概念
    'extended_negative': {
        'commutative',           # 可交换的数值特性
        'composite',             # 复合数的特性
        'extended_nonnegative',  # 扩展非负数的特性
        'extended_nonpositive',  # 扩展非正数的特性
        'extended_nonzero',      # 扩展非零数的特性
        'extended_positive',     # 扩展正数的特性
        'extended_real',         # 扩展实数的特性
        'imaginary',             # 虚数的特性
        'negative',              # 负数的特性
        'nonnegative',           # 非负数的特性
        'positive',              # 正数的特性
        'prime',                 # 素数的特性
        'zero',                  # 零的特性
    },

    # 'extended_nonnegative'的定义，包含一组特性，用于描述一个数值概念
    # 这些特性有助于确定扩展非负数的值
    'extended_nonnegative': {
        'commutative',           # 可交换的数值特性
        'composite',             # 复合数的特性
        'extended_negative',     # 扩展负数的特性
        'extended_positive',     # 扩展正数的特性
        'extended_real',         # 扩展实数的特性
        'imaginary',             # 虚数的特性
        'negative',              # 负数的特性
        'nonnegative',           # 非负数的特性
        'positive',              # 正数的特性
        'prime',                 # 素数的特性
        'zero',                  # 零的特性
    },

    # 'extended_nonpositive'的定义，包含一组特性，用于描述一个数值概念
    # 这些特性有助于确定扩展非正数的值
    'extended_nonpositive': {
        'commutative',           # 可交换的数值特性
        'composite',             # 复合数的特性
        'extended_negative',     # 扩展负数的特性
        'extended_positive',     # 扩展正数的特性
        'extended_real',         # 扩展实数的特性
        'imaginary',             # 虚数的特性
        'negative',              # 负数的特性
        'nonpositive',           # 非正数的特性
        'positive',              # 正数的特性
        'prime',                 # 素数的特性
        'zero',                  # 零的特性
    },

    # 'extended_nonzero'的定义，包含一组特性，用于描述一个数值概念
    # 这些特性有助于确定扩展非零数的值
    'extended_nonzero': {
        'commutative',           # 可交换的数值特性
        'composite',             # 复合数的特性
        'extended_negative',     # 扩展负数的特性
        'extended_positive',     # 扩展正数的特性
        'extended_real',         # 扩展实数的特性
        'imaginary',             # 虚数的特性
        'irrational',            # 无理数的特性
        'negative',              # 负数的特性
        'noninteger',            # 非整数的特性
        'nonzero',               # 非零数的特性
        'odd',                   # 奇数的特性
        'positive',              # 正数的特性
        'prime',                 # 素数的特性
        'zero',                  # 零的特性
    },

    # 'extended_positive'的定义，包含一组特性，用于描述一个数值概念
    # 这些特性有助于确定扩展正数的值
    'extended_positive': {
        'commutative',           # 可交换的数值特性
        'composite',             # 复合数的特性
        'extended_negative',     # 扩展负数的特性
        'extended_nonnegative',  # 扩展非负数的特性
        'extended_nonpositive',  # 扩展非正数的特性
        'extended_nonzero',      # 扩展非零数的特性
        'extended_real',         # 扩展实数的特性
        'imaginary',             # 虚数的特性
        'negative',              # 负数的特性
        'nonpositive',           # 非正数的特性
        'positive',              # 正数的特性
        'prime',                 # 素数的特性
        'zero',                  # 零的特性
    },

    # 'extended_real'的定义，包含一组特性，用于描述一个数值概念
    # 这些特性有助于确定扩展实数的值
    'extended_real': {
        'commutative',           # 可交换的数值特性
        'composite',             # 复合数的特性
        'even',                  # 偶数的特性
        'extended_negative',     # 扩展负数的特性
        'extended_nonnegative',  # 扩展非负数的特性
        'extended_nonpositive',  # 扩展非正数的特性
        'extended_nonzero',      # 扩展非零数的特性
        'extended_positive',     # 扩展正数的特性
        'imaginary',             # 虚数的特性
        'integer',               # 整数的特性
        'irrational',            # 无理数的特性
        'negative',              # 负数的特性
        'noninteger',            # 非整数的特性
        'nonnegative',           # 非负数的特性
        'nonpositive',           # 非正数的特性
        'nonzero',               # 非零数的特性
        'odd',                   # 奇数的特性
        'positive',              # 正数的特性
        'prime',                 # 素数的特性
        'rational',              # 有理数的特性
        'real',                  # 实数的特性
        'zero',                  # 零的特性
    },

    # 'finite'的定义，包含一组特性，用于描述一个数值概念
    # 这些特性有助于确定有限数的值
    'finite': {
        'algebraic',             # 代数数的特性
        'complex',               # 复数的特性
        'composite',             # 复合数的特性
        'even',                  # 偶数的特性
        'imaginary',             # 虚数的特性
        'infinite',              # 无穷大的特性
        'integer',               # 整数的特性
        'irrational',            # 无理数的特性
        'negative',              # 负数的特性
        'nonnegative',           # 非负数的特性
        'nonpositive',           # 非正数的特性
        'nonzero',               # 非零数的特性
        'odd',                   # 奇数的特性
        'positive',              # 正数的特性
        'prime',                 # 素数的特性
        'rational',              # 有理数的特性
        'real',                  # 实数的特性
        'transcendental',        # 超越数的特性
        'zero',                  # 零的特性
    },

    # 'hermitian'的定义，待补充更多信息
    # 定义关于 'hermitian' 的属性集合，表示这些属性的可能取值
    'hermitian': {
        'composite',            # 可能为复合数
        'even',                 # 可能为偶数
        'integer',              # 可能为整数
        'irrational',           # 可能为无理数
        'negative',             # 可能为负数
        'nonnegative',          # 可能为非负数
        'nonpositive',          # 可能为非正数
        'nonzero',              # 可能为非零数
        'odd',                  # 可能为奇数
        'positive',             # 可能为正数
        'prime',                # 可能为质数
        'rational',             # 可能为有理数
        'real',                 # 可能为实数
        'zero',                 # 可能为零
    },

    # 定义关于 'imaginary' 的属性集合，表示这些属性的可能取值
    # 这些属性可能会影响到虚数的值
    'imaginary': {
        'antihermitian',        # 反厄米
        'commutative',          # 可交换
        'complex',              # 复数
        'composite',            # 可能为复合数
        'even',                 # 可能为偶数
        'extended_negative',    # 扩展负数
        'extended_nonnegative', # 扩展非负数
        'extended_nonpositive', # 扩展非正数
        'extended_nonzero',     # 扩展非零数
        'extended_positive',    # 扩展正数
        'extended_real',        # 扩展实数
        'finite',               # 有限
        'infinite',             # 无限
        'integer',              # 可能为整数
        'irrational',           # 可能为无理数
        'negative',             # 可能为负数
        'noninteger',           # 非整数
        'nonnegative',          # 可能为非负数
        'nonpositive',          # 可能为非正数
        'nonzero',              # 可能为非零数
        'odd',                  # 可能为奇数
        'positive',             # 可能为正数
        'prime',                # 可能为质数
        'rational',             # 可能为有理数
        'real',                 # 可能为实数
        'zero',                 # 可能为零
    },

    # 定义关于 'infinite' 的属性集合，表示这些属性的可能取值
    # 这些属性可能会影响到无限的值
    'infinite': {
        'algebraic',            # 代数的
        'complex',              # 复数
        'composite',            # 可能为复合数
        'even',                 # 可能为偶数
        'finite',               # 有限
        'imaginary',            # 可能为虚数
        'integer',              # 可能为整数
        'irrational',           # 可能为无理数
        'negative',             # 可能为负数
        'nonnegative',          # 可能为非负数
        'nonpositive',          # 可能为非正数
        'nonzero',              # 可能为非零数
        'odd',                  # 可能为奇数
        'positive',             # 可能为正数
        'prime',                # 可能为质数
        'rational',             # 可能为有理数
        'real',                 # 可能为实数
        'transcendental',       # 超越的
        'zero',                 # 可能为零
    },

    # 定义关于 'integer' 的属性集合，表示这些属性的可能取值
    # 这些属性可能会影响到整数的值
    'integer': {
        'algebraic',            # 代数的
        'commutative',          # 可交换
        'complex',              # 复数
        'composite',            # 可能为复合数
        'even',                 # 可能为偶数
        'extended_real',        # 扩展实数
        'finite',               # 有限
        'hermitian',            # 厄米
        'imaginary',            # 可能为虚数
        'infinite',             # 可能为无限
        'irrational',           # 可能为无理数
        'noninteger',           # 非整数
        'odd',                  # 可能为奇数
        'prime',                # 可能为质数
        'rational',             # 可能为有理数
        'real',                 # 可能为实数
        'transcendental',       # 超越的
        'zero',                 # 可能为零
    },

    # 定义关于 'irrational' 的属性集合，表示这些属性的可能取值
    # 这些属性可能会影响到无理数的值
    'irrational': {
        'commutative',          # 可交换
        'complex',              # 复数
        'composite',            # 可能为复合数
        'even',                 # 可能为偶数
        'extended_real',        # 扩展实数
        'finite',               # 有限
        'hermitian',            # 厄米
        'imaginary',            # 可能为虚数
        'infinite',             # 可能为无限
        'integer',              # 可能为整数
        'odd',                  # 可能为奇数
        'prime',                # 可能为质数
        'rational',             # 可能为有理数
        'real',                 # 可能为实数
        'zero',                 # 可能为零
    },

    # 定义关于 'negative' 的属性集合，表示这些属性的可能取值
    # 这些属性可能会影响到负数的值
    'negative': {
        'commutative',          # 可交换
        'complex',              # 复数
        'composite',            # 可能为复合数
        'extended_negative',    # 扩展负数
        'extended_nonnegative', # 扩展非负数
        'extended_nonpositive', # 扩展非正数
        'extended_nonzero',     # 扩展非零数
        'extended_positive',    # 扩展正数
        'extended_real',        # 扩展实数
        'finite',               # 有限
        'hermitian',            # 厄米
        'imaginary',            # 可能为虚数
        'infinite',             # 可能为无限
        'nonnegative',          # 可能为非负数
        'nonpositive',          # 可能为非正数
        'nonzero',              # 可能为非零数
        'positive',             # 可能为正数
        'prime',                # 可能为质数
        'real',                 # 可能为实数
        'zero',                 # 可能为零
    },

    # 定义关于 'noninteger' 的属性集合，表示这些属性的可能取值
    # 定义集合，描述“noninteger”数学属性的集合
    'noninteger': {
        'commutative',      # 可交换的
        'composite',        # 复合数
        'even',             # 偶数
        'extended_real',    # 扩展实数
        'imaginary',        # 虚数
        'integer',          # 整数
        'irrational',       # 无理数
        'odd',              # 奇数
        'prime',            # 质数
        'zero',             # 零
    },

    # 描述“nonnegative”数学属性的集合，可能决定非负值的特性
    'nonnegative': {
        'commutative',          # 可交换的
        'complex',              # 复数
        'composite',            # 复合数
        'extended_negative',    # 扩展负数
        'extended_nonnegative', # 扩展非负数
        'extended_real',        # 扩展实数
        'finite',               # 有限的
        'hermitian',            # Hermitian 的
        'imaginary',            # 虚数
        'infinite',             # 无限的
        'negative',             # 负数
        'positive',             # 正数
        'prime',                # 质数
        'real',                 # 实数
        'zero',                 # 零
    },

    # 描述“nonpositive”数学属性的集合，可能决定非正值的特性
    'nonpositive': {
        'commutative',          # 可交换的
        'complex',              # 复数
        'composite',            # 复合数
        'extended_nonpositive', # 扩展非正数
        'extended_positive',    # 扩展正数
        'extended_real',        # 扩展实数
        'finite',               # 有限的
        'hermitian',            # Hermitian 的
        'imaginary',            # 虚数
        'infinite',             # 无限的
        'negative',             # 负数
        'positive',             # 正数
        'prime',                # 质数
        'real',                 # 实数
        'zero',                 # 零
    },

    # 描述“nonzero”数学属性的集合，可能决定非零值的特性
    'nonzero': {
        'commutative',      # 可交换的
        'complex',          # 复数
        'composite',        # 复合数
        'extended_nonzero', # 扩展非零数
        'extended_real',    # 扩展实数
        'finite',           # 有限的
        'hermitian',        # Hermitian 的
        'imaginary',        # 虚数
        'infinite',         # 无限的
        'irrational',       # 无理数
        'negative',         # 负数
        'odd',              # 奇数
        'positive',         # 正数
        'prime',            # 质数
        'real',             # 实数
        'zero',             # 零
    },

    # 描述“odd”数学属性的集合，可能决定奇数的特性
    'odd': {
        'algebraic',        # 代数的
        'commutative',      # 可交换的
        'complex',          # 复数
        'even',             # 偶数
        'extended_real',    # 扩展实数
        'finite',           # 有限的
        'hermitian',        # Hermitian 的
        'imaginary',        # 虚数
        'infinite',         # 无限的
        'integer',          # 整数
        'irrational',       # 无理数
        'noninteger',       # 非整数
        'rational',         # 有理数
        'real',             # 实数
        'transcendental',   # 超越数
        'zero',             # 零
    },

    # 描述“positive”数学属性的集合，可能决定正数的特性
    'positive': {
        'commutative',              # 可交换的
        'complex',                  # 复数
        'composite',                # 复合数
        'extended_negative',        # 扩展负数
        'extended_nonnegative',     # 扩展非负数
        'extended_nonpositive',     # 扩展非正数
        'extended_nonzero',         # 扩展非零数
        'extended_positive',        # 扩展正数
        'extended_real',            # 扩展实数
        'finite',                   # 有限的
        'hermitian',                # Hermitian 的
        'imaginary',                # 虚数
        'infinite',                 # 无限的
        'negative',                 # 负数
        'nonnegative',              # 非负数
        'nonpositive',              # 非正数
        'nonzero',                  # 非零数
        'prime',                    # 质数
        'real',                     # 实数
        'zero',                     # 零
    },

    # 描述“prime”数学属性的集合
    'prime': {
        # 此处还未提供
    },
    # 定义一个包含数学术语集合的字典，每个键对应一个术语，值是一个包含能够确定该术语值的其他术语的集合。
    
    # facts that could determine the value of prime
    'prime': {
        'algebraic',               # 代数的
        'commutative',             # 可交换的
        'complex',                 # 复数的
        'composite',               # 复合数的
        'extended_negative',       # 扩展负数的
        'extended_nonnegative',    # 扩展非负数的
        'extended_nonpositive',    # 扩展非正数的
        'extended_nonzero',        # 扩展非零数的
        'extended_positive',       # 扩展正数的
        'extended_real',           # 扩展实数的
        'finite',                  # 有限的
        'hermitian',               # Hermitian的
        'imaginary',               # 虚数的
        'infinite',                # 无限的
        'integer',                 # 整数的
        'irrational',              # 无理数的
        'negative',                # 负数的
        'noninteger',              # 非整数的
        'nonnegative',             # 非负数的
        'nonpositive',             # 非正数的
        'nonzero',                 # 非零的
        'positive',                # 正数的
        'rational',                # 有理数的
        'real',                    # 实数的
        'transcendental',          # 超越数的
        'zero',                    # 零的
    },
    
    # facts that could determine the value of rational
    'rational': {
        'algebraic',               # 代数的
        'commutative',             # 可交换的
        'complex',                 # 复数的
        'composite',               # 复合数的
        'even',                    # 偶数的
        'extended_real',           # 扩展实数的
        'finite',                  # 有限的
        'hermitian',               # Hermitian的
        'imaginary',               # 虚数的
        'infinite',                # 无限的
        'integer',                 # 整数的
        'irrational',              # 无理数的
        'odd',                     # 奇数的
        'prime',                   # 素数的
        'real',                    # 实数的
        'transcendental',          # 超越数的
        'zero',                    # 零的
    },
    
    # facts that could determine the value of real
    'real': {
        'commutative',             # 可交换的
        'complex',                 # 复数的
        'composite',               # 复合数的
        'even',                    # 偶数的
        'extended_real',           # 扩展实数的
        'finite',                  # 有限的
        'hermitian',               # Hermitian的
        'imaginary',               # 虚数的
        'infinite',                # 无限的
        'integer',                 # 整数的
        'irrational',              # 无理数的
        'negative',                # 负数的
        'nonnegative',             # 非负数的
        'nonpositive',             # 非正数的
        'nonzero',                 # 非零的
        'odd',                     # 奇数的
        'positive',                # 正数的
        'prime',                   # 素数的
        'rational',                # 有理数的
        'zero',                    # 零的
    },
    
    # facts that could determine the value of transcendental
    'transcendental': {
        'algebraic',               # 代数的
        'commutative',             # 可交换的
        'complex',                 # 复数的
        'composite',               # 复合数的
        'even',                    # 偶数的
        'finite',                  # 有限的
        'infinite',                # 无限的
        'integer',                 # 整数的
        'odd',                     # 奇数的
        'prime',                   # 素数的
        'rational',                # 有理数的
        'zero',                    # 零的
    },
    
    # facts that could determine the value of zero
    'zero': {
        'algebraic',               # 代数的
        'commutative',             # 可交换的
        'complex',                 # 复数的
        'composite',               # 复合数的
        'even',                    # 偶数的
        'extended_negative',       # 扩展负数的
        'extended_nonnegative',    # 扩展非负数的
        'extended_nonpositive',    # 扩展非正数的
        'extended_nonzero',        # 扩展非零数的
        'extended_positive',       # 扩展正数的
        'extended_real',           # 扩展实数的
        'finite',                  # 有限的
        'hermitian',               # Hermitian的
        'imaginary',               # 虚数的
        'infinite',                # 无限的
        'integer',                 # 整数的
        'irrational',              # 无理数的
        'negative',                # 负数的
        'noninteger',              # 非整数的
        'nonnegative',             # 非负数的
        'nonpositive',             # 非正数的
        'nonzero',                 # 非零的
        'odd',                     # 奇数的
        'positive',                # 正数的
        'prime',                   # 素数的
        'rational',                # 有理数的
        'real',                    # 实数的
        'transcendental',          # 超越数的
    },
# } # prereq
# 上面这一行可能是注释符号的错位，看起来应该是被误放在这里的，应该是不完整的注释或代码的一部分。

# Note: the order of the beta rules is used in the beta_triggers
# beta_rules 是一个列表，包含了一系列的规则和对应的结论，用于逻辑推断。

beta_rules = [

    # Rules implying composite = True
    # 当 ('even', True)、('positive', True) 和 ('prime', False) 共同存在时，推断 composite = True
    ({('even', True), ('positive', True), ('prime', False)},
        ('composite', True)),

    # Rules implying even = False
    # 当 ('composite', False)、('positive', True) 和 ('prime', False) 共同存在时，推断 even = False
    ({('composite', False), ('positive', True), ('prime', False)},
        ('even', False)),

    # Rules implying even = True
    # 当 ('integer', True) 和 ('odd', False) 共同存在时，推断 even = True
    ({('integer', True), ('odd', False)},
        ('even', True)),

    # Rules implying extended_negative = True
    # 当 ('extended_positive', False)、('extended_real', True) 和 ('zero', False) 共同存在时，推断 extended_negative = True
    ({('extended_positive', False), ('extended_real', True), ('zero', False)},
        ('extended_negative', True)),
    # 当 ('extended_nonpositive', True) 和 ('extended_nonzero', True) 共同存在时，推断 extended_negative = True
    ({('extended_nonpositive', True), ('extended_nonzero', True)},
        ('extended_negative', True)),

    # Rules implying extended_nonnegative = True
    # 当 ('extended_negative', False) 和 ('extended_real', True) 共同存在时，推断 extended_nonnegative = True
    ({('extended_negative', False), ('extended_real', True)},
        ('extended_nonnegative', True)),

    # Rules implying extended_nonpositive = True
    # 当 ('extended_positive', False) 和 ('extended_real', True) 共同存在时，推断 extended_nonpositive = True
    ({('extended_positive', False), ('extended_real', True)},
        ('extended_nonpositive', True)),

    # Rules implying extended_nonzero = True
    # 当 ('extended_real', True) 和 ('zero', False) 共同存在时，推断 extended_nonzero = True
    ({('extended_real', True), ('zero', False)},
        ('extended_nonzero', True)),

    # Rules implying extended_positive = True
    # 当 ('extended_negative', False)、('extended_real', True) 和 ('zero', False) 共同存在时，推断 extended_positive = True
    ({('extended_negative', False), ('extended_real', True), ('zero', False)},
        ('extended_positive', True)),
    # 当 ('extended_nonnegative', True) 和 ('extended_nonzero', True) 共同存在时，推断 extended_positive = True
    ({('extended_nonnegative', True), ('extended_nonzero', True)},
        ('extended_positive', True)),

    # Rules implying extended_real = False
    # 当 ('infinite', False) 和 ('real', False) 共同存在时，推断 extended_real = False
    ({('infinite', False), ('real', False)},
        ('extended_real', False)),
    # 当 ('extended_negative', False)、('extended_positive', False) 和 ('zero', False) 共同存在时，推断 extended_real = False
    ({('extended_negative', False), ('extended_positive', False), ('zero', False)},
        ('extended_real', False)),

    # Rules implying infinite = True
    # 当 ('extended_real', True) 和 ('real', False) 共同存在时，推断 infinite = True
    ({('extended_real', True), ('real', False)},
        ('infinite', True)),

    # Rules implying irrational = True
    # 当 ('rational', False) 和 ('real', True) 共同存在时，推断 irrational = True
    ({('rational', False), ('real', True)},
        ('irrational', True)),

    # Rules implying negative = True
    # 当 ('positive', False)、('real', True) 和 ('zero', False) 共同存在时，推断 negative = True
    ({('positive', False), ('real', True), ('zero', False)},
        ('negative', True)),
    # 当 ('nonpositive', True) 和 ('nonzero', True) 共同存在时，推断 negative = True
    ({('nonpositive', True), ('nonzero', True)},
        ('negative', True)),
    # 当 ('extended_negative', True) 和 ('finite', True) 共同存在时，推断 negative = True
    ({('extended_negative', True), ('finite', True)},
        ('negative', True)),

    # Rules implying noninteger = True
    # 当 ('extended_real', True) 和 ('integer', False) 共同存在时，推断 noninteger = True
    ({('extended_real', True), ('integer', False)},
        ('noninteger', True)),

    # Rules implying nonnegative = True
    # 当 ('negative', False) 和 ('real', True) 共同存在时，推断 nonnegative = True
    ({('negative', False), ('real', True)},
        ('nonnegative', True)),
    # 当 ('extended_nonnegative', True) 和 ('finite', True) 共同存在时，推断 nonnegative = True
    ({('extended_nonnegative', True), ('finite', True)},
        ('nonnegative', True)),

    # Rules implying nonpositive = True
    # 当 ('positive', False) 和 ('real', True) 共同存在时，推断 nonpositive = True
    ({('positive', False), ('real', True)},
        ('nonpositive', True)),
    # 当 ('extended_nonpositive', True) 和 ('finite', True) 共同存在时，推断 nonpositive = True
    ({('extended_nonpositive', True), ('finite', True)},
        ('nonpositive', True)),

    # Rules implying nonzero = True
    # 当 ('extended_nonzero', True) 和 ('finite', True) 共同存在时，推断 nonzero = True
    ({('extended_nonzero', True), ('finite', True)},
        ('nonzero', True)),

    # Rules implying odd = True
    # 当 ('even', False) 和 ('integer', True) 共同存在时，推断 odd = True
    ({('even', False), ('integer', True)},
        ('odd', True)),

    # Rules implying positive = False
    # 集合表示一条规则，包含若干条件，每个条件是一个元组，形如 (属性名, 布尔值)
    ({('composite', False), ('even', True), ('prime', False)},
        ('positive', False)),

    # Rules implying positive = True
    # 包含若干条件，每个条件是一个元组，形如 (属性名, 布尔值)，满足条件时设置 positive = True
    ({('negative', False), ('real', True), ('zero', False)},
        ('positive', True)),
    ({('nonnegative', True), ('nonzero', True)},
        ('positive', True)),
    ({('extended_positive', True), ('finite', True)},
        ('positive', True)),

    # Rules implying prime = True
    # 包含若干条件，每个条件是一个元组，形如 (属性名, 布尔值)，满足条件时设置 prime = True
    ({('composite', False), ('even', True), ('positive', True)},
        ('prime', True)),

    # Rules implying real = False
    # 包含若干条件，每个条件是一个元组，形如 (属性名, 布尔值)，满足条件时设置 real = False
    ({('negative', False), ('positive', False), ('zero', False)},
        ('real', False)),

    # Rules implying real = True
    # 包含若干条件，每个条件是一个元组，形如 (属性名, 布尔值)，满足条件时设置 real = True
    ({('extended_real', True), ('infinite', False)},
        ('real', True)),
    ({('extended_real', True), ('finite', True)},
        ('real', True)),

    # Rules implying transcendental = True
    # 包含若干条件，每个条件是一个元组，形如 (属性名, 布尔值)，满足条件时设置 transcendental = True
    ({('algebraic', False), ('complex', True)},
        ('transcendental', True)),

    # Rules implying zero = True
    # 包含若干条件，每个条件是一个元组，形如 (属性名, 布尔值)，满足条件时设置 zero = True
    ({('extended_negative', False), ('extended_positive', False), ('extended_real', True)},
        ('zero', True)),
    ({('negative', False), ('positive', False), ('real', True)},
        ('zero', True)),
    ({('extended_nonnegative', True), ('extended_nonpositive', True)},
        ('zero', True)),
    ({('nonnegative', True), ('nonpositive', True)},
        ('zero', True)),
# 定义 beta_triggers 变量，存储了各种规则的触发条件及其对应的触发器列表
beta_triggers = {
    # 代数型且非复数情况下的触发器列表
    ('algebraic', False): [32, 11, 3, 8, 29, 14, 25, 13, 17, 7],
    # 代数型且复数情况下的触发器列表
    ('algebraic', True): [10, 30, 31, 27, 16, 21, 19, 22],
    # 反共轭的触发器列表为空
    ('antihermitian', False): [],
    # 可交换的触发器列表为空
    ('commutative', False): [],
    # 复数情况下的触发器列表
    ('complex', False): [10, 12, 11, 3, 8, 17, 7],
    # 复数情况下的触发器列表
    ('complex', True): [32, 10, 30, 31, 27, 16, 21, 19, 22],
    # 复合数情况下的触发器列表
    ('composite', False): [1, 28, 24],
    # 复合数情况下的触发器列表
    ('composite', True): [23, 2],
    # 偶数情况下的触发器列表
    ('even', False): [23, 11, 3, 8, 29, 14, 25, 7],
    # 偶数情况下的触发器列表
    ('even', True): [3, 33, 8, 6, 5, 14, 34, 25, 20, 18, 27, 16, 21, 19, 22, 0, 28, 24, 7],
    # 扩展负数情况下的触发器列表
    ('extended_negative', False): [11, 33, 8, 5, 29, 34, 25, 18],
    # 扩展负数情况下的触发器列表
    ('extended_negative', True): [30, 12, 31, 29, 14, 20, 16, 21, 22, 17],
    # 扩展非负数情况下的触发器列表
    ('extended_nonnegative', False): [11, 3, 6, 29, 14, 20, 7],
    # 扩展非负数情况下的触发器列表
    ('extended_nonnegative', True): [30, 12, 31, 33, 8, 9, 6, 29, 34, 25, 18, 19, 35, 17, 7],
    # 扩展非正数情况下的触发器列表
    ('extended_nonpositive', False): [11, 8, 5, 29, 25, 18, 7],
    # 扩展非正数情况下的触发器列表
    ('extended_nonpositive', True): [30, 12, 31, 3, 33, 4, 5, 29, 14, 34, 20, 21, 35, 17, 7],
    # 扩展非零情况下的触发器列表
    ('extended_nonzero', False): [11, 33, 6, 5, 29, 34, 20, 18],
    # 扩展非零情况下的触发器列表
    ('extended_nonzero', True): [30, 12, 31, 3, 8, 4, 9, 6, 5, 29, 14, 25, 22, 17],
    # 扩展正数情况下的触发器列表
    ('extended_positive', False): [11, 3, 33, 6, 29, 14, 34, 20],
    # 扩展正数情况下的触发器列表
    ('extended_positive', True): [30, 12, 31, 29, 25, 18, 27, 19, 22, 17],
    # 扩展实数情况下的触发器列表为空
    ('extended_real', False): [],
    # 扩展实数情况下的触发器列表
    ('extended_real', True): [30, 12, 31, 3, 33, 8, 6, 5, 17, 7],
    # 有限情况下的触发器列表
    ('finite', False): [11, 3, 8, 17, 7],
    # 有限情况下的触发器列表
    ('finite', True): [10, 30, 31, 27, 16, 21, 19, 22],
    # 共轭数情况下的触发器列表
    ('hermitian', False): [10, 12, 11, 3, 8, 17, 7],
    # 虚数情况下的触发器列表
    ('imaginary', True): [32],
    # 无限情况下的触发器列表
    ('infinite', False): [10, 30, 31, 27, 16, 21, 19, 22],
    # 无限情况下的触发器列表
    ('infinite', True): [11, 3, 8, 17, 7],
    # 整数情况下的触发器列表
    ('integer', False): [11, 3, 8, 29, 14, 25, 17, 7],
    # 整数情况下的触发器列表
    ('integer', True): [23, 2, 3, 33, 8, 6, 5, 14, 34, 25, 20, 18, 27, 16, 21, 19, 22, 7],
    # 无理数情况下的触发器列表
    ('irrational', True): [32, 3, 8, 4, 9, 6, 5, 14, 25, 15, 26, 20, 18, 27, 16, 21, 19],
    # 负数情况下的触发器列表
    ('negative', False): [29, 34, 25, 18],
    # 负数情况下的触发器列表
    ('negative', True): [32, 13, 17],
    # 非整数情况下的触发器列表
    ('noninteger', True): [30, 12, 31, 3, 8, 4, 9, 6, 5, 29, 14, 25, 22],
    # 非负数情况下的触发器列表
    ('nonnegative', False): [11, 3, 8, 29, 14, 20, 7],
    # 非负数情况下的触发器列表
    ('nonnegative', True): [32, 33, 8, 9, 6, 34, 25, 26, 20, 27, 21, 22, 35, 36, 13, 17, 7],
    # 非正数情况下的触发器列表
    ('nonpositive', False): [11, 3, 8, 29, 25, 18, 7],
    # 非正数情况下的触发器列表
    ('nonpositive', True): [32, 3, 33, 4, 5, 14, 34, 15, 18, 16, 19, 22, 35, 36, 13, 17, 7],
    # 非零情况下的触发器列表
    ('nonzero', False): [29, 34, 20, 18],
    # 非零情况下的触发器列表
    ('nonzero', True): [32, 3, 8, 4, 9, 6, 5, 14, 25, 15
    # 定义一个包含元组键和布尔值的字典。每个键代表一个术语和一个布尔值，值是一个整数列表，代表某些关联数字的集合。
    ('real', True): [32, 3, 33, 8, 6, 5, 14, 34, 25, 20, 18, 27, 16, 21, 19, 22, 13, 17, 7],
    ('transcendental', True): [10, 30, 31, 11, 3, 8, 29, 14, 25, 27, 16, 21, 19, 22, 13, 17, 7],
    ('zero', False): [11, 3, 8, 29, 14, 25, 7],
    # 定义一个特殊情况的键 ('zero', True)，它的值是一个空列表，表示没有与此键相关联的数字。
    ('zero', True): [],
} # 结束 beta_triggers 对象的定义


generated_assumptions = {'defined_facts': defined_facts, 'full_implications': full_implications,
               'prereq': prereq, 'beta_rules': beta_rules, 'beta_triggers': beta_triggers}
# 创建一个名为 generated_assumptions 的字典，包含了以下键值对：
# - 'defined_facts': 变量 defined_facts 的值
# - 'full_implications': 变量 full_implications 的值
# - 'prereq': 变量 prereq 的值
# - 'beta_rules': 变量 beta_rules 的值
# - 'beta_triggers': 变量 beta_triggers 的值，即上文定义的 beta_triggers 对象
```