# `D:\src\scipysrc\sympy\sympy\crypto\crypto.py`

```
# 导入必要的模块和函数
from string import whitespace, ascii_uppercase as uppercase, printable
from functools import reduce
import warnings

from itertools import cycle  # 导入 itertools 模块中的 cycle 函数

from sympy.external.gmpy import GROUND_TYPES  # 导入 sympy 外部 gmpy 中的 GROUND_TYPES
from sympy.core import Symbol  # 导入 sympy 核心模块中的 Symbol 类
from sympy.core.numbers import Rational  # 导入 sympy 核心模块中的 Rational 类
from sympy.core.random import _randrange, _randint  # 导入 sympy 核心模块中的随机数生成函数
from sympy.external.gmpy import gcd, invert  # 导入 sympy 外部 gmpy 中的 gcd 和 invert 函数
from sympy.functions.combinatorial.numbers import (totient as _euler,  # 导入 sympy 中的组合数学函数
                                                   reduced_totient as _carmichael)
from sympy.matrices import Matrix  # 导入 sympy 中的矩阵模块中的 Matrix 类
from sympy.ntheory import isprime, primitive_root, factorint  # 导入 sympy 中的数论模块中的函数
from sympy.ntheory.generate import nextprime  # 导入 sympy 中的数论模块中的 nextprime 函数
from sympy.ntheory.modular import crt  # 导入 sympy 中的模运算模块中的 crt 函数
from sympy.polys.domains import FF  # 导入 sympy 中的多项式模块中的 FF 类
from sympy.polys.polytools import Poly  # 导入 sympy 中的多项式工具模块中的 Poly 类
from sympy.utilities.misc import as_int, filldedent, translate  # 导入 sympy 中的实用工具模块中的函数
from sympy.utilities.iterables import uniq, multiset  # 导入 sympy 中的可迭代工具模块中的函数
from sympy.utilities.decorator import doctest_depends_on  # 导入 sympy 中的装饰器模块中的 doctest_depends_on 函数

# 如果使用 flint 类型的 GROUND_TYPES，跳过 lfsr_sequence 的 doctest 测试
if GROUND_TYPES == 'flint':
    __doctest_skip__ = ['lfsr_sequence']


class NonInvertibleCipherWarning(RuntimeWarning):
    """如果密码不可逆，引发的警告。"""
    def __init__(self, msg):
        self.fullMessage = msg

    def __str__(self):
        return '\n\t' + self.fullMessage

    def warn(self, stacklevel=3):
        """发出警告。"""
        warnings.warn(self, stacklevel=stacklevel)


def AZ(s=None):
    """返回字符串 ``s`` 中的大写字母。如果传入多个字符串，将每个字符串处理后返回大写字母的列表。

    Examples
    ========

    >>> from sympy.crypto.crypto import AZ
    >>> AZ('Hello, world!')
    'HELLOWORLD'
    >>> AZ('Hello, world!'.split())
    ['HELLO', 'WORLD']

    See Also
    ========

    check_and_join

    """
    if not s:
        return uppercase
    t = isinstance(s, str)
    if t:
        s = [s]
    rv = [check_and_join(i.upper().split(), uppercase, filter=True)
        for i in s]
    if t:
        return rv[0]
    return rv

bifid5 = AZ().replace('J', '')  # 生成一个不包含 'J' 的大写字母序列
bifid6 = AZ() + '0123456789'  # 生成包含大写字母和数字的序列
bifid10 = printable  # 生成包含所有可打印字符的序列


def padded_key(key, symbols):
    """返回由 ``symbols`` 中的不同字符构成的字符串，其中 ``key`` 中的字符优先出现。
    如果
    a) ``symbols`` 中有重复字符，或者
    b) ``key`` 中有不在 ``symbols`` 中的字符，
    则引发 ValueError 错误。

    Examples
    ========

    >>> from sympy.crypto.crypto import padded_key
    >>> padded_key('PUPPY', 'OPQRSTUVWXY')
    'PUYOQRSTVWX'
    >>> padded_key('RSA', 'ARTIST')

    """
    Traceback (most recent call last):
    ...
    ValueError: duplicate characters in symbols: T

"""
syms = list(uniq(symbols))
# 将 symbols 中的元素去重并转换为列表
if len(syms) != len(symbols):
    # 如果去重后的列表长度与 symbols 长度不一致，说明存在重复字符
    extra = ''.join(sorted({
        i for i in symbols if symbols.count(i) > 1}))
    # 找出所有重复的字符并按字母顺序排序
    raise ValueError('duplicate characters in symbols: %s' % extra)
extra = set(key) - set(syms)
# 计算出在 key 中但不在 symbols 中的字符集合
if extra:
    # 如果存在这样的字符，则抛出异常
    raise ValueError(
        'characters in key but not symbols: %s' % ''.join(
        sorted(extra)))
key0 = ''.join(list(uniq(key)))
# 将 key 中的字符去重并拼接成字符串
# 从 syms 中移除 key0 中的字符
return key0 + translate(''.join(syms), None, key0)
# 返回经过处理后的字符串
# 将字符串或字符串列表``phrase``连接起来，并在给定``symbols``的情况下，如果``phrase``中有任何字符不在``symbols``中，则引发错误。

def check_and_join(phrase, symbols=None, filter=None):
    # 将``phrase``连接成一个字符串
    rv = ''.join(''.join(phrase))
    # 如果提供了``symbols``，则检查``rv``中的字符是否都在``symbols``中
    if symbols is not None:
        # 将``symbols``转换为字符串
        symbols = check_and_join(symbols)
        # 找到``rv``中不在``symbols``中的字符
        missing = ''.join(sorted(set(rv) - set(symbols)))
        # 如果有缺失的字符
        if missing:
            # 如果不需要过滤，则引发 ValueError 错误
            if not filter:
                raise ValueError(
                    'characters in phrase but not symbols: "%s"' % missing)
            # 否则使用``missing``对``rv``进行过滤
            rv = translate(rv, None, missing)
    return rv


# 检查并准备消息``msg``、密钥``key``和字母表``alp``，如果``alp``为空，则使用默认字母表``AZ()``，并对``msg``和``key``进行相应的处理
def _prep(msg, key, alp, default=None):
    if not alp:
        if not default:
            alp = AZ()  # 使用默认字母表``AZ()``
            msg = AZ(msg)  # 将``msg``转换为``AZ``字母表中的内容
            key = AZ(key)  # 将``key``转换为``AZ``字母表中的内容
        else:
            alp = default
    else:
        alp = ''.join(alp)  # 将``alp``连接成一个字符串
    # 检查并连接``key``，确保它只包含``alp``中的字符
    key = check_and_join(key, alp, filter=True)
    # 检查并连接``msg``，确保它只包含``alp``中的字符
    msg = check_and_join(msg, alp, filter=True)
    return msg, key, alp


# 返回``range(n)``列表中元素向左移动``k``个位置后的结果列表
def cycle_list(k, n):
    k = k % n  # 计算``k``对``n``的模
    return list(range(k, n)) + list(range(k))


# 执行移位密码加密过程，返回密文
def encipher_shift(msg, key, symbols=None):
    # 返回``msg``的移位密码加密后的结果
    """
    Performs shift cipher encryption on plaintext msg, and returns the
    ciphertext.

    Parameters
    ==========

    key : int
        The secret key.

    msg : str
        Plaintext of upper-case letters.

    Returns
    =======

    str
        Ciphertext of upper-case letters.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_shift, decipher_shift
    >>> msg = "GONAVYBEATARMY"
    >>> ct = encipher_shift(msg, 1); ct
    'HPOBWZCFBUBSNZ'

    To decipher the shifted text, change the sign of the key:

    >>> encipher_shift(ct, -1)
    'GONAVYBEATARMY'

    There is also a convenience function that does this with the
    original key:

    >>> decipher_shift(ct, 1)
    'GONAVYBEATARMY'

    Notes
    =====
    """
    ```
    """
    # ALGORITHM:
    
    # STEPS:
    #     0. Number the letters of the alphabet from 0, ..., N
    #     1. Compute from the string ``msg`` a list ``L1`` of
    #        corresponding integers.
    #     2. Compute from the list ``L1`` a new list ``L2``, given by
    #        adding ``(k mod 26)`` to each element in ``L1``.
    #     3. Compute from the list ``L2`` a string ``ct`` of
    #        corresponding letters.
    #
    # The shift cipher is also called the Caesar cipher, after
    # Julius Caesar, who, according to Suetonius, used it with a
    # shift of three to protect messages of military significance.
    # Caesar's nephew Augustus reportedly used a similar cipher, but
    # with a right shift of 1.
    
    # References
    # ==========
    #
    # .. [1] https://en.wikipedia.org/wiki/Caesar_cipher
    # .. [2] https://mathworld.wolfram.com/CaesarsMethod.html
    #
    # See Also
    # ========
    #
    # decipher_shift
    
    Compute a Caesar shift cipher encryption of the given message.
    """
    # Prepare the message, remove symbols, and get the alphabet
    msg, _, A = _prep(msg, '', symbols)
    # Calculate the effective shift based on the key and alphabet size
    shift = len(A) - key % len(A)
    # Create the shifted alphabet based on the calculated shift
    key = A[shift:] + A[:shift]
    # Translate the message using the generated key and alphabet
    return translate(msg, key, A)
def decipher_shift(msg, key, symbols=None):
    """
    Return the text by shifting the characters of ``msg`` to the
    left by the amount given by ``key``.
    
    Examples
    ========
    
    >>> from sympy.crypto.crypto import encipher_shift, decipher_shift
    >>> msg = "GONAVYBEATARMY"
    >>> ct = encipher_shift(msg, 1); ct
    'HPOBWZCFBUBSNZ'
    
    To decipher the shifted text, change the sign of the key:
    
    >>> encipher_shift(ct, -1)
    'GONAVYBEATARMY'
    
    Or use this function with the original key:
    
    >>> decipher_shift(ct, 1)
    'GONAVYBEATARMY'
    
    """
    # 使用 encipher_shift 函数对消息 msg 进行解密，通过将 key 取反实现左移
    return encipher_shift(msg, -key, symbols)

def encipher_rot13(msg, symbols=None):
    """
    Performs the ROT13 encryption on a given plaintext ``msg``.
    
    Explanation
    ===========
    
    ROT13 is a substitution cipher which substitutes each letter
    in the plaintext message for the letter furthest away from it
    in the English alphabet.
    
    Equivalently, it is just a Caeser (shift) cipher with a shift
    key of 13 (midway point of the alphabet).
    
    References
    ==========
    
    .. [1] https://en.wikipedia.org/wiki/ROT13
    
    See Also
    ========
    
    decipher_rot13
    encipher_shift
    
    """
    # 使用 encipher_shift 函数进行 ROT13 加密，将 key 设为 13
    return encipher_shift(msg, 13, symbols)

def decipher_rot13(msg, symbols=None):
    """
    Performs the ROT13 decryption on a given plaintext ``msg``.
    
    Explanation
    ============
    
    ``decipher_rot13`` is equivalent to ``encipher_rot13`` as both
    ``decipher_shift`` with a key of 13 and ``encipher_shift`` key with a
    key of 13 will return the same results. Nonetheless,
    ``decipher_rot13`` has nonetheless been explicitly defined here for
    consistency.
    
    Examples
    ========
    
    >>> from sympy.crypto.crypto import encipher_rot13, decipher_rot13
    >>> msg = 'GONAVYBEATARMY'
    >>> ciphertext = encipher_rot13(msg);ciphertext
    'TBANILORNGNEZL'
    >>> decipher_rot13(ciphertext)
    'GONAVYBEATARMY'
    >>> encipher_rot13(msg) == decipher_rot13(msg)
    True
    >>> msg == decipher_rot13(ciphertext)
    True
    
    """
    # 使用 decipher_shift 函数进行 ROT13 解密，将 key 设为 13
    return decipher_shift(msg, 13, symbols)

######## affine cipher examples ############

def encipher_affine(msg, key, symbols=None, _inverse=False):
    r"""
    Performs the affine cipher encryption on plaintext ``msg``, and
    returns the ciphertext.
    
    Explanation
    ===========
    
    Encryption is based on the map `x \rightarrow ax+b` (mod `N`)
    where ``N`` is the number of characters in the alphabet.
    Decryption is based on the map `x \rightarrow cx+d` (mod `N`),
    where `c = a^{-1}` (mod `N`) and `d = -a^{-1}b` (mod `N`).
    In particular, for the map to be invertible, we need
    `\mathrm{gcd}(a, N) = 1` and an error will be raised if this is
    not true.
    
    Parameters
    ==========
    
    msg : str
        Characters that appear in ``symbols``.
    
    a, b : int, int
        A pair integers, with ``gcd(a, N) = 1`` (the secret key).
    
    """
    # 未完成的函数定义，实际使用时会进行加密
    # 解密消息 `msg`，使用仿射密码算法，返回加密后的密文 `ct`
    msg, _, A = _prep(msg, '', symbols)
    # 计算字符集 `A` 的长度 `N`
    N = len(A)
    # 解析密钥 `key` 中的参数 `a` 和 `b`
    a, b = key
    # 确保 `a` 与 `N` 的最大公约数为 1，以确保 `a` 有逆元素
    assert gcd(a, N) == 1
    # 如果 `_inverse` 为真，则计算逆元 `c`，并更新 `a` 和 `b`
    if _inverse:
        c = invert(a, N)
        d = -b*c
        a, b = c, d
    # 使用仿射密码算法生成新的字符集 `B`，其中 `B[i] = A[(a*i + b) % N]`
    B = ''.join([A[(a*i + b) % N] for i in range(N)])
    # 调用 `translate` 函数将原始消息 `msg` 根据字符集 `A` 和 `B` 进行转换，返回密文 `ct`
    return translate(msg, A, B)
# 返回通过用新密钥 `x \rightarrow cx+d` (mod `N`) 重新加密的解密文本，其中 `c = a^{-1}` (mod `N`)，`d = -a^{-1}b` (mod `N`)。
# 这里 `msg` 是要解密的消息，`key` 是原始加密密钥，`symbols` 是可选的符号集。
def decipher_affine(msg, key, symbols=None):
    return encipher_affine(msg, key, symbols, _inverse=True)


# 返回消息 `msg` 的 Atbash 密文。
# Atbash 是一种替换密码，最初用于加密希伯来字母表，它将每个字母映射到其反向字符（例如 a 映射到 z，b 映射到 y）。
# Atbash 在功能上等同于仿射密码，其中 `a = 25`，`b = 25`。
def encipher_atbash(msg, symbols=None):
    return encipher_affine(msg, (25, 25), symbols)


# 解密使用 Atbash 密码加密的消息 `msg` 并返回原始消息。
# `decipher_atbash` 在功能上等同于 `encipher_atbash`，但为了保持一致性，它仍然作为一个单独的函数存在。
def decipher_atbash(msg, symbols=None):
    return decipher_affine(msg, (25, 25), symbols)


#################### substitution cipher ###########################


# 返回替换每个出现在 `old` 中的字符为 `new` 中对应字符所得到的密文。
# 如果 `old` 是一个映射，则忽略 `new`，使用 `old` 定义的替换。
# 这是一个比仿射密码更一般化的替换密码，密钥只能通过确定每个符号的映射来恢复。
# 尽管在实践中，一旦识别了几个符号，其他字符的映射很快就能猜出来。
def encipher_substitution(msg, old, new=None):
    pass  # 这里代码未完成，功能还未实现，因此不返回任何值
    # 使用指定的替换表对消息进行替换加密
    msg = AZ("go navy! beat army!")
    # 调用替换加密函数，使用给定的旧替换表和新替换表进行加密
    ct = encipher_substitution(msg, old, new); ct
    '60N^V4B3^T^RM4'
    
    # 要解密替换加密文本，需将最后两个参数反转
    """
    >>> encipher_substitution(ct, new, old)
    'GONAVYBEATARMY'
    """
    
    # 特殊情况下，如果old和new是顺序2的置换（表示字符的换位），它们的顺序不重要
    """
    >>> old = 'NAVY'
    >>> new = 'ANYV'
    >>> encipher = lambda x: encipher_substitution(x, old, new)
    >>> encipher('NAVY')
    'ANYV'
    >>> encipher(_)
    'NAVY'
    """
    
    # 替换密码一般是一种方法，根据一定的系统，将明文的"单元"（不一定是单个字符）替换为密文
    """
    >>> ords = dict(zip('abc', ['\\%i' % ord(i) for i in 'abc']))
    >>> print(encipher_substitution('abc', ords))
    \97\98\99
    """
    
    # 引用
    """
    .. [1] https://en.wikipedia.org/wiki/Substitution_cipher
    """
    
    # 返回使用指定替换表加密后的消息
    return translate(msg, old, new)
# 定义 Vigenere 密码加密函数，接受消息 msg、密钥 key 和可选的符号集 symbols
def encipher_vigenere(msg, key, symbols=None):
    """
    Performs the Vigenere cipher encryption on plaintext ``msg``, and
    returns the ciphertext.
    """
    # 确定加密使用的字母表或符号集，默认为 None
    # 这里的 symbols 参数可以用来自定义密文所使用的字符集，如果未指定则使用默认的字母表
    """
    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_vigenere, AZ
    >>> key = "encrypt"
    >>> msg = "meet me on monday"
    >>> encipher_vigenere(msg, key)
    'QRGKKTHRZQEBPR'

    Section 1 of the Kryptos sculpture at the CIA headquarters
    uses this cipher and also changes the order of the
    alphabet [2]_. Here is the first line of that section of
    the sculpture:

    >>> from sympy.crypto.crypto import decipher_vigenere, padded_key
    >>> alp = padded_key('KRYPTOS', AZ())
    >>> key = 'PALIMPSEST'
    >>> msg = 'EMUFPHZLRFAXYUSDJKZLDKRNSHGNFIVJ'
    >>> decipher_vigenere(msg, key, alp)
    'BETWEENSUBTLESHADINGANDTHEABSENC'
    """
    # Vigenere 密码的实现是通过使用关键字对明文逐字母加密的过程
    # 具体加密过程会依据密钥中字母在字母表中的位置来进行不同位移
    # 其中 symbols 参数可以用来指定密文使用的符号集

    """
    Explanation
    ===========

    The Vigenere cipher is named after Blaise de Vigenere, a sixteenth
    century diplomat and cryptographer, by a historical accident.
    Vigenere actually invented a different and more complicated cipher.
    The so-called *Vigenere cipher* was actually invented
    by Giovan Batista Belaso in 1553.

    This cipher was used in the 1800's, for example, during the American
    Civil War. The Confederacy used a brass cipher disk to implement the
    Vigenere cipher (now on display in the NSA Museum in Fort
    Meade) [1]_.

    The Vigenere cipher is a generalization of the shift cipher.
    Whereas the shift cipher shifts each letter by the same amount
    (that amount being the key of the shift cipher) the Vigenere
    cipher shifts a letter by an amount determined by the key (which is
    a word or phrase known only to the sender and receiver).

    For example, if the key was a single letter, such as "C", then the
    so-called Vigenere cipher is actually a shift cipher with a
    shift of `2` (since "C" is the 2nd letter of the alphabet, if
    you start counting at `0`). If the key was a word with two
    letters, such as "CA", then the so-called Vigenere cipher will
    shift letters in even positions by `2` and letters in odd positions
    are left alone (shifted by `0`, since "A" is the 0th letter, if
    you start counting at `0`).
    """
    # Vigenere 密码是一种基于密钥字母序列的多表替换密码，不同于 Caesar 密码的固定位移
    # 它使用不同的位移量来加密明文中的每个字母，加密算法依赖于密钥的每个字符的位置
    # 定义一个函数，实现维吉尼亚密码的加密过程
    def vigenere_encrypt(msg, key, symbols):
        # 将明文消息 msg 和密钥 key 转换为对应的整数列表 L2 和 L1
        L2 = [symbols.index(ch) for ch in msg]
        L1 = [symbols.index(ch) for ch in key]
        
        N = len(symbols)  # 字母表的长度
        n2 = len(L2)      # 明文消息的长度
        
        # 对 key 进行扩展，使其与消息长度相同
        extended_key = key + msg[:n2 - len(key)]
        L1 = [symbols.index(ch) for ch in extended_key]  # 更新 L1
        
        # 加密过程的具体步骤
        C = [(L2[i] + L1[i % len(L1)]) % N for i in range(n2)]
        
        # 根据加密后的整数列表 C，生成密文字符串 ct
        ct = ''.join(symbols[c] for c in C)
        
        # 返回加密后的密文字符串 ct
        return ct
    # 函数用于维吉尼亚密码的加密过程
    def encipher_vigenere(msg, key, symbols='ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        # 调用 _prep 函数进行消息、密钥和符号集的预处理
        msg, key, A = _prep(msg, key, symbols)
        
        # 创建符号到索引的映射字典
        map = {c: i for i, c in enumerate(A)}
        
        # 将密钥转换为对应符号在映射字典中的索引列表
        key = [map[c] for c in key]
        
        # 符号集的长度
        N = len(map)
        
        # 密钥的长度
        k = len(key)
        
        # 加密后的消息列表
        rv = []
        
        # 遍历消息中的每个字符
        for i, m in enumerate(msg):
            # 计算当前字符在符号集中的索引并进行加密
            rv.append(A[(map[m] + key[i % k]) % N])
        
        # 将加密后的消息列表转换为字符串并返回
        rv = ''.join(rv)
        return rv
def decipher_vigenere(msg, key, symbols=None):
    """
    Decode using the Vigenere cipher.

    Examples
    ========

    >>> from sympy.crypto.crypto import decipher_vigenere
    >>> key = "encrypt"
    >>> ct = "QRGK kt HRZQE BPR"
    >>> decipher_vigenere(ct, key)
    'MEETMEONMONDAY'

    """
    # 调用_prep函数，准备消息、密钥和符号集合
    msg, key, A = _prep(msg, key, symbols)
    # 创建映射表，将符号映射为它们在A中的索引
    map = {c: i for i, c in enumerate(A)}
    N = len(A)   # A的长度，通常为26
    K = [map[c] for c in key]  # 将密钥转换为对应的索引列表
    n = len(K)  # 密钥的长度
    C = [map[c] for c in msg]  # 将消息转换为对应的索引列表
    # 解密过程
    rv = ''.join([A[(-K[i % n] + c) % N] for i, c in enumerate(C)])
    return rv


#################### Hill cipher  ########################


def encipher_hill(msg, key, symbols=None, pad="Q"):
    r"""
    Return the Hill cipher encryption of ``msg``.

    Explanation
    ===========

    The Hill cipher [1]_, invented by Lester S. Hill in the 1920's [2]_,
    was the first polygraphic cipher in which it was practical
    (though barely) to operate on more than three symbols at once.
    The following discussion assumes an elementary knowledge of
    matrices.

    First, each letter is first encoded as a number starting with 0.
    Suppose your message `msg` consists of `n` capital letters, with no
    spaces. This may be regarded an `n`-tuple M of elements of
    `Z_{26}` (if the letters are those of the English alphabet). A key
    in the Hill cipher is a `k x k` matrix `K`, all of whose entries
    are in `Z_{26}`, such that the matrix `K` is invertible (i.e., the
    linear transformation `K: Z_{N}^k \rightarrow Z_{N}^k`
    is one-to-one).


    Parameters
    ==========

    msg
        Plaintext message of `n` upper-case letters.

    key
        A `k \times k` invertible matrix `K`, all of whose entries are
        in `Z_{26}` (or whatever number of symbols are being used).

    pad
        Character (default "Q") to use to make length of text be a
        multiple of ``k``.

    Returns
    =======

    ct
        Ciphertext of upper-case letters.

    Notes
    =====

    ALGORITHM:

        STEPS:
            0. Number the letters of the alphabet from 0, ..., N
            1. Compute from the string ``msg`` a list ``L`` of
               corresponding integers. Let ``n = len(L)``.
            2. Break the list ``L`` up into ``t = ceiling(n/k)``
               sublists ``L_1``, ..., ``L_t`` of size ``k`` (with
               the last list "padded" to ensure its size is
               ``k``).
            3. Compute new list ``C_1``, ..., ``C_t`` given by
               ``C[i] = K*L_i`` (arithmetic is done mod N), for each
               ``i``.
            4. Concatenate these into a list ``C = C_1 + ... + C_t``.
            5. Compute from ``C`` a string ``ct`` of corresponding
               letters. This has length ``k*t``.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hill_cipher

    """
    # 实现Hill密码的加密过程

    # 调用_prep函数，准备消息、密钥和符号集合
    msg, key, A = _prep(msg, key, symbols)
    # 创建映射表，将符号映射为它们在A中的索引
    map = {c: i for i, c in enumerate(A)}
    N = len(A)   # A的长度，通常为26
    k = len(key)  # 密钥矩阵的大小
    # 将密钥矩阵转换为对应的索引矩阵
    K = [[map[c] for c in row] for row in key]
    n = len(msg)  # 消息长度
    t = (n + k - 1) // k  # 计算分组数量，向上取整

    # 将消息转换为对应的索引列表
    L = [map[c] for c in msg]
    # 对消息进行填充，使其长度为k的整数倍
    L += [map[pad]] * ((t * k) - n)

    # 加密过程
    C = []
    for i in range(t):
        # 取出当前分组的索引列表
        L_i = L[i * k:(i + 1) * k]
        # 计算加密后的结果，使用矩阵乘法并对N取模
        C_i = [(sum(K[j][l] * L_i[l] for l in range(k)) % N) for j in range(k)]
        # 将加密结果添加到C中
        C.extend(C_i)

    # 将加密后的索引列表转换为对应的字符
    ct = ''.join(A[c] for c in C)
    return ct
    # 确保密钥是方阵（即具有相同的行和列数）
    assert key.is_square
    # 确保填充字符 pad 的长度为1
    assert len(pad) == 1
    # 调用 _prep 函数对消息、填充字符和符号集进行预处理，返回处理后的结果
    msg, pad, A = _prep(msg, pad, symbols)
    # 创建从符号集 A 到索引的映射字典
    map = {c: i for i, c in enumerate(A)}
    # 将消息 msg 中的每个字符映射为其在符号集 A 中的索引，形成列表 P
    P = [map[c] for c in msg]
    # 符号集 A 的长度
    N = len(A)
    # 密钥矩阵的列数
    k = key.cols
    # 消息 P 的长度
    n = len(P)
    # 计算消息 P 可以被密钥分成几个块，m 是块数，r 是余数
    m, r = divmod(n, k)
    # 如果有余数 r，则用填充字符 pad 将消息 P 填充到下一个完整块的长度
    if r:
        P = P + [map[pad]]*(k - r)
        m += 1
    # 使用 Hill 密钥对消息 P 进行加密，生成加密后的字符串 rv
    rv = ''.join([A[c % N] for j in range(m) for c in
        list(key*Matrix(k, 1, [P[i]
        for i in range(k*j, k*(j + 1))]))])
    # 返回加密后的消息字符串 rv
    return rv
# 执行 Hill 密码的解密操作，使用给定的密钥矩阵的逆矩阵
def decipher_hill(msg, key, symbols=None):
    """
    Deciphering is the same as enciphering but using the inverse of the
    key matrix.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_hill, decipher_hill
    >>> from sympy import Matrix

    >>> key = Matrix([[1, 2], [3, 5]])
    >>> encipher_hill("meet me on monday", key)
    'UEQDUEODOCTCWQ'
    >>> decipher_hill(_, key)
    'MEETMEONMONDAY'

    When the length of the plaintext (stripped of invalid characters)
    is not a multiple of the key dimension, extra characters will
    appear at the end of the enciphered and deciphered text. In order to
    decipher the text, those characters must be included in the text to
    be deciphered. In the following, the key has a dimension of 4 but
    the text is 2 short of being a multiple of 4 so two characters will
    be added.

    >>> key = Matrix([[1, 1, 1, 2], [0, 1, 1, 0],
    ...               [2, 2, 3, 4], [1, 1, 0, 1]])
    >>> msg = "ST"
    >>> encipher_hill(msg, key)
    'HJEB'
    >>> decipher_hill(_, key)
    'STQQ'
    >>> encipher_hill(msg, key, pad="Z")
    'ISPK'
    >>> decipher_hill(_, key)
    'STZZ'

    If the last two characters of the ciphertext were ignored in
    either case, the wrong plaintext would be recovered:

    >>> decipher_hill("HD", key)
    'ORMV'
    >>> decipher_hill("IS", key)
    'UIKY'

    See Also
    ========

    encipher_hill

    """
    # 断言密钥矩阵是方阵
    assert key.is_square
    # 对消息进行预处理，获得有效字符列表 A 和映射表 map
    msg, _, A = _prep(msg, '', symbols)
    map = {c: i for i, c in enumerate(A)}
    # 将消息字符映射为其在 A 中的索引列表 C
    C = [map[c] for c in msg]
    N = len(A)
    k = key.cols
    n = len(C)
    m, r = divmod(n, k)
    # 如果 r 不为 0，将 C 扩展至 k 的倍数，增加 m 的值
    if r:
        C = C + [0]*(k - r)
        m += 1
    # 计算密钥矩阵的逆矩阵
    key_inv = key.inv_mod(N)
    # 使用逆矩阵解密 C 中的字符并组成解密后的消息字符串 rv
    rv = ''.join([A[p % N] for j in range(m) for p in
        list(key_inv*Matrix(
        k, 1, [C[i] for i in range(k*j, k*(j + 1))]))])
    # 返回解密后的消息字符串
    return rv



# 执行 Bifid 密码的加密操作，返回加密后的密文
def encipher_bifid(msg, key, symbols=None):
    r"""
    Performs the Bifid cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    This is the version of the Bifid cipher that uses an `n \times n`
    Polybius square.

    Parameters
    ==========

    msg
        Plaintext string.

    key
        Short string for key.

        Duplicate characters are ignored and then it is padded with the
        characters in ``symbols`` that were not in the short key.

    symbols
        `n \times n` characters defining the alphabet.

        (default is string.printable)

    Returns
    =======

    ciphertext
        Ciphertext using Bifid5 cipher without spaces.

    See Also
    ========

    decipher_bifid, encipher_bifid5, encipher_bifid6

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bifid_cipher

    """
    # 对消息、密钥和符号集进行预处理，获得有效字符列表 A 和长密钥 long_key
    msg, key, A = _prep(msg, key, symbols, bifid10)
    long_key = ''.join(uniq(key)) or A
    # 计算 A 的大小（假设为 n * n 的 Polybius 方阵）
    n = len(A)**.5
    # 检查变量 n 是否为整数，如果不是则抛出 ValueError 异常
    if n != int(n):
        raise ValueError(
            'Length of alphabet (%s) is not a square number.' % len(A))
    
    # 将 n 转换为整数 N
    N = int(n)
    
    # 如果长密钥 long_key 的长度小于 N 的平方，则将长密钥扩展至至少 N 的平方长度
    if len(long_key) < N**2:
        long_key = list(long_key) + [x for x in A if x not in long_key]

    # 分数化处理
    # 创建一个字典，将长密钥中的字符映射为其在长密钥中的行和列索引
    row_col = {ch: divmod(i, N) for i, ch in enumerate(long_key)}
    
    # 解包 row_col 字典中的行和列索引，以及将消息 msg 中的字符映射为它们的行和列索引
    r, c = zip(*[row_col[x] for x in msg])
    
    # 将 r 和 c 拼接成一个新的列表 rc
    rc = r + c
    
    # 创建一个字典 ch，将行列索引映射回长密钥中的字符
    ch = {i: ch for ch, i in row_col.items()}
    
    # 从 rc 列表中每隔两个取一个索引值，并从字典 ch 中获取对应的字符，将它们连接成字符串 rv
    rv = ''.join(ch[i] for i in zip(rc[::2], rc[1::2]))
    
    # 返回最终加密或解密后的结果字符串 rv
    return rv
def decipher_bifid(msg, key, symbols=None):
    r"""
    Performs the Bifid cipher decryption on ciphertext ``msg``, and
    returns the plaintext.

    This is the version of the Bifid cipher that uses the `n \times n`
    Polybius square.

    Parameters
    ==========

    msg
        Ciphertext string.

    key
        Short string for key.

        Duplicate characters are ignored and then it is padded with the
        characters in symbols that were not in the short key.

    symbols
        `n \times n` characters defining the alphabet.

        (default=string.printable, a `10 \times 10` matrix)

    Returns
    =======

    deciphered
        Deciphered text.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     encipher_bifid, decipher_bifid, AZ)

    Do an encryption using the bifid5 alphabet:

    >>> alp = AZ().replace('J', '')
    >>> ct = AZ("meet me on monday!")
    >>> key = AZ("gold bug")
    >>> encipher_bifid(ct, key, alp)
    'IEILHHFSTSFQYE'

    When entering the text or ciphertext, spaces are ignored so it
    can be formatted as desired. Re-entering the ciphertext from the
    preceding, putting 4 characters per line and padding with an extra
    J, does not cause problems for the deciphering:

    >>> decipher_bifid('''
    ... IEILH
    ... HFSTS
    ... FQYEJ''', key, alp)
    'MEETMEONMONDAY'

    When no alphabet is given, all 100 printable characters will be
    used:

    >>> key = ''
    >>> encipher_bifid('hello world!', key)
    'bmtwmg-bIo*w'
    >>> decipher_bifid(_, key)
    'hello world!'

    If the key is changed, a different encryption is obtained:

    >>> key = 'gold bug'
    >>> encipher_bifid('hello world!', 'gold_bug')
    'hg2sfuei7t}w'

    And if the key used to decrypt the message is not exact, the
    original text will not be perfectly obtained:

    >>> decipher_bifid(_, 'gold pug')
    'heldo~wor6d!'

    """
    # Prepare the message, remove spaces, and convert to matrix format
    msg, _, A = _prep(msg, '', symbols, bifid10)
    
    # Generate the full key from the short key, removing duplicates
    long_key = ''.join(uniq(key)) or A
    
    # Determine the size of the square matrix based on the alphabet size
    n = len(A)**.5
    if n != int(n):
        raise ValueError(
            'Length of alphabet (%s) is not a square number.' % len(A))
    N = int(n)
    
    # If the key is shorter than the required length, pad it with unused characters
    if len(long_key) < N**2:
        long_key = list(long_key) + [x for x in A if x not in long_key]
    
    # Create a dictionary mapping characters to their respective row and column in the matrix
    row_col = {
        ch: divmod(i, N) for i, ch in enumerate(long_key)}
    
    # Apply reverse fractionalization to the message
    rc = [i for c in msg for i in row_col[c]]
    
    # Group the coordinates back into pairs (rows, columns)
    n = len(msg)
    rc = zip(*(rc[:n], rc[n:]))
    
    # Create a dictionary to map coordinates back to characters
    ch = {i: ch for ch, i in row_col.items()}
    
    # Retrieve characters corresponding to the coordinates and join them into the decrypted message
    rv = ''.join(ch[i] for i in rc)
    
    # Return the deciphered text
    return rv
    # 创建一个代表多项式方阵的对象
    Matrix([
    [G, O, L, D, B],  # 第一行：包含 G, O, L, D, B 的矩阵行
    [U, A, C, E, F],  # 第二行：包含 U, A, C, E, F 的矩阵行
    [H, I, K, M, N],  # 第三行：包含 H, I, K, M, N 的矩阵行
    [P, Q, R, S, T],  # 第四行：包含 P, Q, R, S, T 的矩阵行
    [V, W, X, Y, Z]])  # 第五行：包含 V, W, X, Y, Z 的矩阵行
    
    # 查看相关文档
    See Also
    ========
    padded_key
    
    """
    A = ''.join(uniq(''.join(key)))  # 将密钥 key 合并成一个字符串，并去除重复字符，存储在 A 中
    n = len(A)**.5  # 计算 A 的长度的平方根，赋值给 n
    if n != int(n):  # 如果 n 不是整数，即 A 的长度不是平方数
        raise ValueError(
            'Length of alphabet (%s) is not a square number.' % len(A))  # 抛出值错误异常，指出字母表长度不是平方数
    n = int(n)  # 将 n 转换为整数
    f = lambda i, j: Symbol(A[n*i + j])  # 创建一个 lambda 函数 f，根据索引 i, j 返回符号 A 中特定位置的值
    rv = Matrix(n, n, f)  # 使用 lambda 函数 f 创建一个 n x n 的符号矩阵 rv
    return rv  # 返回创建的符号矩阵 rv
def encipher_bifid5(msg, key):
    r"""
    Performs the Bifid cipher encryption on plaintext ``msg``, and
    returns the ciphertext.

    Explanation
    ===========

    This is the version of the Bifid cipher that uses the `5 \times 5`
    Polybius square. The letter "J" is ignored so it must be replaced
    with something else (traditionally an "I") before encryption.

    ALGORITHM: (5x5 case)

        STEPS:
            0. Create the `5 \times 5` Polybius square ``S`` associated
               to ``key`` as follows:

                a) moving from left-to-right, top-to-bottom,
                   place the letters of the key into a `5 \times 5`
                   matrix,
                b) if the key has less than 25 letters, add the
                   letters of the alphabet not in the key until the
                   `5 \times 5` square is filled.

            1. Create a list ``P`` of pairs of numbers which are the
               coordinates in the Polybius square of the letters in
               ``msg``.
            2. Let ``L1`` be the list of all first coordinates of ``P``
               (length of ``L1 = n``), let ``L2`` be the list of all
               second coordinates of ``P`` (so the length of ``L2``
               is also ``n``).
            3. Let ``L`` be the concatenation of ``L1`` and ``L2``
               (length ``L = 2*n``), except that consecutive numbers
               are paired ``(L[2*i], L[2*i + 1])``. You can regard
               ``L`` as a list of pairs of length ``n``.
            4. Let ``C`` be the list of all letters which are of the
               form ``S[i, j]``, for all ``(i, j)`` in ``L``. As a
               string, this is the ciphertext of ``msg``.

    Parameters
    ==========

    msg : str
        Plaintext string.

        Converted to upper case and filtered of anything but all letters
        except J.

    key
        Short string for key; non-alphabetic letters, J and duplicated
        characters are ignored and then, if the length is less than 25
        characters, it is padded with other letters of the alphabet
        (in alphabetical order).

    Returns
    =======

    ct
        Ciphertext (all caps, no spaces).

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     encipher_bifid5, decipher_bifid5)

    "J" will be omitted unless it is replaced with something else:

    >>> round_trip = lambda m, k: \
    ...     decipher_bifid5(encipher_bifid5(m, k), k)
    >>> key = 'a'
    >>> msg = "JOSIE"
    >>> round_trip(msg, key)
    'OSIE'
    >>> round_trip(msg.replace("J", "I"), key)
    'IOSIE'
    >>> j = "QIQ"
    >>> round_trip(msg.replace("J", j), key).replace(j, "J")
    'JOSIE'


    Notes
    =====

    The Bifid cipher was invented around 1901 by Felix Delastelle.
    It is a *fractional substitution* cipher, where letters are
    replaced by pairs of symbols from a smaller alphabet. The
    """
    # 此处开始函数实现
    # 替换掉消息中的 "J"，因为 Bifid 密码不处理 "J"
    msg = msg.replace("J", "I")
    # 创建空的 Polybius 方阵
    S = [['' for i in range(5)] for j in range(5)]
    # 处理密钥，去除非字母字符和重复字符，然后构建 Polybius 方阵
    key = ''.join(sorted(set(filter(str.isalpha, key.upper())), key=lambda x: key.upper().index(x)))
    # 将密钥填充到 Polybius 方阵中
    for i, c in enumerate(key + 'ABCDEFGHIKLMNOPQRSTUVWXYZ'):
        S[i // 5][i % 5] = c
    # 初始化空列表 P 用于存放消息字符在 Polybius 方阵中的坐标
    P = [(r, c) for char in msg.upper() for r, row in enumerate(S) for c, val in enumerate(row) if val == char]
    # 拆分坐标并连接它们以生成 L 列表
    L1, L2 = zip(*P)
    L = L1 + L2
    # 构建密文 C，通过 L 列表中的坐标对应的 Polybius 方阵字符
    ct = ''.join(S[L[i]][L[i + len(L) // 2]] for i in range(len(L) // 2))
    return ct
    cipher uses a `5 \times 5` square filled with some ordering of the
    alphabet, except that "J" is replaced with "I" (this is a so-called
    Polybius square; there is a `6 \times 6` analog if you add back in
    "J" and also append onto the usual 26 letter alphabet, the digits
    0, 1, ..., 9).
    According to Helen Gaines' book *Cryptanalysis*, this type of cipher
    was used in the field by the German Army during World War I.

    See Also
    ========

    decipher_bifid5, encipher_bifid

    """
    将明文消息（msg）和密钥（key）转换为大写形式，并使用 `_prep` 函数进行准备工作，返回元组中的前两个元素，忽略第三个元素。
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid5)
    对密钥进行填充，使其与使用 bifid5 加密算法的要求相匹配。
    key = padded_key(key, bifid5)
    使用 encipher_bifid 函数对明文消息进行加密，返回加密后的结果。
    return encipher_bifid(msg, '', key)
# 对使用 5x5 Polybius 方式的 Bifid 密码进行解密
def decipher_bifid5(msg, key):
    # 准备消息和密钥，转换为大写，并进行预处理
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid5)
    # 对密钥进行填充处理，以适应 Bifid5 密码要求
    key = padded_key(key, bifid5)
    # 调用 encipher_bifid 函数进行解密操作
    return decipher_bifid(msg, '', key)


# 生成 5x5 Polybius 方式的 Bifid 密码的 Polybius 方阵
def bifid5_square(key=None):
    # 如果未提供密钥，则使用默认密钥 bifid5
    if not key:
        key = bifid5
    else:
        # 准备空消息，处理传入的密钥，并进行预处理
        _, key, _ = _prep('', key.upper(), None, bifid5)
        # 对密钥进行填充处理，以适应 Bifid5 密码的要求
        key = padded_key(key, bifid5)
    # 调用 bifid_square 函数生成 Polybius 方阵
    return bifid_square(key)


# 对使用 6x6 Polybius 方式的 Bifid 密码进行加密
def encipher_bifid6(msg, key):
    # 准备消息和密钥，转换为大写，并进行预处理
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid6)
    # 对密钥进行填充处理，以适应 Bifid6 密码要求
    key = padded_key(key, bifid6)
    # 调用 encipher_bifid 函数进行加密操作
    return encipher_bifid(msg, '', key)


# 对使用 6x6 Polybius 方式的 Bifid 密码进行解密
def decipher_bifid6(msg, key):
    # 准备消息和密钥，转换为大写，并进行预处理
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid6)
    # 对密钥进行填充处理，以适应 Bifid6 密码要求
    key = padded_key(key, bifid6)
    # 调用 decipher_bifid 函数进行解密操作
    return decipher_bifid(msg, '', key)
    # 使用 Bifid 加密算法对消息进行解密
    msg, key, _ = _prep(msg.upper(), key.upper(), None, bifid6)
    # 调用 padded_key 函数对密钥进行填充，以适应特定加密算法的要求
    key = padded_key(key, bifid6)
    # 调用 decipher_bifid 函数进行 Bifid 解密操作
    return decipher_bifid(msg, '', key)
# 定义一个生成 6x6 Polybius 方阵的函数，用于 Bifid 密码
def bifid6_square(key=None):
    r"""
    6x6 Polybius square.

    Produces the Polybius square for the `6 \times 6` Bifid cipher.
    Assumes alphabet of symbols is "A", ..., "Z", "0", ..., "9".

    Examples
    ========

    >>> from sympy.crypto.crypto import bifid6_square
    >>> key = "gold bug"
    >>> bifid6_square(key)
    Matrix([
    [G, O, L, D, B, U],
    [A, C, E, F, H, I],
    [J, K, M, N, P, Q],
    [R, S, T, V, W, X],
    [Y, Z, 0, 1, 2, 3],
    [4, 5, 6, 7, 8, 9]])

    """
    # 如果没有指定 key，则使用默认的 bifid6 方阵
    if not key:
        key = bifid6
    else:
        # 使用 _prep 函数准备密钥，将其转换为大写
        _, key, _ = _prep('', key.upper(), None, bifid6)
        # 对密钥进行填充以匹配 bifid6 的要求
        key = padded_key(key, bifid6)
    # 返回生成的 Polybius 方阵
    return bifid_square(key)


#################### RSA  #############################

# 定义一个函数用于使用中国剩余定理解密 RSA 加密的密文
def _decipher_rsa_crt(i, d, factors):
    """Decipher RSA using chinese remainder theorem from the information
    of the relatively-prime factors of the modulus.

    Parameters
    ==========

    i : integer
        Ciphertext

    d : integer
        The exponent component.

    factors : list of relatively-prime integers
        The integers given must be coprime and the product must equal
        the modulus component of the original RSA key.

    Examples
    ========

    How to decrypt RSA with CRT:

    >>> from sympy.crypto.crypto import rsa_public_key, rsa_private_key
    >>> primes = [61, 53]
    >>> e = 17
    >>> args = primes + [e]
    >>> puk = rsa_public_key(*args)
    >>> prk = rsa_private_key(*args)

    >>> from sympy.crypto.crypto import encipher_rsa, _decipher_rsa_crt
    >>> msg = 65
    >>> crt_primes = primes
    >>> encrypted = encipher_rsa(msg, puk)
    >>> decrypted = _decipher_rsa_crt(encrypted, prk[1], primes)
    >>> decrypted
    65
    """
    # 使用指定的指数 d 和模数的相对质数因子来进行 RSA 解密
    moduluses = [pow(i, d, p) for p in factors]

    # 使用中国剩余定理解密，并返回结果
    result = crt(factors, moduluses)
    if not result:
        raise ValueError("CRT failed")
    return result[0]


# 定义一个生成 RSA 密钥的私有子程序函数
def _rsa_key(*args, public=True, private=True, totient='Euler', index=None, multipower=None):
    r"""A private subroutine to generate RSA key

    Parameters
    ==========

    public, private : bool, optional
        Flag to generate either a public key, a private key.

    totient : 'Euler' or 'Carmichael'
        Different notation used for totient.

    multipower : bool, optional
        Flag to bypass warning for multipower RSA.
    """

    # 如果参数数量少于 2，则返回 False
    if len(args) < 2:
        return False

    # 检查 totient 参数是否有效
    if totient not in ('Euler', 'Carmichael'):
        raise ValueError(
            "The argument totient={} should either be " \
            "'Euler', 'Carmichalel'." \
            .format(totient))

    # 根据 totient 类型选择不同的计算函数
    if totient == 'Euler':
        _totient = _euler
    else:
        _totient = _carmichael

    # 如果指定了 index 参数，则必须 totient 为 Carmichael 类型
    if index is not None:
        index = as_int(index)
        if totient != 'Carmichael':
            raise ValueError(
                "Setting the 'index' keyword argument requires totient"
                "notation to be specified as 'Carmichael'.")
    # 检查所有的素数是否都是质数，如果不是，则需要重新计算素因子
    if not all(isprime(p) for p in primes):
        # 初始化一个空列表用于存储新的素因子
        new_primes = []
        # 遍历现有的素数列表
        for i in primes:
            # 将每个素数的素因子添加到新的素因子列表中
            new_primes.extend(factorint(i, multiple=True))
        # 更新素数列表为新的素因子列表
        primes = new_primes

    # 计算所有素数的乘积
    n = reduce(lambda i, j: i*j, primes)

    # 使用多重集合类创建一个多重集合，并检查所有值是否都等于1
    tally = multiset(primes)
    if all(v == 1 for v in tally.values()):
        # 如果所有值都等于1，则计算 Euler 函数 phi(n)
        phi = int(_totient(tally))
    else:
        # 如果存在值大于1的情况
        if not multipower:
            # 如果不允许多重幂，发出非可逆密码警告
            NonInvertibleCipherWarning(
                'Non-distinctive primes found in the factors {}. '
                'The cipher may not be decryptable for some numbers '
                'in the complete residue system Z[{}], but the cipher '
                'can still be valid if you restrict the domain to be '
                'the reduced residue system Z*[{}]. You can pass '
                'the flag multipower=True if you want to suppress this '
                'warning.'
                .format(primes, n, n)
                # 由于大多数用户将调用一个调用这个函数的函数，因此设置 stacklevel=4
                ).warn(stacklevel=4)
        # 计算 Euler 函数 phi(n)
        phi = int(_totient(tally))

    # 检查公钥 e 和 phi(n) 的最大公约数是否为1
    if gcd(e, phi) == 1:
        # 如果是公钥操作且没有私钥
        if public and not private:
            # 如果索引是整数，则对公钥 e 进行调整
            if isinstance(index, int):
                e = e % phi
                e += index * phi
            # 返回 n 和调整后的公钥 e
            return n, e

        # 如果是私钥操作且没有公钥
        if private and not public:
            # 计算私钥 d 的模反元素
            d = invert(e, phi)
            # 如果索引是整数，则对私钥 d 进行调整
            if isinstance(index, int):
                d += index * phi
            # 返回 n 和调整后的私钥 d
            return n, d

    # 如果以上条件都不满足，则返回 False
    return False
def rsa_public_key(*args, **kwargs):
    r"""Return the RSA *public key* pair, `(n, e)`

    Parameters
    ==========

    args : naturals
        If specified as `p, q, e` where `p` and `q` are distinct primes
        and `e` is a desired public exponent of the RSA, `n = p q` and
        `e` will be verified against the totient
        `\phi(n)` (Euler totient) or `\lambda(n)` (Carmichael totient)
        to be `\gcd(e, \phi(n)) = 1` or `\gcd(e, \lambda(n)) = 1`.

        If specified as `p_1, p_2, \dots, p_n, e` where
        `p_1, p_2, \dots, p_n` are specified as primes,
        and `e` is specified as a desired public exponent of the RSA,
        it will be able to form a multi-prime RSA, which is a more
        generalized form of the popular 2-prime RSA.

        It can also be possible to form a single-prime RSA by specifying
        the argument as `p, e`, which can be considered a trivial case
        of a multiprime RSA.

        Furthermore, it can be possible to form a multi-power RSA by
        specifying two or more pairs of the primes to be same.
        However, unlike the two-distinct prime RSA or multi-prime
        RSA, not every numbers in the complete residue system
        (`\mathbb{Z}_n`) will be decryptable since the mapping
        `\mathbb{Z}_{n} \rightarrow \mathbb{Z}_{n}`
        will not be bijective.
        (Only except for the trivial case when
        `e = 1`
        or more generally,

        .. math::
            e \in \left \{ 1 + k \lambda(n)
            \mid k \in \mathbb{Z} \land k \geq 0 \right \}

        when RSA reduces to the identity.)
        However, the RSA can still be decryptable for the numbers in the
        reduced residue system (`\mathbb{Z}_n^{\times}`), since the
        mapping
        `\mathbb{Z}_{n}^{\times} \rightarrow \mathbb{Z}_{n}^{\times}`
        can still be bijective.

        If you pass a non-prime integer to the arguments
        `p_1, p_2, \dots, p_n`, the particular number will be
        prime-factored and it will become either a multi-prime RSA or a
        multi-power RSA in its canonical form, depending on whether the
        product equals its radical or not.
        `p_1 p_2 \dots p_n = \text{rad}(p_1 p_2 \dots p_n)`

    totient : bool, optional
        If ``'Euler'``, it uses Euler's totient `\phi(n)` which is
        :meth:`sympy.functions.combinatorial.numbers.totient` in SymPy.

        If ``'Carmichael'``, it uses Carmichael's totient `\lambda(n)`
        which is :meth:`sympy.functions.combinatorial.numbers.reduced_totient` in SymPy.

        Unlike private key generation, this is a trivial keyword for
        public key generation because
        `\gcd(e, \phi(n)) = 1 \iff \gcd(e, \lambda(n)) = 1`.
    """
    # 实现根据输入参数生成 RSA 公钥的功能
    pass
    # index: 非负整数，可选参数
    # 在指定的 `0, 1, 2, \dots` 索引处返回RSA公钥的任意解决方案。此参数需要与 ``totient='Carmichael'`` 一起指定。
    #
    # 与RSA私钥的非唯一性类似，如 :meth:`rsa_private_key` 中的 `index` 参数文档所述，RSA公钥也不是唯一的，
    # 存在无限数量的RSA公共指数，可以以相同的方式运行。
    #
    # 对于给定的RSA公共指数 `e`，可以存在另一个RSA公共指数 `e + k \lambda(n)`，其中 `k` 是整数，`\lambda` 是Carmichael的欧拉函数。
    #
    # 然而，只考虑正数情况，可以有一个主要的RSA公共指数 `e_0`，其中 `0 < e_0 < \lambda(n)`，并且所有其他解可以以 `e_0 + k \lambda(n)` 的形式被规范化。
    #
    # ``index`` 指定 `k` 表示任何可能值的RSA公钥。
    #
    # 计算任意RSA公钥的示例：
    #
    # >>> from sympy.crypto.crypto import rsa_public_key
    # >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=0)
    # (3233, 17)
    # >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=1)
    # (3233, 797)
    # >>> rsa_public_key(61, 53, 17, totient='Carmichael', index=2)
    # (3233, 1577)

    # multipower: 布尔值，可选参数
    # 在RSA规范中发现的任意一对非不同质数，会限制密码系统的域，如参数 ``args`` 的解释中所述。
    #
    # SymPy RSA密钥生成器可能会在将其作为多幂RSA分派之前发出警告，但是如果将 ``True`` 传递给此关键字，则可以禁用警告。
    
    # 返回值
    # =======

    # (n, e) : int, int
    # `n` 是给定作为参数的任意数量的质数的乘积。
    #
    # `e` 与欧拉函数 `\phi(n)` 互质（互素）。

    # False
    # 如果给出少于两个参数，或 `e` 不与模数互质，则返回。

    # 示例
    # ========

    # >>> from sympy.crypto.crypto import rsa_public_key

    # 一个双质数RSA的公钥：

    # >>> p, q, e = 3, 5, 7
    # >>> rsa_public_key(p, q, e)
    # (15, 7)
    # >>> rsa_public_key(p, q, 30)
    # False

    # 一个多质数RSA的公钥：

    # >>> primes = [2, 3, 5, 7, 11, 13]
    # >>> e = 7
    # >>> args = primes + [e]
    # >>> rsa_public_key(*args)
    # (30030, 7)

    # 注意
    # =====

    # 尽管RSA可以推广到任意模数 `n`，但使用两个大质数已经成为最流行的规范，因为两个大质数的乘积通常是最难相对于 `n` 可能具有的位数进行因式分解的。
    #
    # 然而，可能需要进一步理解时间复杂性。
    # 返回一个 RSA 密钥对中的公钥
    return _rsa_key(*args, public=True, private=False, **kwargs)
# 返回 RSA 的私钥对 (n, d)
def rsa_private_key(*args, **kwargs):
    r"""Return the RSA *private key* pair, `(n, d)`

    Parameters
    ==========

    args : naturals
        The keyword is identical to the ``args`` in
        :meth:`rsa_public_key`.

    totient : bool, optional
        If ``'Euler'``, it uses Euler's totient convention `\phi(n)`
        which is :meth:`sympy.functions.combinatorial.numbers.totient` in SymPy.

        If ``'Carmichael'``, it uses Carmichael's totient convention
        `\lambda(n)` which is
        :meth:`sympy.functions.combinatorial.numbers.reduced_totient` in SymPy.

        There can be some output differences for private key generation
        as examples below.

        Example using Euler's totient:

        >>> from sympy.crypto.crypto import rsa_private_key
        >>> rsa_private_key(61, 53, 17, totient='Euler')
        (3233, 2753)

        Example using Carmichael's totient:

        >>> from sympy.crypto.crypto import rsa_private_key
        >>> rsa_private_key(61, 53, 17, totient='Carmichael')
        (3233, 413)

    index : nonnegative integer, optional
        Returns an arbitrary solution of a RSA private key at the index
        specified at `0, 1, 2, \dots`. This parameter needs to be
        specified along with ``totient='Carmichael'``.

        RSA private exponent is a non-unique solution of
        `e d \mod \lambda(n) = 1` and it is possible in any form of
        `d + k \lambda(n)`, where `d` is an another
        already-computed private exponent, and `\lambda` is a
        Carmichael's totient function, and `k` is any integer.

        However, considering only the positive cases, there can be
        a principal solution of a RSA private exponent `d_0` in
        `0 < d_0 < \lambda(n)`, and all the other solutions
        can be canonicalized in a form of `d_0 + k \lambda(n)`.

        ``index`` specifies the `k` notation to yield any possible value
        an RSA private key can have.

        An example of computing any arbitrary RSA private key:

        >>> from sympy.crypto.crypto import rsa_private_key
        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=0)
        (3233, 413)
        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=1)
        (3233, 1193)
        >>> rsa_private_key(61, 53, 17, totient='Carmichael', index=2)
        (3233, 1973)

    multipower : bool, optional
        The keyword is identical to the ``multipower`` in
        :meth:`rsa_public_key`.

    Returns
    =======

    (n, d) : int, int
        `n` is a product of any arbitrary number of primes given as
        the argument.

        `d` is the inverse of `e` (mod `\phi(n)`) where `e` is the
        exponent given, and `\phi` is a Euler totient.

    False
        Returned if less than two arguments are given, or `e` is
        not relatively prime to the totient of the modulus.

    Examples
    ========

    >>> from sympy.crypto.crypto import rsa_private_key
    # 返回一个 RSA 密钥对中的私钥部分
    return _rsa_key(*args, public=False, private=True, **kwargs)
# 使用 RSA 算法进行加解密操作，根据给定的密钥和因子进行处理
def _encipher_decipher_rsa(i, key, factors=None):
    # 解析密钥
    n, d = key
    # 如果没有给定因子，直接使用标准 RSA 加解密算法
    if not factors:
        return pow(i, d, n)

    # 内部函数，检查列表中的所有数字是否互质
    def _is_coprime_set(l):
        is_coprime_set = True
        for i in range(len(l)):
            for j in range(i+1, len(l)):
                if gcd(l[i], l[j]) != 1:
                    is_coprime_set = False
                    break
        return is_coprime_set

    # 计算给定因子的乘积
    prod = reduce(lambda i, j: i*j, factors)
    # 如果乘积等于模数，并且所有因子都互质，则使用中国剩余定理进行解密
    if prod == n and _is_coprime_set(factors):
        return _decipher_rsa_crt(i, d, factors)
    # 否则继续递归调用该函数
    return _encipher_decipher_rsa(i, key, factors=None)


# 使用 RSA 加密明文
def encipher_rsa(i, key, factors=None):
    r"""Encrypt the plaintext with RSA.

    Parameters
    ==========

    i : integer
        The plaintext to be encrypted for.

    key : (n, e) where n, e are integers
        `n` is the modulus of the key and `e` is the exponent of the
        key. The encryption is computed by `i^e \bmod n`.

        The key can either be a public key or a private key, however,
        the message encrypted by a public key can only be decrypted by
        a private key, and vice versa, as RSA is an asymmetric
        cryptography system.

    factors : list of coprime integers
        This is identical to the keyword ``factors`` in
        :meth:`decipher_rsa`.

    Notes
    =====

    Some specifications may make the RSA not cryptographically
    meaningful.

    For example, `0`, `1` will remain always same after taking any
    number of exponentiation, thus, should be avoided.

    Furthermore, if `i^e < n`, `i` may easily be figured out by taking
    `e` th root.

    And also, specifying the exponent as `1` or in more generalized form
    as `1 + k \lambda(n)` where `k` is an nonnegative integer,
    `\lambda` is a carmichael totient, the RSA becomes an identity
    mapping.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_rsa
    >>> from sympy.crypto.crypto import rsa_public_key, rsa_private_key

    Public Key Encryption:

    >>> p, q, e = 3, 5, 7
    >>> puk = rsa_public_key(p, q, e)
    >>> msg = 12
    >>> encipher_rsa(msg, puk)
    3

    Private Key Encryption:

    >>> p, q, e = 3, 5, 7
    >>> prk = rsa_private_key(p, q, e)
    >>> msg = 12
    >>> encipher_rsa(msg, prk)
    3

    Encryption using chinese remainder theorem:

    >>> encipher_rsa(msg, prk, factors=[p, q])
    3
    """
    # 调用内部函数进行加密/解密操作
    return _encipher_decipher_rsa(i, key, factors=factors)


# 使用 RSA 解密密文
def decipher_rsa(i, key, factors=None):
    r"""Decrypt the ciphertext with RSA.

    Parameters
    ==========

    i : integer
        The ciphertext to be decrypted for.
    # 返回使用 RSA 算法加密或解密后的结果
    return _encipher_decipher_rsa(i, key, factors=factors)
# 定义函数 `kid_rsa_public_key`，实现 Kid RSA 加密算法的公钥生成过程
def kid_rsa_public_key(a, b, A, B):
    # 计算 M = a * b - 1
    M = a * b - 1
    # 计算 e = A * M + a
    e = A * M + a
    # 计算 d = B * M + b
    d = B * M + b
    # 计算 n = (e * d - 1) // M
    n = (e * d - 1) // M
    # 返回公钥 (n, e)
    return n, e


# 定义函数 `kid_rsa_private_key`，实现 Kid RSA 加密算法的私钥生成过程
def kid_rsa_private_key(a, b, A, B):
    # 计算 M = a * b - 1
    M = a * b - 1
    # 计算 e = A * M + a
    e = A * M + a
    # 计算 d = B * M + b
    d = B * M + b
    # 计算 n = (e * d - 1) // M
    n = (e * d - 1) // M
    # 返回私钥 (n, d)
    return n, d


# 定义函数 `encipher_kid_rsa`，实现 Kid RSA 加密算法的加密过程
def encipher_kid_rsa(msg, key):
    # 解析公钥 `key`，得到 n 和 e
    n, e = key
    # 计算并返回密文 c = (msg * e) % n
    return (msg * e) % n


# 定义函数 `decipher_kid_rsa`，实现 Kid RSA 加密算法的解密过程
def decipher_kid_rsa(msg, key):
    # 解析私钥 `key`，得到 n 和 d
    n, d = key
    # 计算并返回明文 p = (msg * d) % n
    return (msg * d) % n


# 定义摩尔斯电码的字母表映射
morse_char = {
    ".-": "A", "-...": "B",
    "-.-.": "C", "-..": "D",
    ".": "E", "..-.": "F",
    "--.": "G", "....": "H",
    "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N",
    "---": "O", ".--.": "P",
    "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T",
    "..-": "U", "...-": "V",
    ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z",
}
    # 定义一个字典，将摩斯电码映射到对应的字符
    "-----": "0",      # 摩斯电码 "-----" 对应字符 "0"
    ".----": "1",      # 摩斯电码 ".----" 对应字符 "1"
    "..---": "2",      # 摩斯电码 "..---" 对应字符 "2"
    "...--": "3",      # 摩斯电码 "...--" 对应字符 "3"
    "....-": "4",      # 摩斯电码 "....-" 对应字符 "4"
    ".....": "5",      # 摩斯电码 "....." 对应字符 "5"
    "-....": "6",      # 摩斯电码 "-...." 对应字符 "6"
    "--...": "7",      # 摩斯电码 "--..." 对应字符 "7"
    "---..": "8",      # 摩斯电码 "---.." 对应字符 "8"
    "----.": "9",      # 摩斯电码 "----." 对应字符 "9"
    ".-.-.-": ".",     # 摩斯电码 ".-.-.-" 对应字符 "."
    "--..--": ",",     # 摩斯电码 "--..--" 对应字符 ","
    "---...": ":",     # 摩斯电码 "---..." 对应字符 ":"
    "-.-.-.": ";",     # 摩斯电码 "-.-.-." 对应字符 ";"
    "..--..": "?",     # 摩斯电码 "..--.." 对应字符 "?"
    "-....-": "-",     # 摩斯电码 "-....-" 对应字符 "-"
    "..--.-": "_",     # 摩斯电码 "..--.-" 对应字符 "_"
    "-.--.": "(",      # 摩斯电码 "-.--." 对应字符 "("
    "-.--.-": ")",      # 摩斯电码 "-.--.-" 对应字符 ")"
    ".----.": "'",     # 摩斯电码 ".----." 对应字符 "'"
    "-...-": "=",      # 摩斯电码 "-...-" 对应字符 "="
    ".-.-.": "+",      # 摩斯电码 ".-.-." 对应字符 "+"
    "-..-.": "/",      # 摩斯电码 "-..-." 对应字符 "/"
    ".--.-.": "@",     # 摩斯电码 ".--.-." 对应字符 "@"
    "...-..-": "$",    # 摩斯电码 "...-..-" 对应字符 "$"
    "-.-.--": "!"      # 摩斯电码 "-.-.--" 对应字符 "!"
}
# 将摩尔斯字符到字符的映射进行反转，以便进行从摩尔斯码到字符的解码
char_morse = {v: k for k, v in morse_char.items()}


def encode_morse(msg, sep='|', mapping=None):
    """
    Encodes a plaintext into popular Morse Code with letters
    separated by ``sep`` and words by a double ``sep``.

    Examples
    ========

    >>> from sympy.crypto.crypto import encode_morse
    >>> msg = 'ATTACK RIGHT FLANK'
    >>> encode_morse(msg)
    '.-|-|-|.-|-.-.|-.-||.-.|..|--.|....|-||..-.|.-..|.-|-.|-.-'

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Morse_code

    """

    # 如果未提供映射表，使用默认的摩尔斯字符到字符的映射表
    mapping = mapping or char_morse
    # 确保分隔符不在映射表中，避免冲突
    assert sep not in mapping
    # 设置单词分隔符为两个分隔符的组合
    word_sep = 2*sep
    # 将空格映射为单词分隔符
    mapping[" "] = word_sep
    # 检查消息是否以空格结尾，以便后续处理
    suffix = msg and msg[-1] in whitespace

    # 规范化消息中的空白字符
    msg = (' ' if word_sep else '').join(msg.split())
    # 删除未映射到的字符
    chars = set(''.join(msg.split()))
    ok = set(mapping.keys())
    msg = translate(msg, None, ''.join(chars - ok))

    # 初始化摩尔斯编码字符串
    morsestring = []
    # 拆分消息为单词，并对每个单词进行摩尔斯编码
    words = msg.split()
    for word in words:
        morseword = []
        for letter in word:
            # 获取字符的摩尔斯编码
            morseletter = mapping[letter]
            morseword.append(morseletter)

        # 将单词内的摩尔斯编码连接起来
        word = sep.join(morseword)
        morsestring.append(word)

    # 将单词之间使用单词分隔符连接起来，并添加结尾分隔符（如果有）
    return word_sep.join(morsestring) + (word_sep if suffix else '')


def decode_morse(msg, sep='|', mapping=None):
    """
    Decodes a Morse Code with letters separated by ``sep``
    (default is '|') and words by `word_sep` (default is '||)
    into plaintext.

    Examples
    ========

    >>> from sympy.crypto.crypto import decode_morse
    >>> mc = '--|---|...-|.||.|.-|...|-'
    >>> decode_morse(mc)
    'MOVE EAST'

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Morse_code

    """

    # 如果未提供映射表，使用默认的字符到摩尔斯字符的映射表
    mapping = mapping or morse_char
    # 设置单词分隔符为两个分隔符的组合
    word_sep = 2*sep
    # 初始化字符串列表，用于存储解码后的字符
    characterstring = []
    # 去除消息开头和结尾的单词分隔符，并按单词分隔符拆分消息
    words = msg.strip(word_sep).split(word_sep)
    for word in words:
        # 拆分单词内的摩尔斯编码，解码为字符
        letters = word.split(sep)
        chars = [mapping[c] for c in letters]
        # 将解码后的字符连接成单词
        word = ''.join(chars)
        characterstring.append(word)
    # 将解码后的单词列表连接成最终的消息字符串
    rv = " ".join(characterstring)
    return rv


#################### LFSRs  ##########################################


@doctest_depends_on(ground_types=['python', 'gmpy'])
def lfsr_sequence(key, fill, n):
    r"""
    This function creates an LFSR sequence.

    Parameters
    ==========

    key : list
        A list of finite field elements, `[c_0, c_1, \ldots, c_k].`

    fill : list
        The list of the initial terms of the LFSR sequence,
        `[x_0, x_1, \ldots, x_k].`

    n
        Number of terms of the sequence that the function returns.

    Returns
    =======

    L
        The LFSR sequence defined by
        `x_{n+1} = c_k x_n + \ldots + c_0 x_{n-k}`, for
        `n \leq k`.

    Notes
    =====

    S. Golomb [G]_ gives a list of three statistical properties a
    sequence of numbers `a = \{a_n\}_{n=1}^\infty`,
    `a_n \in \{0,1\}`, should display to be considered
    "random". Define the autocorrelation of `a` to be
    """
    # 检查 key 是否为列表，如果不是则引发类型错误异常
    if not isinstance(key, list):
        raise TypeError("key must be a list")
    # 检查 fill 是否为列表，如果不是则引发类型错误异常
    if not isinstance(fill, list):
        raise TypeError("fill must be a list")
    # 确定 key 的域，并创建对应的有限域对象
    p = key[0].modulus()
    F = FF(p)
    # 将 fill 复制给序列 s
    s = fill
    # 获取 fill 的长度作为 k
    k = len(fill)
    # 初始化空列表 L，用于存储生成的序列
    L = []
    # 迭代 n 次，生成序列
    for i in range(n):
        # 复制当前状态 s 到 s0
        s0 = s[:]
        # 将当前状态 s 的第一个元素添加到序列 L 中
        L.append(s[0])
        # 更新状态 s 为去除第一个元素后的子序列
        s = s[1:k]
        # 计算新元素 x，为 key 和 s0 对应位置的乘积之和，使用有限域运算
        x = sum(int(key[i]*s0[i]) for i in range(k))
        # 将计算结果 x 转换为有限域中的元素，并添加到状态 s 的末尾
        s.append(F(x))
    # 返回生成的序列 L
    return L       # use [int(x) for x in L] for int version
    # 初始化有限域的模数
    p = s[0].modulus()
    # 创建符号变量 x
    x = Symbol("x")
    # 初始化连接多项式 C(x) 为 1
    C = 1*x**0
    # 初始化 B(x) 为 1
    B = 1*x**0
    # 初始化 m 为 1
    m = 1
    # 初始化 b(x) 为 1
    b = 1*x**0
    # 初始化 L 为 0
    L = 0
    # 初始化 N 为 0
    N = 0
    # 当 N 小于字符串 s 的长度时，执行循环
    while N < len(s):
        # 如果 L 大于 0，则计算多项式 C 的次数
        if L > 0:
            dC = Poly(C).degree()
            # 计算 r，即 L+1 和 dC+1 中较小的值
            r = min(L + 1, dC + 1)
            # 获取多项式 C 的系数列表
            coeffsC = [C.subs(x, 0)] + [C.coeff(x**i) for i in range(1, dC + 1)]
            # 计算 d，使用 s[N] 和多项式 C 的系数进行计算
            d = (int(s[N]) + sum(coeffsC[i] * int(s[N - i]) for i in range(1, r))) % p
        # 如果 L 等于 0，则设置 d 为 s[N] 乘以 x 的 0 次方（即 s[N]）
        if L == 0:
            d = int(s[N]) * x**0
        # 如果 d 等于 0，则增加 m 和 N 的值，并继续下一次循环
        if d == 0:
            m += 1
            N += 1
        # 如果 d 大于 0
        if d > 0:
            # 如果 2*L 大于 N，则更新多项式 C，并增加 m 和 N 的值
            if 2 * L > N:
                C = (C - d * ((b**(p - 2)) % p) * x**m * B).expand()
                m += 1
                N += 1
            # 否则，进行变量交换和更新操作
            else:
                T = C
                C = (C - d * ((b**(p - 2)) % p) * x**m * B).expand()
                L = N + 1 - L
                m = 1
                b = d
                B = T
                N += 1
    # 计算多项式 C 的次数
    dC = Poly(C).degree()
    # 获取多项式 C 的系数列表
    coeffsC = [C.subs(x, 0)] + [C.coeff(x**i) for i in range(1, dC + 1)]
    # 计算并返回多项式的结果
    return sum(coeffsC[i] % p * x**i for i in range(dC + 1) if coeffsC[i] is not None)
#################### ElGamal  #############################

# 定义生成 ElGamal 加密算法私钥的函数
def elgamal_private_key(digit=10, seed=None):
    """
    Return three number tuple as private key.

    Explanation
    ===========

    Elgamal encryption is based on the mathematical problem
    called the Discrete Logarithm Problem (DLP). For example,

    `a^{b} \equiv c \pmod p`

    In general, if ``a`` and ``b`` are known, ``ct`` is easily
    calculated. If ``b`` is unknown, it is hard to use
    ``a`` and ``ct`` to get ``b``.

    Parameters
    ==========

    digit : int
        Minimum number of binary digits for key.

    Returns
    =======

    tuple : (p, r, d)
        p = prime number.

        r = primitive root.

        d = random number.

    Notes
    =====

    For testing purposes, the ``seed`` parameter may be set to control
    the output of this routine. See sympy.core.random._randrange.

    Examples
    ========

    >>> from sympy.crypto.crypto import elgamal_private_key
    >>> from sympy.ntheory import is_primitive_root, isprime
    >>> a, b, _ = elgamal_private_key()
    >>> isprime(a)
    True
    >>> is_primitive_root(b, a)
    True

    """
    # 使用 _randrange 函数生成随机数
    randrange = _randrange(seed)
    # 生成一个大于 2^digit 的素数 p
    p = nextprime(2**digit)
    # 生成 p 的原根 r
    return p, primitive_root(p), randrange(2, p)


# 定义生成 ElGamal 加密算法公钥的函数
def elgamal_public_key(key):
    """
    Return three number tuple as public key.

    Parameters
    ==========

    key : (p, r, e)
        Tuple generated by ``elgamal_private_key``.

    Returns
    =======

    tuple : (p, r, e)
        `e = r**d \bmod p`

        `d` is a random number in private key.

    Examples
    ========

    >>> from sympy.crypto.crypto import elgamal_public_key
    >>> elgamal_public_key((1031, 14, 636))
    (1031, 14, 212)

    """
    p, r, e = key
    # 计算公钥中的 e = r^d mod p
    return p, r, pow(r, e, p)


# 定义使用 ElGamal 加密算法加密消息的函数
def encipher_elgamal(i, key, seed=None):
    """
    Encrypt message with public key.

    Explanation
    ===========

    ``i`` is a plaintext message expressed as an integer.
    ``key`` is public key (p, r, e). In order to encrypt
    a message, a random number ``a`` in ``range(2, p)``
    is generated and the encryped message is returned as
    `c_{1}` and `c_{2}` where:

    `c_{1} \equiv r^{a} \pmod p`

    `c_{2} \equiv m e^{a} \pmod p`

    Parameters
    ==========

    msg
        int of encoded message.

    key
        Public key.

    Returns
    =======

    tuple : (c1, c2)
        Encipher into two number.

    Notes
    =====

    For testing purposes, the ``seed`` parameter may be set to control
    the output of this routine. See sympy.core.random._randrange.

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_elgamal, elgamal_private_key, elgamal_public_key
    >>> pri = elgamal_private_key(5, seed=[3]); pri
    (37, 2, 3)
    >>> pub = elgamal_public_key(pri); pub
    (37, 2, 8)
    >>> msg = 36
    >>> encipher_elgamal(msg, pub, seed=[3])
    (8, 6)

    """
    p, r, e = key
    # 如果 i 小于 0 或者大于等于 p，抛出数值错误异常，提示消息应该在范围内
    if i < 0 or i >= p:
        raise ValueError(
            'Message (%s) should be in range(%s)' % (i, p))
    
    # 使用给定种子生成一个随机数生成器函数
    randrange = _randrange(seed)
    
    # 生成一个随机数 a，范围在 2 到 p 之间（不包括 p）
    a = randrange(2, p)
    
    # 返回 r 的 a 次方对 p 取模，并且返回 i * e 的 a 次方对 p 取模
    return pow(r, a, p), i*pow(e, a, p) % p
# 解密 ElGamal 加密的消息
def decipher_elgamal(msg, key):
    r"""
    Decrypt message with private key.

    `msg = (c_{1}, c_{2})`

    `key = (p, r, d)`

    根据扩展欧几里得定理计算：
    `u c_{1}^{d} + p n = 1`

    计算模 p 下的逆元：
    `u \equiv 1/{{c_{1}}^d} \pmod p`

    计算解密后的明文消息：
    `u c_{2} \equiv \frac{1}{c_{1}^d} c_{2} \equiv \frac{1}{r^{ad}} c_{2} \pmod p`

    返回解密后的明文消息：
    `\frac{1}{r^{ad}} m e^a \equiv \frac{1}{r^{ad}} m {r^{d a}} \equiv m \pmod p`

    Examples
    ========

    >>> from sympy.crypto.crypto import decipher_elgamal
    >>> from sympy.crypto.crypto import encipher_elgamal
    >>> from sympy.crypto.crypto import elgamal_private_key
    >>> from sympy.crypto.crypto import elgamal_public_key

    >>> pri = elgamal_private_key(5, seed=[3])
    >>> pub = elgamal_public_key(pri); pub
    (37, 2, 8)
    >>> msg = 17
    >>> decipher_elgamal(encipher_elgamal(msg, pub), pri) == msg
    True

    """
    p, _, d = key
    c1, c2 = msg
    # 计算 c1 的 d 次方的模 p 的逆元
    u = pow(c1, -d, p)
    # 返回解密后的消息
    return u * c2 % p


################ Diffie-Hellman Key Exchange  #########################

# 生成 Diffie-Hellman 密钥对中的私钥
def dh_private_key(digit=10, seed=None):
    r"""
    Return three integer tuple as private key.

    Explanation
    ===========

    Diffie-Hellman key exchange is based on the mathematical problem
    called the Discrete Logarithm Problem (see ElGamal).

    Diffie-Hellman key exchange is divided into the following steps:

    *   Alice and Bob agree on a base that consist of a prime ``p``
        and a primitive root of ``p`` called ``g``
    *   Alice choses a number ``a`` and Bob choses a number ``b`` where
        ``a`` and ``b`` are random numbers in range `[2, p)`. These are
        their private keys.
    *   Alice then publicly sends Bob `g^{a} \pmod p` while Bob sends
        Alice `g^{b} \pmod p`
    *   They both raise the received value to their secretly chosen
        number (``a`` or ``b``) and now have both as their shared key
        `g^{ab} \pmod p`

    Parameters
    ==========

    digit
        Minimum number of binary digits required in key.

    Returns
    =======

    tuple : (p, g, a)
        p = prime number.

        g = primitive root of p.

        a = random number from 2 through p - 1.

    Notes
    =====

    For testing purposes, the ``seed`` parameter may be set to control
    the output of this routine. See sympy.core.random._randrange.

    Examples
    ========

    >>> from sympy.crypto.crypto import dh_private_key
    >>> from sympy.ntheory import isprime, is_primitive_root
    >>> p, g, _ = dh_private_key()
    >>> isprime(p)
    True
    >>> is_primitive_root(g, p)
    True
    >>> p, g, _ = dh_private_key(5)
    >>> isprime(p)
    True
    >>> is_primitive_root(g, p)
    True

    """
    # 生成一个至少具有 digit 位的素数 p
    p = nextprime(2**digit)
    # 找到 p 的原根 g
    g = primitive_root(p)
    # 使用随机数生成器 _randrange(seed) 生成范围在 [2, p) 内的随机数 a
    randrange = _randrange(seed)
    a = randrange(2, p)
    # 返回生成的私钥 p, g, a
    return p, g, a


def dh_public_key(key):
    r"""
    Return three number tuple as public key.

    This is the tuple that Alice sends to Bob.

    Parameters
    ==========

    ```
    key : (p, g, a)
        A tuple generated by ``dh_private_key``.

    Returns
    =======
    
    tuple : int, int, int
        A tuple of `(p, g, g^a \mod p)` with `p`, `g` and `a` given as
        parameters.

    Examples
    ========

    >>> from sympy.crypto.crypto import dh_private_key, dh_public_key
    >>> p, g, a = dh_private_key();
    >>> _p, _g, x = dh_public_key((p, g, a))
    >>> p == _p and g == _g
    True
    >>> x == pow(g, a, p)
    True

    """
    p, g, a = key
    # 返回元组 (p, g, g^a mod p)，其中 p、g、a 是给定的参数
    return p, g, pow(g, a, p)
# 定义函数 dh_shared_key，计算 Diffie-Hellman 密钥交换的共享密钥
def dh_shared_key(key, b):
    """
    Return an integer that is the shared key.

    This is what Bob and Alice can both calculate using the public
    keys they received from each other and their private keys.

    Parameters
    ==========

    key : (p, g, x)
        Tuple `(p, g, x)` generated by ``dh_public_key``.

    b
        Random number in the range of `2` to `p - 1`
        (Chosen by second key exchange member (Bob)).

    Returns
    =======

    int
        A shared key.

    Examples
    ========

    >>> from sympy.crypto.crypto import (
    ...     dh_private_key, dh_public_key, dh_shared_key)
    >>> prk = dh_private_key();
    >>> p, g, x = dh_public_key(prk);
    >>> sk = dh_shared_key((p, g, x), 1000)
    >>> sk == pow(x, 1000, p)
    True

    """
    p, _, x = key
    # 检查 b 是否在 1 和 p 之间，若不是则引发 ValueError 异常
    if 1 >= b or b >= p:
        raise ValueError(filldedent('''
            Value of b should be greater 1 and less
            than prime %s.''' % p))

    # 计算并返回 x^b % p，作为共享密钥
    return pow(x, b, p)


################ Goldwasser-Micali Encryption  #########################


# 定义函数 _legendre，计算 Legendre 符号
def _legendre(a, p):
    """
    Returns the legendre symbol of a and p
    assuming that p is a prime.

    i.e. 1 if a is a quadratic residue mod p
        -1 if a is not a quadratic residue mod p
         0 if a is divisible by p

    Parameters
    ==========

    a : int
        The number to test.

    p : prime
        The prime to test ``a`` against.

    Returns
    =======

    int
        Legendre symbol (a / p).

    """
    # 计算 a^(p-1)/2 % p，根据结果确定 Legendre 符号的值并返回
    sig = pow(a, (p - 1)//2, p)
    if sig == 1:
        return 1
    elif sig == 0:
        return 0
    else:
        return -1


# 定义函数 _random_coprime_stream，生成与给定整数互质的随机数生成器
def _random_coprime_stream(n, seed=None):
    randrange = _randrange(seed)
    while True:
        y = randrange(n)
        # 检查 y 和 n 是否互质，是则 yield y
        if gcd(y, n) == 1:
            yield y


# 定义函数 gm_private_key，生成 Goldwasser-Micali 加密的私钥
def gm_private_key(p, q, a=None):
    r"""
    Check if ``p`` and ``q`` can be used as private keys for
    the Goldwasser-Micali encryption. The method works
    roughly as follows.

    Explanation
    ===========

    #. Pick two large primes $p$ and $q$.
    #. Call their product $N$.
    #. Given a message as an integer $i$, write $i$ in its bit representation $b_0, \dots, b_n$.
    #. For each $k$,

     if $b_k = 0$:
        let $a_k$ be a random square
        (quadratic residue) modulo $p q$
        such that ``jacobi_symbol(a, p*q) = 1``
     if $b_k = 1$:
        let $a_k$ be a random non-square
        (non-quadratic residue) modulo $p q$
        such that ``jacobi_symbol(a, p*q) = 1``

    returns $\left[a_1, a_2, \dots\right]$

    $b_k$ can be recovered by checking whether or not
    $a_k$ is a residue. And from the $b_k$'s, the message
    can be reconstructed.

    The idea is that, while ``jacobi_symbol(a, p*q)``
    can be easily computed (and when it is equal to $-1$ will
    tell you that $a$ is not a square mod $p q$), quadratic
    residuosity modulo a composite number is hard to compute
    without knowing its factorization.

    """
    # 函数未被实现，文档中描述了 GM 加密的私钥生成方法
    # 检查输入的两个参数 p 和 q 是否相等，若相等则抛出 ValueError 异常
    if p == q:
        raise ValueError("expected distinct primes, "
                         "got two copies of %i" % p)
    # 检查输入的两个参数 p 和 q 是否为素数，若不是则抛出 ValueError 异常
    elif not isprime(p) or not isprime(q):
        raise ValueError("first two arguments must be prime, "
                         "got %i of %i" % (p, q))
    # 检查输入的两个参数 p 和 q 是否为偶数，若是则抛出 ValueError 异常
    elif p == 2 or q == 2:
        raise ValueError("first two arguments must not be even, "
                         "got %i of %i" % (p, q))
    # 若以上条件均不满足，则返回参数 p 和 q 组成的元组
    return p, q
# 定义函数 gm_public_key，用于计算 Goldwasser-Micali 加密的公钥
def gm_public_key(p, q, a=None, seed=None):
    """
    Compute public keys for ``p`` and ``q``.
    Note that in Goldwasser-Micali Encryption,
    public keys are randomly selected.

    Parameters
    ==========

    p, q, a : int, int, int
        Initialization variables.

    Returns
    =======

    tuple : (a, N)
        ``a`` is the input ``a`` if it is not ``None`` otherwise
        some random integer coprime to ``p`` and ``q``.

        ``N`` is the product of ``p`` and ``q``.

    """
    # 调用 gm_private_key 函数获取私钥 p, q
    p, q = gm_private_key(p, q)
    # 计算 N 作为 p 和 q 的乘积
    N = p * q

    # 如果 a 为 None，则随机选择与 p 和 q 互质的整数
    if a is None:
        # 调用 _randrange(seed) 函数生成随机数
        randrange = _randrange(seed)
        # 循环直到找到符合条件的 a
        while True:
            a = randrange(N)
            # 检查 a 是否满足 Legendre 符号条件
            if _legendre(a, p) == _legendre(a, q) == -1:
                break
    else:
        # 如果给定的 a 不符合条件，返回 False
        if _legendre(a, p) != -1 or _legendre(a, q) != -1:
            return False
    # 返回公钥 (a, N)
    return (a, N)


# 定义函数 encipher_gm，用于 Goldwasser-Micali 加密
def encipher_gm(i, key, seed=None):
    """
    Encrypt integer 'i' using public_key 'key'
    Note that gm uses random encryption.

    Parameters
    ==========

    i : int
        The message to encrypt.

    key : (a, N)
        The public key.

    Returns
    =======

    list : list of int
        The randomized encrypted message.

    """
    # 如果 i 小于 0，抛出 ValueError 异常
    if i < 0:
        raise ValueError(
            "message must be a non-negative "
            "integer: got %d instead" % i)
    # 解包公钥 key 中的 a 和 N
    a, N = key
    bits = []
    # 将整数 i 转换为二进制形式
    while i > 0:
        bits.append(i % 2)
        i //= 2

    # 生成与 N 互质的随机数流
    gen = _random_coprime_stream(N, seed)
    # 反转 bits 列表
    rev = reversed(bits)
    # 定义加密函数 encode
    encode = lambda b: next(gen)**2*pow(a, b) % N
    # 返回加密后的列表
    return [ encode(b) for b in rev ]


# 定义函数 decipher_gm，用于解密 Goldwasser-Micali 加密的消息
def decipher_gm(message, key):
    """
    Decrypt message 'message' using public_key 'key'.

    Parameters
    ==========

    message : list of int
        The randomized encrypted message.

    key : (p, q)
        The private key.

    Returns
    =======

    int
        The encrypted message.

    """
    # 解包私钥 key 中的 p 和 q
    p, q = key
    # 定义函数 res，用于判断 Legendre 符号是否大于 0
    res = lambda m, p: _legendre(m, p) > 0
    # 根据加密消息 message 计算 bits 列表
    bits = [res(m, p) * res(m, q) for m in message]
    m = 0
    # 恢复原始消息 m
    for b in bits:
        m <<= 1
        m += not b
    # 返回解密后的整数消息 m
    return m



########### RailFence Cipher #############

# 定义函数 encipher_railfence，用于 Rail Fence 加密
def encipher_railfence(message,rails):
    """
    Performs Railfence Encryption on plaintext and returns ciphertext

    Examples
    ========

    >>> from sympy.crypto.crypto import encipher_railfence
    >>> message = "hello world"
    >>> encipher_railfence(message,3)
    'horel ollwd'

    Parameters
    ==========

    message : string, the message to encrypt.
    rails : int, the number of rails.

    Returns
    =======

    The Encrypted string message.

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/Rail_fence_cipher

    """
    # 初始化 rails 数组，创建 cycle 对象 p
    r = list(range(rails))
    p = cycle(r + r[-2:0:-1])
    # 对消息进行 Rail Fence 加密并返回
    return ''.join(sorted(message, key=lambda i: next(p)))


# 定义函数 decipher_railfence，用于解密 Rail Fence 加密的消息
def decipher_railfence(ciphertext,rails):
    """
    Decrypt the message using the given rails

    Examples
    ========

    >>> from sympy.crypto.crypto import decipher_railfence

    """
    Decrypt the message using the given rails

    Examples
    ========

    >>> from sympy.crypto.crypto import decipher_railfence
    >>> ciphertext = "horel ollwd"
    >>> decipher_railfence(ciphertext, 3)
    'hello world'

    Parameters
    ==========

    ciphertext : string, the message to decrypt.
    rails : int, the number of rails.

    Returns
    =======

    The Decrypted string message.
    """
    # 略，待补充完整的解密逻辑
    # 使用 railfence 解密算法对密文进行解密
    r = list(range(rails))
    # 创建一个循环迭代器，用于生成 railfence 加密解密时的索引顺序
    p = cycle(r + r[-2:0:-1])

    # 根据 p 生成的索引顺序对密文的字符进行排序
    idx = sorted(range(len(ciphertext)), key=lambda i: next(p))
    # 根据排序后的索引顺序重新排列密文字符，得到解密后的结果
    res = [''] * len(ciphertext)
    for i, c in zip(idx, ciphertext):
        res[i] = c
    # 将列表中的字符连接成最终的解密字符串并返回
    return ''.join(res)
################ Blum-Goldwasser cryptosystem  #########################

# 检查给定的 p 和 q 是否可以作为 Blum-Goldwasser 加密系统的私钥
def bg_private_key(p, q):
    """
    检查 p 和 q 是否可以作为 Blum-Goldwasser 加密系统的私钥。

    Explanation
    ===========

    对于 p 和 q 来说，必须通过以下三个必要检查才能作为私钥使用：
        1. p 和 q 必须都是素数
        2. p 和 q 必须不同
        3. p 和 q 必须模 4 余 3

    Parameters
    ==========

    p, q
        需要检查的密钥。

    Returns
    =======

    p, q
        输入的值。

    Raises
    ======

    ValueError
        如果 p 和 q 不满足上述条件。

    """

    # 检查 p 和 q 是否都是素数
    if not isprime(p) or not isprime(q):
        raise ValueError("the two arguments must be prime, "
                         "got %i and %i" %(p, q))
    # 检查 p 和 q 是否相等
    elif p == q:
        raise ValueError("the two arguments must be distinct, "
                         "got two copies of %i. " %p)
    # 检查 p 和 q 是否模 4 余 3
    elif (p - 3) % 4 != 0 or (q - 3) % 4 != 0:
        raise ValueError("the two arguments must be congruent to 3 mod 4, "
                         "got %i and %i" %(p, q))
    # 返回通过检查的 p 和 q
    return p, q

# 根据私钥计算公钥
def bg_public_key(p, q):
    """
    根据私钥计算公钥。

    Explanation
    ===========

    函数首先检查作为参数传递的私钥的有效性，
    然后返回它们的乘积。

    Parameters
    ==========

    p, q
        私钥。

    Returns
    =======

    N
        公钥。

    """
    # 调用私钥检查函数
    p, q = bg_private_key(p, q)
    # 计算公钥 N
    N = p * q
    return N

# 使用公钥和种子加密消息
def encipher_bg(i, key, seed=None):
    """
    使用公钥和种子加密消息。

    Explanation
    ===========

    算法：
        1. 将 i 编码为长度为 L 的比特串 m。
        2. 选择一个 1 < r < key 的随机数 r，并计算 x = r^2 mod key。
        3. 使用 BBS 伪随机数生成器以初始种子 x 生成长度为 L 的随机比特串 b。
        4. 加密消息，c_i = m_i XOR b_i，其中 1 <= i <= L。
        5. 计算 x_L = x^(2^L) mod key。
        6. 返回 (加密消息, x_L)。

    Parameters
    ==========

    i
        消息，非负整数。

    key
        公钥。

    Returns
    =======

    Tuple
        (加密后的消息, x_L)

    Raises
    ======

    ValueError
        如果 i 是负数。

    """

    # 检查消息是否为非负整数
    if i < 0:
        raise ValueError(
            "message must be a non-negative "
            "integer: got %d instead" % i)

    # 将消息 i 编码为比特串
    enc_msg = []
    while i > 0:
        enc_msg.append(i % 2)
        i //= 2
    enc_msg.reverse()
    L = len(enc_msg)

    # 选择随机数 r 并计算 x = r^2 mod key
    r = _randint(seed)(2, key - 1)
    x = r**2 % key
    x_L = pow(int(x), int(2**L), int(key))

    # 使用 BBS 伪随机数生成器生成随机比特串
    rand_bits = []
    for _ in range(L):
        rand_bits.append(x % 2)
        x = x**2 % key

    # 加密消息
    encrypt_msg = [m ^ b for (m, b) in zip(enc_msg, rand_bits)]

    return (encrypt_msg, x_L)

# 解密 Blum-Goldwasser 加密系统中的消息
def decipher_bg(message, key):
    """
    解密 Blum-Goldwasser 加密系统中的消息。

    """
    # 使用私钥解密消息。

    # 算法:
    #     1. 假设 c 是加密消息，y 是接收到的第二个数字，p 和 q 是私钥。
    #     2. 计算 r_p = y^((p+1)/4 ^ L) mod p 和 r_q = y^((q+1)/4 ^ L) mod q。
    #     3. 计算 x_0 = (q(q^-1 mod p)r_p + p(p^-1 mod q)r_q) mod N。
    #     4. 根据加密算法重新计算位，使用BBS生成器。
    #     5. 通过对 c 和 b 进行异或运算计算原始消息。

    # 参数:
    #     message
    #         加密消息的元组，包含加密消息和一个非负整数。
    #     key
    #         私钥的元组。
    
    # 返回:
    #     orig_msg
    #         原始消息。

    p, q = key
    encrypt_msg, y = message
    public_key = p * q  # 计算公钥 N
    L = len(encrypt_msg)  # 计算加密消息的长度
    p_t = ((p + 1)/4)**L  # 计算 p_t
    q_t = ((q + 1)/4)**L  # 计算 q_t
    r_p = pow(int(y), int(p_t), int(p))  # 计算 r_p
    r_q = pow(int(y), int(q_t), int(q))  # 计算 r_q

    x = (q * invert(q, p) * r_p + p * invert(p, q) * r_q) % public_key  # 计算 x

    orig_bits = []
    for _ in range(L):
        orig_bits.append(x % 2)  # 将 x 的最低位添加到 orig_bits
        x = x**2 % public_key  # 更新 x 的值

    orig_msg = 0
    for (m, b) in zip(encrypt_msg, orig_bits):
        orig_msg = orig_msg * 2  # 将 orig_msg 左移一位
        orig_msg += (m ^ b)  # 将 m 与 b 进行异或运算，并加到 orig_msg 中

    return orig_msg  # 返回解密后的原始消息
```