# `D:\src\scipysrc\sympy\sympy\crypto\tests\test_crypto.py`

```
# 导入 sympy 库中的符号模块，用于处理数学符号
from sympy.core import symbols
# 导入 sympy 库中的密码学模块，包括多个加密和解密函数
from sympy.crypto.crypto import (cycle_list,           # 导入列表循环函数
      encipher_shift, encipher_affine, encipher_substitution,   # 凯撒密码、仿射密码、替换密码加密函数
      check_and_join, encipher_vigenere, decipher_vigenere,     # 维吉尼亚密码加密解密函数
      encipher_hill, decipher_hill, encipher_bifid5, encipher_bifid6,  # Hill密码、Bifid5、Bifid6加密函数
      bifid5_square, bifid6_square, bifid5, bifid6,                 # Bifid5、Bifid6相关函数和方阵
      decipher_bifid5, decipher_bifid6, encipher_kid_rsa,           # Bifid5、Bifid6、Kid RSA加密解密函数
      decipher_kid_rsa, kid_rsa_private_key, kid_rsa_public_key,     # Kid RSA私钥、公钥相关函数
      decipher_rsa, rsa_private_key, rsa_public_key, encipher_rsa,  # RSA加密解密函数及其私钥、公钥函数
      lfsr_connection_polynomial, lfsr_autocorrelation, lfsr_sequence,  # LFSR（线性反馈移位寄存器）相关函数
      encode_morse, decode_morse, elgamal_private_key, elgamal_public_key,  # Morse编码解码、Elgamal密码私钥、公钥函数
      encipher_elgamal, decipher_elgamal, dh_private_key, dh_public_key,    # Elgamal密码加密解密、DH密钥相关函数
      dh_shared_key, decipher_shift, decipher_affine, encipher_bifid,        # DH共享密钥、凯撒密码、仿射密码、Bifid加密解密函数
      decipher_bifid, bifid_square, padded_key, uniq, decipher_gm,           # Bifid方阵、填充密钥、唯一性函数、GM密码解密函数
      encipher_gm, gm_public_key, gm_private_key, encipher_bg, decipher_bg,  # GM密码加密函数、BG密码加密解密函数及其密钥
      bg_private_key, bg_public_key, encipher_rot13, decipher_rot13,         # BG密码私钥、公钥、ROT13加密解密函数
      encipher_atbash, decipher_atbash, NonInvertibleCipherWarning,           # Atbash密码加密解密函数、不可逆密码警告
      encipher_railfence, decipher_railfence)                               # 栅栏密码加密解密函数

# 导入 sympy.external.gmpy 库中的最大公约数函数 gcd
from sympy.external.gmpy import gcd
# 导入 sympy 库中的矩阵模块
from sympy.matrices import Matrix
# 导入 sympy 库中的数论模块，包括判断素数和原根的函数
from sympy.ntheory import isprime, is_primitive_root
# 导入 sympy 库中的多项式环模块
from sympy.polys.domains import FF
# 导入 sympy.testing.pytest 中的异常处理函数 raises 和警告函数 warns
from sympy.testing.pytest import raises, warns

# 导入 sympy.core.random 库中的随机数生成函数 randrange
from sympy.core.random import randrange

# 定义测试函数 test_encipher_railfence，测试栅栏密码加密功能
def test_encipher_railfence():
    # 验证栅栏密码加密结果与预期结果是否一致
    assert encipher_railfence("hello world",2) == "hlowrdel ol"
    assert encipher_railfence("hello world",3) == "horel ollwd"
    assert encipher_railfence("hello world",4) == "hwe olordll"

# 定义测试函数 test_decipher_railfence，测试栅栏密码解密功能
def test_decipher_railfence():
    # 验证栅栏密码解密结果与预期结果是否一致
    assert decipher_railfence("hlowrdel ol",2) == "hello world"
    assert decipher_railfence("horel ollwd",3) == "hello world"
    assert decipher_railfence("hwe olordll",4) == "hello world"


# 定义测试函数 test_cycle_list，测试列表循环函数 cycle_list
def test_cycle_list():
    # 验证列表循环函数的输出结果是否符合预期
    assert cycle_list(3, 4) == [3, 0, 1, 2]
    assert cycle_list(-1, 4) == [3, 0, 1, 2]
    assert cycle_list(1, 4) == [1, 2, 3, 0]


# 定义测试函数 test_encipher_shift，测试凯撒密码加密和解密函数
def test_encipher_shift():
    # 验证凯撒密码加密和解密函数的正确性
    assert encipher_shift("ABC", 0) == "ABC"
    assert encipher_shift("ABC", 1) == "BCD"
    assert encipher_shift("ABC", -1) == "ZAB"
    assert decipher_shift("ZAB", -1) == "ABC"

# 定义测试函数 test_encipher_rot13，测试 ROT13 加密和解密函数
def test_encipher_rot13():
    # 验证 ROT13 加密和解密函数的正确性
    assert encipher_rot13("ABC") == "NOP"
    assert encipher_rot13("NOP") == "ABC"
    assert decipher_rot13("ABC") == "NOP"
    assert decipher_rot13("NOP") == "ABC"


# 定义测试函数 test_encipher_affine，测试仿射密码加密和解密函数
def test_encipher_affine():
    # 验证仿射密码加密和解密函数的正确性
    assert encipher_affine("ABC", (1, 0)) == "ABC"
    assert encipher_affine("ABC", (1, 1)) == "BCD"
    assert encipher_affine("ABC", (-1, 0)) == "AZY"
    assert encipher_affine("ABC", (-1, 1), symbols="ABCD") == "BAD"
    assert encipher_affine("123", (-1, 1), symbols="1234") == "214"
    assert encipher_affine("ABC", (3, 16)) == "QTW"
    assert decipher_affine("QTW", (3, 16)) == "ABC"

# 定义测试函数 test_encipher_atbash，测试 Atbash 密码加密和解密函数
def test_encipher_atbash():
    # 验证 Atbash 密码加密和解密函数的正确性
    assert encipher_atbash("ABC") == "ZYX"
    assert encipher_atbash("ZYX") == "ABC"
    assert decipher_atbash("ABC") == "ZYX"
    # 使用 assert 断言函数 decipher_atbash("ZYX") 的返回结果是否等于 "ABC"
    assert decipher_atbash("ZYX") == "ABC"
# 测试函数，用于测试 encipher_substitution 函数
def test_encipher_substitution():
    # 断言加密函数 encipher_substitution 的返回结果是否符合预期
    assert encipher_substitution("ABC", "BAC", "ABC") == "BAC"
    assert encipher_substitution("123", "1243", "1234") == "124"

# 测试函数，用于测试 check_and_join 函数
def test_check_and_join():
    # 断言 check_and_join 函数对不同输入的返回结果是否符合预期
    assert check_and_join("abc") == "abc"
    assert check_and_join(uniq("aaabc")) == "abc"
    assert check_and_join("ab c".split()) == "abc"
    assert check_and_join("abc", "a", filter=True) == "a"
    # 断言 check_and_join 函数对异常情况的处理是否正确
    raises(ValueError, lambda: check_and_join('ab', 'a'))

# 测试函数，用于测试 encipher_vigenere 函数
def test_encipher_vigenere():
    # 断言加密函数 encipher_vigenere 的返回结果是否符合预期
    assert encipher_vigenere("ABC", "ABC") == "ACE"
    assert encipher_vigenere("ABC", "ABC", symbols="ABCD") == "ACA"
    assert encipher_vigenere("ABC", "AB", symbols="ABCD") == "ACC"
    assert encipher_vigenere("AB", "ABC", symbols="ABCD") == "AC"
    assert encipher_vigenere("A", "ABC", symbols="ABCD") == "A"

# 测试函数，用于测试 decipher_vigenere 函数
def test_decipher_vigenere():
    # 断言解密函数 decipher_vigenere 的返回结果是否符合预期
    assert decipher_vigenere("ABC", "ABC") == "AAA"
    assert decipher_vigenere("ABC", "ABC", symbols="ABCD") == "AAA"
    assert decipher_vigenere("ABC", "AB", symbols="ABCD") == "AAC"
    assert decipher_vigenere("AB", "ABC", symbols="ABCD") == "AA"
    assert decipher_vigenere("A", "ABC", symbols="ABCD") == "A"

# 测试函数，用于测试 encipher_hill 函数
def test_encipher_hill():
    # 创建 2x2 矩阵 A 作为加密密钥
    A = Matrix(2, 2, [1, 2, 3, 5])
    # 断言加密函数 encipher_hill 的返回结果是否符合预期
    assert encipher_hill("ABCD", A) == "CFIV"
    A = Matrix(2, 2, [1, 0, 0, 1])
    assert encipher_hill("ABCD", A) == "ABCD"
    assert encipher_hill("ABCD", A, symbols="ABCD") == "ABCD"
    A = Matrix(2, 2, [1, 2, 3, 5])
    assert encipher_hill("ABCD", A, symbols="ABCD") == "CBAB"
    assert encipher_hill("AB", A, symbols="ABCD") == "CB"
    # 断言加密函数 encipher_hill 对消息长度不是密钥长度倍数的情况的处理是否正确
    assert encipher_hill("ABA", A) == "CFGC"
    assert encipher_hill("ABA", A, pad="Z") == "CFYV"

# 测试函数，用于测试 decipher_hill 函数
def test_decipher_hill():
    # 创建 2x2 矩阵 A 作为解密密钥
    A = Matrix(2, 2, [1, 2, 3, 5])
    # 断言解密函数 decipher_hill 的返回结果是否符合预期
    assert decipher_hill("CFIV", A) == "ABCD"
    A = Matrix(2, 2, [1, 0, 0, 1])
    assert decipher_hill("ABCD", A) == "ABCD"
    assert decipher_hill("ABCD", A, symbols="ABCD") == "ABCD"
    A = Matrix(2, 2, [1, 2, 3, 5])
    assert decipher_hill("CBAB", A, symbols="ABCD") == "ABCD"
    assert decipher_hill("CB", A, symbols="ABCD") == "AB"
    # 断言解密函数 decipher_hill 对消息长度不是密钥长度倍数的情况的处理是否正确
    assert decipher_hill("CFA", A) == "ABAA"

# 测试函数，用于测试 encipher_bifid5 函数
def test_encipher_bifid5():
    # 断言加密函数 encipher_bifid5 的返回结果是否符合预期
    assert encipher_bifid5("AB", "AB") == "AB"
    assert encipher_bifid5("AB", "CD") == "CO"
    assert encipher_bifid5("ab", "c") == "CH"
    assert encipher_bifid5("a bc", "b") == "BAC"

# 测试函数，用于测试 bifid5_square 函数
def test_bifid5_square():
    # 获取 bifid5 矩阵的符号方阵 M
    A = bifid5
    f = lambda i, j: symbols(A[5*i + j])
    M = Matrix(5, 5, f)
    # 断言 bifid5_square 函数返回的矩阵是否符合预期
    assert bifid5_square("") == M

# 测试函数，用于测试 decipher_bifid5 函数
def test_decipher_bifid5():
    # 断言解密函数 decipher_bifid5 的返回结果是否符合预期
    assert decipher_bifid5("AB", "AB") == "AB"
    assert decipher_bifid5("CO", "CD") == "AB"
    assert decipher_bifid5("ch", "c") == "AB"
    assert decipher_bifid5("b ac", "b") == "ABC"

# 测试函数，用于测试 encipher_bifid6 函数
def test_encipher_bifid6():
    # 断言加密函数 encipher_bifid6 的返回结果是否符合预期
    assert encipher_bifid6("AB", "AB") == "AB"
    assert encipher_bifid6("AB", "CD") == "CP"
    assert encipher_bifid6("ab", "c") == "CI"
    # 断言测试函数 encipher_bifid6 的结果是否符合预期
    assert encipher_bifid6("a bc", "b") == "BAC"
def test_decipher_bifid6():
    # 测试 decipher_bifid6 函数，验证其对输入的不同参数返回预期的解密结果
    assert decipher_bifid6("AB", "AB") == "AB"
    assert decipher_bifid6("CP", "CD") == "AB"
    assert decipher_bifid6("ci", "c") == "AB"
    assert decipher_bifid6("b ac", "b") == "ABC"


def test_bifid6_square():
    # 测试 bifid6_square 函数，确保其能够正确生成一个 6x6 的矩阵 M
    A = bifid6
    f = lambda i, j: symbols(A[6*i + j])
    M = Matrix(6, 6, f)
    assert bifid6_square("") == M


def test_rsa_public_key():
    # 测试 rsa_public_key 函数，验证其根据给定的参数能够正确计算公钥
    assert rsa_public_key(2, 3, 1) == (6, 1)
    assert rsa_public_key(5, 3, 3) == (15, 3)

    # 测试非可逆加密警告
    with warns(NonInvertibleCipherWarning):
        assert rsa_public_key(2, 2, 1) == (4, 1)
        assert rsa_public_key(8, 8, 8) is False


def test_rsa_private_key():
    # 测试 rsa_private_key 函数，验证其根据给定的参数能够正确计算私钥
    assert rsa_private_key(2, 3, 1) == (6, 1)
    assert rsa_private_key(5, 3, 3) == (15, 3)
    assert rsa_private_key(23,29,5) == (667,493)

    # 测试非可逆加密警告
    with warns(NonInvertibleCipherWarning):
        assert rsa_private_key(2, 2, 1) == (4, 1)
        assert rsa_private_key(8, 8, 8) is False


def test_rsa_large_key():
    # 测试 rsa_public_key 和 rsa_private_key 函数，验证其对于大数值的处理是否正确
    # 示例来源于指定的网站链接
    p = int('101565610013301240713207239558950144682174355406589305284428666'\
        '903702505233009')
    q = int('894687191887545488935455605955948413812376003053143521429242133'\
        '12069293984003')
    e = int('65537')
    d = int('893650581832704239530398858744759129594796235440844479456143566'\
        '6999402846577625762582824202269399672579058991442587406384754958587'\
        '400493169361356902030209')
    assert rsa_public_key(p, q, e) == (p*q, e)
    assert rsa_private_key(p, q, e) == (p*q, d)


def test_encipher_rsa():
    # 测试 encipher_rsa 函数，验证其能够正确加密消息
    puk = rsa_public_key(2, 3, 1)
    assert encipher_rsa(2, puk) == 2
    puk = rsa_public_key(5, 3, 3)
    assert encipher_rsa(2, puk) == 8

    # 测试非可逆加密警告
    with warns(NonInvertibleCipherWarning):
        puk = rsa_public_key(2, 2, 1)
        assert encipher_rsa(2, puk) == 2


def test_decipher_rsa():
    # 测试 decipher_rsa 函数，验证其能够正确解密消息
    prk = rsa_private_key(2, 3, 1)
    assert decipher_rsa(2, prk) == 2
    prk = rsa_private_key(5, 3, 3)
    assert decipher_rsa(8, prk) == 2

    # 测试非可逆加密警告
    with warns(NonInvertibleCipherWarning):
        prk = rsa_private_key(2, 2, 1)
        assert decipher_rsa(2, prk) == 2


def test_mutltiprime_rsa_full_example():
    # 测试多质数的 RSA 示例，验证加密、解密过程是否正确
    # 示例来源于指定的网站链接和文献
    puk = rsa_public_key(2, 3, 5, 7, 11, 13, 7)
    prk = rsa_private_key(2, 3, 5, 7, 11, 13, 7)
    assert puk == (30030, 7)
    assert prk == (30030, 823)

    msg = 10
    encrypted = encipher_rsa(2 * msg - 15, puk)
    assert encrypted == 18065
    decrypted = (decipher_rsa(encrypted, prk) + 15) / 2
    assert decrypted == msg

    # 另一个文献的示例
    puk1 = rsa_public_key(53, 41, 43, 47, 41)
    prk1 = rsa_private_key(53, 41, 43, 47, 41)
    puk2 = rsa_public_key(53, 41, 43, 47, 97)
    prk2 = rsa_private_key(53, 41, 43, 47, 97)

    assert puk1 == (4391633, 41)
    # 确认 prk1 的值是否等于 (4391633, 294041)
    assert prk1 == (4391633, 294041)
    # 确认 puk2 的值是否等于 (4391633, 97)
    assert puk2 == (4391633, 97)
    # 确认 prk2 的值是否等于 (4391633, 455713)
    assert prk2 == (4391633, 455713)

    # 设定消息内容为整数 12321
    msg = 12321
    # 使用 puk1 对消息进行 RSA 加密，并再次使用 puk2 进行 RSA 加密
    encrypted = encipher_rsa(encipher_rsa(msg, puk1), puk2)
    # 确认加密后的消息是否等于 1081588
    assert encrypted == 1081588
    # 对加密后的消息进行 RSA 解密，先使用 prk2，然后再使用 prk1
    decrypted = decipher_rsa(decipher_rsa(encrypted, prk2), prk1)
    # 确认解密后的消息是否等于初始设定的消息内容
    assert decrypted == msg
def test_rsa_crt_extreme():
    p = int(
        '10177157607154245068023861503693082120906487143725062283406501' \
        '54082258226204046999838297167140821364638180697194879500245557' \
        '65445186962893346463841419427008800341257468600224049986260471' \
        '92257248163014468841725476918639415726709736077813632961290911' \
        '0256421232977833028677441206049309220354796014376698325101693')

    q = int(
        '28752342353095132872290181526607275886182793241660805077850801' \
        '75689512797754286972952273553128181861830576836289738668745250' \
        '34028199691128870676414118458442900035778874482624765513861643' \
        '27966696316822188398336199002306588703902894100476186823849595' \
        '103239410527279605442148285816149368667083114802852804976893')

    r = int(
        '17698229259868825776879500736350186838850961935956310134378261' \
        '89771862186717463067541369694816245225291921138038800171125596' \
        '07315449521981157084370187887650624061033066022458512942411841' \
        '18747893789972315277160085086164119879536041875335384844820566' \
        '0287479617671726408053319619892052000850883994343378882717849')

    s = int(
        '68925428438585431029269182233502611027091755064643742383515623' \
        '64321310582896893395529367074942808353187138794422745718419645' \
        '28291231865157212604266903677599180789896916456120289112752835' \
        '98502265889669730331688206825220074713977607415178738015831030' \
        '364290585369150502819743827343552098197095520550865360159439'
    )

    t = int(
        '69035483433453632820551311892368908779778144568711455301541094' \
        '31487047642322695357696860925747923189635033183069823820910521' \
        '71172909106797748883261493224162414050106920442445896819806600' \
        '15448444826108008217972129130625571421904893252804729877353352' \
        '739420480574842850202181462656251626522910618936534699566291'
    )

    e = 65537
    # 使用给定的素数和指数生成 RSA 公钥
    puk = rsa_public_key(p, q, r, s, t, e)
    # 使用给定的素数和指数生成 RSA 私钥
    prk = rsa_private_key(p, q, r, s, t, e)

    plaintext = 1000
    # 使用公钥加密明文，返回密文
    ciphertext_1 = encipher_rsa(plaintext, puk)
    # 使用公钥加密明文，传入素数参数的列表，返回密文
    ciphertext_2 = encipher_rsa(plaintext, puk, [p, q, r, s, t])
    # 断言两种加密方式得到的密文相同
    assert ciphertext_1 == ciphertext_2
    # 使用私钥解密密文，返回解密后的明文，不传入素数参数列表
    assert decipher_rsa(ciphertext_1, prk) == \
        decipher_rsa(ciphertext_1, prk, [p, q, r, s, t])


def test_rsa_exhaustive():
    p, q = 61, 53
    e = 17
    # 使用给定的素数和指数生成 RSA 公钥，使用 Carmichael 函数计算欧拉数
    puk = rsa_public_key(p, q, e, totient='Carmichael')
    # 使用给定的素数和指数生成 RSA 私钥，使用 Carmichael 函数计算欧拉数
    prk = rsa_private_key(p, q, e, totient='Carmichael')

    for msg in range(puk[0]):
        # 使用公钥加密消息，返回密文
        encrypted = encipher_rsa(msg, puk)
        # 使用私钥解密密文，返回解密后的明文
        decrypted = decipher_rsa(encrypted, prk)
        try:
            # 断言解密后的明文与原始消息相同
            assert decrypted == msg
        except AssertionError:
            # 如果断言失败，抛出带有详细信息的 AssertionError
            raise AssertionError(
                "The RSA is not correctly decrypted " \
                "(Original : {}, Encrypted : {}, Decrypted : {})" \
                .format(msg, encrypted, decrypted)
                )


def test_rsa_multiprime_exhanstive():
    # 此函数尚未完整定义，在后续的实现中需要补充完整
    pass
    # 定义一个质数列表
    primes = [3, 5, 7, 11]
    # 定义变量 e 并赋值为 7
    e = 7
    # 将质数列表 primes 和变量 e 合并成一个参数列表 args
    args = primes + [e]
    # 使用参数调用 rsa_public_key 函数生成 RSA 公钥
    puk = rsa_public_key(*args, totient='Carmichael')
    # 使用参数调用 rsa_private_key 函数生成 RSA 私钥
    prk = rsa_private_key(*args, totient='Carmichael')
    # 从公钥中获取模数 n
    n = puk[0]

    # 对于从 0 到 n 的每一个整数 msg
    for msg in range(n):
        # 使用公钥 puk 对消息 msg 进行 RSA 加密
        encrypted = encipher_rsa(msg, puk)
        # 使用私钥 prk 对加密后的消息进行 RSA 解密
        decrypted = decipher_rsa(encrypted, prk)
        # 尝试断言解密后的消息应该等于原始消息 msg
        try:
            assert decrypted == msg
        # 如果断言失败，则抛出 AssertionError 异常，并显示详细错误信息
        except AssertionError:
            raise AssertionError(
                "The RSA is not correctly decrypted " \
                "(Original : {}, Encrypted : {}, Decrypted : {})" \
                .format(msg, encrypted, decrypted)
                )
# 测试 RSA 多幂尽量全面
def test_rsa_multipower_exhanstive():
    # 定义素数列表
    primes = [5, 5, 7]
    # 设置 RSA 加密指数 e
    e = 7
    # 将素数列表和 e 合并为参数列表
    args = primes + [e]
    # 调用 RSA 公钥生成函数，启用多幂选项
    puk = rsa_public_key(*args, multipower=True)
    # 调用 RSA 私钥生成函数，启用多幂选项
    prk = rsa_private_key(*args, multipower=True)
    # 获取公钥的模数 n
    n = puk[0]

    # 遍历从 0 到 n 的消息
    for msg in range(n):
        # 若消息 msg 与 n 的最大公约数不为 1，则继续下一个消息
        if gcd(msg, n) != 1:
            continue

        # 使用 RSA 公钥加密消息
        encrypted = encipher_rsa(msg, puk)
        # 使用 RSA 私钥解密消息
        decrypted = decipher_rsa(encrypted, prk)
        # 检查解密后的消息是否与原始消息相等
        try:
            assert decrypted == msg
        except AssertionError:
            raise AssertionError(
                "The RSA is not correctly decrypted " \
                "(Original : {}, Encrypted : {}, Decrypted : {})" \
                .format(msg, encrypted, decrypted)
                )


# 测试 Kid RSA 公钥生成函数
def test_kid_rsa_public_key():
    # 验证 Kid RSA 公钥生成函数的输出结果
    assert kid_rsa_public_key(1, 2, 1, 1) == (5, 2)
    assert kid_rsa_public_key(1, 2, 2, 1) == (8, 3)
    assert kid_rsa_public_key(1, 2, 1, 2) == (7, 2)


# 测试 Kid RSA 私钥生成函数
def test_kid_rsa_private_key():
    # 验证 Kid RSA 私钥生成函数的输出结果
    assert kid_rsa_private_key(1, 2, 1, 1) == (5, 3)
    assert kid_rsa_private_key(1, 2, 2, 1) == (8, 3)
    assert kid_rsa_private_key(1, 2, 1, 2) == (7, 4)


# 测试 Kid RSA 加密函数
def test_encipher_kid_rsa():
    # 验证 Kid RSA 加密函数的输出结果
    assert encipher_kid_rsa(1, (5, 2)) == 2
    assert encipher_kid_rsa(1, (8, 3)) == 3
    assert encipher_kid_rsa(1, (7, 2)) == 2


# 测试 Kid RSA 解密函数
def test_decipher_kid_rsa():
    # 验证 Kid RSA 解密函数的输出结果
    assert decipher_kid_rsa(2, (5, 3)) == 1
    assert decipher_kid_rsa(3, (8, 3)) == 1
    assert decipher_kid_rsa(2, (7, 4)) == 1


# 测试 Morse 编码函数
def test_encode_morse():
    # 验证 Morse 编码函数的输出结果
    assert encode_morse('ABC') == '.-|-...|-.-.'
    assert encode_morse('SMS ') == '...|--|...||'
    assert encode_morse('SMS\n') == '...|--|...||'
    assert encode_morse('') == ''
    assert encode_morse(' ') == '||'
    assert encode_morse(' ', sep='`') == '``'
    assert encode_morse(' ', sep='``') == '````'
    assert encode_morse('!@#$%^&*()_+') == '-.-.--|.--.-.|...-..-|-.--.|-.--.-|..--.-|.-.-.'
    assert encode_morse('12345') == '.----|..---|...--|....-|.....'
    assert encode_morse('67890') == '-....|--...|---..|----.|-----'


# 测试 Morse 解码函数
def test_decode_morse():
    # 验证 Morse 解码函数的输出结果
    assert decode_morse('-.-|.|-.--') == 'KEY'
    assert decode_morse('.-.|..-|-.||') == 'RUN'
    # 测试解码未知 Morse 代码引发 KeyError 的情况
    raises(KeyError, lambda: decode_morse('.....----'))


# 测试 LFSR 序列生成函数
def test_lfsr_sequence():
    # 测试输入参数类型错误时是否会引发 TypeError
    raises(TypeError, lambda: lfsr_sequence(1, [1], 1))
    raises(TypeError, lambda: lfsr_sequence([1], 1, 1))
    # 创建有限域对象 F
    F = FF(2)
    # 验证 LFSR 序列生成函数的输出结果
    assert lfsr_sequence([F(1)], [F(1)], 2) == [F(1), F(1)]
    assert lfsr_sequence([F(0)], [F(1)], 2) == [F(1), F(0)]
    F = FF(3)
    assert lfsr_sequence([F(1)], [F(1)], 2) == [F(1), F(1)]
    assert lfsr_sequence([F(0)], [F(2)], 2) == [F(2), F(0)]
    assert lfsr_sequence([F(1)], [F(2)], 2) == [F(2), F(2)]


# 测试 LFSR 自相关函数
def test_lfsr_autocorrelation():
    # 测试输入参数类型错误时是否会引发 TypeError
    raises(TypeError, lambda: lfsr_autocorrelation(1, 2, 3))
    # 创建有限域对象 F
    F = FF(2)
    # 生成 LFSR 序列 s
    s = lfsr_sequence([F(1), F(0)], [F(0), F(1)], 5)
    # 验证 LFSR 自相关函数的输出结果
    assert lfsr_autocorrelation(s, 2, 0) == 1
    assert lfsr_autocorrelation(s, 2, 1) == -1


# 测试 LFSR 连接多项式函数
def test_lfsr_connection_polynomial():
    # 创建有限域对象 F
    F = FF(2)
    # 定义符号变量 x
    x = symbols("x")
    # 生成 LFSR 序列 s
    s = lfsr_sequence([F(1), F(0)], [F(0), F(1)], 5)
    # 断言：验证 lfsr_connection_polynomial(s) 的返回结果是否等于 x**2 + 1
    assert lfsr_connection_polynomial(s) == x**2 + 1
    
    # 使用给定的初始状态和反馈多项式生成长度为 5 的线性反馈移位寄存器（LFSR）序列
    s = lfsr_sequence([F(1), F(1)], [F(0), F(1)], 5)
    
    # 断言：验证生成的 LFSR 序列 s 的连接多项式是否等于 x**2 + x + 1
    assert lfsr_connection_polynomial(s) == x**2 + x + 1
# 测试 ElGamal 算法生成私钥的函数
def test_elgamal_private_key():
    # 调用 elgamal_private_key 函数，返回元组 (a, b, _)
    a, b, _ = elgamal_private_key(digit=100)
    # 断言 a 是素数
    assert isprime(a)
    # 断言 b 是 a 的原根
    assert is_primitive_root(b, a)
    # 断言 a 的二进制表示长度至少为 102
    assert len(bin(a)) >= 102


# 测试 ElGamal 算法加密解密功能
def test_elgamal():
    # 生成 ElGamal 算法的私钥
    dk = elgamal_private_key(5)
    # 根据私钥生成 ElGamal 算法的公钥
    ek = elgamal_public_key(dk)
    # 取出公钥的第一个元素 P
    P = ek[0]
    # 断言 P-1 经过加密和解密后等于 P-1
    assert P - 1 == decipher_elgamal(encipher_elgamal(P - 1, ek), dk)
    # 断言加密 P 会引发 ValueError 异常
    raises(ValueError, lambda: encipher_elgamal(P, dk))
    # 断言加密 -1 会引发 ValueError 异常
    raises(ValueError, lambda: encipher_elgamal(-1, dk))


# 测试 Diffie-Hellman 算法生成私钥的函数
def test_dh_private_key():
    # 调用 dh_private_key 函数，返回元组 (p, g, _)
    p, g, _ = dh_private_key(digit=100)
    # 断言 p 是素数
    assert isprime(p)
    # 断言 g 是模 p 的原根
    assert is_primitive_root(g, p)
    # 断言 p 的二进制表示长度至少为 102
    assert len(bin(p)) >= 102


# 测试 Diffie-Hellman 算法生成公钥的函数
def test_dh_public_key():
    # 生成 Diffie-Hellman 算法的私钥 (p1, g1, a)
    p1, g1, a = dh_private_key(digit=100)
    # 根据私钥生成 Diffie-Hellman 算法的公钥 (p2, g2, ga)
    p2, g2, ga = dh_public_key((p1, g1, a))
    # 断言 p1 和 p2 相等
    assert p1 == p2
    # 断言 g1 和 g2 相等
    assert g1 == g2
    # 断言 ga 等于 g1^a mod p1
    assert ga == pow(g1, a, p1)


# 测试 Diffie-Hellman 算法生成共享密钥的函数
def test_dh_shared_key():
    # 生成 Diffie-Hellman 算法的私钥 (p, _, ga)
    prk = dh_private_key(digit=100)
    # 根据私钥生成 Diffie-Hellman 算法的公钥 (p, _, ga)
    p, _, ga = dh_public_key(prk)
    # 生成随机整数 b，范围在 2 到 p-1 之间
    b = randrange(2, p)
    # 计算共享密钥 sk = ga^b mod p
    sk = dh_shared_key((p, _, ga), b)
    # 断言 sk 等于 ga^b mod p
    assert sk == pow(ga, b, p)
    # 断言对于不合法参数 (1031, 14, 565)，调用 dh_shared_key 会引发 ValueError 异常
    raises(ValueError, lambda: dh_shared_key((1031, 14, 565), 2000))


# 测试字符串填充函数 padded_key
def test_padded_key():
    # 断言 padded_key('b', 'ab') 返回 'ba'
    assert padded_key('b', 'ab') == 'ba'
    # 断言对于不合法参数 ('ab', 'ace')，调用 padded_key 会引发 ValueError 异常
    raises(ValueError, lambda: padded_key('ab', 'ace'))
    # 断言对于不合法参数 ('ab', 'abba')，调用 padded_key 会引发 ValueError 异常
    raises(ValueError, lambda: padded_key('ab', 'abba'))


# 测试 Bifid 加密算法相关函数
def test_bifid():
    # 断言对于不合法参数 ('abc', 'b', 'abcde')，调用 encipher_bifid 会引发 ValueError 异常
    raises(ValueError, lambda: encipher_bifid('abc', 'b', 'abcde'))
    # 断言 encipher_bifid('abc', 'b', 'abcd') 返回 'bdb'
    assert encipher_bifid('abc', 'b', 'abcd') == 'bdb'
    # 断言对于不合法参数 ('bdb', 'b', 'abcde')，调用 decipher_bifid 会引发 ValueError 异常
    raises(ValueError, lambda: decipher_bifid('bdb', 'b', 'abcde'))
    # 断言 decipher_bifid('bdb', 'b', 'abcd') 返回 'abc'
    assert encipher_bifid('bdb', 'b', 'abcd') == 'abc'
    # 断言对于不合法参数 ('abcde')，调用 bifid_square 会引发 ValueError 异常
    raises(ValueError, lambda: bifid_square('abcde'))
    # 断言 bifid5_square("B") 等于 bifid5_square('BACDEFGHIKLMNOPQRSTUVWXYZ')
    assert bifid5_square("B") == bifid5_square('BACDEFGHIKLMNOPQRSTUVWXYZ')
    # 断言 bifid6_square('B0') 等于 bifid6_square('B0ACDEFGHIJKLMNOPQRSTUVWXYZ123456789')
    assert bifid6_square('B0') == bifid6_square('B0ACDEFGHIJKLMNOPQRSTUVWXYZ123456789')


# 测试 Goldwasser-Micali 加密算法的加密解密过程
def test_encipher_decipher_gm():
    # 待测试的素数列表 ps 和 相应的 q 列表 qs
    ps = [131, 137, 139, 149, 151, 157, 163, 167,
          173, 179, 181, 191, 193, 197, 199]
    qs = [89, 97, 101, 103, 107, 109, 113, 127,
          131, 137, 139, 149, 151, 157, 47]
    # 待测试的消息列表 messages
    messages = [
        0, 32855, 34303, 14805, 1280, 75859, 38368,
        724, 60356, 51675, 76697, 61854, 18661,
    ]
    # 遍历素数 ps 和 相应的 q 列表 qs
    for p, q in zip(ps, qs):
        # 生成 Goldwasser-Micali 算法的私钥
        pri = gm_private_key(p, q)
        # 遍历消息列表 messages
        for msg in messages:
            # 生成 Goldwasser-Micali 算法的公钥
            pub = gm_public_key(p, q)
            # 使用公钥对消息进行加密
            enc = encipher_gm(msg, pub)
            # 使用私钥对加密消息进行解密
            dec = decipher_gm(enc, pri)
            # 断言解密后的消息等于原始消息
            assert dec == msg


# 测试 Goldwasser-Micali 算法生成私钥的函数
def test_gm_private_key():
    # 断言对于不合法参数 (13, 15)，调用 gm_public_key 会引发 ValueError 异常
    raises(ValueError, lambda: gm_public_key(13, 15))
    # 断言对于不合法参数 (0, 0)，调用 gm_public_key 会引发 ValueError 异常
    raises(ValueError, lambda: gm_public_key(0, 0))
    # 断言对于不合法参数 (0, 5)，调用 gm_public_key 会引发 ValueError 异常
    raises(ValueError, lambda: gm_public_key(0, 5))
    # 断言 gm_public_key(17, 19) 返回 (17, 19)
    assert 17, 19 == gm_public_key(17
    # 定义一组整数列表，表示待加密的消息列表
    messages = [
        0, 328, 343, 148, 1280, 758, 383,
        724, 603, 516, 766, 618, 186,
    ]
    
    # 使用 zip 函数将 ps 和 qs 中的元素一一配对，分别赋给 p 和 q
    for p, q in zip(ps, qs):
        # 根据 p 和 q 计算出的私钥
        pri = bg_private_key(p, q)
        
        # 遍历消息列表中的每个消息
        for msg in messages:
            # 根据 p 和 q 计算出的公钥
            pub = bg_public_key(p, q)
            
            # 使用公钥 pub 对消息 msg 进行加密
            enc = encipher_bg(msg, pub)
            
            # 使用私钥 pri 对加密后的消息 enc 进行解密
            dec = decipher_bg(enc, pri)
            
            # 断言解密后的消息应该与原始消息 msg 相同
            assert dec == msg
# 测试函数，用于验证 bg_private_key 函数的行为
def test_bg_private_key():
    # 检查当输入参数分别为 8 和 16 时，是否引发 ValueError 异常
    raises(ValueError, lambda: bg_private_key(8, 16))
    # 检查当输入参数分别为 8 和 8 时，是否引发 ValueError 异常
    raises(ValueError, lambda: bg_private_key(8, 8))
    # 检查当输入参数分别为 13 和 17 时，是否引发 ValueError 异常
    raises(ValueError, lambda: bg_private_key(13, 17))
    # 检查返回值是否等于 (23, 31) 的 bg_private_key 函数的返回值，但实际上此断言有误，应该修正如下：
    # 检查 bg_private_key(23, 31) 的返回值是否为 (23, 31)，修正如下：
    assert 23, 31 == bg_private_key(23, 31)

# 测试函数，用于验证 bg_public_key 函数的行为
def test_bg_public_key():
    # 检查 bg_public_key(67, 79) 的返回值是否为 5293
    assert 5293 == bg_public_key(67, 79)
    # 检查 bg_public_key(23, 31) 的返回值是否为 713
    assert 713 == bg_public_key(23, 31)
    # 检查当输入参数分别为 13 和 17 时，是否引发 ValueError 异常（应该是调用 bg_public_key 而不是 bg_private_key）
    raises(ValueError, lambda: bg_private_key(13, 17))
```