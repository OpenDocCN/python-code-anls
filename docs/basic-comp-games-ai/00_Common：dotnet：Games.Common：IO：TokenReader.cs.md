# `00_Common\dotnet\Games.Common\IO\TokenReader.cs`

```
# 导入必要的模块
import System
import Collections.Generic
import Linq
from Games.Common.IO import Strings

# 命名空间
namespace Games.Common.IO;

# 类的注释
"""
从输入中读取并组装给定数量的值或标记，可能跨多个输入行。
"""

# 内部类 TokenReader
class TokenReader:
    # 构造函数，接受 TextIO 对象和 Token 判断函数作为参数
    def __init__(self, io, isTokenValid):
        # 初始化 TextIO 对象
        self._io = io
        # 初始化 Token 判断函数，如果未提供则默认为返回 True
        self._isTokenValid = isTokenValid if isTokenValid else (lambda t: True)
    /// <summary>
    /// Creates a <see cref="TokenReader" /> which reads string tokens.
    /// </summary>
    /// <param name="io">A <see cref="TextIO" /> instance.</param>
    /// <returns>The new <see cref="TokenReader" /> instance.</returns>
    public static TokenReader ForStrings(TextIO io) => new(io, t => true);
```
这段代码是一个静态方法，用于创建一个TokenReader实例，该实例用于读取字符串类型的token。它接受一个TextIO实例作为参数，并返回一个新的TokenReader实例。

```
    /// <summary>
    /// Creates a <see cref="TokenReader" /> which reads tokens and validates that they can be parsed as numbers.
    /// </summary>
    /// <param name="io">A <see cref="TextIO" /> instance.</param>
    /// <returns>The new <see cref="TokenReader" /> instance.</returns>
    public static TokenReader ForNumbers(TextIO io) => new(io, t => t.IsNumber);
```
这段代码也是一个静态方法，用于创建一个TokenReader实例，该实例用于读取token并验证它们是否可以解析为数字。它接受一个TextIO实例作为参数，并返回一个新的TokenReader实例。

```
    /// <summary>
    /// Reads valid tokens from one or more input lines and builds a list with the required quantity.
    /// </summary>
    /// <param name="prompt">The string used to prompt the user for input.</param>
    /// <param name="quantityNeeded">The number of tokens required.</param>
```
这段代码是一个方法，用于从一个或多个输入行中读取有效的token，并构建一个包含所需数量的token的列表。它接受一个用于提示用户输入的字符串和所需的token数量作为参数。
    # 读取标记的序列
    def ReadTokens(prompt, quantityNeeded):
        # 如果需要的数量为0，则抛出参数异常
        if quantityNeeded == 0:
            raise ValueError(f"'{quantityNeeded}' must be greater than zero.")
        
        # 创建一个标记列表
        tokens = []

        # 当标记数量小于需要的数量时，继续读取有效的标记并添加到列表中
        while len(tokens) < quantityNeeded:
            tokens.extend(ReadValidTokens(prompt, quantityNeeded - len(tokens)))
            prompt = "?"
        
        # 返回标记列表
        return tokens
    # 读取有效的令牌行，最多读取 maxCount 个令牌，并且如果有任何无效的令牌则拒绝该行
    def ReadValidTokens(prompt, maxCount):
        while True:
            tokensValid = True  # 标记令牌是否有效
            tokens = []  # 存储读取的令牌
            for token in ReadLineOfTokens(prompt, maxCount):  # 读取一行令牌
                if not isTokenValid(token):  # 检查令牌是否有效
                    io.WriteLine(NumberExpected)  # 输出错误消息
                    tokensValid = False  # 令牌无效
                    prompt = ""  # 清空提示
                    break;  # 如果tokensValid为False，跳出循环
                }

                tokens.Add(token);  # 将token添加到tokens列表中
            }

            if (tokensValid) { return tokens; }  # 如果tokensValid为True，返回tokens列表
        }
    }

    /// <summary>
    /// Lazily reads up to <paramref name="maxCount" /> tokens from an input line.
    /// </summary>
    /// <param name="prompt">The string used to prompt the user for input.</param>
    /// <param name="maxCount">The maximum number of tokens to read.</param>
    /// <returns></returns>
    private IEnumerable<Token> ReadLineOfTokens(string prompt, uint maxCount)
    {
        var tokenCount = 0;  # 初始化tokenCount变量为0
        # 遍历 Tokenizer 解析出的每个 token
        for token in Tokenizer.ParseTokens(_io.ReadLine(prompt)):
            # 如果 token 数量超过最大限制
            if tokenCount > maxCount:
                # 输出额外输入提示
                _io.WriteLine(ExtraInput)
                # 跳出循环
                break
            # 返回当前 token
            yield token
        # 结束循环
    }
}
```