# `basic-computer-games\00_Common\dotnet\Games.Common\IO\TokenReader.cs`

```py
using System;
using System.Collections.Generic;
using System.Linq;
using static Games.Common.IO.Strings;

namespace Games.Common.IO;
{
    /// <summary>
    /// 从输入中读取并组装给定数量的值或标记，可能跨越多个输入行。
    /// </summary>
    internal class TokenReader
    {
        private readonly TextIO _io;
        private readonly Predicate<Token> _isTokenValid;

        private TokenReader(TextIO io, Predicate<Token> isTokenValid)
        {
            _io = io;
            _isTokenValid = isTokenValid ?? (t => true);
        }

        /// <summary>
        /// 创建一个读取字符串标记的 <see cref="TokenReader" />。
        /// </summary>
        /// <param name="io">一个 <see cref="TextIO" /> 实例。</param>
        /// <returns>新的 <see cref="TokenReader" /> 实例。</returns>
        public static TokenReader ForStrings(TextIO io) => new(io, t => true);

        /// <summary>
        /// 创建一个读取标记并验证其是否可以解析为数字的 <see cref="TokenReader" />。
        /// </summary>
        /// <param name="io">一个 <see cref="TextIO" /> 实例。</param>
        /// <returns>新的 <see cref="TokenReader" /> 实例。</returns>
        public static TokenReader ForNumbers(TextIO io) => new(io, t => t.IsNumber);

        /// <summary>
        /// 从一个或多个输入行中读取有效的标记，并构建一个包含所需数量的列表。
        /// </summary>
        /// <param name="prompt">用于提示用户输入的字符串。</param>
        /// <param name="quantityNeeded">所需的标记数量。</param>
        /// <returns>读取的标记序列。</returns>
        public IEnumerable<Token> ReadTokens(string prompt, uint quantityNeeded)
    {
        // 如果需要的数量为0，则抛出参数超出范围的异常
        if (quantityNeeded == 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(quantityNeeded),
                $"'{nameof(quantityNeeded)}' must be greater than zero.");
        }
    
        // 创建一个 Token 列表
        var tokens = new List<Token>();
    
        // 当 Token 列表中的数量小于需要的数量时，循环读取有效的 Token
        while (tokens.Count < quantityNeeded)
        {
            // 读取有效的 Token，并添加到 Token 列表中
            tokens.AddRange(ReadValidTokens(prompt, quantityNeeded - (uint)tokens.Count));
            // 重置提示信息
            prompt = "?";
        }
    
        // 返回 Token 列表
        return tokens;
    }
    
    /// <summary>
    /// 读取一行 Token，最多读取 <paramref name="maxCount" /> 个 Token，并且如果有无效的 Token，则拒绝该行输入。
    /// </summary>
    /// <param name="prompt">用于提示用户输入的字符串。</param>
    /// <param name="maxCount">要读取的 Token 的最大数量。</param>
    /// <returns>读取到的 Token 序列。</returns>
    private IEnumerable<Token> ReadValidTokens(string prompt, uint maxCount)
    {
        // 循环读取 Token
        while (true)
        {
            // 标记 Token 是否有效
            var tokensValid = true;
            // 创建 Token 列表
            var tokens = new List<Token>();
            // 遍历读取一行 Token
            foreach (var token in ReadLineOfTokens(prompt, maxCount))
            {
                // 如果 Token 无效，则输出错误信息，并标记 Token 无效
                if (!_isTokenValid(token))
                {
                    _io.WriteLine(NumberExpected);
                    tokensValid = false;
                    prompt = "";
                    break;
                }
    
                // 将有效的 Token 添加到 Token 列表中
                tokens.Add(token);
            }
    
            // 如果 Token 有效，则返回 Token 列表
            if (tokensValid) { return tokens; }
        }
    }
    
    /// <summary>
    /// 从输入行中延迟读取最多 <paramref name="maxCount" /> 个 Token。
    /// </summary>
    /// <param name="prompt">用于提示用户输入的字符串。</param>
    /// <param name="maxCount">要读取的 Token 的最大数量。</param>
    /// <returns></returns>
    private IEnumerable<Token> ReadLineOfTokens(string prompt, uint maxCount)
    {
        // 初始化令牌计数器
        var tokenCount = 0;

        // 遍历通过分词器解析输入行得到的令牌
        foreach (var token in Tokenizer.ParseTokens(_io.ReadLine(prompt)))
        {
            // 如果令牌计数超过最大数量限制
            if (++tokenCount > maxCount)
            {
                // 输出额外输入提示信息
                _io.WriteLine(ExtraInput);
                // 跳出循环
                break;
            }

            // 返回当前令牌
            yield return token;
        }
    }
# 闭合前面的函数定义
```