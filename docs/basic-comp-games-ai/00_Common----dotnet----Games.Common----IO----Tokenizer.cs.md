# `basic-computer-games\00_Common\dotnet\Games.Common\IO\Tokenizer.cs`

```
// 命名空间声明，定义了代码所在的命名空间
using System;
using System.Collections.Generic;

namespace Games.Common.IO
{
    /// <summary>
    /// 一个简单的状态机，用于从输入行中解析标记。
    /// </summary>
    internal class Tokenizer
    {
        // 定义常量 Quote 和 Separator
        private const char Quote = '"';
        private const char Separator = ',';

        // 声明私有字段 _characters，用于存储输入的字符队列
        private readonly Queue<char> _characters;

        // 私有构造函数，接受输入并初始化 _characters 字段
        private Tokenizer(string input) => _characters = new Queue<char>(input);

        // 公共静态方法，用于解析输入并返回标记的集合
        public static IEnumerable<Token> ParseTokens(string input)
        {
            // 检查输入是否为空，为空则抛出参数异常
            if (input is null) { throw new ArgumentNullException(nameof(input)); }

            // 创建 Tokenizer 对象并调用其 ParseTokens 方法
            return new Tokenizer(input).ParseTokens();
        }

        // 私有方法，用于解析输入并返回标记的集合
        private IEnumerable<Token> ParseTokens()
        {
            // 循环解析标记并返回
            while (true)
            {
                var (token, isLastToken) = Consume(_characters);
                yield return token;

                if (isLastToken) { break; }
            }
        }

        // 公共方法，用于从字符队列中消费字符并返回标记
        public (Token, bool) Consume(Queue<char> characters)
        {
            // 初始化标记构建器和状态
            var tokenBuilder = new Token.Builder();
            var state = ITokenizerState.LookForStartOfToken;

            // 循环消费字符并更新状态和标记构建器
            while (characters.TryDequeue(out var character))
            {
                (state, tokenBuilder) = state.Consume(character, tokenBuilder);
                if (state is AtEndOfTokenState) { return (tokenBuilder.Build(), false); }
            }

            return (tokenBuilder.Build(), true);
        }

        // 定义接口，表示状态机的状态
        private interface ITokenizerState
        {
            public static ITokenizerState LookForStartOfToken { get; } = new LookForStartOfTokenState();

            (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder);
        }

        // 定义结构体，表示状态机的初始状态
        private struct LookForStartOfTokenState : ITokenizerState
    {
        // 定义一个公共方法，用于根据输入字符和当前的 Token.Builder 状态来消费字符，并返回新的状态和更新后的 Token.Builder
        public (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder) =>
            // 根据输入字符进行匹配
            character switch
            {
                Separator => (new AtEndOfTokenState(), tokenBuilder),  // 如果是分隔符，则返回 AtEndOfTokenState 状态和更新后的 Token.Builder
                Quote => (new InQuotedTokenState(), tokenBuilder.SetIsQuoted()),  // 如果是引号，则返回 InQuotedTokenState 状态和设置为引用状态的 Token.Builder
                _ when char.IsWhiteSpace(character) => (this, tokenBuilder),  // 如果是空白字符，则保持当前状态，返回原始的 Token.Builder
                _ => (new InTokenState(), tokenBuilder.Append(character))  // 其他情况返回 InTokenState 状态和追加字符后的 Token.Builder
            };
    }
    
    private struct InTokenState : ITokenizerState
    {
        // 实现 ITokenizerState 接口的 Consume 方法
        public (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder) =>
            character == Separator
                ? (new AtEndOfTokenState(), tokenBuilder)  // 如果是分隔符，则返回 AtEndOfTokenState 状态和更新后的 Token.Builder
                : (this, tokenBuilder.Append(character));  // 否则保持当前状态，返回追加字符后的 Token.Builder
    }
    
    // 其他结构体的注释和上面的 InTokenState 类似
# 闭合前面的函数定义
```