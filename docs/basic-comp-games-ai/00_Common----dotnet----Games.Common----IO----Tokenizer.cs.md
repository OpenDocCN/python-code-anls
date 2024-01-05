# `00_Common\dotnet\Games.Common\IO\Tokenizer.cs`

```
using System;  // 导入 System 命名空间
using System.Collections.Generic;  // 导入 System.Collections.Generic 命名空间

namespace Games.Common.IO;  // 声明 Games.Common.IO 命名空间

/// <summary>
/// A simple state machine which parses tokens from a line of input.
/// </summary>
internal class Tokenizer  // 声明 Tokenizer 类
{
    private const char Quote = '"';  // 声明常量 Quote 并赋值为双引号字符
    private const char Separator = ',';  // 声明常量 Separator 并赋值为逗号字符

    private readonly Queue<char> _characters;  // 声明只读字段 _characters，类型为 Queue<char>

    private Tokenizer(string input) => _characters = new Queue<char>(input);  // Tokenizer 类的构造函数，接受一个字符串参数 input，并将其转换为字符队列赋值给 _characters

    public static IEnumerable<Token> ParseTokens(string input)  // 声明公共静态方法 ParseTokens，接受一个字符串参数 input，并返回一个 Token 类型的可枚举集合
    {
        if (input is null) { throw new ArgumentNullException(nameof(input)); }  // 如果 input 为 null，则抛出 ArgumentNullException 异常
        return new Tokenizer(input).ParseTokens();  # 调用 Tokenizer 类的 ParseTokens 方法，并返回其结果

    }

    private IEnumerable<Token> ParseTokens()  # 定义一个私有方法 ParseTokens，返回一个 Token 类型的可枚举集合
    {
        while (true)  # 进入一个无限循环
        {
            var (token, isLastToken) = Consume(_characters);  # 调用 Consume 方法，将返回的 token 和 isLastToken 赋值给变量
            yield return token;  # 返回当前的 token

            if (isLastToken) { break; }  # 如果是最后一个 token，则跳出循环
        }
    }

    public (Token, bool) Consume(Queue<char> characters)  # 定义一个公共方法 Consume，接受一个字符队列作为参数，返回一个元组类型的 Token 和 bool 值
    {
        var tokenBuilder = new Token.Builder();  # 创建一个 Token.Builder 对象
        var state = ITokenizerState.LookForStartOfToken;  # 创建一个状态变量，并赋初值为 LookForStartOfToken
        while (characters.TryDequeue(out var character))
        {
            // 从字符队列中取出字符，然后根据当前状态消耗字符并更新状态和tokenBuilder
            (state, tokenBuilder) = state.Consume(character, tokenBuilder);
            // 如果状态为AtEndOfTokenState，则返回当前构建的token和false
            if (state is AtEndOfTokenState) { return (tokenBuilder.Build(), false); }
        }

        // 返回当前构建的token和true
        return (tokenBuilder.Build(), true);
    }

    // 定义接口ITokenizerState
    private interface ITokenizerState
    {
        // 定义静态属性LookForStartOfToken，初始值为LookForStartOfTokenState实例
        public static ITokenizerState LookForStartOfToken { get; } = new LookForStartOfTokenState();

        // 定义Consume方法，根据输入的字符和当前的tokenBuilder消耗字符并更新状态和tokenBuilder
        (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder);
    }

    // 定义结构体LookForStartOfTokenState，实现ITokenizerState接口
    private struct LookForStartOfTokenState : ITokenizerState
    {
        // 根据输入的字符和当前的tokenBuilder消耗字符并更新状态和tokenBuilder
        public (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder) =>
            character switch
            {
                Separator => (new AtEndOfTokenState(), tokenBuilder),  // 如果遇到分隔符，切换到结束状态，并返回当前的 tokenBuilder
                Quote => (new InQuotedTokenState(), tokenBuilder.SetIsQuoted()),  // 如果遇到引号，切换到引号内状态，并设置 tokenBuilder 为引号内状态
                _ when char.IsWhiteSpace(character) => (this, tokenBuilder),  // 如果遇到空白字符，保持当前状态，并返回当前的 tokenBuilder
                _ => (new InTokenState(), tokenBuilder.Append(character))  // 其他情况切换到普通 token 状态，并将当前字符添加到 tokenBuilder
            };
    }

    private struct InTokenState : ITokenizerState
    {
        public (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder) =>
            character == Separator
                ? (new AtEndOfTokenState(), tokenBuilder)  // 如果遇到分隔符，切换到结束状态，并返回当前的 tokenBuilder
                : (this, tokenBuilder.Append(character));  // 否则保持当前状态，并将当前字符添加到 tokenBuilder
    }

    private struct InQuotedTokenState : ITokenizerState
    {
        public (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder) =>
            character == Quote  // 如果遇到引号
                ? (new AtEndOfTokenState(), tokenBuilder)  // 切换到结束状态，并返回当前的 tokenBuilder
// 创建一个新的 ExpectSeparatorState 状态对象，并使用 tokenBuilder 构建器
                ? (new ExpectSeparatorState(), tokenBuilder)
                // 否则，保持当前状态，并将字符追加到 tokenBuilder 中
                : (this, tokenBuilder.Append(character));
    }

    // 定义一个期望分隔符状态
    private struct ExpectSeparatorState : ITokenizerState
    {
        // 根据输入的字符进行状态转换和 tokenBuilder 的操作
        public (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder) =>
            // 如果输入的字符是分隔符，则转换到 AtEndOfTokenState 状态，并返回 tokenBuilder
            character == Separator
                ? (new AtEndOfTokenState(), tokenBuilder)
                // 否则，转换到 IgnoreRestOfLineState 状态，并返回 tokenBuilder
                : (new IgnoreRestOfLineState(), tokenBuilder);
    }

    // 定义一个忽略行尾状态
    private struct IgnoreRestOfLineState : ITokenizerState
    {
        // 根据输入的字符进行状态转换和 tokenBuilder 的操作
        public (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder) =>
            // 保持当前状态，并返回 tokenBuilder
            (this, tokenBuilder);
    }

    // 定义一个到达标记结尾状态
    private struct AtEndOfTokenState : ITokenizerState
# 定义一个公共方法，接受字符和 Token.Builder 对象作为参数，返回一个包含 ITokenizerState 和 Token.Builder 对象的元组
public (ITokenizerState, Token.Builder) Consume(char character, Token.Builder tokenBuilder) =>
    # 抛出一个 InvalidOperationException 异常，表示当前方法不可用
    throw new InvalidOperationException();
# 结束当前类的定义
}
# 结束命名空间的定义
}
```