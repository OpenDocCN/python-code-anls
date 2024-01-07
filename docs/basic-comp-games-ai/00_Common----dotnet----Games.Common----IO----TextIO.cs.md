# `basic-computer-games\00_Common\dotnet\Games.Common\IO\TextIO.cs`

```

// 使用 Games.Common.Numbers 命名空间
using Games.Common.Numbers;

// 使用 Games.Common.IO 命名空间
namespace Games.Common.IO;

/// <inheritdoc />
/// <summary>
/// 使用 TextReader 读取输入并使用 TextWriter 写入输出来实现 IReadWrite 接口
/// </summary>
/// <remarks>
/// 这个实现重现了 Vintage BASIC 输入体验，当输入不完整时会提示多次，根据需要拒绝非数字输入，警告忽略额外输入等。
/// </remarks>
public class TextIO : IReadWrite
}

```