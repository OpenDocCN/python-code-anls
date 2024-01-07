# `basic-computer-games\00_Common\dotnet\Games.Common\Randomness\IRandom.cs`

```

// 命名空间声明，表示该接口属于 Games.Common.Randomness 命名空间
namespace Games.Common.Randomness;

/// <summary>
/// 提供对随机数生成器的访问
/// </summary>
// 定义一个接口 IRandom
public interface IRandom
{
    /// <summary>
    /// 获取一个随机的 <see cref="float" />，使得 0 &lt;= n &lt; 1.
    /// </summary>
    /// <returns>随机数。</returns>
    // 定义一个方法 NextFloat，返回一个随机的浮点数
    float NextFloat();

    /// <summary>
    /// 获取上一次调用 <see cref="NextFloat" /> 返回的 <see cref="float" />。
    /// </summary>
    /// <returns>上一个随机数。</returns>
    // 定义一个方法 PreviousFloat，返回上一次生成的随机浮点数
    float PreviousFloat();

    /// <summary>
    /// 重新设置随机数生成器的种子。
    /// </summary>
    /// <param name="seed">种子。</param>
    // 定义一个方法 Reseed，重新设置随机数生成器的种子
    void Reseed(int seed);
}

```