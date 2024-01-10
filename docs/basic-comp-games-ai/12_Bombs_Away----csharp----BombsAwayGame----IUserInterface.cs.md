# `basic-computer-games\12_Bombs_Away\csharp\BombsAwayGame\IUserInterface.cs`

```
// 命名空间声明，表示BombsAwayGame命名空间
namespace BombsAwayGame;

/// <summary>
/// 代表为游戏提供数据的接口。
/// </summary>
/// <remarks>
/// 抽象化UI允许我们集中其关注点在代码的一个部分，并且在不创建任何改变游戏逻辑的风险的情况下更改UI行为。
/// 它还允许我们为测试提供自动化UI。
/// </remarks>
// 声明用户界面接口
public interface IUserInterface
{
    /// <summary>
    /// 显示给定的消息。
    /// </summary>
    /// <param name="message">要显示的消息。</param>
    // 输出消息的方法声明
    void Output(string message);

    /// <summary>
    /// 从给定的选择中选择一个项目。
    /// </summary>
    /// <param name="message">要显示的消息。</param>
    /// <param name="choices">要选择的选项。</param>
    /// <returns>用户在<paramref name="choices"/>中选择的选项的索引。</returns>
    // 选择项目的方法声明
    int Choose(string message, IList<string> choices);

    /// <summary>
    /// 允许用户选择是或否。
    /// </summary>
    /// <param name="message">要显示的消息。</param>
    /// <returns>如果用户选择是，则为true，如果用户选择否，则为false。</returns>
    // 选择是或否的方法声明
    bool ChooseYesOrNo(string message);

    /// <summary>
    /// 从用户获取整数。
    /// </summary>
    /// <returns>用户提供的整数。</returns>
    // 获取用户输入整数的方法声明
    int InputInteger();
}
```