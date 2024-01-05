# `d:/src/tocomm/basic-computer-games\17_Bullfight\csharp\Mediator.cs`

```
// 命名空间 Game
namespace Game
{
    /// <summary>
    /// 便于在两个游戏循环之间发送消息。
    /// </summary>
    /// <remarks>
    /// 这个类在主程序循环和斗牛协程之间起到了一点点粘合作用。当主程序调用其方法之一时，中介者创建适当的输入数据，斗牛协程稍后使用 <see cref="GetInput{T}"/> 检索。
    /// </remarks>
    public class Mediator
    {
        private object? m_input;

        // 躲避方法，接收风险级别作为参数
        public void Dodge(RiskLevel riskLevel) =>
            m_input = (Action.Dodge, riskLevel);
```
- 命名空间 Game，表示该类所属的命名空间。
- 便于在两个游戏循环之间发送消息。在类的注释中解释了该类的作用。
- 中介者类，用于在主程序循环和斗牛协程之间起到粘合作用。
- 当主程序调用其方法之一时，中介者创建适当的输入数据，斗牛协程稍后使用 <see cref="GetInput{T}"/> 检索。
- 私有成员变量 m_input，用于存储输入数据。
- Dodge 方法，接收风险级别作为参数，并将动作和风险级别存储在 m_input 中。
        // 设置输入为杀死风险级别
        public void Kill(RiskLevel riskLevel) =>
            m_input = (Action.Kill, riskLevel);

        // 设置输入为恐慌
        public void Panic() =>
            m_input = (Action.Panic, default(RiskLevel));

        // 设置输入为逃离戒备
        public void RunFromRing() =>
            m_input = true;

        // 设置输入为继续战斗
        public void ContinueFighting() =>
            m_input = false;

        /// <summary>
        /// 从用户获取下一个输入。
        /// </summary>
        /// <typeparam name="T">
        /// 要接收的输入类型。
        /// </typeparam>
        public T GetInput<T>()
        {
            Debug.Assert(m_input is not null, "No input received");  # 检查输入是否为空，如果为空则抛出异常
            Debug.Assert(m_input.GetType() == typeof(T), "Invalid input received");  # 检查输入的类型是否与指定类型T相同，如果不同则抛出异常
            var result = (T)m_input;  # 将输入转换为指定类型T的变量result
            m_input = null;  # 将输入置空
            return result;  # 返回转换后的结果
        }
    }
}
```