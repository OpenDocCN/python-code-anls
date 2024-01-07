# `basic-computer-games\17_Bullfight\csharp\Mediator.cs`

```

// 命名空间 Game，包含 Mediator 类
namespace Game
{
    /// <summary>
    /// 便于在两个游戏循环之间发送消息。
    /// </summary>
    /// <remarks>
    /// 这个类作为主程序循环和斗牛协程之间的一小段粘合剂。当主程序调用其方法时，中介者创建适当的输入数据，斗牛协程稍后使用 <see cref="GetInput{T}"/> 检索。
    /// </remarks>
    public class Mediator
    {
        private object? m_input; // 用于存储输入数据的私有成员变量

        // 设置闪避动作和风险级别的输入数据
        public void Dodge(RiskLevel riskLevel) =>
            m_input = (Action.Dodge, riskLevel);

        // 设置击杀动作和风险级别的输入数据
        public void Kill(RiskLevel riskLevel) =>
            m_input = (Action.Kill, riskLevel);

        // 设置恐慌动作的输入数据
        public void Panic() =>
            m_input = (Action.Panic, default(RiskLevel));

        // 设置逃离斗技场的输入数据
        public void RunFromRing() =>
            m_input = true;

        // 设置继续战斗的输入数据
        public void ContinueFighting() =>
            m_input = false;

        /// <summary>
        /// 获取用户的下一个输入。
        /// </summary>
        /// <typeparam name="T">
        /// 要接收的输入类型。
        /// </typeparam>
        public T GetInput<T>()
        {
            Debug.Assert(m_input is not null, "未收到输入");
            Debug.Assert(m_input.GetType() == typeof(T), "收到无效输入");
            var result = (T)m_input;
            m_input = null;
            return result;
        }
    }
}

```