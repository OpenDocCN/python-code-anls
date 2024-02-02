# `basic-computer-games\17_Bullfight\csharp\Mediator.cs`

```py
namespace Game
{
    /// <summary>
    /// Facilitates sending messages between the two game loops.
    /// </summary>
    /// <remarks>
    /// This class serves as a little piece of glue in between the main program
    /// loop and the bull fight coroutine.  When the main program calls one of
    /// its methods, the mediator creates the appropriate input data that the
    /// bull fight coroutine later retrieves with <see cref="GetInput{T}"/>.
    /// </remarks>
    public class Mediator
    {
        private object? m_input;

        // 设置闪避动作及风险级别
        public void Dodge(RiskLevel riskLevel) =>
            m_input = (Action.Dodge, riskLevel);

        // 设置击杀动作及风险级别
        public void Kill(RiskLevel riskLevel) =>
            m_input = (Action.Kill, riskLevel);

        // 设置恐慌动作
        public void Panic() =>
            m_input = (Action.Panic, default(RiskLevel));

        // 设置逃离擂台动作
        public void RunFromRing() =>
            m_input = true;

        // 设置继续战斗动作
        public void ContinueFighting() =>
            m_input = false;

        /// <summary>
        /// Gets the next input from the user.
        /// </summary>
        /// <typeparam name="T">
        /// The type of input to receive.
        /// </typeparam>
        public T GetInput<T>()
        {
            // 断言确保已接收到输入
            Debug.Assert(m_input is not null, "No input received");
            // 断言确保接收到的输入类型与期望类型相符
            Debug.Assert(m_input.GetType() == typeof(T), "Invalid input received");
            var result = (T)m_input;
            m_input = null;
            return result;
        }
    }
}
```