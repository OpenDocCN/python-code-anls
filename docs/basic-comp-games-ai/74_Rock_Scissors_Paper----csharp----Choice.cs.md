# `basic-computer-games\74_Rock_Scissors_Paper\csharp\Choice.cs`

```py
# 定义名为 RockScissorsPaper 的命名空间
namespace RockScissorsPaper
{
    # 定义名为 Choice 的公共类
    public class Choice
    {
        # 定义 Selector 属性，可供外部获取，但只能在类内部设置
        public string Selector {get; private set; }
        # 定义 Name 属性，可供外部获取，但只能在类内部设置
        public string Name { get; private set; }
        # 定义内部 CanBeat 属性，可在类内部设置
        internal Choice CanBeat { get; set; }

        # 定义构造函数，初始化 Selector 和 Name 属性
        public Choice(string selector, string name) {
            Selector = selector;
            Name = name;
        }

        # 定义 Beats 方法，判断当前 Choice 对象是否能击败传入的 Choice 对象
        public bool Beats(Choice choice)
        {
            return choice == CanBeat;
        }
    }
}
```