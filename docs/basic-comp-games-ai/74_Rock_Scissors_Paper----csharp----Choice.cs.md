# `basic-computer-games\74_Rock_Scissors_Paper\csharp\Choice.cs`

```

// 命名空间 RockScissorsPaper
namespace RockScissorsPaper
{
    // Choice 类
    public class Choice
    {
        // Selector 属性，可读不可写
        public string Selector {get; private set; }
        // Name 属性，可读不可写
        public string Name { get; private set; }
        // CanBeat 属性，内部可设置
        internal Choice CanBeat { get; set; }

        // Choice 类的构造函数，接受选择器和名称作为参数
        public Choice(string selector, string name) {
            Selector = selector;
            Name = name;
        }

        // Beats 方法，判断当前选择是否能战胜传入的选择
        public bool Beats(Choice choice)
        {
            return choice == CanBeat;
        }
    }
}

```