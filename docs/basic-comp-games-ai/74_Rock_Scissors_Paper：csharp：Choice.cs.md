# `d:/src/tocomm/basic-computer-games\74_Rock_Scissors_Paper\csharp\Choice.cs`

```
namespace RockScissorsPaper
{
    // 创建一个名为 Choice 的类
    public class Choice
    {
        // 创建一个名为 Selector 的公共属性，只能在类内部设置值
        public string Selector {get; private set; }
        // 创建一个名为 Name 的公共属性，只能在类内部设置值
        public string Name { get; private set; }
        // 创建一个名为 CanBeat 的内部属性，可以被设置值
        internal Choice CanBeat { get; set; }

        // 创建一个构造函数，接受两个参数 selector 和 name
        public Choice(string selector, string name) {
            // 设置 Selector 属性的值为传入的 selector 参数
            Selector = selector;
            // 设置 Name 属性的值为传入的 name 参数
            Name = name;
        }

        // 创建一个名为 Beats 的方法，接受一个 Choice 类型的参数 choice
        public bool Beats(Choice choice)
        {
            // 返回当前 Choice 对象的 CanBeat 属性是否等于传入的 choice 参数
            return choice == CanBeat;
        }
    }
}
```