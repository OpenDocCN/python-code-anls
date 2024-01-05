# `74_Rock_Scissors_Paper\csharp\Choices.cs`

```
using System;  // 导入 System 命名空间，以便使用其中的类和方法

namespace RockScissorsPaper  // 声明 RockScissorsPaper 命名空间
{
    public class Choices  // 声明 Choices 类
    {
        public static readonly Choice Rock = new Choice("3", "Rock");  // 创建一个名为 Rock 的 Choice 对象，值为 "3" 和 "Rock"
        public static readonly Choice Scissors = new Choice("2", "Scissors");  // 创建一个名为 Scissors 的 Choice 对象，值为 "2" 和 "Scissors"
        public static readonly Choice Paper = new Choice("1", "Paper");  // 创建一个名为 Paper 的 Choice 对象，值为 "1" 和 "Paper"

        private static readonly Choice[] _allChoices;  // 声明一个私有的 Choice 数组 _allChoices
        private static readonly Random _random = new Random();  // 创建一个 Random 对象 _random

        static Choices()  // 静态构造函数
        {
            Rock.CanBeat = Scissors;  // 设置 Rock 对象的 CanBeat 属性为 Scissors
            Scissors.CanBeat = Paper;  // 设置 Scissors 对象的 CanBeat 属性为 Paper
            Paper.CanBeat = Rock;  // 设置 Paper 对象的 CanBeat 属性为 Rock

            _allChoices = new[] { Rock, Scissors, Paper };  // 将 Rock、Scissors、Paper 对象组成的数组赋值给 _allChoices
        }

        public static Choice GetRandom()
        {
            // 从_allChoices数组中随机选择一个Choice对象并返回
            return _allChoices[_random.Next(_allChoices.GetLength(0))];
        }

        public static bool TryGetBySelector(string selector, out Choice choice)
        {
            // 遍历_allChoices数组，查找与给定selector相匹配的Choice对象
            foreach (var possibleChoice in _allChoices)
            {
                if (string.Equals(possibleChoice.Selector, selector))
                {
                    // 如果找到匹配的Choice对象，则将其赋值给choice并返回true
                    choice = possibleChoice;
                    return true;
                }
            }
            // 如果未找到匹配的Choice对象，则将choice赋值为null并返回false
            choice = null;
            return false;
        }
    }
```

这部分代码是一个缩进错误，应该删除。
```