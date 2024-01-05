# `d:/src/tocomm/basic-computer-games\03_Animal\csharp\Branch.cs`

```
// 命名空间声明，表示该类属于 Animal 命名空间
namespace Animal
{
    // 定义 Branch 类
    public class Branch
    {
        // 声明 Text 属性，用于存储文本信息
        public string Text { get; set; }

        // 声明 IsEnd 属性，用于判断该分支是否为结束节点
        public bool IsEnd => Yes == null && No == null;

        // 声明 Yes 属性，用于存储指向下一个分支的引用
        public Branch Yes { get; set; }

        // 声明 No 属性，用于存储指向下一个分支的引用
        public Branch No { get; set; }

        // 重写 ToString 方法，返回分支的文本信息和是否为结束节点的信息
        public override string ToString()
        {
            return $"{Text} : IsEnd {IsEnd}";
        }
    }
}
```