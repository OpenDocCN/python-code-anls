# `basic-computer-games\03_Animal\csharp\Branch.cs`

```
// 命名空间 Animal
namespace Animal
{
    // 定义 Branch 类
    public class Branch
    {
        // 文本属性
        public string Text { get; set; }

        // 是否为叶子节点的属性
        public bool IsEnd => Yes == null && No == null;

        // 是的分支
        public Branch Yes { get; set; }

        // 否的分支
        public Branch No { get; set; }

        // 重写 ToString 方法，返回文本和是否为叶子节点的信息
        public override string ToString()
        {
            return $"{Text} : IsEnd {IsEnd}";
        }
    }
}
```