# `d:/src/tocomm/basic-computer-games\38_Fur_Trader\csharp\GameState.cs`

```
# 命名空间声明
namespace FurTrader
{
    # 游戏状态类
    internal class GameState
    {
        # 游戏是否结束的标志
        internal bool GameOver { get; set; }

        # 存款金额
        internal double Savings { get; set; }

        # 远征次数
        internal int ExpeditionCount { get; set; }

        # 未分配的毛皮数量
        internal int UnasignedFurCount { get; set; }

        # 毛皮数组
        internal int[] Pelts { get; private set; }

        # 获得或设置貂皮数量
        internal int MinkPelts { get { return this.Pelts[0]; } set { this.Pelts[0] = value; } }

        # 获得或设置海狸皮数量
        internal int BeaverPelts { get { return this.Pelts[1]; } set { this.Pelts[1] = value; } }
        internal int ErminePelts { get { return this.Pelts[2]; } set { this.Pelts[2] = value; } } // 定义属性 ErminePelts，用于获取和设置索引为2的皮毛数量
        internal int FoxPelts { get { return this.Pelts[3]; } set { this.Pelts[3] = value; } } // 定义属性 FoxPelts，用于获取和设置索引为3的皮毛数量

        internal int SelectedFort { get; set; } // 定义属性 SelectedFort，用于获取和设置选定的堡垒

        internal GameState() // 构造函数，初始化游戏状态
        {
            this.Savings = 600; // 设置初始储蓄为600
            this.ExpeditionCount = 0; // 设置初始探险次数为0
            this.UnasignedFurCount = 190; // 设置初始未分配的皮毛数量为190
            this.Pelts = new int[4]; // 初始化长度为4的皮毛数组
            this.SelectedFort = 0; // 设置初始选定的堡垒为0
        }

        internal void StartTurn() // 开始新的回合方法
        {
            this.SelectedFort = 0; // 重置选定的堡垒为默认值
            this.UnasignedFurCount = 190; // 每个回合开始时未分配的皮毛数量为190
            this.Pelts = new int[4]; // 重置每个回合的皮毛数量
        }
    }
}
```

这部分代码是一个函数的结束标志，表示函数的定义结束。在Python中，函数的定义使用关键字def开始，然后是函数的代码块，最后以冒号(:)结尾。在示例中，这部分代码表示read_zip函数的定义结束。
```