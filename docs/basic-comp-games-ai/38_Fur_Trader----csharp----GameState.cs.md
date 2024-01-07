# `basic-computer-games\38_Fur_Trader\csharp\GameState.cs`

```

// 引入命名空间
using System;
using System.Collections.Generic;
using System.Text;

// 定义内部类 GameState
namespace FurTrader
{
    internal class GameState
    {
        // 游戏是否结束的标志
        internal bool GameOver { get; set; }

        // 存款金额
        internal double Savings { get; set; }

        // 远征次数
        internal int ExpeditionCount { get; set; }

        // 未分配的毛皮数量
        internal int UnasignedFurCount { get; set; }

        // 毛皮数组
        internal int[] Pelts { get; private set; }

        // 貂皮数量
        internal int MinkPelts { get { return this.Pelts[0]; } set { this.Pelts[0] = value; } }
        // 海狸皮数量
        internal int BeaverPelts { get { return this.Pelts[1]; } set { this.Pelts[1] = value; } }
        // 白鼬皮数量
        internal int ErminePelts { get { return this.Pelts[2]; } set { this.Pelts[2] = value; } }
        // 狐狸皮数量
        internal int FoxPelts { get { return this.Pelts[3]; } set { this.Pelts[3] = value; } }

        // 选择的堡垒
        internal int SelectedFort { get; set; }

        // 构造函数，初始化游戏状态
        internal GameState()
        {
            this.Savings = 600;
            this.ExpeditionCount = 0;
            this.UnasignedFurCount = 190;
            this.Pelts = new int[4];
            this.SelectedFort = 0;
        }

        // 开始回合的方法
        internal void StartTurn()
        {
            this.SelectedFort = 0;              // 重置为默认值
            this.UnasignedFurCount = 190;       // 每个回合开始时有190个毛皮
            this.Pelts = new int[4];            // 每个回合重置毛皮数量
        }
    }
}

```