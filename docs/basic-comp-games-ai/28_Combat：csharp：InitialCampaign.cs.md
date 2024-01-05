# `d:/src/tocomm/basic-computer-games\28_Combat\csharp\InitialCampaign.cs`

```
// 命名空间 Game，表示游戏的命名空间
namespace Game
{
    /// <summary>
    /// 表示战争初始阶段的游戏状态
    /// </summary>
    public sealed class InitialCampaign : WarState
    {
        /// <summary>
        /// 初始化 InitialCampaign 类的新实例
        /// </summary>
        /// <param name="computerForces">
        /// 计算机的力量
        /// </param>
        /// <param name="playerForces">
        /// 玩家的力量
        /// </param>
        public InitialCampaign(ArmedForces computerForces, ArmedForces playerForces)
            : base(computerForces, playerForces)
        {
        }
        // 重写父类的 AttackWithArmy 方法，传入攻击规模参数，返回战争状态和消息字符串
        protected override (WarState nextState, string message) AttackWithArmy(int attackSize)
        {
            // BUG: 为什么我们要将攻击规模与自己的军队规模进行比较？
            //   如果我们的军队很小，这会导致一些非常荒谬的结果。
            if (attackSize < PlayerForces.Army / 3)
            {
                return
                (
                    // 创建一个 FinalCampaign 对象，传入电脑军队和更新后的玩家军队
                    new FinalCampaign(
                        ComputerForces,
                        PlayerForces with
                        {
                            Army = PlayerForces.Army - attackSize
                        }),
                    // 返回消息字符串，指示玩家失去了一部分军队
                    $"YOU LOST {attackSize} MEN FROM YOUR ARMY."
                );
            }
            else
            # 如果攻击规模小于玩家兵力的2/3
            if (attackSize < 2 * PlayerForces.Army / 3)
            {
                # 返回一个最终战役对象，其中计算机兵力为0，玩家兵力减少攻击规模的1/3
                return
                (
                    new FinalCampaign(
                        ComputerForces with
                        {
                            # BUG: 明显不符合我们下面所声称的...
                            Army = 0
                        },
                        PlayerForces with
                        {
                            Army = PlayerForces.Army - attackSize / 3
                        }),
                    $"YOU LOST {attackSize / 3} MEN, BUT I LOST {2 * ComputerForces.Army / 3}"
                );
            }
            else
            {
                # BUG？这与攻击时的第三种结果相同
// 返回最终的战役结果和消息
return
(
    // 创建最终的战役对象，使用计算机部队和玩家部队的新值
    new FinalCampaign(
        // 计算机部队的新值，海军的值为原值的2/3
        ComputerForces with
        {
            Navy = 2 * ComputerForces.Navy / 3
        },
        // 玩家部队的新值，陆军和空军的值为原值的1/3
        PlayerForces with
        {
            Army     = PlayerForces.Army / 3,
            AirForce = PlayerForces.AirForce / 3
        }),
    // 返回消息
    "YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO\n" +
    "OF YOUR AIR FORCE BASES AND 3 ARMY BASES."
);
        }

        # 用海军进行攻击
        protected override (WarState nextState, string message) AttackWithNavy(int attackSize)
        {
            # 如果攻击规模小于计算机海军的三分之一
            if (attackSize < ComputerForces.Navy / 3)
            {
                # 返回新的战争状态和消息
                return
                (
                    # 创建新的最终战役对象，更新玩家部队的海军数量
                    new FinalCampaign(
                        ComputerForces,
                        PlayerForces with
                        {
                            Navy = PlayerForces.Navy - attackSize
                        }),
                    "YOUR ATTACK WAS STOPPED!"
                );
            }
            # 如果攻击规模小于计算机海军的三分之二
            else if (attackSize < 2 * ComputerForces.Navy / 3)
            {
                return  # 返回最终结果
                (
                    new FinalCampaign(  # 创建一个新的FinalCampaign对象
                        ComputerForces with  # 使用ComputerForces对象的属性
                        {
                            Navy = ComputerForces.Navy / 3  # 将Navy属性的值设为原来的1/3
                        },
                        PlayerForces),  # 使用PlayerForces对象
                    $"YOU DESTROYED {2 * ComputerForces.Navy / 3} OF MY ARMY."  # 返回一个字符串，包含计算后的值
                );
            }
            else  # 如果条件不成立
            {
                return  # 返回最终结果
                (
                    new FinalCampaign(  # 创建一个新的FinalCampaign对象
                        ComputerForces with  # 使用ComputerForces对象的属性
                        {
                            Navy = 2 * ComputerForces.Navy / 3  # 将Navy属性的值设为原来的2/3
                        },
                        PlayerForces with
                        {
                            Army     = PlayerForces.Army / 3,  # 将玩家部队的陆军数量更新为原来的三分之一
                            AirForce = PlayerForces.AirForce / 3  # 将玩家部队的空军数量更新为原来的三分之一
                        }),
                    "YOU SUNK ONE OF MY PATROL BOATS, BUT I WIPED OUT TWO\n" +  # 返回攻击结果的消息
                    "OF YOUR AIR FORCE BASES AND 3 ARMY BASES."
                );
            }
        }

        protected override (WarState nextState, string message) AttackWithAirForce(int attackSize)
        {
            // BUG: Why are we comparing the attack size to the size of
            //  our own air force?  Surely we meant to compare to the
            //  computer's air force.
            if (attackSize < PlayerForces.AirForce / 3)  # 如果攻击大小小于玩家部队空军数量的三分之一
            {
                return
                (
                    # 创建一个新的 FinalCampaign 对象，传入计算机势力和玩家势力作为参数
                    new FinalCampaign(
                        ComputerForces,
                        PlayerForces with
                        {
                            # 更新玩家势力的空军数量，减去攻击规模
                            AirForce = PlayerForces.AirForce - attackSize
                        }),
                    "YOUR ATTACK WAS WIPED OUT."
                );
            }
            else
            # 如果攻击规模小于玩家势力空军数量的 2/3
            if (attackSize < 2 * PlayerForces.AirForce / 3)
            {
                return
                (
                    # 创建一个新的 FinalCampaign 对象，传入更新后的计算机势力作为参数
                    new FinalCampaign(
                        ComputerForces with
                        {
                            # 更新计算机势力的陆军数量为原数量的 2/3
                            Army     = 2 * ComputerForces.Army / 3,
                            # 更新计算机势力的海军数量为原数量的 1/3
                            Navy     = ComputerForces.Navy / 3,
                            # 更新计算机势力的空军数量为原数量的 1/3
                            AirForce = ComputerForces.AirForce / 3
# 创建一个新的最终战役对象，其中计算机势力的军队数量为原来的2/3，玩家势力的军队数量为原来的1/4，海军数量为原来的1/3
new FinalCampaign(
    ComputerForces with
    {
        Army = 2 * ComputerForces.Army / 3
    },
    PlayerForces with
    {
        Army = PlayerForces.Army / 4,
        Navy = PlayerForces.Navy / 3
    }),
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 使用 open 函数以二进制模式打开文件，读取文件内容，然后使用 BytesIO 封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用 BytesIO 对象创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 使用字典推导式遍历 ZIP 对象的文件名列表，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```