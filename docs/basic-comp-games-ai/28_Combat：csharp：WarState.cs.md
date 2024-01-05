# `d:/src/tocomm/basic-computer-games\28_Combat\csharp\WarState.cs`

```
/// <summary>
/// Represents the current state of the war.
/// </summary>
public abstract class WarState
{
    /// <summary>
    /// Gets the computer's armed forces.
    /// </summary>
    public ArmedForces ComputerForces { get; }

    /// <summary>
    /// Gets the player's armed forces.
    /// </summary>
    public ArmedForces PlayerForces { get; }

    /// <summary>
```

- 表示当前战争状态的抽象类
- 获取计算机的武装力量
- 获取玩家的武装力量
        /// Gets a flag indicating whether this state represents absolute
        /// victory for the player.
        /// </summary>
        public virtual bool IsAbsoluteVictory => false;  // 返回一个布尔值，表示玩家是否取得了绝对胜利

        /// <summary>
        /// Gets the final outcome of the war.
        /// </summary>
        /// <remarks>
        /// If the war is ongoing, this property will be null.
        /// </remarks>
        public virtual WarResult? FinalOutcome => null;  // 返回战争的最终结果，如果战争仍在进行中，则返回空值

        /// <summary>
        /// Initializes a new instance of the state class.
        /// </summary>
        /// <param name="computerForces">
        /// The computer's forces.
        /// </param>
        /// <param name="playerForces">
```  // 初始化状态类的新实例，传入计算机的力量和玩家的力量作为参数
        /// The player's forces.
        /// </param>
        public WarState(ArmedForces computerForces, ArmedForces playerForces) =>
            (ComputerForces, PlayerForces) = (computerForces, playerForces);

        /// <summary>
        /// Launches an attack.
        /// </summary>
        /// <param name="branch">
        /// The branch of the military to use for the attack.
        /// </param>
        /// <param name="attackSize">
        /// The number of men and women to use for the attack.
        /// </param>
        /// <returns>
        /// The new state of the game resulting from the attack and a message
        /// describing the result.
        /// </returns>
        public (WarState nextState, string message) LaunchAttack(MilitaryBranch branch, int attackSize) =>
            branch switch
```

在这段代码中，我们定义了一个名为WarState的类，它包含了两个武装力量（computerForces和playerForces）作为参数的构造函数。另外，还定义了一个LaunchAttack方法，用于发动攻击，接受军事分支和攻击规模作为参数，并返回攻击后的新游戏状态和描述结果的消息。
                {
                    MilitaryBranch.Army     => AttackWithArmy(attackSize),  // 如果军种是陆军，则调用AttackWithArmy方法进行攻击
                    MilitaryBranch.Navy     => AttackWithNavy(attackSize),  // 如果军种是海军，则调用AttackWithNavy方法进行攻击
                    MilitaryBranch.AirForce => AttackWithAirForce(attackSize),  // 如果军种是空军，则调用AttackWithAirForce方法进行攻击
                    _               => throw new ArgumentException("INVALID BRANCH")  // 如果军种无效，则抛出参数异常
                };

            /// <summary>
            /// Conducts an attack with the player's army.
            /// </summary>
            /// <param name="attackSize">
            /// The number of men and women used in the attack.
            /// </param>
            /// <returns>
            /// The new game state and a message describing the result.
            /// </returns>
            protected abstract (WarState nextState, string message) AttackWithArmy(int attackSize);  // 抽象方法，用于实现使用玩家的陆军进行攻击

            /// <summary>
            /// Conducts an attack with the player's navy.
/// </summary>
/// <param name="attackSize">
/// The number of men and women used in the attack.
/// </param>
/// <returns>
/// The new game state and a message describing the result.
/// </returns>
protected abstract (WarState nextState, string message) AttackWithNavy(int attackSize);
```
这段代码是一个抽象方法，用于进行海军攻击。它接受一个攻击规模参数，并返回一个元组，包含新的游戏状态和描述结果的消息。

```
/// <summary>
/// Conducts an attack with the player's air force.
/// </summary>
/// <param name="attackSize">
/// The number of men and women used in the attack.
/// </param>
/// <returns>
/// The new game state and a message describing the result.
/// </returns>
protected abstract (WarState nextState, string message) AttackWithAirForce(int attackSize);
```
这段代码是一个抽象方法，用于进行空军攻击。它接受一个攻击规模参数，并返回一个元组，包含新的游戏状态和描述结果的消息。
# 关闭 ZIP 对象
zip.close()  # 关闭 ZIP 对象，释放资源
```