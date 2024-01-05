# `d:/src/tocomm/basic-computer-games\15_Boxing\java\Player.java`

```
/**
 * The Player class model the user and compuer player
 * Player类模拟用户和计算机玩家
 */
public class Player {
    private final String name; // 玩家的名字
    private final Punch bestPunch; // 玩家最佳出拳
    private final Punch vulnerability; // 玩家的弱点
    private boolean isPlayer = false; // 是否是玩家，默认为false

    public Player(String name, Punch bestPunch, Punch vulnerability) {
        this.name = name; // 初始化玩家名字
        this.bestPunch = bestPunch; // 初始化玩家最佳出拳
        this.vulnerability = vulnerability; // 初始化玩家的弱点
        this.isPlayer = true; // 设置为玩家
    }

    /**
     * Player with random Best Punch and Vulnerability
     * 随机生成最佳出拳和弱点的玩家
     */
    public Player(String name) {
        this.name = name;  # 将传入的name赋值给对象的name属性

        int b1;  # 声明一个整型变量b1
        int d1;  # 声明一个整型变量d1

        do {
            b1 = Basic.randomOf(4);  # 使用Basic类的randomOf方法生成一个0到3的随机数赋值给b1
            d1 = Basic.randomOf(4);  # 使用Basic类的randomOf方法生成一个0到3的随机数赋值给d1
        } while (b1 == d1);  # 当b1等于d1时重复执行上述步骤，直到b1不等于d1

        this.bestPunch = Punch.fromCode(b1);  # 使用Punch类的fromCode方法根据b1生成一个拳头对象赋值给bestPunch
        this.vulnerability = Punch.fromCode(d1);  # 使用Punch类的fromCode方法根据d1生成一个拳头对象赋值给vulnerability
    }

    public boolean isPlayer() { return isPlayer; }  # 返回isPlayer属性的值
    public String getName() { return  name; }  # 返回name属性的值
    public Punch getBestPunch() { return bestPunch; }  # 返回bestPunch属性的值

    public boolean hitVulnerability(Punch punch) {  # 定义一个方法，接受一个Punch类型的参数punch
        return vulnerability == punch;  # 返回vulnerability属性是否等于参数punch
    }
```

这部分代码是一个缩进错误，应该删除。
```