# `basic-computer-games\15_Boxing\java\Player.java`

```
/**
 * The Player class model the user and compuer player
 */
public class Player {
    // 玩家姓名
    private final String name;
    // 最佳出拳
    private final Punch bestPunch;
    // 弱点
    private final Punch vulnerability;
    // 是否为玩家
    private boolean isPlayer = false;

    // 构造函数，指定姓名、最佳出拳和弱点
    public Player(String name, Punch bestPunch, Punch vulnerability) {
        this.name = name;
        this.bestPunch = bestPunch;
        this.vulnerability = vulnerability;
        this.isPlayer = true;
    }

    /**
     * Player with random Best Punch and Vulnerability
     */
    // 构造函数，随机生成最佳出拳和弱点
    public Player(String name) {
        this.name = name;

        int b1;
        int d1;

        // 随机生成不同的最佳出拳和弱点
        do {
            b1 = Basic.randomOf(4);
            d1 = Basic.randomOf(4);
        } while (b1 == d1);

        this.bestPunch = Punch.fromCode(b1);
        this.vulnerability = Punch.fromCode(d1);
    }

    // 返回是否为玩家
    public boolean isPlayer() { return isPlayer; }
    // 返回姓名
    public String getName() { return  name; }
    // 返回最佳出拳
    public Punch getBestPunch() { return bestPunch; }

    // 判断是否命中弱点
    public boolean hitVulnerability(Punch punch) {
        return vulnerability == punch;
    }
}
```