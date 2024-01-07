# `basic-computer-games\15_Boxing\java\Player.java`

```

/**
 * The Player class model the user and compuer player
 * Player类模拟用户和计算机玩家
 */
public class Player {
    private final String name; // 玩家姓名
    private final Punch bestPunch; // 最佳出拳
    private final Punch vulnerability; // 弱点
    private boolean isPlayer = false; // 是否是玩家，默认为false

    public Player(String name, Punch bestPunch, Punch vulnerability) {
        this.name = name; // 初始化玩家姓名
        this.bestPunch = bestPunch; // 初始化最佳出拳
        this.vulnerability = vulnerability; // 初始化弱点
        this.isPlayer = true; // 设置为玩家
    }

    /**
     * Player with random Best Punch and Vulnerability
     * 随机生成最佳出拳和弱点的玩家
     */
    public Player(String name) {
        this.name = name; // 初始化玩家姓名

        int b1;
        int d1;

        do {
            b1 = Basic.randomOf(4); // 生成随机数作为最佳出拳
            d1 = Basic.randomOf(4); // 生成随机数作为弱点
        } while (b1 == d1);

        this.bestPunch = Punch.fromCode(b1); // 根据随机数生成最佳出拳
        this.vulnerability = Punch.fromCode(d1); // 根据随机数生成弱点
    }

    public boolean isPlayer() { return isPlayer; } // 返回是否是玩家
    public String getName() { return  name; } // 返回玩家姓名
    public Punch getBestPunch() { return bestPunch; } // 返回最佳出拳

    public boolean hitVulnerability(Punch punch) {
        return vulnerability == punch; // 判断是否命中弱点
    }
}

```