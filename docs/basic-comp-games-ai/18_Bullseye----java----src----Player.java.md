# `basic-computer-games\18_Bullseye\java\src\Player.java`

```
/**
 * 游戏中的玩家 - 包括姓名和分数
 *
 */
public class Player {

    // 玩家姓名，一旦赋值就不可更改
    private final String name;

    // 玩家分数
    private int score;

    // 构造函数，初始化玩家姓名和分数
    Player(String name) {
        this.name = name;
        this.score = 0;
    }

    // 增加玩家分数
    public void addScore(int score) {
        this.score += score;
    }

    // 获取玩家姓名
    public String getName() {
        return name;
    }

    // 获取玩家分数
    public int getScore() {
        return score;
    }
}
```