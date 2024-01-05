# `d:/src/tocomm/basic-computer-games\16_Bug\java\src\PlayerBug.java`

```
// 创建一个名为PlayerBug的类，继承自Insect类

public class PlayerBug extends Insect {

    // 为玩家创建特定的消息

    // 构造函数，调用父类的构造函数进行初始化
    public PlayerBug() {
        super();
        // 为触角添加消息
        addMessages(new String[]{"I NOW GIVE YOU A FEELER.", "YOU HAVE " + MAX_FEELERS + " FEELERS ALREADY.", "YOU DO NOT HAVE A HEAD."}, PARTS.FEELERS);
        // 为头部添加消息
        addMessages(new String[]{"YOU NEEDED A HEAD.", "YOU HAVE A HEAD.", "YOU DO NOT HAVE A NECK."}, PARTS.HEAD);
        // 为颈部添加消息
        addMessages(new String[]{"YOU NOW HAVE A NECK.", "YOU DO NOT NEED A NECK.", "YOU DO NOT HAVE A BODY."}, PARTS.NECK);
        // 为身体添加消息
        addMessages(new String[]{"YOU NOW HAVE A BODY.", "YOU DO NOT NEED A BODY."}, PARTS.BODY);
        // 为尾部添加消息
        addMessages(new String[]{"I NOW GIVE YOU A TAIL.", "YOU ALREADY HAVE A TAIL.", "YOU DO NOT HAVE A BODY."}, PARTS.TAIL);
        // 为腿部添加消息
        addMessages(new String[]{"YOU NOW HAVE ^^^ LEG", "YOU HAVE " + MAX_LEGS + " FEET ALREADY.", "YOU DO NOT HAVE A BODY."}, PARTS.LEGS);
    }
}
```