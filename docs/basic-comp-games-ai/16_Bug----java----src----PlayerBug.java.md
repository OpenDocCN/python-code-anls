# `basic-computer-games\16_Bug\java\src\PlayerBug.java`

```

// 创建一个名为 PlayerBug 的类，继承自 Insect 类

public class PlayerBug extends Insect {

    // 创建特定于玩家的消息

    // 构造函数，调用父类的构造函数进行初始化
    public PlayerBug() {
        super();
        // 添加关于触角的消息
        addMessages(new String[]{"I NOW GIVE YOU A FEELER.", "YOU HAVE " + MAX_FEELERS + " FEELERS ALREADY.", "YOU DO NOT HAVE A HEAD."}, PARTS.FEELERS);
        // 添加关于头部的消息
        addMessages(new String[]{"YOU NEEDED A HEAD.", "YOU HAVE A HEAD.", "YOU DO NOT HAVE A NECK."}, PARTS.HEAD);
        // 添加关于颈部的消息
        addMessages(new String[]{"YOU NOW HAVE A NECK.", "YOU DO NOT NEED A NECK.", "YOU DO NOT HAVE A BODY."}, PARTS.NECK);
        // 添加关于身体的消息
        addMessages(new String[]{"YOU NOW HAVE A BODY.", "YOU DO NOT NEED A BODY."}, PARTS.BODY);
        // 添加关于尾部的消息
        addMessages(new String[]{"I NOW GIVE YOU A TAIL.", "YOU ALREADY HAVE A TAIL.", "YOU DO NOT HAVE A BODY."}, PARTS.TAIL);
        // 添加关于腿部的消息
        addMessages(new String[]{"YOU NOW HAVE ^^^ LEG", "YOU HAVE " + MAX_LEGS + " FEET ALREADY.", "YOU DO NOT HAVE A BODY."}, PARTS.LEGS);
    }
}

```