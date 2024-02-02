# `basic-computer-games\16_Bug\java\src\ComputerBug.java`

```py
public class ComputerBug extends Insect {

    // 创建特定于计算机玩家的消息。

    public ComputerBug() {
        // 调用超类构造函数进行初始化。
        super();
        // 添加关于触角的消息
        addMessages(new String[]{"I GET A FEELER.", "I HAVE " + MAX_FEELERS + " FEELERS ALREADY.", "I DO NOT HAVE A HEAD."}, PARTS.FEELERS);
        // 添加关于头部的消息
        addMessages(new String[]{"I NEEDED A HEAD.", "I DO NOT NEED A HEAD.", "I DO NOT HAVE A NECK."}, PARTS.HEAD);
        // 添加关于颈部的消息
        addMessages(new String[]{"I NOW HAVE A NECK.", "I DO NOT NEED A NECK.", "I DO NOT HAVE A BODY."}, PARTS.NECK);
        // 添加关于身体的消息
        addMessages(new String[]{"I NOW HAVE A BODY.", "I DO NOT NEED A BODY."}, PARTS.BODY);
        // 添加关于尾部的消息
        addMessages(new String[]{"I NOW HAVE A TAIL.", "I DO NOT NEED A TAIL.", "I DO NOT HAVE A BODY."}, PARTS.TAIL);
        // 添加关于腿部的消息
        addMessages(new String[]{"I NOW HAVE ^^^" + " LEG", "I HAVE " + MAX_LEGS + " FEET.", "I DO NOT HAVE A BODY."}, PARTS.LEGS);
    }
}
```