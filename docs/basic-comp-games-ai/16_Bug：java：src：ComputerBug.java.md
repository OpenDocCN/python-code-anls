# `16_Bug\java\src\ComputerBug.java`

```
public class ComputerBug extends Insect {

    // Create messages specific to the computer player.

    public ComputerBug() {
        // 调用父类构造函数进行初始化
        super();
        // 为感觉器官添加消息
        addMessages(new String[]{"I GET A FEELER.", "I HAVE " + MAX_FEELERS + " FEELERS ALREADY.", "I DO NOT HAVE A HEAD."}, PARTS.FEELERS);
        // 为头部添加消息
        addMessages(new String[]{"I NEEDED A HEAD.", "I DO NOT NEED A HEAD.", "I DO NOT HAVE A NECK."}, PARTS.HEAD);
        // 为颈部添加消息
        addMessages(new String[]{"I NOW HAVE A NECK.", "I DO NOT NEED A NECK.", "I DO NOT HAVE A BODY."}, PARTS.NECK);
        // 为身体添加消息
        addMessages(new String[]{"I NOW HAVE A BODY.", "I DO NOT NEED A BODY."}, PARTS.BODY);
        // 为尾部添加消息
        addMessages(new String[]{"I NOW HAVE A TAIL.", "I DO NOT NEED A TAIL.", "I DO NOT HAVE A BODY."}, PARTS.TAIL);
        // 为腿部添加消息
        addMessages(new String[]{"I NOW HAVE ^^^" + " LEG", "I HAVE " + MAX_LEGS + " FEET.", "I DO NOT HAVE A BODY."}, PARTS.LEGS);
    }
}
```