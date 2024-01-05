# `16_Bug\java\src\Insect.java`

```
# 导入必要的模块
import java.util.ArrayList;
import java.util.Arrays;

/**
 * This tracks the insect (bug) and has methods to
 * add body parts, create an array of output so it
 * can be drawn and to determine if a bug is complete.
 * N.B. This is a super class for ComputerBug and PlayerBug
 */
public class Insect {

    # 定义常量，表示感觉器的最大数量和腿的最大数量
    public static final int MAX_FEELERS = 2;
    public static final int MAX_LEGS = 6;

    # 定义常量，表示添加成功、未添加和缺失
    public static final int ADDED = 0;
    public static final int NOT_ADDED = 1;
    public static final int MISSING = 2;

    // Various parts of the bug
    # 定义枚举类型，表示昆虫的各个部分
    public enum PARTS {
        FEELERS,  // 定义感觉器官
        HEAD,  // 定义头部
        NECK,  // 定义颈部
        BODY,  // 定义身体
        TAIL,  // 定义尾部
        LEGS  // 定义腿部
    }

    // Tracks what parts of the bug have been added
    // 跟踪虫子的哪些部分已经添加
    private boolean body;  // 身体部分是否已添加
    private boolean neck;  // 颈部是否已添加
    private boolean head;  // 头部是否已添加
    private int feelers;  // 感觉器官的数量
    private boolean tail;  // 尾部是否已添加
    private int legs;  // 腿部的数量

    // Messages about for various body parts
    // These are set in the subclass ComputerBug or PlayerBug
    // 关于各种身体部位的消息
    // 这些消息在子类ComputerBug或PlayerBug中设置
    private String[] bodyMessages;  // 身体部分的消息
    private String[] neckMessages;  // 颈部的消息
    # 定义私有成员变量，分别存储头部、触角、尾部和腿部的消息
    private String[] headMessages;
    private String[] feelerMessages;
    private String[] tailMessages;
    private String[] legMessages;

    # 构造函数，初始化昆虫的各个部分消息
    public Insect() {
        init();
    }

    /**
     * 如果昆虫还没有身体，添加一个身体。
     *
     * @return 返回关于操作状态的适当消息。
     */
    public String addBody() {

        # 当前状态标志，初始值为假
        boolean currentState = false;

        # 如果还没有身体，将身体状态标志设为真
        if (!body) {
            body = true;
            currentState = true;  # 设置当前状态为真

        }

        return addBodyMessage(currentState);  # 调用addBodyMessage函数并传入currentState参数作为返回结果
    }

    /**
     * 根据是否已经添加过body来创建输出消息
     *
     * @return 包含输出消息的字符串
     */

    private String addBodyMessage(boolean wasAdded) {

        // 根据body是否已经添加过来返回相应的消息
        if (wasAdded) {
            return bodyMessages[ADDED];  # 如果已经添加过body，则返回ADDED索引处的消息
        } else {
            return bodyMessages[NOT_ADDED];  # 如果还未添加过body，则返回NOT_ADDED索引处的消息
    /**
     * 如果之前已经添加了身体，并且之前没有添加过脖子，则添加一个脖子。
     *
     * @return 包含操作状态的文本
     */
    public String addNeck() {

        int status = NOT_ADDED;  // 默认状态为未添加

        if (!body) {
            // 没有身体，无法添加脖子
            status = MISSING;
        } else if (!neck) {
            neck = true;
            status = ADDED;
        }
        return neckMessages[status];
    }
```

这段代码是一个方法的结尾，根据操作的结果返回相应的文本信息。

```
    /**
     * Add a head to the bug if a) there already exists a neck and
     * b) a head has not previously been added
     *
     * @return text outlining the success of the operation
     */
    public String addHead() {
```

这是一个方法的声明，用于给虫子添加头部。方法的注释说明了添加头部的条件和返回的文本信息。

```
        int status = NOT_ADDED;  // Default is not added
```

定义一个整型变量status，初始值为NOT_ADDED，表示头部还未被添加。

```
        if (!neck) {
            // No neck, cannot add a head
            status = MISSING;
        } else if (!head) {
            head = true;
            status = ADDED;
```

如果虫子没有颈部，则无法添加头部，将status设置为MISSING；如果虫子有颈部但没有头部，则将头部设置为true，并将status设置为ADDED。

```
        return neckMessages[status];
    }
```

根据status的值返回相应的文本信息。
        }

        return headMessages[status];
    }
```

这部分代码是一个方法的结束和返回语句，根据status的值返回对应的消息。

```
    /**
     * Add a feeler to the head if a) there has been a head added to
     * the bug previously, and b) there are not already 2 (MAX_FEELERS)
     * feelers previously added to the bug.
     *
     * @return text outlining the status of the operation
     */
    public String addFeelers() {

        int status = NOT_ADDED;  // Default is not added

        if (!head) {
            // No head, cannot add a feeler
            status = MISSING;
        } else if (feelers < MAX_FEELERS) {
```

这部分代码是一个方法的开始和定义，注释说明了这个方法的作用和返回值。然后定义了一个整型变量status，并初始化为NOT_ADDED。接下来是一个条件判断，如果head为false，则将status赋值为MISSING；如果feelers小于MAX_FEELERS，则执行下面的代码块。
            feelers++;  # 增加触角数量
            status = ADDED;  # 修改状态为已添加
        }

        return feelerMessages[status];  # 返回操作状态的文本
    }

    /**
     * 如果已经添加了身体部分并且尚未添加尾部，则向 bug 添加尾部。
     *
     * @return 描述操作状态的文本。
     */
    public String addTail() {

        int status = NOT_ADDED;  // 默认状态为未添加

        if (!body) {
            // 没有身体部分，无法添加尾部
            status = MISSING;  # 修改状态为缺失
        } else if (!tail) {  // 如果尾部还没有添加
            tail = true;  // 将尾部标记为已添加
            status = ADDED;  // 修改状态为已添加
        }

        return tailMessages[status];  // 返回操作状态的文本
    }

    /**
     * Add a leg to the bug if a) there is already a body previously added
     * b) there are less than 6 (MAX_LEGS) previously added.
     *
     * @return text outlining status of the operation.
     */
    public String addLeg() {

        int status = NOT_ADDED;  // 默认状态为未添加

        if (!body) {  // 如果没有身体，无法添加腿部
            status = MISSING;  # 将状态设置为MISSING
        } else if (legs < MAX_LEGS) {  # 如果腿的数量小于最大腿的数量
            legs++;  # 增加腿的数量
            status = ADDED;  # 将状态设置为ADDED
        }

        String message = "";  # 创建一个空字符串message

        // Create a string showing the result of the operation
        // 创建一个字符串来显示操作的结果

        switch(status) {  # 根据状态进行切换
            case ADDED:  # 如果状态是ADDED
                // Replace # with number of legs
                // 用腿的数量替换#
                message = legMessages[status].replace("^^^", String.valueOf(legs));
                // Add text S. if >1 leg, or just . if one leg.
                // 如果腿的数量大于1，则添加文本S.，否则添加.
                if (legs > 1) {
                    message += "S.";
                } else {
                    message += ".";
                }
                break;  // 结束当前的 case，跳出 switch 语句

            case NOT_ADDED:
                // 故意落入下一个 case，因为要执行相同的代码
                // Deliberate fall through to next case as its the
                // same code to be executed
            case MISSING:
                message = legMessages[status];  // 根据 status 状态获取对应的消息
                break;  // 结束当前的 case，跳出 switch 语句
        }

        return message;  // 返回消息

    }

    /**
     * Initialise
     */
    public void init() {
        body = false;  // 初始化 body 变量为 false
        neck = false;  // 初始化 neck 变量为 false
        head = false;  // 初始化头部为假
        feelers = 0;   // 初始化触角数量为0
        tail = false;  // 初始化尾部为假
        legs = 0;      // 初始化腿的数量为0
    }

    /**
     * 根据玩家类型添加唯一的消息
     * 这个类的一个子类调用这个方法
     * 例如，参见ComputerBug或PlayerBug类
     *
     * @param messages 一个消息数组
     * @param bodyPart  消息所关联的身体部位
     */
    public void addMessages(String[] messages, PARTS bodyPart) {

        switch (bodyPart) {
            case FEELERS:  // 如果是触角部位
                feelerMessages = messages;  // 将消息数组赋值给触角消息数组
                break;
# 当条件为HEAD时，将messages赋值给headMessages
case HEAD:
    headMessages = messages;
    break;

# 当条件为NECK时，将messages赋值给neckMessages
case NECK:
    neckMessages = messages;
    break;

# 当条件为BODY时，将messages赋值给bodyMessages
case BODY:
    bodyMessages = messages;
    break;

# 当条件为TAIL时，将messages赋值给tailMessages
case TAIL:
    tailMessages = messages;
    break;

# 当条件为LEGS时，将messages赋值给legMessages
case LEGS:
    legMessages = messages;
    break;
    }

    /**
     * Returns a string array containing
     * the "bug" that can be output to console
     *
     * @return the bug ready to draw
     */
    public ArrayList<String> draw() {
        ArrayList<String> bug = new ArrayList<>();  // 创建一个字符串数组列表用于存储“bug”图案
        StringBuilder lineOutput;  // 创建一个字符串构建器用于构建每一行的输出

        // Feelers
        if (feelers > 0) {  // 如果触角数量大于0
            for (int i = 0; i < 4; i++) {  // 循环4次
                lineOutput = new StringBuilder(addSpaces(10));  // 在每行前面添加10个空格
                for (int j = 0; j < feelers; j++) {  // 循环触角数量次
                    lineOutput.append("A ");  // 在每行后面添加“A ”
                bug.add(lineOutput.toString());  # 将lineOutput的内容转换为字符串并添加到bug列表中
            }
        }

        if (head):  # 如果head为真
            lineOutput = new StringBuilder(addSpaces(8) + "HHHHHHH");  # 创建一个包含8个空格和"HHHHHHH"的字符串
            bug.add(lineOutput.toString());  # 将lineOutput的内容转换为字符串并添加到bug列表中
            lineOutput = new StringBuilder(addSpaces(8) + "H" + addSpaces(5) + "H");  # 创建一个包含8个空格、一个"H"、5个空格和一个"H"的字符串
            bug.add(lineOutput.toString());  # 将lineOutput的内容转换为字符串并添加到bug列表中
            lineOutput = new StringBuilder(addSpaces(8) + "H O O H");  # 创建一个包含8个空格、"H O O H"的字符串
            bug.add(lineOutput.toString());  # 将lineOutput的内容转换为字符串并添加到bug列表中
            lineOutput = new StringBuilder(addSpaces(8) + "H" + addSpaces(5) + "H");  # 创建一个包含8个空格、一个"H"、5个空格和一个"H"的字符串
            bug.add(lineOutput.toString());  # 将lineOutput的内容转换为字符串并添加到bug列表中
            lineOutput = new StringBuilder(addSpaces(8) + "H" + addSpaces(2) + "V" + addSpaces(2) + "H");  # 创建一个包含8个空格、一个"H"、2个空格、"V"、2个空格和一个"H"的字符串
            bug.add(lineOutput.toString());  # 将lineOutput的内容转换为字符串并添加到bug列表中
            lineOutput = new StringBuilder(addSpaces(8) + "HHHHHHH");  # 创建一个包含8个空格和"HHHHHHH"的字符串
            bug.add(lineOutput.toString());  # 将lineOutput的内容转换为字符串并添加到bug列表中
        }

        if (neck):  # 如果neck为真
        for (int i = 0; i < 2; i++) {
            // 创建一个新的 StringBuilder 对象，用于构建输出行，行首添加 10 个空格，然后添加 "N N"
            lineOutput = new StringBuilder(addSpaces(10) + "N N");
            // 将构建好的行添加到 bug 列表中
            bug.add(lineOutput.toString());
        }
        // 如果 body 为真，则执行以下代码块
        if (body) {
            // 创建一个新的 StringBuilder 对象，用于构建输出行，行首添加 5 个空格，然后添加 "BBBBBBBBBBBB"
            lineOutput = new StringBuilder(addSpaces(5) + "BBBBBBBBBBBB");
            // 将构建好的行添加到 bug 列表中
            bug.add(lineOutput.toString());
            for (int i = 0; i < 2; i++) {
                // 创建一个新的 StringBuilder 对象，用于构建输出行，行首添加 5 个空格，然后添加 "B"，再添加 10 个空格，最后再添加一个 "B"
                lineOutput = new StringBuilder(addSpaces(5) + "B" + addSpaces(10) + "B");
                // 将构建好的行添加到 bug 列表中
                bug.add(lineOutput.toString());
            }
            // 如果 tail 为真，则执行以下代码块
            if (tail) {
                // 创建一个新的 StringBuilder 对象，用于构建输出行，内容为 "TTTTTB"，然后添加 10 个空格，最后再添加一个 "B"
                lineOutput = new StringBuilder("TTTTTB" + addSpaces(10) + "B");
                // 将构建好的行添加到 bug 列表中
                bug.add(lineOutput.toString());
            }
            // 创建一个新的 StringBuilder 对象，用于构建输出行，行首添加 5 个空格，然后添加 "BBBBBBBBBBBB"
            lineOutput = new StringBuilder(addSpaces(5) + "BBBBBBBBBBBB");
            // 将构建好的行添加到 bug 列表中
            bug.add(lineOutput.toString());
        }
        if (legs > 0) {  # 检查bug是否有腿
            for (int i = 0; i < 2; i++) {  # 循环两次
                lineOutput = new StringBuilder(addSpaces(5));  # 创建一个包含5个空格的StringBuilder对象
                for (int j = 0; j < legs; j++) {  # 循环bug的腿数次
                    lineOutput.append(" L");  # 在StringBuilder对象中添加一个空格和字母L
                }
                bug.add(lineOutput.toString());  # 将StringBuilder对象转换为字符串并添加到bug列表中
            }
        }

        return bug;  # 返回bug列表
    }

    /**
     * Check if the bug is complete i.e. it has
     * 2 (MAX_FEELERS) feelers, a head, a neck, a body
     * a tail and 6 (MAX_FEET) feet.
     *
     * @return true if complete.  # 如果bug完整则返回true
    */
    // 检查是否所有部件都完整
    public boolean complete() {
        return (feelers == MAX_FEELERS)  // 检查触角数量是否达到最大值
                && head  // 检查是否有头部
                && neck  // 检查是否有颈部
                && body  // 检查是否有身体
                && tail  // 检查是否有尾部
                && (legs == MAX_LEGS);  // 检查腿的数量是否达到最大值
    }

    /**
     * 通过创建包含 X 个空格的字符串来模拟制表符。
     *
     * @param number 包含所需空格数量的参数
     * @return 包含空格的字符串
     */
    private String addSpaces(int number) {
        char[] spaces = new char[number];  // 创建一个包含指定数量空格的字符数组
        Arrays.fill(spaces, ' ');  // 用空格填充字符数组
        return new String(spaces);  // 将字符数组转换为字符串并返回
    }
    }  # 结束遍历循环
}  # 结束函数定义
```