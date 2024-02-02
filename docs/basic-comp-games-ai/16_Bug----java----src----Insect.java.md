# `basic-computer-games\16_Bug\java\src\Insect.java`

```py
import java.util.ArrayList;
import java.util.Arrays;

/**
 * This tracks the insect (bug) and has methods to
 * add body parts, create an array of output so it
 * can be drawn and to determine if a bug is complete.
 * N.B. This is a super class for ComputerBug and PlayerBug
 */
public class Insect {

    public static final int MAX_FEELERS = 2;
    public static final int MAX_LEGS = 6;

    public static final int ADDED = 0;
    public static final int NOT_ADDED = 1;
    public static final int MISSING = 2;

    // Various parts of the bug
    public enum PARTS {
        FEELERS,
        HEAD,
        NECK,
        BODY,
        TAIL,
        LEGS
    }

    // Tracks what parts of the bug have been added
    private boolean body;  // Track if the body has been added
    private boolean neck;  // Track if the neck has been added
    private boolean head;  // Track if the head has been added
    private int feelers;  // Track the number of feelers added
    private boolean tail;  // Track if the tail has been added
    private int legs;  // Track the number of legs added

    // Messages about for various body parts
    // These are set in the subclass ComputerBug or PlayerBug
    private String[] bodyMessages;  // Messages related to the body
    private String[] neckMessages;  // Messages related to the neck
    private String[] headMessages;  // Messages related to the head
    private String[] feelerMessages;  // Messages related to the feelers
    private String[] tailMessages;  // Messages related to the tail
    private String[] legMessages;  // Messages related to the legs

    public Insect() {
        init();  // Initialize the insect
    }

    /**
     * Add a body to the bug if there is not one already added.
     *
     * @return return an appropriate message about the status of the operation.
     */
    public String addBody() {
        boolean currentState = false;  // Initialize the current state as false

        if (!body) {  // If body has not been added
            body = true;  // Set body as added
            currentState = true;  // Update current state
        }

        return addBodyMessage(currentState);  // Return the message about the status of the operation
    }

    /**
     * Create output based on adding the body or it being already added previously
     *
     * @return contains the output message
     */
    // 根据是否添加了身体，返回相应的消息
    private String addBodyMessage(boolean wasAdded) {

        // 根据身体是否添加返回相应的消息
        if (wasAdded) {
            return bodyMessages[ADDED];
        } else {
            return bodyMessages[NOT_ADDED];
        }
    }

    /**
     * 如果之前已经添加了身体，并且之前没有添加过脖子，则添加一个脖子
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

    /**
     * 如果已经存在脖子，并且之前没有添加过头部，则为虫子添加头部
     *
     * @return 描述操作成功与否的文本
     */
    public String addHead() {

        int status = NOT_ADDED;  // 默认状态为未添加

        if (!neck) {
            // 没有脖子，无法添加头部
            status = MISSING;
        } else if (!head) {
            head = true;
            status = ADDED;
        }

        return headMessages[status];
    }

    /**
     * 如果之前已经添加了头部，并且之前添加的触角数量不超过2（MAX_FEELERS）个，则为虫子添加触角
     *
     * @return 描述操作状态的文本
     */
    public String addFeelers() {

        int status = NOT_ADDED;  // 默认状态为未添加

        if (!head) {
            // 没有头部，无法添加触角
            status = MISSING;
        } else if (feelers < MAX_FEELERS) {
            feelers++;
            status = ADDED;
        }

        return feelerMessages[status];
    }
    /**
     * Add a tail to the bug if a) there is already a body previously added
     * to the bug and b) there is not already a tail added.
     *
     * @return text outlining the status of the operation.
     */
    public String addTail() {

        int status = NOT_ADDED;  // Default is not added

        if (!body) {
            // No body, cannot add a tail
            status = MISSING;
        } else if (!tail) {
            tail = true;
            status = ADDED;
        }

        return tailMessages[status];
    }

    /**
     * Add a leg to the bug if a) there is already a body previously added
     * b) there are less than 6 (MAX_LEGS) previously added.
     *
     * @return text outlining status of the operation.
     */
    public String addLeg() {

        int status = NOT_ADDED;  // Default is not added

        if (!body) {
            // No body, cannot add a leg
            status = MISSING;
        } else if (legs < MAX_LEGS) {
            legs++;
            status = ADDED;
        }

        String message = "";

        // Create a string showing the result of the operation

        switch(status) {
            case ADDED:
                // Replace # with number of legs
                message = legMessages[status].replace("^^^", String.valueOf(legs));
                // Add text S. if >1 leg, or just . if one leg.
                if (legs > 1) {
                    message += "S.";
                } else {
                    message += ".";
                }
                break;

            case NOT_ADDED:

                // Deliberate fall through to next case as its the
                // same code to be executed
            case MISSING:
                message = legMessages[status];
                break;
        }

        return message;
    }

    /**
     * Initialise
     */
    // 初始化各个身体部位的状态
    public void init() {
        body = false;
        neck = false;
        head = false;
        feelers = 0;
        tail = false;
        legs = 0;
    }

    /**
     * 根据玩家类型添加不同的消息
     * 这个方法由该类的子类调用
     * 例如：参见 ComputerBug 或 PlayerBug 类
     *
     * @param messages 消息数组
     * @param bodyPart 消息所关联的身体部位
     */
    public void addMessages(String[] messages, PARTS bodyPart) {

        switch (bodyPart) {
            case FEELERS:
                feelerMessages = messages;
                break;

            case HEAD:
                headMessages = messages;
                break;

            case NECK:
                neckMessages = messages;
                break;

            case BODY:
                bodyMessages = messages;
                break;

            case TAIL:
                tailMessages = messages;
                break;

            case LEGS:
                legMessages = messages;
                break;
        }
    }

    /**
     * 返回一个包含“bug”的字符串数组，可以输出到控制台
     *
     * @return 准备绘制的bug
     */
    }

    /**
     * 检查bug是否完整，即是否有2个（MAX_FEELERS）触角，一个头部，一个颈部，一个身体，一个尾巴和6个（MAX_LEGS）脚。
     *
     * @return 如果完整则返回true
     */
    public boolean complete() {
        return (feelers == MAX_FEELERS)
                && head
                && neck
                && body
                && tail
                && (legs == MAX_LEGS);
    }

    /**
     * 通过创建包含X个空格的字符串来模拟制表符
     *
     * @param number 包含所需空格数的数字
     * @return 包含空格的字符串
     */
    private String addSpaces(int number) {
        char[] spaces = new char[number];
        Arrays.fill(spaces, ' ');
        return new String(spaces);

    }
# 闭合前面的函数定义
```