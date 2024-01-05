# `61_Math_Dice\javascript\mathdice.js`

```
// MATH DICE
// 该程序是一个名为MATH DICE的游戏，这段代码是将BASIC语言转换为Javascript语言
// 作者是Oscar Toledo G. (nanochess)

function print(str)
// 定义一个名为print的函数，用于在页面上输出文本
{
    document.getElementById("output").appendChild(document.createTextNode(str));
    // 在id为"output"的元素中添加一个文本节点，内容为传入的str参数
}

function input()
// 定义一个名为input的函数
{
    var input_element;
    var input_str;
    // 声明两个变量input_element和input_str

    return new Promise(function (resolve) {
    // 返回一个Promise对象，用于处理异步操作
        input_element = document.createElement("INPUT");
        // 创建一个input元素

        print("? ");
        // 在页面上输出"?"

        input_element.setAttribute("type", "text");
        // 设置input元素的类型为文本输入框
```
```python
# 设置输入元素的长度为50
input_element.setAttribute("length", "50");
# 将输入元素添加到 id 为 "output" 的元素中
document.getElementById("output").appendChild(input_element);
# 让输入元素获得焦点
input_element.focus();
# 初始化输入字符串为 undefined
input_str = undefined;
# 添加键盘按下事件监听器，当按下回车键时执行相应操作
input_element.addEventListener("keydown", function (event) {
    # 如果按下的键是回车键
    if (event.keyCode == 13) {
        # 将输入元素的值赋给输入字符串
        input_str = input_element.value;
        # 从 id 为 "output" 的元素中移除输入元素
        document.getElementById("output").removeChild(input_element);
        # 打印输入字符串
        print(input_str);
        # 打印换行符
        print("\n");
        # 解析并返回输入字符串
        resolve(input_str);
    }
});
# 结束键盘按下事件监听器的添加
});
}

# 定义一个函数，用于生成指定数量的空格
function tab(space)
{
    var str = "";
    # 当 space 大于 0 时，循环添加空格到 str 中
    while (space-- > 0)
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回处理后的字符串

}

// Main program
async function main()
{
    print(tab(31) + "MATH DICE\n");  // 打印标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");  // 打印创意计算的信息
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    print("THIS PROGRAM GENERATES SUCCESSIVE PICTURES OF TWO DICE.\n");  // 打印程序功能介绍
    print("WHEN TWO DICE AND AN EQUAL SIGN FOLLOWED BY A QUESTION\n");  // 打印提示信息
    print("MARK HAVE BEEN PRINTED, TYPE YOUR ANSWER AND THE RETURN KEY.\n"),  // 打印提示信息
    print("TO CONCLUDE THE LESSON, TYPE ZERO AS YOUR ANSWER.\n");  // 打印提示信息
    print("\n");  // 打印空行
    print("\n");  // 打印空行
    n = 0;  // 初始化变量 n 为 0
    while (1) {  // 进入无限循环
        n++;  // 增加 n 的值
        d = Math.floor(6 * Math.random() + 1);  // 生成一个 1 到 6 之间的随机整数并赋值给 d
        print(" ----- \n");  // 打印一行字符
        if (d == 1)  // 如果 d 等于 1
            print("I     I\n");  // 打印特定格式的字符
        else if (d == 2 || d == 3)  // 如果 d 等于 2 或者等于 3
            print("I *   I\n");  // 打印特定格式的字符
        else  // 否则
            print("I * * I\n");  // 打印特定格式的字符
        if (d == 2 || d == 4)  // 如果 d 等于 2 或者等于 4
            print("I     I\n");  // 打印特定格式的字符
        else if (d == 6)  // 如果 d 等于 6
            print("I * * I\n");  // 打印特定格式的字符
        else  // 否则
            print("I  *  I\n");  // 打印特定格式的字符
        if (d == 1)  // 如果 d 等于 1
            print("I     I\n");  // 打印特定格式的字符
        else if (d == 2 || d == 3)  // 如果 d 等于 2 或者等于 3
            print("I   * I\n");  // 打印特定格式的字符
        else  // 否则
            # 打印特定的字符串
            print("I * * I\n");
        # 打印特定的字符串
        print(" ----- \n");
        # 打印空行
        print("\n");
        # 如果 n 不等于 2，则执行下面的操作
        if (n != 2) {
            # 打印特定的字符串
            print("   +\n");
            # 打印空行
            print("\n");
            # 将变量 a 的值赋给变量 d
            a = d;
            # 继续循环
            continue;
        # 将变量 d 和 a 相加，结果赋给变量 t
        t = d + a;
        # 打印特定的字符串
        print("      =");
        # 将用户输入的值转换为整数类型，赋给变量 t1
        t1 = parseInt(await input());
        # 如果 t1 等于 0，则跳出循环
        if (t1 == 0)
            break;
        # 如果 t1 不等于 t，则执行下面的操作
        if (t1 != t) {
            # 打印特定的字符串
            print("NO, COUNT THE SPOTS AND GIVE ANOTHER ANSWER.\n");
            # 打印特定的字符串
            print("      =");
            # 将用户输入的值转换为整数类型，赋给变量 t1
            t1 = parseInt(await input());
            # 如果 t1 不等于 t，则执行下面的操作
            if (t1 != t) {
                # 打印特定的字符串
                print("NO, THE ANSWER IS " + t + "\n");
            }
        }
        if (t1 == t) {  # 如果 t1 等于 t
            print("RIGHT!\n");  # 打印 "RIGHT!\n"
        }
        print("\n");  # 打印换行
        print("THE DICE ROLL AGAIN...\n");  # 打印 "THE DICE ROLL AGAIN...\n"
        print("\n");  # 打印换行
        n = 0;  # 将 n 设为 0
    }
}

main();  # 调用 main 函数
```