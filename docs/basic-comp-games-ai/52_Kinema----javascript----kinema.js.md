# `52_Kinema\javascript\kinema.js`

```
// 创建一个新的 Promise 对象，用于处理异步操作
// 创建一个 input 元素，用于用户输入
// 在页面上打印提示符 "? "
// 设置 input 元素的类型为文本输入类型
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
# 结束函数定义

# 定义一个名为 tab 的函数，参数为 space
function tab(space)
{
    # 初始化一个空字符串
    var str = "";
    # 当 space 大于 0 时执行循环
    while (space-- > 0)
```
        str += " ";  // 将空格添加到字符串末尾
    return str;  // 返回修改后的字符串
}

var q;  // 声明变量 q

function evaluate_answer(str, a)  // 定义函数 evaluate_answer，接受两个参数 str 和 a
{
    g = parseFloat(str);  // 将 str 转换为浮点数并赋值给变量 g
    if (Math.abs((g - a) / a) < 0.15) {  // 如果 g 与 a 的相对误差小于 0.15
        print("CLOSE ENOUGH.\n");  // 打印 "CLOSE ENOUGH."
        q++;  // 变量 q 自增 1
    } else {
        print("NOT EVEN CLOSE....\n");  // 打印 "NOT EVEN CLOSE...."
    }
    print("CORRECT ANSWER IS " + a + "\n\n");  // 打印 "CORRECT ANSWER IS " 后跟变量 a 的值和两个换行符
}

// Main program  // 主程序
async function main()  // 异步函数 main
    {
        # 打印 "KINEMA"，并在前面加上 33 个空格
        print(tab(33) + "KINEMA\n");
        # 打印 "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY"，并在前面加上 15 个空格
        print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
        # 打印三个空行
        print("\n");
        print("\n");
        print("\n");
        # 进入无限循环
        while (1) {
            # 打印两个空行
            print("\n");
            print("\n");
            # 将变量 q 设为 0
            q = 0;
            # 将变量 v 设为 5 加上 35 与 0 之间的随机整数
            v = 5 + Math.floor(35 * Math.random());
            # 打印 "A BALL IS THROWN UPWARDS AT " 后面加上 v 的值和 " METERS PER SECOND."
            print("A BALL IS THROWN UPWARDS AT " + v + " METERS PER SECOND.\n");
            # 计算 a 的值为 0.5 乘以 v 的平方
            a = 0.5 * Math.pow(v, 2);
            # 打印 "HOW HIGH WILL IT GO (IN METERS)"
            print("HOW HIGH WILL IT GO (IN METERS)");
            # 等待用户输入，并将输入的值赋给变量 str
            str = await input();
            # 调用 evaluate_answer 函数，传入用户输入的值和 a 的值作为参数
            evaluate_answer(str, a);
            # 将 a 的值设为 v 除以 5
            a = v / 5;
            # 打印 "HOW LONG UNTIL IT RETURNS (IN SECONDS)"
            print("HOW LONG UNTIL IT RETURNS (IN SECONDS)");
            # 等待用户输入，并将输入的值赋给变量 str
            str = await input();
        evaluate_answer(str, a); // 调用 evaluate_answer 函数，传入参数 str 和 a
        t = 1 + Math.floor(2 * v * Math.random()) / 10; // 计算 t 的值
        a = v - 10 * t; // 计算 a 的值
        print("WHAT WILL ITS VELOCITY BE AFTER " + t + " SECONDS"); // 打印输出提示信息
        str = await input(); // 等待用户输入，并将输入值赋给 str
        evaluate_answer(str, a); // 调用 evaluate_answer 函数，传入参数 str 和 a
        print("\n"); // 打印输出空行
        print(q + " RIGHT OUT OF 3."); // 打印输出正确答案数量
        if (q < 2) // 如果正确答案数量小于 2
            continue; // 继续循环
        print("  NOT BAD.\n"); // 打印输出提示信息
    }
}

main(); // 调用 main 函数
```