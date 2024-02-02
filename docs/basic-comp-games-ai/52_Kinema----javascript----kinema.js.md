# `basic-computer-games\52_Kinema\javascript\kinema.js`

```py
// 定义一个打印函数，将字符串输出到指定的元素中
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义一个输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建一个输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 输出输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串并返回
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义一个全局变量 q
var q;

// 根据输入的字符串和答案进行评估
function evaluate_answer(str, a)
{
    g = parseFloat(str);
    // 判断输入的值是否接近答案
    if (Math.abs((g - a) / a) < 0.15) {
        print("CLOSE ENOUGH.\n");
        q++;
    } else {
        print("NOT EVEN CLOSE....\n");
    }
    // 输出正确答案
    print("CORRECT ANSWER IS " + a + "\n\n");
}

// 主程序
async function main()
{
    // 输出标题
    print(tab(33) + "KINEMA\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
}
    # 打印换行符
    print("\n");
    # 进入无限循环
    while (1) {
        # 打印两个换行符
        print("\n");
        print("\n");
        # 初始化变量 q 为 0
        q = 0;
        # 生成一个 5 到 40 之间的随机数，并赋值给变量 v
        v = 5 + Math.floor(35 * Math.random());
        # 打印提示信息，说明球以 v 米每秒的速度向上抛出
        print("A BALL IS THROWN UPWARDS AT " + v + " METERS PER SECOND.\n");
        print("\n");
        # 根据公式计算球抛出的高度，并赋值给变量 a
        a = 0.5 * Math.pow(v, 2);
        # 打印提示信息，要求输入球抛出的最高高度
        print("HOW HIGH WILL IT GO (IN METERS)");
        # 等待用户输入，并将输入的值赋给变量 str
        str = await input();
        # 调用 evaluate_answer 函数，判断输入的值是否正确
        evaluate_answer(str, a);
        # 根据公式计算球返回所需的时间，并赋值给变量 a
        a = v / 5;
        # 打印提示信息，要求输入球返回所需的时间
        print("HOW LONG UNTIL IT RETURNS (IN SECONDS)");
        # 等待用户输入，并将输入的值赋给变量 str
        str = await input();
        # 调用 evaluate_answer 函数，判断输入的值是否正确
        evaluate_answer(str, a);
        # 生成一个 1 到 2v/10 之间的随机数，并赋值给变量 t
        t = 1 + Math.floor(2 * v * Math.random()) / 10;
        # 根据公式计算 t 秒后球的速度，并赋值给变量 a
        a = v - 10 * t;
        # 打印提示信息，要求输入 t 秒后球的速度
        print("WHAT WILL ITS VELOCITY BE AFTER " + t + " SECONDS");
        # 等待用户输入，并将输入的值赋给变量 str
        str = await input();
        # 调用 evaluate_answer 函数，判断输入的值是否正确
        evaluate_answer(str, a);
        # 打印两个换行符
        print("\n");
        # 打印变量 q 的值和提示信息
        print(q + " RIGHT OUT OF 3.");
        # 如果 q 小于 2，则继续循环，否则跳出循环
        if (q < 2)
            continue;
        # 打印提示信息
        print("  NOT BAD.\n");
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```