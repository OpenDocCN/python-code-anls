# `basic-computer-games\52_Kinema\javascript\kinema.js`

```

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
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 输出输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的字符串
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义一个生成指定数量空格的函数
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 定义全局变量 q
var q;

// 根据输入的字符串和正确答案进行评估
function evaluate_answer(str, a)
{
    g = parseFloat(str);
    // 判断输入的值是否接近正确答案
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
    print("\n");
    while (1) {
        print("\n");
        print("\n");
        q = 0;
        // 生成一个随机速度
        v = 5 + Math.floor(35 * Math.random());
        // 输出问题
        print("A BALL IS THROWN UPWARDS AT " + v + " METERS PER SECOND.\n");
        print("\n");
        // 计算正确答案
        a = 0.5 * Math.pow(v, 2);
        print("HOW HIGH WILL IT GO (IN METERS)");
        // 等待输入
        str = await input();
        // 评估答案
        evaluate_answer(str, a);
        // 重新计算正确答案
        a = v / 5;
        print("HOW LONG UNTIL IT RETURNS (IN SECONDS)");
        // 等待输入
        str = await input();
        // 评估答案
        evaluate_answer(str, a);
        // 生成一个随机时间
        t = 1 + Math.floor(2 * v * Math.random()) / 10;
        // 重新计算正确答案
        a = v - 10 * t;
        print("WHAT WILL ITS VELOCITY BE AFTER " + t + " SECONDS");
        // 等待输入
        str = await input();
        // 评估答案
        evaluate_answer(str, a);
        print("\n");
        // 输出答对的数量
        print(q + " RIGHT OUT OF 3.");
        // 如果答对的数量小于2，则继续循环
        if (q < 2)
            continue;
        // 输出提示
        print("  NOT BAD.\n");
    }
}

// 调用主程序
main();

```