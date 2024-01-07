# `basic-computer-games\61_Math_Dice\javascript\mathdice.js`

```

// MATH DICE
//
// Converted from BASIC to Javascript by Oscar Toledo G. (nanochess)
//

// 定义打印函数，将字符串输出到指定元素
function print(str)
{
    document.getElementById("output").appendChild(document.createTextNode(str));
}

// 定义输入函数，返回一个 Promise 对象
function input()
{
    var input_element;
    var input_str;

    return new Promise(function (resolve) {
                       // 创建输入框元素
                       input_element = document.createElement("INPUT");

                       // 输出提示符
                       print("? ");
                       // 设置输入框属性
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到指定元素
                       document.getElementById("output").appendChild(input_element);
                       // 输入框获取焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 输出输入值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入值
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 定义制表符函数，返回指定数量的空格字符串
function tab(space)
{
    var str = "";
    while (space-- > 0)
        str += " ";
    return str;
}

// 主程序，使用 async 函数定义
async function main()
{
    // 输出标题
    print(tab(31) + "MATH DICE\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("THIS PROGRAM GENERATES SUCCESSIVE PICTURES OF TWO DICE.\n");
    print("WHEN TWO DICE AND AN EQUAL SIGN FOLLOWED BY A QUESTION\n");
    print("MARK HAVE BEEN PRINTED, TYPE YOUR ANSWER AND THE RETURN KEY.\n"),
    print("TO CONCLUDE THE LESSON, TYPE ZERO AS YOUR ANSWER.\n");
    print("\n");
    print("\n");
    n = 0;
    // 循环生成骰子图片
    while (1) {
        n++;
        d = Math.floor(6 * Math.random() + 1);
        print(" ----- \n");
        if (d == 1)
            print("I     I\n");
        else if (d == 2 || d == 3)
            print("I *   I\n");
        else
            print("I * * I\n");
        if (d == 2 || d == 4)
            print("I     I\n");
        else if (d == 6)
            print("I * * I\n");
        else
            print("I  *  I\n");
        if (d == 1)
            print("I     I\n");
        else if (d == 2 || d == 3)
            print("I   * I\n");
        else
            print("I * * I\n");
        print(" ----- \n");
        print("\n");
        if (n != 2) {
            print("   +\n");
            print("\n");
            a = d;
            continue;
        }
        t = d + a;
        print("      =");
        // 等待用户输入
        t1 = parseInt(await input());
        if (t1 == 0)
            break;
        if (t1 != t) {
            print("NO, COUNT THE SPOTS AND GIVE ANOTHER ANSWER.\n");
            print("      =");
            t1 = parseInt(await input());
            if (t1 != t) {
                print("NO, THE ANSWER IS " + t + "\n");
            }
        }
        if (t1 == t) {
            print("RIGHT!\n");
        }
        print("\n");
        print("THE DICE ROLL AGAIN...\n");
        print("\n");
        n = 0;
    }
}

// 调用主程序
main();

```