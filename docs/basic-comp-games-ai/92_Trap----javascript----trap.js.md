# `basic-computer-games\92_Trap\javascript\trap.js`

```

// 定义一个打印函数，将字符串添加到输出元素中
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

                       // 打印提示符
                       print("? ");
                       // 设置输入框类型和长度
                       input_element.setAttribute("type", "text");
                       input_element.setAttribute("length", "50");
                       // 将输入框添加到输出元素中
                       document.getElementById("output").appendChild(input_element);
                       // 让输入框获得焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件，当按下回车键时，获取输入值并解析为字符串
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      input_str = input_element.value;
                                                      // 从输出元素中移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入值并传递给 resolve 函数
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

// 主控制部分，使用 async 函数定义
async function main()
{
    // 打印标题
    print(tab(34) + "TRAP\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 初始化变量 g 和 n
    g = 6;
    n = 100;
    // 打印游戏说明
    print("INSTRUCTIONS");
    // 等待用户输入，并将输入值赋给 str
    str = await input();
    // 根据用户输入的第一个字符判断是否开始游戏
    if (str.substr(0, 1) == "Y") {
        // 打印游戏规则和提示
        print("I AM THINKING OF A NUMBER BETWEEN 1 AND " + n + "\n");
        // ...
    }
    // 游戏循环
    while (1) {
        // 生成一个 1 到 n 之间的随机数
        x = Math.floor(n * Math.random()) + 1;
        // 循环进行猜数游戏
        for (q = 1; q <= g; q++) {
            // ...
        }
        // 打印提示，重新开始游戏
        print("\n");
        print("TRY AGAIN.\n");
        print("\n");
    }
}

// 调用主函数开始游戏
main();

```