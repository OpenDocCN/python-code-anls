# `basic-computer-games\91_Train\javascript\train.js`

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
                       // 输入框获取焦点
                       input_element.focus();
                       input_str = undefined;
                       // 监听键盘事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的值
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 输出输入的值
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析输入的值
                                                      resolve(input_str);
                                                      }
                                                      });
                       });
}

// 生成指定数量的空格字符串
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
    // 输出标题
    print(tab(33) + "TRAIN\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    print("TIME - SPEED DISTANCE EXERCISE\n");
    print("\n ");
    // 循环进行问题求解
    while (1) {
        // 生成随机的速度和时间
        c = Math.floor(25 * Math.random()) + 40;
        d = Math.floor(15 * Math.random()) + 5;
        t = Math.floor(19 * Math.random()) + 20;
        // 输出问题
        print(" A CAR TRAVELING " + c + " MPH CAN MAKE A CERTAIN TRIP IN\n");
        print(d + " HOURS LESS THAN A TRAIN TRAVELING AT " + t + " MPH.\n");
        print("HOW LONG DOES THE TRIP TAKE BY CAR");
        // 获取用户输入的值
        a = parseFloat(await input());
        // 计算并输出结果
        v = d * t / (c - t);
        e = Math.floor(Math.abs((v - a) * 100 / a) + 0.5);
        if (e > 5) {
            print("SORRY.  YOU WERE OFF BY " + e + " PERCENT.\n");
        } else {
            print("GOOD! ANSWER WITHIN " + e + " PERCENT.\n");
        }
        print("CORRECT ANSWER IS " + v + " HOURS.\n");
        print("\n");
        // 询问用户是否继续
        print("ANOTHER PROBLEM (YES OR NO)\n");
        str = await input();
        print("\n");
        // 判断用户输入，决定是否继续循环
        if (str.substr(0, 1) != "Y")
            break;
    }
}

// 调用主函数
main();

```