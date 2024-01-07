# `basic-computer-games\24_Chemist\javascript\chemist.js`

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
                       // 设置输入框属性
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
                                                      // 解析输入的字符串
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

// 主程序
async function main()
{
    // 输出标题
    print(tab(33) + "CHEMIST\n");
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    print("\n");
    print("\n");
    print("\n");
    // 输出游戏介绍
    print("THE FICTITIOUS CHECMICAL KRYPTOCYANIC ACID CAN ONLY BE\n");
    print("DILUTED BY THE RATIO OF 7 PARTS WATER TO 3 PARTS ACID.\n");
    print("IF ANY OTHER RATIO IS ATTEMPTED, THE ACID BECOMES UNSTABLE\n");
    print("AND SOON EXPLODES.  GIVEN THE AMOUNT OF ACID, YOU MUST\n");
    print("DECIDE WHO MUCH WATER TO ADD FOR DILUTION.  IF YOU MISS\n");
    print("YOU FACE THE CONSEQUENCES.\n");
    t = 0;
    // 循环游戏
    while (1) {
        // 生成随机数作为酸的数量
        a = Math.floor(Math.random() * 50);
        // 根据酸的数量计算需要的水的数量
        w = 7 * a / 3;
        // 输出提示并等待用户输入
        print(a + " LITERS OF KRYPTOCYANIC ACID.  HOW MUCH WATER");
        r = parseFloat(await input());
        // 计算用户输入的水量与正确水量的差值
        d = Math.abs(w - r);
        // 判断用户是否成功
        if (d > w / 20) {
            // 输出失败信息
            print(" SIZZLE!  YOU HAVE JUST BEEN DESALINATED INTO A BLOB\n");
            print(" OF QUIVERING PROTOPLASM!\n");
            t++;
            if (t == 9)
                break;
            print(" HOWEVER, YOU MAY TRY AGAIN WITH ANOTHER LIFE.\n");
        } else {
            // 输出成功信息
            print(" GOOD JOB! YOU MAY BREATHE NOW, BUT DON'T INHALE THE FUMES!\n");
            print("\n");
        }
    }
    // 输出游戏结束信息
    print(" YOUR 9 LIVES ARE USED, BUT YOU WILL BE LONG REMEMBERED FOR\n");
    print(" YOUR CONTRIBUTIONS TO THE FIELD OF COMIC BOOK CHEMISTRY.\n");
}

// 调用主程序
main();

```