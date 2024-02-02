# `basic-computer-games\86_Target\javascript\target.js`

```py
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
                       // 监听键盘按下事件
                       input_element.addEventListener("keydown", function (event) {
                                                      if (event.keyCode == 13) {
                                                      // 获取输入的字符串
                                                      input_str = input_element.value;
                                                      // 移除输入框
                                                      document.getElementById("output").removeChild(input_element);
                                                      // 打印输入的字符串
                                                      print(input_str);
                                                      print("\n");
                                                      // 解析 Promise 对象
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

// 主程序
async function main()
{
    // 打印标题
    print(tab(33) + "TARGET\n");
    // 打印副标题
    print(tab(15) + "CREATIVE COMPUTING  MORRISTOWN, NEW JERSEY\n");
    // 打印空行
    print("\n");
    print("\n");
    print("\n");
    // 初始化变量
    r = 0;  // 1 in original
    r1 = 57.296;
    p = Math.PI;
    // 打印提示信息
    print("YOU ARE THE WEAPONS OFFICER ON THE STARSHIP ENTERPRISE\n");
    print("AND THIS IS A TEST TO SEE HOW ACCURATE A SHOT YOU\n");
}
    # 打印提示信息，告知用户将进行三维范围内的操作
    print("ARE IN A THREE-DIMENSIONAL RANGE.  YOU WILL BE TOLD\n");
    # 打印提示信息，告知用户将会得到X和Z轴的弧度偏移、目标在三维直角坐标系中的位置等信息
    print("THE RADIAN OFFSET FOR THE X AND Z AXES, THE LOCATION\n");
    print("OF THE TARGET IN THREE DIMENSIONAL RECTANGULAR COORDINATES,\n");
    print("THE APPROXIMATE NUMBER OF DEGREES FROM THE X AND Z\n");
    print("AXES, AND THE APPROXIMATE DISTANCE TO THE TARGET.\n");
    # 打印提示信息，告知用户将进行射击直到目标被摧毁
    print("YOU WILL THEN PROCEEED TO SHOOT AT THE TARGET UNTIL IT IS\n");
    print("DESTROYED!\n");
    # 打印祝福信息
    print("\n");
    print("GOOD LUCK!!\n");
    print("\n");
    print("\n");
    }
# 结束当前的函数定义
}

# 调用名为main的函数
main();
```