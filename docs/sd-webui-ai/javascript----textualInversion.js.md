# `stable-diffusion-webui\javascript\textualInversion.js`

```py
# 开始文本反转训练的函数
function start_training_textual_inversion() {
    # 清空错误信息
    gradioApp().querySelector('#ti_error').innerHTML = '';

    # 生成随机的ID
    var id = randomId();
    # 请求进度信息，更新输出和画廊
    requestProgress(id, gradioApp().getElementById('ti_output'), gradioApp().getElementById('ti_gallery'), function() {}, function(progress) {
        # 更新进度信息
        gradioApp().getElementById('ti_progress').innerHTML = progress.textinfo;
    });

    # 将参数转换为数组
    var res = Array.from(arguments);

    # 将ID存储在数组的第一个位置
    res[0] = id;

    # 返回结果数组
    return res;
}
```