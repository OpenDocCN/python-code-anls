# `D:\src\scipysrc\matplotlib\lib\matplotlib\_animation_data.py`

```
// JavaScript模板用于HTMLWriter

// 包含外部样式表的链接，加载FontAwesome图标库
JS_INCLUDE = """
<link rel="stylesheet"
href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">
<script language="javascript">
  // 检测浏览器是否为Internet Explorer
  function isInternetExplorer() {
    ua = navigator.userAgent;
    /* MSIE用于检测旧版浏览器，Trident用于检测较新版IE内核浏览器 */
    return ua.indexOf("MSIE ") > -1 || ua.indexOf("Trident/") > -1;
  }

  /* 定义Animation类 */
  function Animation(frames, img_id, slider_id, interval, loop_select_id){
    this.img_id = img_id;
    this.slider_id = slider_id;
    this.loop_select_id = loop_select_id;
    this.interval = interval;
    this.current_frame = 0;
    this.direction = 0;
    this.timer = null;
    this.frames = new Array(frames.length);

    // 预加载动画帧图片
    for (var i=0; i<frames.length; i++)
    {
     this.frames[i] = new Image();
     this.frames[i].src = frames[i];
    }
    // 获取滑块元素，并设置最大值为帧数减一
    var slider = document.getElementById(this.slider_id);
    slider.max = this.frames.length - 1;
    // 如果是Internet Explorer，修改事件处理，以兼容不符合W3C规范的旧版本IE
    if (isInternetExplorer()) {
        // 从oninput改为onchange，因为IE <= 11不支持oninput，而onchange的行为类似于oninput。
        slider.setAttribute('onchange', slider.getAttribute('oninput'));
        slider.setAttribute('oninput', null);
    }
    // 设置初始帧
    this.set_frame(this.current_frame);
  }

  // 获取循环状态的方法
  Animation.prototype.get_loop_state = function(){
    var button_group = document[this.loop_select_id].state;
    for (var i = 0; i < button_group.length; i++) {
        var button = button_group[i];
        if (button.checked) {
            return button.value;
        }
    }
    return undefined;
  }

  // 设置当前帧的方法
  Animation.prototype.set_frame = function(frame){
    this.current_frame = frame;
    // 更新显示当前帧的图片
    document.getElementById(this.img_id).src =
            this.frames[this.current_frame].src;
    // 更新滑块的值为当前帧
    document.getElementById(this.slider_id).value = this.current_frame;
  }

  // 显示下一帧的方法
  Animation.prototype.next_frame = function()
  {
    this.set_frame(Math.min(this.frames.length - 1, this.current_frame + 1));
  }

  // 显示上一帧的方法
  Animation.prototype.previous_frame = function()
  {
    this.set_frame(Math.max(0, this.current_frame - 1));
  }

  // 显示第一帧的方法
  Animation.prototype.first_frame = function()
  {
    this.set_frame(0);
  }

  // 显示最后一帧的方法
  Animation.prototype.last_frame = function()
  {
    this.set_frame(this.frames.length - 1);
  }

  // 减慢动画速度的方法
  Animation.prototype.slower = function()
  {
    this.interval /= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  // 加快动画速度的方法
  Animation.prototype.faster = function()
  {
    this.interval *= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  // 前进一步的动画方法
  Animation.prototype.anim_step_forward = function()
  {
    this.current_frame += 1;
    if(this.current_frame < this.frames.length){
      this.set_frame(this.current_frame);
  }else{
    // 获取循环状态
    var loop_state = this.get_loop_state();
    // 如果是循环状态，跳转到第一帧
    if(loop_state == "loop"){
      this.first_frame();
    }else if(loop_state == "reflect"){
      // 如果是反射状态，跳转到最后一帧并反向播放动画
      this.last_frame();
      this.reverse_animation();
    }else{
      // 否则暂停动画，并跳转到最后一帧
      this.pause_animation();
      this.last_frame();
    }
  }

  // 动画倒退一步
  Animation.prototype.anim_step_reverse = function()
  {
    // 当前帧数减一
    this.current_frame -= 1;
    // 如果当前帧数仍大于等于0
    if(this.current_frame >= 0){
      // 设置动画到当前帧
      this.set_frame(this.current_frame);
    }else{
      // 获取循环状态
      var loop_state = this.get_loop_state();
      // 如果是循环状态，跳转到最后一帧
      if(loop_state == "loop"){
        this.last_frame();
      }else if(loop_state == "reflect"){
        // 如果是反射状态，跳转到第一帧并播放动画
        this.first_frame();
        this.play_animation();
      }else{
        // 否则暂停动画，并跳转到第一帧
        this.pause_animation();
        this.first_frame();
      }
    }
  }

  // 暂停动画
  Animation.prototype.pause_animation = function()
  {
    // 设置动画方向为静止
    this.direction = 0;
    // 清除定时器，停止动画播放
    if (this.timer){
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  // 播放动画
  Animation.prototype.play_animation = function()
  {
    // 先暂停当前动画
    this.pause_animation();
    // 设置动画方向为正向
    this.direction = 1;
    var t = this;
    // 如果定时器不存在，创建并启动动画播放定时器
    if (!this.timer) this.timer = setInterval(function() {
        t.anim_step_forward();
    }, this.interval);
  }

  // 反向播放动画
  Animation.prototype.reverse_animation = function()
  {
    // 先暂停当前动画
    this.pause_animation();
    // 设置动画方向为反向
    this.direction = -1;
    var t = this;
    // 如果定时器不存在，创建并启动反向播放动画定时器
    if (!this.timer) this.timer = setInterval(function() {
        t.anim_step_reverse();
    }, this.interval);
  }
# HTML模板字符串，用于HTMLWriter类
DISPLAY_TEMPLATE = """
<div class="animation">
  <img id="_anim_img{id}">
  <div class="anim-controls">
    <input id="_anim_slider{id}" type="range" class="anim-slider"
           name="points" min="0" max="1" step="1" value="0"
           oninput="anim{id}.set_frame(parseInt(this.value));">
    <div class="anim-buttons">
      <button title="减慢速度" aria-label="减慢速度" onclick="anim{id}.slower()">
          <i class="fa fa-minus"></i></button>
      <button title="第一帧" aria-label="第一帧" onclick="anim{id}.first_frame()">
        <i class="fa fa-fast-backward"></i></button>
      <button title="上一帧" aria-label="上一帧" onclick="anim{id}.previous_frame()">
          <i class="fa fa-step-backward"></i></button>
      <button title="倒放动画" aria-label="倒放动画" onclick="anim{id}.reverse_animation()">
          <i class="fa fa-play fa-flip-horizontal"></i></button>
      <button title="暂停" aria-label="暂停" onclick="anim{id}.pause_animation()">
          <i class="fa fa-pause"></i></button>
      <button title="播放" aria-label="播放" onclick="anim{id}.play_animation()">
          <i class="fa fa-play"></i></button>
      <button title="下一帧" aria-label="下一帧" onclick="anim{id}.next_frame()">
          <i class="fa fa-step-forward"></i></button>
      <button title="最后一帧" aria-label="最后一帧" onclick="anim{id}.last_frame()">
          <i class="fa fa-fast-forward"></i></button>
      <button title="加快速度" aria-label="加快速度" onclick="anim{id}.faster()">
          <i class="fa fa-plus"></i></button>
    </div>
    <form title="重复模式" aria-label="重复模式" action="#n" name="_anim_loop_select{id}"
          class="anim-state">
      <input type="radio" name="state" value="once" id="_anim_radio1_{id}"
             {once_checked}>
      <label for="_anim_radio1_{id}">一次</label>
      <input type="radio" name="state" value="loop" id="_anim_radio2_{id}"
             {loop_checked}>
      <label for="_anim_radio2_{id}">循环</label>
      <input type="radio" name="state" value="reflect" id="_anim_radio3_{id}"
             {reflect_checked}>
      <label for="_anim_radio3_{id}">反射</label>
    </form>
  </div>
</div>
"""

# JavaScript语言脚本，用于实例化Animation类并设置相关控件
SCRIPT_INCLUDE = """
<script language="javascript">
  /* 实例化Animation类。 */
  /* 给定的ID应该与上述模板中使用的ID匹配。 */
  (function() {{
    var img_id = "_anim_img{id}";
"""
    // 生成动画滑块的唯一标识符
    var slider_id = "_anim_slider{id}";
    // 生成动画循环选择器的唯一标识符
    var loop_select_id = "_anim_loop_select{id}";
    // 创建一个数组来存储帧的数据，数组长度为 {Nframes}
    var frames = new Array({Nframes});
    {fill_frames}

    /* 设置一个延时操作确保所有上述元素在对象初始化之前都已创建 */

    setTimeout(function() {{
        // 创建动画对象并初始化，使用帧数据、图像 ID、滑块 ID、循环选择器 ID 和间隔参数
        anim{id} = new Animation(frames, img_id, slider_id, {interval},
                                 loop_select_id);
    }}, 0);
</script>
"""  # noqa: E501

# 闭合 JavaScript 脚本标签和多行字符串的引号，忽略 PEP 8 编码最大行长度检查。


INCLUDED_FRAMES = """
  for (var i=0; i<{Nframes}; i++){{
    frames[i] = "{frame_dir}/frame" + ("0000000" + i).slice(-7) +
                ".{frame_format}";
  }}
"""

# 定义包含帧的 JavaScript 代码字符串，通过循环生成帧的路径，并使用变量插入帧目录和格式。
```