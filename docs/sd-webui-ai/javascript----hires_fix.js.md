# `stable-diffusion-webui\javascript\hires_fix.js`

```py
function onCalcResolutionHires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y) {
    // 定义内部函数，用于设置元素的激活状态
    function setInactive(elem, inactive) {
        // 根据 inactive 参数来切换元素的 inactive 类
        elem.classList.toggle('inactive', !!inactive);
    }

    // 获取指定 ID 的元素
    var hrUpscaleBy = gradioApp().getElementById('txt2img_hr_scale');
    var hrResizeX = gradioApp().getElementById('txt2img_hr_resize_x');
    var hrResizeY = gradioApp().getElementById('txt2img_hr_resize_y');

    // 根据条件设置元素的显示状态
    gradioApp().getElementById('txt2img_hires_fix_row2').style.display = opts.use_old_hires_fix_width_height ? "none" : "";

    // 根据条件设置元素的激活状态
    setInactive(hrUpscaleBy, opts.use_old_hires_fix_width_height || hr_resize_x > 0 || hr_resize_y > 0);
    setInactive(hrResizeX, opts.use_old_hires_fix_width_height || hr_resize_x == 0);
    setInactive(hrResizeY, opts.use_old_hires_fix_width_height || hr_resize_y == 0);

    // 返回包含参数值的数组
    return [enable, width, height, hr_scale, hr_resize_x, hr_resize_y];
}
```