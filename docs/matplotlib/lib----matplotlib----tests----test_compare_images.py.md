# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_compare_images.py`

```py
# 引入必要的模块和库：Path和shutil用于文件路径操作和复制，pytest用于测试框架，approx用于近似比较，compare_images和_image_directories来自matplotlib.testing用于图像比较。
from pathlib import Path  # 导入路径操作模块Path
import shutil  # 导入文件操作模块shutil

import pytest  # 导入pytest测试框架
from pytest import approx  # 导入approx用于近似比较

from matplotlib.testing.compare import compare_images  # 导入图像比较函数compare_images
from matplotlib.testing.decorators import _image_directories  # 导入图像目录处理函数_image_directories


# 图像比较算法的测试函数
@pytest.mark.parametrize(
    'im1, im2, tol, expect_rms',
    [
        # 对比一张图像及其有轻微变化的版本。
        # 预期这两张图在正常容差下相同，并且有较小的均方根误差（RMS）。
        ('basn3p02.png', 'basn3p02-minorchange.png', 10, None),
        # 现在测试无容差的情况。
        ('basn3p02.png', 'basn3p02-minorchange.png', 0, 6.50646),
        # 对比一张图像及其在X轴上偏移1像素的版本。
        ('basn3p02.png', 'basn3p02-1px-offset.png', 0, 90.15611),
        # 对比一张图像及其一半像素在X轴上偏移1像素的版本。
        ('basn3p02.png', 'basn3p02-half-1px-offset.png', 0, 63.75),
        # 对比一张图像及其完全被打乱顺序的版本。
        # 预期这两张图完全不同，均方根误差（RMS）非常大。
        # 注意：图像按照一种特定方式打乱，每个像素的每个颜色分量都随机放置在图像的某个位置。
        # 图像包含完全相同数量的每种颜色分量的像素，但位置完全不同。
        # 测试无容差，确保即使有非常小的均方根误差（RMS），也能检测到。
        ('basn3p02.png', 'basn3p02-scrambled.png', 0, 172.63582),
        # 对比一张图像及其稍微亮度更高的版本。
        # 这两张图都是单色，第二张图比第一张亮度高1个颜色值。
        # 预期这两张图在正常容差下相同，并且均方根误差（RMS）为1。
        ('all127.png', 'all128.png', 0, 1),
        # 现在测试反向对比。
        ('all128.png', 'all127.png', 0, 1),
    ])
def test_image_comparison_expect_rms(im1, im2, tol, expect_rms, tmp_path,
                                     monkeypatch):
    """
    比较两张图像，预期得到特定的均方根误差（RMS）。

    im1 和 im2 是相对于 baseline_dir 目录的文件名。

    tol 是传递给 compare_images 的容差。

    expect_rms 是期望的均方根误差值，如果为 None，则测试将在 compare_images 成功时通过。
    否则，如果 compare_images 失败并返回的均方根误差几乎等于此值，则测试将通过。
    """
    # 使用 monkeypatch 改变工作目录，以使用临时测试专用目录
    monkeypatch.chdir(tmp_path)
    # 获取基准目录和结果目录，映射为Path对象
    baseline_dir, result_dir = map(Path, _image_directories(lambda: "dummy"))
    # 将基准目录下的 im2 图像复制到结果目录下，这样 compare_images 将把差异写入结果目录，而不是源树
    result_im2 = result_dir / im1
    shutil.copyfile(baseline_dir / im2, result_im2)
    # 调用 compare_images 函数，比较基准图像和结果图像的差异
    results = compare_images(
        baseline_dir / im1, result_im2, tol=tol, in_decorator=True)

    # 如果期望的均方根误差为 None，则断言结果应该为 None
    if expect_rms is None:
        assert results is None
    else:
        # 否则，断言结果不为 None
        assert results is not None
        # 断言返回的结果中的均方根误差（rms）应该接近期望值，允许误差为 1e-4
        assert results['rms'] == approx(expect_rms, abs=1e-4)
```