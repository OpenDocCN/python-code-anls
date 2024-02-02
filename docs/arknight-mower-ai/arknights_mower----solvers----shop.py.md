# `arknights-mower\arknights_mower\solvers\shop.py`

```py
# 导入未来版本的注解特性
from __future__ import annotations

# 导入相关模块
from ..data import shop_items
from ..ocr import ocr_rectify, ocrhandle
from ..utils import segment
from ..utils.device import Device
from ..utils.image import scope2slice
from ..utils.log import logger
from ..utils.recognize import RecognizeError, Scene
from ..utils.solver import BaseSolver, Recognizer

# 定义商店求解器类
class ShopSolver(BaseSolver):
    """
    自动使用信用点购买物资
    """

    # 初始化方法
    def __init__(self, device: Device = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)

    # 运行方法
    def run(self, priority: list[str] = None) -> None:
        """
        :param priority: list[str], 使用信用点购买东西的优先级, 若无指定则默认购买第一件可购买的物品
        """
        # 设置优先级
        self.priority = priority
        self.buying = None
        # 输出日志信息
        logger.info('Start: 商店')
        logger.info('购买期望：%s' % priority if priority else '无，购买到信用点用完为止')
        # 调用父类的运行方法
        super().run()
    # 定义状态转换方法，返回布尔值
    def transition(self) -> bool:
        # 如果当前场景是INDEX，则点击index_shop元素
        if self.scene() == Scene.INDEX:
            self.tap_element('index_shop')
        # 如果当前场景是SHOP_OTHERS，则点击shop_credit_2元素
        elif self.scene() == Scene.SHOP_OTHERS:
            self.tap_element('shop_credit_2')
        # 如果当前场景是SHOP_CREDIT
        elif self.scene() == Scene.SHOP_CREDIT:
            # 查找shop_collect元素，如果存在则点击，否则调用shop_credit方法
            collect = self.find('shop_collect')
            if collect is not None:
                self.tap(collect)
            else:
                return self.shop_credit()
        # 如果当前场景是SHOP_CREDIT_CONFIRM
        elif self.scene() == Scene.SHOP_CREDIT_CONFIRM:
            # 如果找不到shop_credit_not_enough元素，则点击shop_cart元素
            if self.find('shop_credit_not_enough') is None:
                self.tap_element('shop_cart')
            # 如果优先级列表长度大于0
            elif len(self.priority) > 0:
                # 从优先级列表中移除无法购买的物品
                self.priority.remove(self.buying) 
                logger.info('信用点不足，放弃购买%s，看看别的...' % self.buying)
                self.back()
            else:
                return True
        # 如果当前场景是SHOP_ASSIST，则返回上一级
        elif self.scene() == Scene.SHOP_ASSIST:
            self.back()
        # 如果当前场景是MATERIEL，则点击materiel_ico元素
        elif self.scene() == Scene.MATERIEL:
            self.tap_element('materiel_ico')
        # 如果当前场景是LOADING，则休眠3秒
        elif self.scene() == Scene.LOADING:
            self.sleep(3)
        # 如果当前场景是CONNECTING，则休眠3秒
        elif self.scene() == Scene.CONNECTING:
            self.sleep(3)
        # 如果有导航，则点击nav_shop元素
        elif self.get_navigation():
            self.tap_element('nav_shop')
        # 如果当前场景不是UNKNOWN，则返回到INDEX
        elif self.scene() != Scene.UNKNOWN:
            self.back_to_index()
        # 否则，抛出识别错误
        else:
            raise RecognizeError('Unknown scene')
    def shop_credit(self) -> bool:
        """ 购买物品逻辑 """
        # 从图像中提取信用商店的各个区域
        segments = segment.credit(self.recog.img)
        valid = []
        # 遍历各个区域，检查是否有可购买的物品
        for seg in segments:
            # 如果在当前区域找不到已售出的标志，则认为该区域内有可购买的物品
            if self.find('shop_sold', scope=seg) is None:
                # 重新定义区域范围，只取区域高度的四分之一
                scope = (seg[0], (seg[1][0], seg[0][1] + (seg[1][1]-seg[0][1])//4))
                # 从图像中切割出新的区域
                img = self.recog.img[scope2slice(scope)]
                # 使用 OCR 模型识别切割出的区域
                ocr = ocrhandle.predict(img)
                # 如果 OCR 识别结果为空，则抛出识别错误
                if len(ocr) == 0:
                    raise RecognizeError
                ocr = ocr[0]
                # 如果 OCR 识别结果不在可购买物品列表中，则进行纠正
                if ocr[1] not in shop_items:
                    ocr[1] = ocr_rectify(img, ocr, shop_items, '物品名称')
                # 将有效的区域和识别结果添加到列表中
                valid.append((seg, ocr[1]))
        # 记录商店内可购买的物品
        logger.info(f'商店内可购买的物品：{[x[1] for x in valid]}')
        # 如果没有可购买的物品，则返回 True
        if len(valid) == 0:
            return True
        # 获取购买优先级
        priority = self.priority
        # 如果有购买优先级，则按照优先级排序
        if priority is not None:
            valid.sort(
                key=lambda x: 9999 if x[1] not in priority else priority.index(x[1]))
            # 如果最优先的物品不在优先级列表中，则返回 True
            if valid[0][1] not in priority:
                return True
        # 记录实际购买顺序
        logger.info(f'实际购买顺序：{[x[1] for x in valid]}')
        # 设置要购买的物品，并点击购买
        self.buying = valid[0][1]
        self.tap(valid[0][0], interval=3)
```