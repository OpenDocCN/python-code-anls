# `.\pandas-ta\tests\test_ext_indicator_volume.py`

```
# 导入所需的模块
from .config import sample_data
from .context import pandas_ta
# 导入测试用例相关的模块
from unittest import TestCase
# 从 pandas 模块中导入 DataFrame 类
from pandas import DataFrame

# 定义测试类 TestVolumeExtension，继承自 TestCase 类
class TestVolumeExtension(TestCase):
    # 在整个测试类执行之前调用，设置测试所需的数据
    @classmethod
    def setUpClass(cls):
        # 获取示例数据
        cls.data = sample_data
        # 获取示例数据中的 "open" 列
        cls.open = cls.data["open"]

    # 在整个测试类执行之后调用，清理测试所用到的数据
    @classmethod
    def tearDownClass(cls):
        # 删除示例数据
        del cls.data
        # 删除示例数据中的 "open" 列
        del cls.open

    # 在每个测试方法执行之前调用，可用于设置测试前的准备工作
    def setUp(self): pass
    
    # 在每个测试方法执行之后调用，可用于清理测试后的工作
    def tearDown(self): pass

    # 测试 AD 指标扩展功能
    def test_ad_ext(self):
        # 调用 ad 方法计算 AD 指标并将其追加到数据帧中
        self.data.ta.ad(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据帧的最后一列为 "AD"
        self.assertEqual(self.data.columns[-1], "AD")

    # 测试 AD 指标中 open 参数扩展功能
    def test_ad_open_ext(self):
        # 调用 ad 方法计算 AD 指标，传入 open_ 参数，并将其追加到数据帧中
        self.data.ta.ad(open_=self.open, append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据帧的最后一列为 "ADo"
        self.assertEqual(self.data.columns[-1], "ADo")

    # 测试 ADOSC 指标扩展功能
    def test_adosc_ext(self):
        # 调用 adosc 方法计算 ADOSC 指标并将其追加到数据帧中
        self.data.ta.adosc(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据帧的最后一列为 "ADOSC_3_10"
        self.assertEqual(self.data.columns[-1], "ADOSC_3_10")

    # 测试 AOBV 指标扩展功能
    def test_aobv_ext(self):
        # 调用 aobv 方法计算 AOBV 指标并将其追加到数据帧中
        self.data.ta.aobv(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据帧的最后七列为给定列表中的列名
        self.assertEqual(
            list(self.data.columns[-7:]),
            ["OBV", "OBV_min_2", "OBV_max_2", "OBVe_4", "OBVe_12", "AOBV_LR_2", "AOBV_SR_2"],
        )
        # 移除 "OBV" 列，以免干扰 test_obv_ext() 测试方法
        self.data.drop("OBV", axis=1, inplace=True)

    # 测试 CMF 指标扩展功能
    def test_cmf_ext(self):
        # 调用 cmf 方法计算 CMF 指标并将其追加到数据帧中
        self.data.ta.cmf(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据帧的最后一列为 "CMF_20"
        self.assertEqual(self.data.columns[-1], "CMF_20")

    # 测试 EFI 指标扩展功能
    def test_efi_ext(self):
        # 调用 efi 方法计算 EFI 指标并将其追加到数据帧中
        self.data.ta.efi(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据帧的最后一列为 "EFI_13"
        self.assertEqual(self.data.columns[-1], "EFI_13")

    # 测试 EOM 指标扩展功能
    def test_eom_ext(self):
        # 调用 eom 方法计算 EOM 指标并将其追加到数据帧中
        self.data.ta.eom(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据帧的最后一列为 "EOM_14_100000000"
        self.assertEqual(self.data.columns[-1], "EOM_14_100000000")

    # 测试 KVO 指标扩展功能
    def test_kvo_ext(self):
        # 调用 kvo 方法计算 KVO 指标并将其追加到数据帧中
        self.data.ta.kvo(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据帧的倒数第二和最后一列为给定列表中的列名
        self.assertEqual(list(self.data.columns[-2:]), ["KVO_34_55_13", "KVOs_34_55_13"])

    # 测试 MFI 指标扩展功能
    def test_mfi_ext(self):
        # 调用 mfi 方法计算 MFI 指标并将其追加到数据帧中
        self.data.ta.mfi(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据帧的最后一列为 "MFI_14"
        self.assertEqual(self.data.columns[-1], "MFI_14")

    # 测试 NVI 指标扩展功能
    def test_nvi_ext(self):
        # 调用 nvi 方法计算 NVI 指标并将其追加到数据帧中
        self.data.ta.nvi(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言数据帧的最后一
    # 测试 pvol 方法是否正常运行并将结果附加到数据框中
    def test_pvol_ext(self):
        self.data.ta.pvol(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为 "PVOL"
    
    # 测试 pvr 方法是否正常运行并将结果附加到数据框中
    def test_pvr_ext(self):
        self.data.ta.pvr(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为 "PVR"
    
    # 测试 pvt 方法是否正常运行并将结果附加到数据框中
    def test_pvt_ext(self):
        self.data.ta.pvt(append=True)
        # 断言数据类型为 DataFrame
        self.assertIsInstance(self.data, DataFrame)
        # 断言最后一列的列名为 "PVT"
    
    # 测试 vp 方法是否正常运行并返回 DataFrame
    def test_vp_ext(self):
        result = self.data.ta.vp()
        # 断言返回结果的数据类型为 DataFrame
        self.assertIsInstance(result, DataFrame)
        # 断言返回结果的名称为 "VP_10"
```