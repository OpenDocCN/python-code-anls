# `.\DB-GPT-src\web\pages\models\index.tsx`

```py
// 引入所需的模块和组件
import { apiInterceptors, getModelList } from '@/client/api';
import ModelCard from '@/components/model/model-card';
import ModelForm from '@/components/model/model-form';
import { IModelData } from '@/types/model';
import { Button, Modal } from 'antd';  // 导入按钮和模态框组件
import { useEffect, useState } from 'react';  // 导入 React 的副作用钩子和状态钩子
import { useTranslation } from 'react-i18next';  // 导入国际化翻译钩子

function Models() {
  const { t } = useTranslation();  // 获取翻译函数
  const [models, setModels] = useState<Array<IModelData>>([]);  // 使用状态钩子定义模型数据和设置函数
  const [isModalOpen, setIsModalOpen] = useState(false);  // 使用状态钩子定义模态框的开启状态

  // 异步函数，获取模型数据
  async function getModels() {
    const [, res] = await apiInterceptors(getModelList());
    setModels(res ?? []);  // 设置获取到的模型数据，若为空则设置为一个空数组
  }

  // 在组件加载时获取模型数据
  useEffect(() => {
    getModels();
  }, []);  // 空依赖数组确保只在组件加载时调用一次

  // 渲染模型管理界面
  return (
    <div className="p-4 md:p-6 overflow-y-auto">
      {/* 创建模型按钮 */}
      <Button
        className="mb-4"
        type="primary"
        onClick={() => {
          setIsModalOpen(true);
        }}
      >
        {t('create_model')}  {/* 按钮文字，使用国际化翻译 */}
      </Button>
      <div className="flex flex-wrap gap-2 md:gap-4">
        {/* 显示每个模型的卡片 */}
        {models.map((item) => (
          <ModelCard info={item} key={item.model_name} />
        ))}
      </div>
      {/* 创建模型的模态框 */}
      <Modal
        width={800}
        open={isModalOpen}
        title={t('create_model')}  // 模态框标题，使用国际化翻译
        onCancel={() => {
          setIsModalOpen(false);
        }}
        footer={null}
      >
        {/* 模型创建表单 */}
        <ModelForm
          onCancel={() => {
            setIsModalOpen(false);
          }}
          onSuccess={() => {
            setIsModalOpen(false);
            getModels();  // 成功后重新获取模型数据
          }}
        />
      </Modal>
    </div>
  );
}

export default Models;  // 导出模型管理组件
```