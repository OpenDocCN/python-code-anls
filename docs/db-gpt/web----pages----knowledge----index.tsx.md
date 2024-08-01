# `.\DB-GPT-src\web\pages\knowledge\index.tsx`

```py
import React, { useState, useEffect } from 'react';
// 导入 Ant Design 提供的图标和组件
import { PlusOutlined } from '@ant-design/icons';
import { Button, Modal, Steps } from 'antd';
// 导入自定义的 SpaceCard 组件
import SpaceCard from '@/components/knowledge/space-card';
// 导入类型定义
import { File, ISpace, IStorage, SpaceConfig, StepChangeParams } from '@/types/knowledge';
// 导入 API 请求函数和拦截器
import { apiInterceptors, getSpaceConfig, getSpaceList } from '@/client/api';
// 导入多语言翻译函数
import { useTranslation } from 'react-i18next';
// 导入文档上传表单组件
import DocUploadForm from '@/components/knowledge/doc-upload-form';
// 导入空间表单组件
import SpaceForm from '@/components/knowledge/space-form';
// 导入文档类型选择表单组件
import DocTypeForm from '@/components/knowledge/doc-type-form';
// 导入分割组件
import Segmentation from '@/components/knowledge/segmentation';
// 导入 classNames 库
import classNames from 'classnames';

const Knowledge = () => {
  // 状态管理器：存储空间列表
  const [spaceList, setSpaceList] = useState<Array<ISpace> | null>([]);
  // 状态管理器：控制添加模态框显示
  const [isAddShow, setIsAddShow] = useState<boolean>(false);
  // 状态管理器：当前活动步骤
  const [activeStep, setActiveStep] = useState<number>(0);
  // 状态管理器：当前空间名称
  const [spaceName, setSpaceName] = useState<string>('');
  // 状态管理器：当前文件列表
  const [files, setFiles] = useState<Array<File>>([]);
  // 状态管理器：当前文档类型
  const [docType, setDocType] = useState<string>('');
  // 状态管理器：当前空间配置信息
  const [spaceConfig, setSpaceConfig] = useState<IStorage | null>(null);

  // 多语言翻译函数
  const { t } = useTranslation();

  // 添加知识的步骤数组
  const addKnowledgeSteps = [
    { title: t('Knowledge_Space_Config') },
    { title: t('Choose_a_Datasource_type') },
    { title: t('Upload') },
    { title: t('Segmentation') },
  ];

  // 异步函数：获取空间列表
  async function getSpaces() {
    const [_, data] = await apiInterceptors(getSpaceList());
    setSpaceList(data);  // 更新空间列表状态
  }

  // 异步函数：获取空间配置信息
  async function getSpaceConfigs() {
    const [_, data] = await apiInterceptors(getSpaceConfig());
    if (!data) return null;
    setSpaceConfig(data.storage);  // 更新空间配置状态
  }

  // 组件挂载时执行：获取空间列表和空间配置信息
  useEffect(() => {
    getSpaces();
    getSpaceConfigs();
  }, []);

  // 处理步骤改变的函数
  const handleStepChange = ({ label, spaceName, docType = '', files, pace = 1 }: StepChangeParams) => {
    if (label === 'finish') {
      setIsAddShow(false);  // 关闭添加模态框
      getSpaces();  // 获取最新的空间列表
      setSpaceName('');  // 清空当前空间名称
      setDocType('');  // 清空当前文档类型
      getSpaces();  // 再次获取最新的空间列表
    } else if (label === 'forward') {
      activeStep === 0 && getSpaces();  // 如果是第一步，获取空间列表
      setActiveStep((step) => step + pace);  // 增加步骤
    } else {
      setActiveStep((step) => step - pace);  // 减少步骤
    }
    files && setFiles(files);  // 更新文件列表状态
    spaceName && setSpaceName(spaceName);  // 更新空间名称状态
    docType && setDocType(docType);  // 更新文档类型状态
  };

  // 添加文档的函数
  function onAddDoc(spaceName: string) {
    const space = spaceList?.find((item) => item?.name === spaceName);
    setSpaceName(spaceName);  // 设置当前空间名称
    setActiveStep(space?.domain_type === 'FinancialReport' ? 2 : 1);  // 根据空间类型设置当前步骤
    setIsAddShow(true);  // 打开添加模态框
    if (space?.domain_type === 'FinancialReport') {
      setDocType('DOCUMENT');  // 如果是财务报告空间，设置文档类型
    }
  }

  return (
    <div className="bg-[#FAFAFA] dark:bg-transparent w-full h-full">
      {/* 背景样式设置为灰色，dark 模式下为透明，占据整个宽度和高度 */}
      <div className="page-body p-4 md:p-6 h-full overflow-auto">
        {/* 页面主体部分设置内边距，占据整个高度并允许纵向滚动 */}
        <Button
          type="primary"
          className="flex items-center"
          icon={<PlusOutlined />}
          onClick={() => {
            setIsAddShow(true);
          }}
        >
          Create
        </Button>
        {/* 创建按钮，带有加号图标，点击时设置显示添加模态框 */}
        <div className="flex flex-wrap mt-4 gap-2 md:gap-4">
          {/* 显示空间列表，每个空间用 SpaceCard 组件展示 */}
          {spaceList?.map((space: ISpace) => (
            <SpaceCard key={space.id} space={space} onAddDoc={onAddDoc} getSpaces={getSpaces} />
          ))}
        </div>
      </div>
      {/* 添加知识模态框 */}
      <Modal
        title="Add Knowledge"
        centered
        open={isAddShow}
        destroyOnClose={true}
        onCancel={() => {
          setIsAddShow(false);
        }}
        width={1000}
        afterClose={() => {
          setActiveStep(0);
          getSpaces();
        }}
        footer={null}
      >
        {/* 显示添加知识的步骤 */}
        <Steps current={activeStep} items={addKnowledgeSteps} />
        {/* 第一步：显示空间表单 */}
        {activeStep === 0 && <SpaceForm handleStepChange={handleStepChange} spaceConfig={spaceConfig} />}
        {/* 第二步：显示文档类型表单 */}
        {activeStep === 1 && <DocTypeForm handleStepChange={handleStepChange} />}
        {/* 第三步：显示文档上传表单 */}
        <DocUploadForm
          className={classNames({ hidden: activeStep !== 2 })}
          spaceName={spaceName}
          docType={docType}
          handleStepChange={handleStepChange}
        />
        {/* 第四步：显示分段处理组件 */}
        {activeStep === 3 && <Segmentation spaceName={spaceName} docType={docType} uploadFiles={files} handleStepChange={handleStepChange} />}
      </Modal>
    </div>
};

export default Knowledge;
```