# `.\DB-GPT-src\web\components\knowledge\doc-type-form.tsx`

```py
import { StepChangeParams } from '@/types/knowledge';
import { Card } from 'antd';
import { useTranslation } from 'react-i18next';
import DocIcon from './doc-icon';

type IProps = {
  handleStepChange: (params: StepChangeParams) => void;
};

// 定义并导出组件 DocTypeForm，接收 handleStepChange 函数作为参数
export default function DocTypeForm(props: IProps) {
  // 使用 react-i18next 中的 useTranslation hook 获取 t 函数
  const { t } = useTranslation();
  // 从 props 中解构出 handleStepChange 函数
  const { handleStepChange } = props;

  // 定义文档类型列表
  const docTypeList = [
    {
      type: 'TEXT',
      title: t('Text'),             // 使用 t 函数翻译文本 'Text'
      subTitle: t('Fill your raw text'),  // 使用 t 函数翻译文本 'Fill your raw text'
      iconType: 'TEXT',
    },
    {
      type: 'URL',
      title: t('URL'),              // 使用 t 函数翻译文本 'URL'
      subTitle: t('Fetch_the_content_of_a_URL'),  // 使用 t 函数翻译文本 'Fetch the content of a URL'
      iconType: 'WEBPAGE',
    },
    {
      type: 'DOCUMENT',
      title: t('Document'),         // 使用 t 函数翻译文本 'Document'
      subTitle: t('Upload_a_document'),  // 使用 t 函数翻译文本 'Upload a document'
      iconType: 'DOCUMENT',
    },
  ];

  // 返回 JSX 元素，展示文档类型列表
  return (
    <>
      {docTypeList.map((type, index) => (
        <Card
          key={index}                        // 设置 Card 组件的唯一 key 属性
          className="mt-4 mb-4 cursor-pointer"  // 添加 CSS 类名，设定样式
          onClick={() => {
            // 点击 Card 组件时调用 handleStepChange 函数，传递 label 和 docType 参数
            handleStepChange({ label: 'forward', docType: type.type });
          }}
        >
          <div className="font-semibold">
            <DocIcon type={type.iconType} />  // 渲染 DocIcon 组件，传递 iconType 属性
            {type.title}                     // 显示文档类型标题
          </div>
          <div>{type.subTitle}</div>         // 显示文档类型副标题
        </Card>
      ))}
    </>
  );
}
```