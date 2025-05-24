## 基于AI的贷款需求匹配与审批预测的应用工具
🔍 通过ZHIPU AI大模型匹配与用户的贷款申请相似类别和情绪的案例，通过建立逻辑回归模型对Income, Credit Score, Loan amount, DTI_Ratio, Employment Status进行分析，为用户预测贷款审批结果。

### 核心功能
* 语义搜索：输入贷款申请描述，类别，情绪 → 基于ZHIPU AI大模型的案例向量数据库，实时匹配历史相似需求。
* 智能预测：综合分析Income, Credit Score, Loan amount, DTI_Ratio, Employment Status等关键特征，利用案列训练的逻辑回归模型进行预测，理论上准确率可达90%以上。

### 安装说明
git clone 项目仓库
```bash
git clone https://github.com/JP3000/Loan-Or-Not.git
cd .venv/
```
安装依赖
```bash
pip install -r requirements.txt
```

### 使用说明
.env文件配置
```bash
ZHIPUAI_API_KEY=your_api_key
```

运行主程序
```bash
cd .venv/
streamlit run app.py
```

### 技术栈
* 申请文本匹配：langchain Zhipuai chroma 向量数据库
* 金融特征预测：sklearn 逻辑回归模型
* demo展示：streamlit

### 项目展示
<img src="./demoShow.png" alt="项目演示" width="400" />
