import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler

# 加载模型
filename = 'loan_approval_model.pkl'
model_dir = 'model'  # 模型存储目录
model_path = os.path.join(model_dir, filename)  # 模型存储路径
try:
    loaded_model = pickle.load(open(model_path, 'rb'))
    print("模型已加载")
except FileNotFoundError:
    st.error(f"错误：找不到模型文件 {model_path}，请检查路径是否正确")
    exit()

# 加载 scaler
scaler_path = os.path.join(model_dir, 'scaler.pkl')
try:
    scaler, feature_names = pickle.load(open(scaler_path, 'rb'))
    print("Scaler已加载")
except FileNotFoundError:
    st.error(f"错误：找不到Scaler文件 {scaler_path}，请确保在训练模型时保存了Scaler")
    exit()

st.title("审批预测")

# 创建输入框
col1, col2, col3 = st.columns(3)
with col1:
    income = st.number_input("收入", value=60000)
with col2:
    credit_score = st.number_input("信用评分", value=700)
with col3:
    loan_amount = st.number_input("贷款金额", value=20000)

col4, col5 = st.columns(2)
with col4:
    dti_ratio = st.number_input("DTI 比率", value=37)
with col5:
    employment_status = st.selectbox("就业状态", [1, 0], format_func=lambda x: "已就业" if x == 1 else "未就业")

# 添加预测按钮
if st.button("预测"):
    # 准备新数据
    new_data = pd.DataFrame({
        'Income': [income],
        'Credit_Score': [credit_score],
        'Loan_Amount': [loan_amount],
        'DTI_Ratio': [dti_ratio],
        'Employment_Status': [employment_status]
    })

    # 确保新数据的列名与训练数据一致
    new_data = new_data[feature_names]

    # 缩放新数据
    new_data_scaled = scaler.transform(new_data)
    new_data_scaled = pd.DataFrame(new_data_scaled, columns=feature_names)

    # 进行预测
    new_predictions = loaded_model.predict(new_data_scaled)

    # 显示预测结果
    if new_predictions[0] == 1:
        st.success("预测结果：贷款已批准")
    else:
        st.error("预测结果：贷款已拒绝")