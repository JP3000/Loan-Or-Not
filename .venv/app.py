import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# 加载模型 (Load the model)
filename = 'loan_approval_model.pkl'
model_dir = 'model'  # 模型存储目录 (Model storage directory)
model_path = os.path.join(model_dir, filename)  # 模型存储路径 (Model storage path)
try:
    loaded_model = pickle.load(open(model_path, 'rb'))
    print("模型已加载 (Model loaded)")
except FileNotFoundError:
    st.error(f"错误：找不到模型文件 {model_path}，请检查路径是否正确 (Error: Model file not found at {model_path}, please check the path)")
    exit()

# 加载 scaler (Load the scaler)
scaler_path = os.path.join(model_dir, 'scaler.pkl')
try:
    scaler, feature_names = pickle.load(open(scaler_path, 'rb'))
    print("Scaler已加载 (Scaler loaded)")
except FileNotFoundError:
    st.error(f"错误：找不到Scaler文件 {scaler_path}，请确保在训练模型时保存了Scaler (Error: Scaler file not found at {scaler_path}, please ensure it was saved during model training)")
    exit()

# 加载文本匹配相关数据 (Load text matching related data)
df = pd.read_csv("data/loan_data_with_sentiment_analysis.csv")

embeddings_zp = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
)

db_load = Chroma(
    persist_directory="db/vectorstore_loan",
    embedding_function=embeddings_zp
)

# 检索相似申请 (Retrieve similar applications)
def retrieve_similar_applications(
        query: str,
        top_k: int = 5,
        predicted_category: str = None,
        employment_status: str = None,
        emotion: str = None
) -> pd.DataFrame:
    recs = db_load.similarity_search(query, k = top_k)

    text_list = []

    for i in range(0, len(recs)):
        text_list += [int(recs[i].page_content.strip('"').split()[0])]

    filtered_df = df[df["id"].isin(text_list)].copy()

    if predicted_category:
        filtered_df = filtered_df[filtered_df["predicted_category"] == predicted_category]

    if employment_status:
        filtered_df = filtered_df[filtered_df["Employment_Status"] == employment_status]

   # 情感过滤 (Emotion filtering)
    if emotion:
        if emotion == "anger":
            filtered_df.sort_values(by="anger", ascending=False, inplace=True)
        elif emotion == "disgust":
            filtered_df.sort_values(by="disgust", ascending=False, inplace=True)
        elif emotion == "fear":
            filtered_df.sort_values(by="fear", ascending=False, inplace=True)
        elif emotion == "joy":
            filtered_df.sort_values(by="joy", ascending=False, inplace=True)
        elif emotion == "sadness":
            filtered_df.sort_values(by="sadness", ascending=False, inplace=True)
        elif emotion == "surprise":
            filtered_df.sort_values(by="surprise", ascending=False, inplace=True)
        elif emotion == "neutral":
            filtered_df.sort_values(by="neutral", ascending=False, inplace=True)

    # 移除不需要显示的列 (Remove unnecessary columns)
    columns_to_hide = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral', 'id', 'tagged_text']
    columns_to_display = [col for col in filtered_df.columns if col not in columns_to_hide]
    filtered_df = filtered_df[columns_to_display]

    return filtered_df

st.title("Loan Approval Prediction")

# 创建输入框 (Create input fields)
col1, col2, col3 = st.columns(3)
with col1:
    income = st.number_input("Income", value=60000)
with col2:
    credit_score = st.number_input("Credit Score", value=700)
with col3:
    loan_amount = st.number_input("Loan Amount", value=20000)

col4, col5 = st.columns(2)
with col4:
    dti_ratio = st.number_input("DTI Ratio", value=0.3)
with col5:
    employment_status = st.selectbox("Employment Status", [1, 0], format_func=lambda x: "Employed" if x == 1 else "Unemployed")

# 添加预测按钮 (Add prediction button)
if st.button("Predict"):
    # 准备新数据 (Prepare new data)
    new_data = pd.DataFrame({
        'Income': [income],
        'Credit_Score': [credit_score],
        'Loan_Amount': [loan_amount],
        'DTI_Ratio': [dti_ratio],
        'Employment_Status': [employment_status]
    })

    # 确保新数据具有与训练数据相同的列名 (Ensure new data has the same column names as the training data)
    new_data = new_data[feature_names]

    # 缩放新数据 (Scale the new data)
    new_data_scaled = scaler.transform(new_data)
    new_data_scaled = pd.DataFrame(new_data_scaled, columns=feature_names)

    # 进行预测 (Make a prediction)
    new_predictions = loaded_model.predict(new_data_scaled)

    # 显示预测结果 (Display the prediction result)
    if new_predictions[0] == 1:
        st.success("Prediction: Loan Approved")
    else:
        st.error("Prediction: Loan Rejected")

st.header("Text Matching")

# 创建文本输入框和匹配选项 (Create input fields and matching options)
query = st.text_input("Enter application text", "I want buy a car", max_chars=100)
col6, col7, col8, col9 = st.columns(4)
with col6:
    predicted_categories = df["predicted_category"].unique().tolist()
    predicted_category = st.selectbox("Category", options=[None] + predicted_categories)
with col7:
    employment_statuses = df["Employment_Status"].unique().tolist()
    employment_status = st.selectbox("Employment Status", options=[None] + employment_statuses)
with col8:
    emotion_choices = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral', None]
    emotion = st.selectbox("Emotion", options=emotion_choices)
with col9:
    top_k = st.slider("Number of Matches", min_value=1, max_value=20, value=5)

# 添加匹配按钮 (Add matching button)
if st.button("Match"):
    # 获取匹配结果 (Get matching results)
    similar_applications = retrieve_similar_applications(query, top_k, predicted_category, employment_status, emotion)

    # 显示匹配结果 (Display matching results)
    st.subheader("Matching Results")
    st.dataframe(similar_applications)