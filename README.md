## AI-Powered Loan Demand Matching and Approval Prediction Application Tool

🔍 Matches cases with similar categories and sentiments to the user's loan application through the ZHIPU AI large model. Analyzes Income, Credit Score, Loan Amount, DTI_Ratio, and Employment Status by building a logistic regression model to predict loan approval results for users.

### Core Features

*   **Semantic Search:** Input loan application description, category, sentiment → Real-time matching of historical similar demands based on the ZHIPU AI large model case vector database.
*   **Intelligent Prediction:** Comprehensive analysis of key features such as Income, Credit Score, Loan Amount, DTI_Ratio, and Employment Status. Utilizes a logistic regression model trained on cases for prediction, with a theoretical accuracy of over 90%.

### Installation Instructions

Clone the project repository:

```bash
git clone https://github.com/JP3000/Loan-Or-Not.git
cd .venv/
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage Instructions
Configure the .env file:
```bash
ZHIPUAI_API_KEY=your_api_key
```

Run the main program:
```bash
cd .venv/
streamlit run app.py
```

### Technology Stack
* Application Text Matching: langchain, ZhipuAI, chroma vector database
* Financial Feature Prediction: sklearn logistic regression model
* Demo Presentation: streamlit

### Project Demo
<img src="./demoShow.png" alt="Project Demo" width="400" />
