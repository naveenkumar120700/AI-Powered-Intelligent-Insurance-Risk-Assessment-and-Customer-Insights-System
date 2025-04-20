import streamlit as st
from streamlit_option_menu import option_menu
# Set page config
st.set_page_config(page_title="Insurance ML Suite", page_icon="💼", layout="wide")

# Sidebar menu with emojis using streamlit_option_menu
with st.sidebar:
    app_mode = option_menu(
        menu_title="🏠 Navigation",
        options=[
            "🏡 Home",
            "📈 Risk Score Prediction",
            "💰 Claim Amount Prediction",
            "👥 Customer Segmentation",
            "🕵️ Fraud Detection",
            "😊 Sentiment Analysis",
            "🌐 Translation",
            "📝 Summarization",
            "🤖 Chatbot"
        ],
        menu_icon="cast",  # Optional top icon for the sidebar menu title
        default_index=0,
        orientation="vertical"
    )


# Home Page Summary
if app_mode == "🏡 Home":
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap');

        .custom-title {
            font-family: 'Great Vibes', cursive;
            font-size: 70px;
            color: #ffffff;
            text-align: center;
            text-shadow: 2px 2px 4px #000000;
            padding: 10px;
            background: linear-gradient(90deg,rgb(55, 160, 64) ,rgb(82, 205, 239));
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .disease-name {
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            display: block;
            margin-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


    # Display Title with custom spacing
    st.markdown("<h1 class='custom-title'>Insurance AI Project</h1>", unsafe_allow_html=True)

    st.markdown(
        """
        <hr style="border: 1px solid black;">
        """,
        unsafe_allow_html=True
    )
    import streamlit as st
    from PIL import Image

    # Title section
    st.markdown("""
    <div style='font-size:18px; line-height:1.6'>
    Welcome to the <strong>AI-Powered Insurance Intelligence Suite</strong> — your all-in-one hub for smart, efficient, and data-driven insurance workflows. <br>
    From predicting risks to detecting fraud, understanding customers, and enhancing communication, this suite brings together the power of machine learning to revolutionize the way insurance works.
    </div>
    <br><hr><br>
    """, unsafe_allow_html=True)

    
    # List of features with image paths
    features = [
        {"title": "Risk Score Prediction", "desc": "Instantly evaluate risk levels using customer data.", "img": "D:/Insurance_AI_project/assets/risk.jpeg"},
        {"title": "Claim Amount Estimation", "desc": "Accurately forecast expected claim payouts.", "img": "D:/Insurance_AI_project/assets/claim.jpeg"},
        {"title": "Customer Segmentation", "desc": "Discover meaningful customer clusters based on behavior or demographics.", "img": "D:/Insurance_AI_project/assets/customer segment.jpeg"},
        {"title": "Fraud Detection", "desc": "Catch suspicious claims before they slip through the cracks.", "img": "D:/Insurance_AI_project/assets/fraud.jpeg"},
        {"title": "Sentiment Analysis", "desc": "Decode the emotion behind customer feedback.", "img": "D:/Insurance_AI_project/assets/sentiment.jpeg"},
        {"title": "Language Translation", "desc": "Seamlessly translate content across global languages.", "img": "D:/Insurance_AI_project/assets/translator.jpeg"},
        {"title": "Text Summarization", "desc": "Shrink long documents into bite-sized insights.", "img": "D:/Insurance_AI_project/assets/summary.jpeg"},
        {"title": "AI Chatbot", "desc": "Get instant answers from your smart virtual insurance assistant.", "img": "D:/Insurance_AI_project/assets/chatbot.jpeg"},
    ]

    # Custom CSS for styling
    st.markdown("""
    <style>
        .custom-title {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        .feature-title {
            text-align: center;
            font-size: 20px;
            font-weight: 600;
            margin-top: 10px;
            margin-bottom: 5px;
        }
        .feature-desc {
            text-align: center;
            font-size: 14px;
            line-height: 1.5;
            color: #333;
        }
    </style>
    """, unsafe_allow_html=True)

    # Display 3 cards per row using columns
    for i in range(0, len(features), 3):
        cols = st.columns(3)
        for col, feature in zip(cols, features[i:i+3]):
            with col:
                st.image(feature["img"], use_container_width=True)
                st.markdown(f"<div class='feature-title'>{feature['title']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='feature-desc'>{feature['desc']}</div>", unsafe_allow_html=True)




elif app_mode == "📈 Risk Score Prediction":
    
    import streamlit as st
    import numpy as np
    import pickle
    import pandas as pd

    # Streamlit UI
    st.markdown("""
    <h1 style='text-align: center; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
               color: #1A237E; font-size: 48px;'>
        🚀 <span style='color:#303F9F;'>Risk Score</span> <span style='color:#1A237E;'>Prediction</span>
    </h1>
    """, unsafe_allow_html=True)


    # Load pre-trained RandomForest model
    model_path = "D:/Insurance_AI_project/Model/random_forest_risk_model.sav"
    scaler_path = "D:/Insurance_AI_project/Model/scaler_risk.pkl"

    with open(model_path, "rb") as f:
        loaded_rf_model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        loaded_scaler = pickle.load(f)

    # Define feature columns (Ensure same order as training)
    feature_columns = loaded_rf_model.feature_names_in_

    # Manual Prediction Section
    st.write("### 🔍 Enter Policy Details")

    annual_income = st.number_input("💰 Annual Income", min_value=0.0, format="%.2f")
    claim_amount = st.number_input("💵 Claim Amount", min_value=0.0, format="%.2f")
    premium_amount = st.number_input("📅 Premium Amount", min_value=0.0, format="%.2f")
    claim_history = st.number_input("📊 Claim History (Number of claims)", min_value=0, step=1)

    policy_type = st.selectbox("📜 Policy Type", ["Health", "Auto", "Life", "Property"])
    gender = st.selectbox("🧑 Gender", ["Male", "Female", "Other"])

    if st.button("🔮 Predict Risk Score"):
        # Create input dictionary with one-hot encoding
        input_dict = {
            'Annual_Income': [annual_income],
            'Claim_Amount': [claim_amount],
            'Premium_Amount': [premium_amount],
            'Claim_History': [claim_history],
            'Policy_Type_Auto': [1 if policy_type == "Auto" else 0],
            'Policy_Type_Health': [1 if policy_type == "Health" else 0],
            'Policy_Type_Life': [1 if policy_type == "Life" else 0],
            'Policy_Type_Property': [1 if policy_type == "Property" else 0],
            'Gender_Female': [1 if gender == "Female" else 0],
            'Gender_Male': [1 if gender == "Male" else 0],
            'Gender_Other': [1 if gender == "Other" else 0]
        }
        input_df = pd.DataFrame(input_dict)

        # Ensure input DataFrame matches training features *in the correct order*
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Apply MinMaxScaler (same as used during training)
        scaled_features = loaded_scaler.transform(input_df[["Annual_Income", "Premium_Amount", "Claim_Amount"]])
        input_df[["Annual_Income", "Premium_Amount", "Claim_Amount"]] = scaled_features

        # Make prediction
        prediction = loaded_rf_model.predict(input_df)
        
        # Risk Score Mapping
        risk_mapping = {0: "🟢 Low", 1: "🟡 Medium", 2: "🔴 High"}
        predicted_risk = risk_mapping.get(prediction[0], "Unknown")
        
        st.success(f"### *Predicted Risk Score: {predicted_risk} ({prediction[0]})*") 

elif app_mode == "💰 Claim Amount Prediction":
    
    import streamlit as st
    import pickle
    import numpy as np
    import pandas as pd
    import os

    # Model and scaler paths
    MODEL_PATH = r"D:\Insurance_AI_project\Model\random_forest_claims.pkl"
    SCALER_PATH = r"D:\Insurance_AI_project\Model\scaler_claims.pkl"
    # Load the model
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()

    # Load the scaler
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Scaler file not found at: {SCALER_PATH}")
        st.stop()

    # Streamlit UI
    #st.set_page_config(page_title="Insurance Claim Predictor", layout="centered")
    st.markdown("""
    <h1 style='text-align: center; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
               color: #3E2723; font-size: 48px;'>
        💰 <span style='color:#5D4037;'>Claim Amount</span> <span style='color:#3E2723;'>Predictor</span>
    </h1>
    """, unsafe_allow_html=True)


    st.markdown("### 🧾 Fill in the customer details below:")

    # Input fields with icons, no min/max restrictions
    age = st.number_input("🎂 Customer Age", value=0, step=1)
    income = st.number_input("💵 Annual Income", value=0, step=1000)
    premium = st.number_input("📄 Premium Amount", value=0, step=500)

    gender = st.selectbox("⚧️ Gender", ["Select Gender", "Male", "Female", "Other"])
    policy = st.selectbox("📑 Policy Type", ["Select Policy Type", "Health", "Life", "Auto", "Home"])
    claim_hist = st.selectbox("📁 Claim History", ["Select", "Yes", "No"])

    # Predict button
    if st.button("Predict Claim Amount"):
        if None in [age, income, premium] or gender == "Select Gender" or policy == "Select Policy Type" or claim_hist == "Select":
            st.warning("⚠️ Please fill out all fields before predicting.")
        else:
            # Manual encoding
            gender_map = {
                "Gender_Female": int(gender == "Female"),
                "Gender_Male": int(gender == "Male"),
                "Gender_Other": int(gender == "Other")
            }

            policy_map = {
                "Policy_Type_Auto": int(policy == "Auto"),
                "Policy_Type_Health": int(policy == "Health"),
                "Policy_Type_Home": int(policy == "Home"),
                "Policy_Type_Life": int(policy == "Life")
            }

            claim_hist_encoded = 1 if claim_hist == "Yes" else 0

            # Construct input DataFrame
            user_df = pd.DataFrame([{
                "Customer_Age": age,
                "Annual_Income": income,
                "Claim_History": claim_hist_encoded,
                "Premium_Amount": premium,
                **gender_map,
                **policy_map
            }])

            # Scale numeric features in correct order
            scale_cols = ["Annual_Income", "Premium_Amount", "Customer_Age"]
            user_df[scale_cols] = scaler.transform(user_df[scale_cols])

            # Reorder columns to match model training
            final_cols = [
                'Customer_Age', 'Annual_Income', 'Claim_History', 'Premium_Amount',
                'Gender_Female', 'Gender_Male', 'Gender_Other',
                'Policy_Type_Auto', 'Policy_Type_Health', 'Policy_Type_Home', 'Policy_Type_Life'
            ]
            user_df = user_df[final_cols]

            # Make prediction
            prediction = model.predict(user_df)[0]
            st.success(f"💵 Predicted Claim Amount: ₹{prediction:,.2f}") 

elif app_mode == "👥 Customer Segmentation":
    
    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    # Load saved models
    with open(r"D:\Insurance_AI_project\Model\kmeans_segment.pkl", "rb") as file:
        kmeans = pickle.load(file)

    # Load the trained PCA model
    pca = joblib.load(r"D:\Insurance_AI_project\Model\pca_segment.pkl")  # Load PCA model

    # Load the trained scaler
    scaler = joblib.load(r"D:\Insurance_AI_project\Model\scaler_segment.pkl")  # Load StandardScaler

    # Define function to assign segment labels
    cluster_labels = {
        0: "High-Value, High-Claim Customers",
        1: "Young and Growing Customers",
        2: "Senior Customers with High Premiums",
        3: "Low Engagement, Low-Risk Customers"
    }

    def assign_segment(label):
        return cluster_labels.get(label, "Unknown Segment")

    # Streamlit UI
    st.markdown("""
    <h1 style='text-align: center; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
               color: #2E7D32; font-size: 48px;'>
        👥 <span style='color:#1B5E20;'>Customer Segmentation</span> <span style='color:#2E7D32;'>Predictor</span>
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("### 🧾 Provide customer information for segmentation:")

    # User Inputs with icons and no input restrictions
    age = st.number_input("🎂 Age", value=None, step=1)
    annual_income = st.number_input("💵 Annual Income", value=None, format="%.2f")
    policy_count = st.number_input("📑 Number of Active Policies", value=None, step=1)
    total_premium_paid = st.number_input("💳 Total Premium Paid ($)", value=None, format="%.2f")
    claim_frequency = st.number_input("📁 Number of Claims Filed", value=None, step=1)
    policy_upgrades = st.number_input("🔄 Number of Policy Changes", value=None, step=1)


    # Predict Segment
    if st.button("Predict Segment"):
        user_data = np.array([[age, annual_income, policy_count, total_premium_paid, claim_frequency, policy_upgrades]])
        user_data_scaled = scaler.transform(user_data)
        user_data_pca = pca.transform(user_data_scaled)
        predicted_cluster = kmeans.predict(user_data_pca)[0]
        segment = assign_segment(predicted_cluster)
        st.success(f"Predicted Segment: {segment}") 

elif app_mode == "🕵️ Fraud Detection":
    
    import streamlit as st
    import pickle
    import numpy as np

    # Load models and scaler using absolute paths
    with open("D:/Insurance_AI_project/scriptfiles/new_random_forest_model.pkl", "rb") as f:
        rf_model = pickle.load(f)

    with open("D:/Insurance_AI_project/scriptfiles/new_neural_network_model.pkl", "rb") as f:
        nn_model = pickle.load(f)

    # Load Scaler
    with open("D:/Insurance_AI_project/scriptfiles/fraud_new_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    st.markdown("""
    <h1 style='text-align: center; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
               color: #D32F2F; font-size: 48px;'>
        🚨 <span style='color:#212121;'>Fraud Claim</span> <span style='color:#D32F2F;'>Prediction</span>
    </h1>
    """, unsafe_allow_html=True)


    st.markdown("### 🧾 Enter claim details to predict fraud probability:")

    # Input fields (initially empty, with icons)
    claim_amount = st.number_input("💰 Claim Amount", value=None)
    annual_income = st.number_input("💵 Annual Income", value=None)

    # Compute claim-to-income ratio safely
    if claim_amount is not None and annual_income:
        claim_to_income_ratio = claim_amount / annual_income
    else:
        claim_to_income_ratio = None

    claim_within_short = st.selectbox("⏱️ Claim Filed Shortly After Policy Issuance?", ["Select", 1, 0])
    suspicious_flag = st.selectbox("🚩 Suspicious Flag?", ["Select", 1, 0])

    # Claim type input field
    claim_type = st.selectbox("📝 Claim Type", ["Select", 'Auto', 'Home', 'Life', 'Medical'])

    claim_type_auto = 1 if claim_type == 'Auto' else 0
    claim_type_home = 1 if claim_type == 'Home' else 0
    claim_type_life = 1 if claim_type == 'Life' else 0
    claim_type_medical = 1 if claim_type == 'Medical' else 0

    # Simplified input for anomaly detection
    is_anomalous = st.selectbox("Is the claim detected as anomalous?", [0, 1])

    if st.button("Predict Fraud Probability"):
        # Prepare input array in correct feature order
        input_data = np.array([[
            claim_amount, suspicious_flag, annual_income, claim_to_income_ratio,
            claim_within_short, claim_type_auto, claim_type_home,
            claim_type_life, claim_type_medical, is_anomalous
        ]])

        # Scale only claim_amount and annual_income (columns 0 and 2)
        input_data[:, [0, 2]] = scaler.transform(input_data[:, [0, 2]])

        # Predict probabilities
        rf_prob = rf_model.predict_proba(input_data)[0][1]
        nn_prob = nn_model.predict_proba(input_data)[0][1]
        fraud_score = (rf_prob + nn_prob) / 2

        # Display results
        #st.markdown(f"### 🧮 Predicted Fraud Probability: *{fraud_score:.2f}*")
        #st.markdown("✅ High scores indicate a higher likelihood of fraud.")

        prediction = 1 if fraud_score > 0.15 else 0


        result = "Fraudulent Claim 🚨" if prediction == 1 else "Genuine Claim ✅"
        st.markdown(f"### 🔍 Prediction: *{result}*")  

elif app_mode == "😊 Sentiment Analysis":
    
    import streamlit as st
    import torch
    import torch.nn as nn
    import numpy as np
    from transformers import BertTokenizer, BertModel

    # Define the Sentiment Analysis Model
    class SentimentModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SentimentModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x  # No softmax here

    # Load BERT tokenizer and model
    @st.cache_resource
    def load_bert():
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert = BertModel.from_pretrained("bert-base-uncased")
        return tokenizer, bert

    tokenizer, bert_model = load_bert()

    # Function to extract BERT embeddings
    def get_bert_embedding(text):
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            output = bert_model(**tokens)
        return output.last_hidden_state.mean(dim=1).squeeze().numpy()

    # Load trained model from .pth using state_dict
    @st.cache_resource
    def load_model():
        try:
            input_dim = 768  # BERT embedding size
            output_dim = 3   # Sentiment classes: Negative, Neutral, Positive
            model = SentimentModel(input_dim, output_dim)
            model.load_state_dict(torch.load("D:/Insurance_AI_project/Model/sentiment_mlp_model.pth", map_location=torch.device("cpu")))
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

    model = load_model()

    # Streamlit UI
    st.markdown("""
    <h1 style='text-align: center; font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
               font-size: 48px;'>
        🧠 <span style='background: linear-gradient(45deg, #FF0000, #FFEB3B); -webkit-background-clip: text; color: transparent;'>Sentiment</span>
        <span style='background: linear-gradient(45deg, #4CAF50, #FFEB3B); -webkit-background-clip: text; color: transparent;'>Analysis</span>
    </h1>
    """, unsafe_allow_html=True)


    
    input_text = st.text_area("✏️ Enter your text for sentiment prediction:")

    if st.button("🔍 Predict Sentiment"):
        if not input_text.strip():
            st.warning("Please enter some text for prediction.")
        elif model:
            # Get BERT embedding
            embedding = get_bert_embedding(input_text)
            input_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

            # Predict sentiment
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()

            sentiment_labels = ["Negative", "Neutral", "Positive"]
            st.success(f"### Predicted Sentiment: **{sentiment_labels[predicted_class]}** 🎯")
        else:
            st.error("Model failed to load.")
 

elif app_mode == "🌐 Translation":
    
    import streamlit as st
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # Use CPU since no GPU is available
    device = "cpu"

    # Cache model loading
    @st.cache_resource
    def load_model():
        MODEL_PATH = r"D:\fine_tuned_translation_model"  # Ensure correct local path
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
        return tokenizer, model

    tokenizer, model = load_model()

    def translate_text(input_text, target_lang):
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to CPU
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Streamlit UI
    st.markdown("""
    <h1 style='text-align: center; font-family: "Trebuchet MS", sans-serif; 
               color: #1E88E5; font-size: 48px;'>
        🌐 AI-Powered <span style='color:#43A047;'>Translation</span>
    </h1>
    """, unsafe_allow_html=True)


    #st.write("### ✍️ Enter English text and get translations in French and Spanish:")

    input_text = st.text_area("✍️ Enter English text:")
    if st.button("Translate"):
        if input_text.strip():
            french_translation = translate_text(input_text, "fr_XX")
            spanish_translation = translate_text(input_text, "es_XX")
            
            st.success("### Translations:")
            st.write("*French:*", french_translation)
            st.write("*Spanish:*", spanish_translation)
        else:
            st.warning("Please enter some text for translation.") 


elif app_mode == "📝 Summarization":
    
    import streamlit as st
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    # Load model and tokenizer with caching
    @st.cache_resource()
    def load_model():
        MODEL_PATH = "D:\\fine_tuned_model"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        return tokenizer, model

    tokenizer, model = load_model()

    # Streamlit App Title
    st.markdown("""
    <h1 style='text-align: center; font-family: "Comic Sans MS", cursive, sans-serif; color: #4BAF65; font-size: 50px;'>📝 Text Summarization</h1>
    """, unsafe_allow_html=True)



    # Input Text Area
    input_text = st.text_area("Enter text to summarize:", height=200)

    def summarize_text(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            summary_ids = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    # Summarize Button
    if st.button("Summarize"):
        if input_text.strip():
            summary = summarize_text(input_text)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text to summarize.") 


elif app_mode == "🤖 Chatbot":
    
    import streamlit as st
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    # 🟠 Must be the first Streamlit command
    #st.set_page_config(page_title="Insurance Chatbot 🤖", layout="centered")

    # 🎯 Load model and tokenizer
    MODEL_PATH = "D:/insurance_t5_model/content/insurance_t5_model"

    @st.cache_resource
    def load_model():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        return tokenizer, model

    tokenizer, model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 🎨 Custom CSS
    st.markdown("""
        <style>
            .main { background-color: #f8f9fa; }
            .stTextInput>div>div>input {
                border-radius: 10px;
                padding: 10px;
                border: 1px solid #ccc;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                border: none;
                border-radius: 10px;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    # 💾 Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []  # List of (user_msg, bot_reply) pairs

    # 💬 App Title
    st.markdown("""
    <h1 style='text-align: center; font-family: "Impact", "Arial Black", sans-serif;
               color: #0D47A1; font-size: 52px; letter-spacing: 1px;'>
        🤖 <span style='color:#1976D2;'>Insurance Query</span> <span style='color:#0D47A1;'>Chatbot</span>
    </h1>
    """, unsafe_allow_html=True)



    st.write("Ask me anything about your insurance policies, claims, or coverage!")

    # 🧠 Chatbox-style conversation
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your Question", placeholder="e.g. How do I file a car insurance claim?")
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        input_text = "insurance query: " + user_input.strip()
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(**inputs, max_length=100)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Save the conversation
        st.session_state.messages.append(("You", user_input.strip()))
        st.session_state.messages.append(("Bot", decoded_output))

    # 🪄 Display entire conversation
    if st.session_state.messages:
        st.markdown("### 💬 Chat History")
        for sender, msg in st.session_state.messages:
            if sender == "You":
                st.markdown(f"*🧑 {sender}:* {msg}")
            else:
                st.markdown(f"*🤖 {sender}:* {msg}")

        st.markdown("---")
        st.info("Ask another question or close the app anytime.")  
