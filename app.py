import os
import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Insurance Charges Prediction", page_icon="💰", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: black;
            color: white;
        }
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        .stDataFrame, .stTable {
            background-color: #1c1c1c;
            color: white;
        }
        .css-1v0mbdj, .stButton>button {
            background-color: #262730;
            color: white;
        }
        h1, h2, h3, h4, h5, h6, p {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# File paths
model_path = ("data_pickle_2.pkl")
data_path = ("insurance.csv")

# Load model and data
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at {model_path}")
elif not os.path.exists(data_path):
    st.error(f"❌ Data file not found at {data_path}")
else:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(data_path)

    # App title
    st.title("💰 Insurance Charges Prediction App")
    st.markdown("Predict the insurance charges based on several factors.")
    st.sidebar.header("Enter Patient Details")

    # Sidebar inputs
    age = st.sidebar.number_input("Age", min_value=18, max_value=100)
    sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
    bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)
    children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10)
    smoker = st.sidebar.selectbox("Smoker", ("Yes", "No"))
    region = st.sidebar.selectbox("Region", ("northeast", "northwest", "southeast", "southwest"))

    if st.sidebar.button("Predict"):
        input_df = pd.DataFrame([{
            'age': age,
            'sex': sex.lower(),
            'bmi': bmi,
            'children': children,
            'smoker': smoker.lower(),
            'region': region.lower()
        }])

        try:
            prediction = model.predict(input_df)[0]
            st.subheader("Prediction Result")
            st.success(f"💰 Predicted Insurance Charge: ${prediction:,.2f}")

            # Profile chart
            profile_data = pd.DataFrame({
                'Feature': ['Age', 'Sex (1=Male)', 'BMI', 'Children', 'Smoker (1=Yes)'],
                'Value': [
                    age,
                    1 if sex.lower() == "male" else 0,
                    bmi,
                    children,
                    1 if smoker.lower() == "yes" else 0
                ]
            })

            fig_profile, ax_profile = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Feature', y='Value', data=profile_data, ax=ax_profile)
            ax_profile.set_title("Patient Profile", color='white')
            ax_profile.set_ylabel("Value", color='white')
            ax_profile.set_xlabel("Feature", color='white')
            ax_profile.tick_params(colors='white')
            st.pyplot(fig_profile)

            # Region display
            st.markdown(f"**Region:** {region.title()}")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

    # Dataset section
    st.subheader("Dataset Overview")
    st.dataframe(df.head())
    st.markdown(f"The dataset contains {df.shape[0]} records and {df.shape[1]} features.")
