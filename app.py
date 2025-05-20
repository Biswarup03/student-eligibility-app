
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Student Eligibility Checker", layout="centered")

st.title("📘 Student Exam Eligibility Checker")
st.markdown("Upload a CSV with columns **Name** and **Total** (format: `Attended|Total`)")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

threshold = st.slider("Attendance Threshold (%)", 0, 100, 70, step=1)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if "Name" not in df.columns or "Total" not in df.columns:
            st.error("CSV must contain 'Name' and 'Total' columns.")
        else:
            df = df.dropna(subset=["Total"])
            valid_format = df["Total"].astype(str).str.contains(r"\d+\|\d+")
            df = df[valid_format].copy()
            df[['Attended', 'Total_Classes']] = df['Total'].str.split('|', expand=True).astype(int)
            df['Attendance_Percentage'] = (df['Attended'] / df['Total_Classes']) * 100
            df['Allowed'] = (df['Attendance_Percentage'] >= threshold).astype(int)

            X = df[['Attendance_Percentage']]
            y = df['Allowed']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model = LogisticRegression()
            model.fit(X_train_scaled, y_train)

            st.success("✅ Model trained successfully!")

            student_name = st.text_input("Enter student name to check eligibility")

            if student_name:
                match = df[df['Name'].str.lower() == student_name.strip().lower()]
                if not match.empty:
                    attendance = match.iloc[0]['Attendance_Percentage']
                    scaled = scaler.transform([[attendance]])
                    prediction = model.predict(scaled)[0]
                    status = "✅ Allowed" if prediction == 1 else "❌ Not Allowed"
                    st.markdown(f"""
                        **Student:** {match.iloc[0]['Name']}  
                        **Attendance:** {attendance:.2f}%  
                        **Prediction:** {status}
                    """)
                else:
                    st.error(f"Student '{student_name}' not found.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Upload a CSV file to begin.")
