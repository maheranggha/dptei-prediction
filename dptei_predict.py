import streamlit as st
import pandas as pd 
import joblib

# Load the saved model
joblib_file = "rf_model.pkl"
loaded_model = joblib.load(joblib_file)


# Define a function to make predictions using the loaded model
def make_prediction(input_data):
    # Convert input_data to DataFrame or the appropriate format for the model
    input_df = pd.DataFrame([input_data])
    prediction = loaded_model.predict(input_df)
    return prediction

# Streamlit app code
st.title("On-Time Graduation Prediction 🎓")

# Instructions section with a collapsible box
with st.expander("About This App", expanded=False):
    st.markdown("""This app predicts whether a student will graduate on time based on various demographic and academic performance factors. 
The model was trained on a dataset containing information about students' demographics and their academic performance. 
Initially, dataset was collected from DPTEI UNY from 2019 to 2023 academic calendar years. 
It consists of 140 features which include students' demographic and students' all courses in 4 years. 
The following machine learning models were considered: Random Forest, kNN, Decision Tree, SVM, Naive Bayes, Logistic Regression, Gradient Boosting, Neural Network, and Stochastic Gradient Descent.
After evaluation, the Random Forest model was selected as the best performing model with 85% Accuracy and 0.88 of ROC AUC.
    """)

st.write("""
### Instruction
Please, fill in the details in the form below to predict whether you will graduate on time.
These features are chosen based on the *Feature Importances* score from Random Forest model. 
""")

# Create two columns
col1, col2 = st.columns(2)

# Column 1: Demographic inputs
with col1:
    st.write("### Demographic")
    extra_curr = st.selectbox("Are you involved in Extra Curricular or Student Club?", ['Yes', 'No'])
    sex = st.selectbox("Gender", ['Male', 'Female'])
    M_Job = st.selectbox("What is your Mother's Job?", ['Labour', 'Government Worker', 'Farmer', 'Employee', 'Freelancer', 'Seller', 'Entrepreneur', 'Deceased', 'Retired', 'Not Working', 'Others'])
    F_Job = st.selectbox("What is your Father's Job?", ['Labour', 'Government Worker', 'Farmer', 'Employee', 'Freelancer', 'Seller', 'Entrepreneur', 'Deceased', 'Retired', 'Not Working', 'Others'])
    school_major = st.selectbox("High School Major*", ['Science', 'Social Science', 'EEC', 'Non-EEC'])
    st.markdown("*EEC = Electrical, Electronics, or Computer-related major (Vocational)")

# Column 2: Student Performance inputs
with col2:
    st.write("### Student Performance")
    gpa4 = st.number_input("GPA Semester 4", min_value=0.00, max_value=4.00, step=0.01, format="%.2f")
    cgpa = st.number_input("Cumulative GPA", min_value=0.00, max_value=4.00, step=0.01, format="%.2f")
    toefl = st.number_input("TOEFL Score", min_value=310, max_value=677, step=1)
    Social_Science = st.number_input("Social Science Score", min_value=0.00, max_value=4.00, step=0.01, format="%.2f")
    Programming = st.number_input("Programming Score", min_value=0.00, max_value=4.00, step=0.01, format="%.2f")

# Function to generate suggestions based on input data
def generate_suggestions(input_data):
    suggestions = []
    if input_data['gpa4'] < 3.4:
        suggestions.append("Try to improve your next GPA. Aim for a GPA above 3.40!")
    if input_data['cgpa'] < 3.29:
        suggestions.append("Work on raising your Cumulative GPA. A CGPA above 3.30 is preferable.")
    if input_data['toefl'] < 425:
        suggestions.append("Consider improving your TOEFL score. Minimum require for graduation is 425!")
    if input_data['Social_Science'] < 3.67:
        suggestions.append("Enhance your Social Science Courses score!")
    if input_data['Programming'] < 3.68:
        suggestions.append("Work on your Programming skills. A score above 3.6 is advisable.")
    if input_data['extra_curr'] == 'yes':
        suggestions.append("You might need to spend less time in Extra Curricular or Student Club and focus more in your study!")
    if not suggestions:
        suggestions.append("Keep up the good work and maintain your current performance levels.")
    return suggestions

if st.button("Predict"):
    # Input validation
    error_message = ""
    if not (0.00 <= gpa4 <= 4.00):
        error_message += "GPA Semester 4 must be between 0.00 and 4.00.\n"
    if not (0.00 <= cgpa <= 4.00):
        error_message += "Cumulative GPA must be between 0.00 and 4.00.\n"
    if not (310 <= toefl <= 677):
        error_message += "TOEFL Score must be between 310 and 677.\n"
    if not (0.00 <= Social_Science <= 4.00):
        error_message += "Social Science Score must be between 0.00 and 4.00.\n"
    if not (0.00 <= Programming <= 4.00):
        error_message += "Programming Score must be between 0.00 and 4.00.\n"
    
    if error_message:
        st.error(error_message)
    else:
        # Mapping to match with trained data
        sex_mapping = {'Male': 'M', 'Female':'F'}
        extra_curr_mapping = {'Yes': 'yes', 'No': 'no'}
        input_data = {
            'extra_curr': extra_curr_mapping[extra_curr],
            'sex': sex_mapping[sex],
            'M_Job': M_Job,
            'F_Job': F_Job,
            'school_major': school_major,
            'gpa4': gpa4,
            'cgpa': cgpa,
            'toefl': toefl,
            'Social_Science': Social_Science,
            'Programming': Programming
        }
        result = make_prediction(input_data)
        graduation_status = 'Yes' if result[0] == 1 else 'No'
        
        # Generate suggestions for improvement
        suggestions = generate_suggestions(input_data)
        
        # Display the result in an appealing way with suggestions
        if graduation_status == 'Yes':
            st.success(f"The predicted graduation status is: **ON TIME** 🎓")
            st.balloons()
            st.write("### Keep up the great work! You're on track to graduate on time.")
        else:
            st.warning(f"The predicted graduation status is: **NOT ON TIME** 😢")
            st.write("Here are some suggestions to improve your chances of graduating on time:")
            for suggestion in suggestions:
                st.warning(f"- {suggestion}")