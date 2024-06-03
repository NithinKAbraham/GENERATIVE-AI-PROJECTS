import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai

# Replace with your actual API key
API_KEY = "AIzaSyBAq7FQNuaOGGbyGUdv61eEs7pg1kCAgM0"

def analyze_dataframe(file_path):
    data = {}  # Dictionary to store printed information

    # Read the CSV file
    df = pd.read_csv(file_path, encoding="latin-1")

    # Store the original DataFrame in data
    data['Original DataFrame'] = df

    # Store the head of DataFrame
    data['Head of DataFrame'] = df.head()

    # Store the tail of DataFrame
    data['Tail of DataFrame'] = df.tail()

    # Store a random sample of DataFrame
    data['Random Sample of DataFrame'] = df.sample(5)

    # Store the shape of DataFrame
    data['Shape of DataFrame'] = df.shape

    # Store data types of columns
    data['Data Types of Columns'] = df.dtypes

    # Store the sum of missing values in each column
    data['Missing Values in DataFrame'] = df.isna().sum()

    # Identify categorical and numerical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
    data['Categorical Columns'] = categorical_columns
    data['Numerical Columns'] = numerical_columns

    # Encode categorical columns using LabelEncoder
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Store the cleaned DataFrame (df1)
    df1 = df.dropna()
    data['Cleaned DataFrame'] = df1

    # Store the number of duplicated rows in the cleaned DataFrame (df1 remains unchanged)
    data['Number of Duplicated Rows in Cleaned DataFrame'] = df1.duplicated().sum()

    # Create a new DataFrame (df3) from the original DataFrame (df)
    df3 = df.copy()
    data['New DataFrame'] = df3

    # Calculate correlation between variables in df3
    correlation_matrix = df3.corr()
    data['Correlation Matrix'] = correlation_matrix

    return data

def analyze_text_data(text_data, purpose):
    analysis_report = ""
    try:
        # Configure API key
        genai.configure(api_key=API_KEY)

        # Load Gemini Pro model
        model = genai.GenerativeModel("gemini-pro")  # Replace with actual model name if needed

        # Define the analysis prompt
        prompt = f"""
        You are a data analyst with good knowledge and experience, you can analyze any data of various domains.
        Now I will give you text data as: {text_data}

        The purpose of the data is: {purpose}
        
        Based on the purpose, please do the following without using stars or asterisks for bullet points:
        
        -show the dataframe for better understanding
        - Analyze the data and find key insights from it. Use numbered points for the insights.
        - Inferential statistics make inferences about populations based on sample data: analyze the data using this method and deliver key findings from it
        - Find key points and craft a story about the data. Use numbered points for the key points.
        - Provide recommendations for better performance. Use numbered points for the recommendations.
        - Create a table for showcasing pros and cons for the {purpose} for our data. The table should be in a rectangle, and pros and cons should be on different sides of the rectangle
        - Point out key information and improvements for better performance. Use numbered points for these as well.
        """

        # Generate content (analysis report)
        response = model.generate_content(prompt)
        analysis_report = response.text

    except Exception as e:
        st.error(f"Error analyzing text: {e}")

    return analysis_report

def format_report(report):
    lines = report.split("\n")
    formatted_lines = []
    for line in lines:
        # Replace stars at the beginning of the line with numbers or empty spaces
        if line.startswith("* "):
            line = line.replace("*", "").strip()
        formatted_lines.append(line)
    return "\n".join(formatted_lines)

def main():
    st.set_page_config(page_title="Data Analysis App")
    st.header("Data Analysis App")

    uploaded_file = st.file_uploader("Upload a CSV file...", type=["csv"])
    purpose = st.text_input("Enter the purpose of the data analysis:")

    if uploaded_file is not None and purpose:
        with st.spinner('Analyzing data...'):
            data = analyze_dataframe(uploaded_file)
            text_data = ""
            for key, value in data.items():
                text_data += f"{key}:\n{value}\n\n"
            
            analysis_report = analyze_text_data(text_data, purpose)
            formatted_report = format_report(analysis_report)
        
        st.subheader("DataFrame Analysis")
        st.write(data['Head of DataFrame'])
        st.write(data['Tail of DataFrame'])
        st.write(data['Random Sample of DataFrame'])
        st.write("Shape:", data['Shape of DataFrame'])
        st.write("Data Types of Columns:", data['Data Types of Columns'])
        st.write("Missing Values:", data['Missing Values in DataFrame'])
        st.write("Categorical Columns:", data['Categorical Columns'])
        st.write("Numerical Columns:", data['Numerical Columns'])
        st.write("Cleaned DataFrame:", data['Cleaned DataFrame'])
        st.write("Number of Duplicated Rows:", data['Number of Duplicated Rows in Cleaned DataFrame'])
        st.write("Correlation Matrix:", data['Correlation Matrix'])
        
        st.subheader("Analysis Report")
        st.write(formatted_report)

if __name__ == "__main__":
    main()
