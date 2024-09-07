import os
import streamlit as st
import pandas as pd
import ssl
from io import BytesIO
from langchain_openai import OpenAI
from langchain.prompts import load_prompt
from langchain.chains import LLMChain
from dotenv import load_dotenv
import json
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_icon=":mag:")

# Debug: Print to check if the API key is loaded correctly
api_key = os.getenv("OPENAI_API_KEY")
print(f"Loaded OPENAI_API_KEY: {api_key}")  # Check the API key

# Verify API key format
if not api_key or not api_key.startswith("sk-"):
    raise ValueError("Invalid OpenAI API key provided.")

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Streamlit configuration to suppress deprecation warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

def load_css(file_path):
    try:
        with open(file_path, 'r') as file:
            css = file.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
            print("CSS loaded successfully")
    except FileNotFoundError:
        st.warning(f"CSS file '{file_path}' not found. Using default styles.")

load_css(".streamlit/styles.css")

# Define the function to load the dataset
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Define a function to summarize the dataset
@st.cache_data
def summarize_data(data):
    sample_data = data.sample(min(5, len(data))).to_string(index=False)
    summary = data.describe(include='all').to_string()
    columns = data.columns.to_list()
    columns_str = ', '.join(columns)
    return f"Sample data:\n{sample_data}\n\nSummary statistics:\n{summary}\n\nColumns: {columns_str}"

# Function to load memory from file
def load_memory(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []

# Function to save memory to file
def save_memory(memory, file_path):
    with open(file_path, 'w') as file:
        json.dump(memory, file)

# Load memory from file
memory_file_path = "memory.json"
memory = load_memory(memory_file_path)

# Chatbot integration at the top of the sidebar
st.sidebar.markdown("### Chatbot :speech_balloon:")

# User input for the chatbot
user_input = st.sidebar.text_input("Ask a question about the data:", key="user_input")

# Container to display the AI response
response_container = st.sidebar.empty()

# Load the dataset
@st.cache_resource
def get_dataset(use_iris, uploaded_file):
    if use_iris:
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
        df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    elif uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        df = None
    return df

st.sidebar.markdown("### Upload or Select a Dataset :open_file_folder:")

# Option to use Iris dataset
use_iris = st.sidebar.checkbox("Use Iris dataset")
if use_iris:
    st.sidebar.info("Using the Iris dataset.")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

df = get_dataset(use_iris, uploaded_file)

if api_key:
    try:
        # Initialize Langchain with OpenAI
        llm = OpenAI(api_key=api_key)

        # Define a prompt for Langchain
        prompt = load_prompt("prompt_file.json")

        # Define a function to generate responses
        def generate_response(question, data=None):
            chain = LLMChain(llm=llm, prompt=prompt)
            if data is not None:
                try:
                    data_summary = summarize_data(data)
                    question_with_context = f"Given the following dataset summary:\n{data_summary}\n\nQuestion: {question}"
                    response = chain.run(question_with_context)
                except Exception:
                    response = chain.run(question)
            else:
                response = chain.run(question)
            response = response.strip()
            # Debugging: Print the response to inspect it
            print(f"Raw Response: {response}")
            # Remove any leading question marks or unwanted text
            if response.startswith("?"):
                response = response[1:].strip()
            if response.lower().startswith("answer:"):
                response = response[7:].strip()
            return response

        if user_input:
            response = generate_response(user_input, df)
            response_html = f"""
            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
                <strong>AI Response:</strong><br>{response}
            </div>
            """
            response_container.markdown(response_html, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"Error generating response: {e}")
else:
    st.sidebar.error("API key not found. Please set the OPENAI_API_KEY in the .env file.")

if df is not None:
    # Streamlit app title and header
    st.title('Exploratory Data Analysis :mag:')
    st.header('This app allows you to explore, visualize and chat with data')

    # Sidebar for user inputs
    selected_column = st.sidebar.selectbox('Select a column to visualize :chart_with_upwards_trend:', df.columns)

    # Tabs for charts
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "DataFrame :clipboard:", 
        "Profiling Report :page_facing_up:",
        "Area Chart :globe_with_meridians:", 
        "Bar Chart :bar_chart:", 
        "Line Chart :chart_with_upwards_trend:", 
        "Scatter Chart :large_blue_diamond:", 
        "Correlation Matrix :link:", 
        "Boxplot :package:", 
        "Export :file_folder:"
    ])

    with tab1:
        st.subheader("DataFrame :clipboard:")
        st.dataframe(df)
        
        with st.expander("Description of DataFrame"):
            st.write("This table displays the dataset. You can view and scroll through all the rows and columns in the dataset.")
    
    with tab2:
        st.subheader("Profiling Report :page_facing_up:")
        profile = ProfileReport(df, title="Profiling Report")
        st_profile_report(profile)
        with st.expander("Description of Profiling Report"):
            st.write("""
            **Profiling Report Explanation**:
            - The profiling report generates an extensive report for a pandas DataFrame.
            - It includes a detailed description of the dataset, such as missing values, variable types, statistics, and correlations.
            - This report helps to understand the data and identify potential data quality issues quickly.
            """)

    with tab3:
        st.subheader("Area Chart :globe_with_meridians:")
        st.area_chart(df[selected_column])
        with st.expander("Description of Area Chart"):
            st.write("""
            **Area Chart Explanation**:
            - An area chart displays quantitative data over time.
            - The x-axis represents the time or other continuous variable, and the y-axis represents the quantitative value.
            - The area between the axis and the line is shaded to indicate volume.
            """)
    
    with tab4:
        st.subheader("Bar Chart :bar_chart:")
        st.bar_chart(df[selected_column])
        with st.expander("Description of Bar Chart"):
            st.write("""
            **Bar Chart Explanation**:
            - A bar chart represents categorical data with rectangular bars.
            - Each bar's length is proportional to the value it represents.
            - Bar charts are useful for comparing different categories.
            """)
    
    with tab5:
        st.subheader("Line Chart :chart_with_upwards_trend:")
        st.line_chart(df[selected_column])
        with st.expander("Description of Line Chart"):
            st.write("""
            **Line Chart Explanation**:
            - A line chart displays information as a series of data points called 'markers' connected by straight line segments.
            - It is useful for showing trends over time.
            """)
    
    with tab6:
        st.subheader("Scatter Chart :large_blue_diamond:")
        x_axis = st.selectbox('Select the x-axis', df.columns, index=0, key="scatter_x")
        y_axis = st.selectbox('Select the y-axis', df.columns, index=1, key="scatter_y")
        scatter_data = pd.DataFrame({
            x_axis: df[x_axis],
            y_axis: df[y_axis]
        })
        st.scatter_chart(scatter_data)
        with st.expander("Description of Scatter Chart"):
            st.write("""
            **Scatter Chart Explanation**:
            - A scatter chart displays values for typically two variables for a set of data.
            - Each point represents an observation, with its position determined by the values of the selected x and y variables.
            - Scatter plots are useful for identifying correlations, patterns, and outliers in the data.
            """)
    
    with tab7:
        st.subheader("Correlation Matrix :link:")
        numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Filter out non-numeric columns
        corr = numeric_df.corr()
        st.write(corr.style.background_gradient(cmap='coolwarm').format(precision=2))
        with st.expander("Description of Correlation Matrix"):
            st.write("""
            **Correlation Matrix Explanation**:
            - A correlation matrix shows the pairwise correlations between numeric variables in the dataset.
            - Each cell in the matrix represents the correlation coefficient between two variables.
            - The values range from -1 to 1, where -1 indicates a perfect negative correlation, 1 indicates a perfect positive correlation, and 0 indicates no correlation.
            - A heatmap is often used to visualize the correlation matrix, with colors indicating the strength and direction of the correlations.
            """)
    
    with tab8:
        st.subheader("Boxplot :package:")
        fig = px.box(df, y=selected_column, title=f'Box plot of {selected_column}')
        st.plotly_chart(fig)
        with st.expander("Description of Boxplot"):
            st.write("""
            **Boxplot Explanation**:
            - A boxplot, also known as a box-and-whisker plot, displays the distribution of a numeric variable.
            - It shows the median, quartiles, and potential outliers in the data.
            - The box represents the interquartile range (IQR), which contains the middle 50% of the data.
            - The line inside the box represents the median value.
            - The whiskers extend to the smallest and largest values within 1.5 times the IQR from the lower and upper quartiles.
            - Points outside this range are considered outliers and are shown as individual points.
            """)

    with tab9:
        st.subheader("Export :file_folder:")

        # CSV export
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='dataset.csv',
            mime='text/csv',
        )

        # Excel export
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        processed_data = output.getvalue()
        st.download_button(
            label="Download data as Excel",
            data=processed_data,
            file_name='dataset.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )

else:
    st.title('Exploratory Data Analysis :mag:')
    st.header('This app allows you to explore, visualize and chat with structured data.')
    st.write("Please upload a CSV file to begin or use the Iris dataset.")
