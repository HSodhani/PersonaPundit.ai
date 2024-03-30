import streamlit as st
import pandas as pd
import snowflake.connector
from openai import OpenAI
import re
import os

from langchain.retrievers.web_research import WebResearchRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
import faiss

# Assuming the necessary API keys and connection details are stored in Streamlit's secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
os.environ['GOOGLE_CSE_ID'] = st.secrets["GOOGLE_CSE_ID"]
snowflake_user = st.secrets["connections"]["snowflake"]["user"]
snowflake_password = st.secrets["connections"]["snowflake"]["password"]
snowflake_account = st.secrets["connections"]["snowflake"]["account"]
snowflake_warehouse = st.secrets["connections"]["snowflake"]["warehouse"]
snowflake_database = st.secrets["connections"]["snowflake"]["database"]
snowflake_schema = st.secrets["connections"]["snowflake"]["schema"]

st.title("👳🏽‍♂ PersonaPundit.ai")

search = GoogleSearchAPIWrapper()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

# Initialize FAISS and embeddings
def initialize_faiss_and_embeddings():
    embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    return vectorstore

vectorstore_public = initialize_faiss_and_embeddings()

# Initialize WebResearchRetriever
def initialize_web_research_retriever(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", api_key=openai_api_key, temperature=0, streaming=True)
    
    # Now, GoogleSearchAPIWrapper is initialized without explicit keys
    search = GoogleSearchAPIWrapper()
    
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore,
        llm=llm, 
        search=search, 
        num_search_results=3
    )
    return web_retriever

web_retriever = initialize_web_research_retriever(vectorstore_public)

# Fetch review data from Snowflake
def fetch_review_data(reviewer_id):
    ctx = snowflake.connector.connect(
        user=snowflake_user,
        password=snowflake_password,
        account=snowflake_account,
        warehouse=snowflake_warehouse,
        database=snowflake_database,
        schema=snowflake_schema
    )
    cs = ctx.cursor()
    try:
        query = f"""
            SELECT REVIEWERNAME, REVIEWTEXT, SUMMARY, TITLE, FEATURE, DESCRIPTION, BRAND, PRICE
            FROM AMAZONREVIEW.TEST_VIEW
            WHERE REVIEWERID = '{reviewer_id}'
        """
        cs.execute(query)
        df = pd.DataFrame(cs.fetchall(), columns=[x[0] for x in cs.description])
        return df
    finally:
        cs.close()
        ctx.close()

def analyze_review_data(review_data):
    # Implement the logic to analyze the review data and generate insights.
    # This could include NLP processing, sentiment analysis, extracting key phrases, etc.
    # For now, let's just concatenate the data into a simple string.
    persona_details = f"""
    Reviewer Name: {review_data['REVIEWERNAME'].iloc[0]}
    Product Title: {review_data['TITLE'].iloc[0]}
    Brand: {review_data['BRAND'].iloc[0]}
    Price: {review_data['PRICE'].iloc[0]}
    Review Summary: {review_data['SUMMARY'].iloc[0]}
    Review Text: {review_data['REVIEWTEXT'].iloc[0]}
    """
    return persona_details

# Function to generate a persona from review data using chat.completions
def generate_persona(reviewer_id):
    review_data = fetch_review_data(reviewer_id)
    if review_data.empty:
        return "No data found for this Reviewer ID."

    # Now you have the DataFrame, you can proceed to extract details and generate the persona
    reviewer_name = review_data["REVIEWERNAME"].iloc[0]
    price = review_data["PRICE"].iloc[0]
    review_text = review_data["REVIEWTEXT"].iloc[0]
    title = review_data["TITLE"].iloc[0]
    description = review_data["DESCRIPTION"].iloc[0]
    summary = review_data["SUMMARY"].iloc[0]
    brand = review_data["BRAND"].iloc[0]
    messages = [
        {"role": "system", "content": "You are a wise sage providing insights into customer personas based on their reviews and brand"},
        {"role": "user", "content": f"Based on this review by {reviewer_name} (print this on the top: Name: {reviewer_name}), who paid {price} for the product named {title} by {brand} which is described by {description}. The summary{summary}. Also, here is the review text: {review_text}. Generate a customer persona considering demographics, psychographics, behavioral traits, needs, goals, and pain points. Dive deeper into review text and get the deepest of Insights and use the {title} to suggest what other products the user might buy. NOTE: Give me 5 Points for each block."}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content

# Detecting persona request from user input
def detect_persona_request(prompt):
    match = re.search(r"reviewerID[:]\s(\S+)", prompt, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

# Add a function to process general knowledge questions
def handle_general_query(query):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Make sure to use the correct model ID
            messages=[{"role": "system", "content": "I am an AI trained to provide information and answer questions."},
                      {"role": "user", "content": query}]
        )
        
        # Accessing the response correctly
        if response.choices:
            first_choice = response.choices[0]
            return first_choice.message.content  # Correctly access the content attribute
        else:
            return "No response was generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Update the function to detect the type of query and route it accordingly
def process_input(user_input):
    reviewer_id_match = re.search(r"reviewerID[:]\s(\S+)", user_input, re.IGNORECASE)
    if reviewer_id_match:
        reviewer_id = reviewer_id_match.group(1).strip()
        persona = generate_persona(reviewer_id)
        return persona, 'persona'
    else:
        # Call the updated function here
        general_response = handle_general_query(user_input)
        return general_response, 'general'

# Streamlit UI handling logic
st.subheader("Ask me anything:")
user_input = st.text_area("Enter your request here:")

# Handling the input directly, without using session state for history
if user_input:
    response, response_type = process_input(user_input)
    
    if response_type == 'persona':
        st.write("Generated Persona:")
        st.write(response)
    elif response_type == 'general':
        st.write("General Knowledge Answer:")
        st.write(response)


# Move instructions to a sidebar or a static section on the main page
st.sidebar.header("Instructions:")
st.sidebar.markdown("""
- **To generate a user persona**, please type a request that includes a specific reviewer ID.
- **Example request**: "Generate persona for reviewerID: A1JMSX54DO3LOP".

**Note**:
- The persona generation leverages both the analysis of review data from Snowflake and enriched insights through web research.
- Please ensure that the reviewer ID you provide matches an existing record in the Snowflake database for accurate persona generation.
""")

# Below this line, you could add more functionality or information about how the personas are generated,
# tips for interacting with your application, or any additional features you provide.

# Example of adding more interactivity or information
st.sidebar.header("About PersonaPundit.ai")
st.sidebar.info("""
PersonaPundit.ai uses advanced AI techniques to generate detailed user personas based on product review data.
By analyzing reviews and supplementing this analysis with web research, PersonaPundit.ai provides insights into
the demographics, preferences, and behavior of users, helping businesses understand their customers better.
""")