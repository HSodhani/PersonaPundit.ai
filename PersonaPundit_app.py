from openai import OpenAI
import re
import streamlit as st
import pandas as pd
import snowflake.connector
from prompts import get_system_prompt

st.title("👳🏽‍♂️ PersonaPundit.ai")

# Initialize the chat messages history
client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": get_system_prompt()}]

# Function to fetch review data from Snowflake
def fetch_review_data(reviewer_id):
    ctx = snowflake.connector.connect(
        user=st.secrets["connections"]["snowflake"]["user"],
        password=st.secrets["connections"]["snowflake"]["password"],
        account=st.secrets["connections"]["snowflake"]["account"],
        warehouse=st.secrets["connections"]["snowflake"]["warehouse"],
        database=st.secrets["connections"]["snowflake"]["database"],
        schema="AMAZONREVIEW"
    )
    cs = ctx.cursor()
    try:
        cs.execute(f"""
            SELECT REVIEWERNAME, REVIEWTEXT, SUMMARY, TITLE, FEATURE, DESCRIPTION, BRAND, PRICE
            FROM AMAZONREVIEW.TEST_VIEW
            WHERE REVIEWERID = '{reviewer_id}'
        """)
        df = pd.DataFrame(cs.fetchall(), columns=[x[0] for x in cs.description])
        return df
    finally:
        cs.close()
        ctx.close()

# Function to generate a persona from review data using chat.completions
def generate_persona(review_data):
    reviewer_name = review_data["REVIEWERNAME"].iloc[0]
    price = review_data["PRICE"].iloc[0]
    review_text = review_data["REVIEWTEXT"].iloc[0]
    title =  review_data["TITLE"].iloc[0]
    description = review_data["DESCRIPTION"].iloc[0]
    summary = review_data["SUMMARY"].iloc[0]
    brand = review_data["BRAND"].iloc[0]
    messages = [
        {"role": "system", "content": "You are a wise sage providing insights into customer personas based on their reviews and brand"},
        {"role": "user", "content": f"Based on this review by {reviewer_name} (print this on the top: Name: {reviewer_name}), who paid {price} for the product named {title} by {brand} which is described by {description}. The summary{summary}. Also, here is the review text: {review_text}. Generate a customer persona considering demographics, psychographics, behavioral traits, needs, goals, and pain points."}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content

# Improved detection for generating a persona
def detect_persona_request(prompt):
    # Check if the prompt mentions generating a persona
    if "user persona" in prompt.lower():
        # Attempt to extract a ReviewerID from the prompt
        reviewer_id_match = re.search(r"reviewerID[:]*\s*(\S+)", prompt, re.IGNORECASE)
        if reviewer_id_match:
            return reviewer_id_match.group(1).strip()
    return None

# Main chat input handling
if prompt := st.chat_input():
    reviewer_id = detect_persona_request(prompt)
    if reviewer_id:
        review_data = fetch_review_data(reviewer_id)
        if not review_data.empty:
            persona = generate_persona(review_data)
            st.session_state.messages.append({"role": "user", "content": prompt, "persona": persona})
        else:
            st.write("No data found for this Reviewer ID.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

# Display existing chat messages and personas if generated
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "persona" in message:
            st.write("Generated Persona:", message["persona"])
        if "results" in message:
            st.dataframe(message["results"])

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = ""
        resp_container = st.empty()
        for delta in client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            stream=True,
        ):
            response += (delta.choices[0].delta.content or "")
            resp_container.markdown(response)

        message = {"role": "assistant", "content": response}
        # Attempt to parse the response for a SQL query and execute it if available
        sql_match = re.search(r"sql\n(.*)\n", response, re.DOTALL)
        if sql_match:
            sql = sql_match.group(1)
            conn = st.connection("snowflake")
            message["results"] = conn.query(sql)
            st.dataframe(message["results"])
        st.session_state.messages.append(message)
