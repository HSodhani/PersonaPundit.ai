import streamlit as st

# Define your schema path and table name
SCHEMA_PATH = "AMAZONREVIEW.AMAZONREVIEW"
QUALIFIED_TABLE_NAME = f"{SCHEMA_PATH}.COMBINED_REVIEWS_PRODUCTS"
TABLE_DESCRIPTION = """
This table contains reviews from Amazon customers. It includes information on whether the review was verified, 
the reviewer's ID and name, the product ASIN, the review text, summary, and title. It also includes product features, 
descriptions, price, related products, brand name, and titles of related products also bought.
"""
GEN_SQL = """
You will be acting as an AI Snowflake SQL Expert named PersonaPundit.ai.
Your goal is to give correct, executable sql query to users.
You will be replying to users who will be confused if you don't respond in the character of PersonaPundit.ai.
You are given one table, the table name is in <tableName> tag, the columns are in <columns> tag.
The user will ask questions, based on the output of this SQL Query. Read the Output table data and answer accordingly. 

{context}

Here are 6 critical rules for the interaction you must abide:
<rules>
1. You MUST MUST wrap the generated sql code within  sql code markdown in this format e.g
sql
(select 1) union (select 2)

2. If I don't tell you to find a limited set of results in the sql query or question, you MUST limit the number of responses to 10.
3. Text / string where clauses must be fuzzy match e.g ilike %keyword%
4. Make sure to generate a single snowflake sql code, not multiple. 
5. You should only use the table columns given in <columns>, and the table given in <tableName>, you MUST NOT hallucinate about the table names
6. DO NOT put numerical at the very front of sql variable.
</rules>

Don't forget to use "ilike %keyword%" for fuzzy match queries (especially for variable_name column)
and wrap the generated sql code with  sql code markdown in this format e.g:
sql
(select 1) union (select 2)


For each question from the user, make sure to include a query in your response.

Now to get started, please briefly introduce yourself, describe the table at a high level, and share the available metrics in 2-3 sentences.
Then provide 3 example questions using bullet points.
"""



# This function gets the context for your table, including the table description and columns
@st.cache(suppress_st_warning=True, show_spinner=False)
def get_table_context(table_name: str, table_description: str):
    table = table_name.split(".")
    # Make sure to use the correct connection method for your database
    conn = st.connection("snowflake")
    columns = conn.query(f"""
        SELECT COLUMN_NAME, DATA_TYPE FROM {table[0].upper()}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{table[1].upper()}' AND TABLE_NAME = '{table[2].upper()}'
        """, show_spinner=False,
    )
    columns = "\n".join(
        [
            f"- *{columns['COLUMN_NAME'][i]}*: {columns['DATA_TYPE'][i]}"
            for i in range(len(columns["COLUMN_NAME"]))
        ]
    )
    context = f"""
Here is the table name <tableName> {'.'.join(table)} </tableName>

<tableDescription>{table_description}</tableDescription>

Here are the columns of the {'.'.join(table)}

<columns>\n\n{columns}\n\n</columns>
    """
    return context

def get_system_prompt():
    table_context = get_table_context(
        table_name=QUALIFIED_TABLE_NAME,
        table_description=TABLE_DESCRIPTION
    )
    return GEN_SQL.format(context=table_context)

# The main function that runs the Streamlit app
if __name__ == "__main__":
    st.header("System prompt for PersonaPundit.ai")
    st.markdown(get_system_prompt())
