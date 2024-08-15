# 👳🏽‍♂ PersonaPundit.ai
## Overview :

__PersonaPundit.ai__ is a Streamlit-based web application that leverages advanced AI techniques, including OpenAI's GPT-3.5 and Gemini models, to generate detailed user personas from product review data. This application uses a variety of data sources and APIs, such as Snowflake for database management, Amazon S3 for data storage, and Google API for enhanced web research, providing businesses with insights into customer demographics, preferences, and behaviors.

## Features:

- __Persona Generation:__ Generate detailed customer personas based on individual review data stored in Snowflake, including demographics, psychographics, and behavioral traits.
- __Group Persona Insights:__ Aggregate and analyze personas to generate insights for a group of users sharing common interests.
- __Dynamic Interactivity:__ Users can manage personas, save conversations, and retrieve historical data directly from the user interface.
- __Real-Time Data Handling:__ Integrates with AWS S3 for real-time data storage and retrieval of conversation histories.
- __Advanced Text Analysis:__ Utilizes OpenAI's GPT-3.5 for text generation and analysis, and Gemini model for generating enriched content based on user reviews.

## How it Works:

- __Data Retrieval:__ The app retrieves user review data from a Snowflake database using provided reviewer IDs.
- __Persona Creation:__ Based on the review data, the app uses AI models to generate a persona that includes detailed demographic and behavioral traits.
- __Web Research:__ Augments persona creation with web-based research to enrich the insights provided, using an integrated Google Search API wrapper.
- __Interactive UI__: Users interact with the application through a friendly web interface built with Streamlit, where they can input requests, view generated personas, and manage stored data.

## Installation

To run PersonaPundit.ai locally or deploy it to a server, follow these steps:

- #### Clone the repository:
```bash
git clone https://github.com/yourusername/PersonaPundit.git
cd PersonaPundit
```
- #### Set up a virtual environment (Optional):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

- #### Install dependencies:
```
pip install -r requirements.txt
```

- #### Configure API Keys and Database Connections:
Place your API keys and database credentials in secrets.toml or set them as environment variables.

- #### Run the Streamlit Application:
```
streamlit run login.py
```


## Usage:

- __Generate Persona:__ Enter a reviewer ID and click on "Generate Persona". The app will fetch the review data and display the generated persona.
- __Manage Conversations:__ Save, delete, and load conversation histories for ongoing analysis.
- __Explore Group Insights:__ Generate insights for groups based on common interests by entering a relevant query.

## Youtube Video:
https://youtu.be/FaLBxfkfEJg

## Contribute:

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. You can also open issues for bugs you've found or features you think would add value to the project.

## Contact Us:
- Hardik Sodhani (NUID: 002770306) (Linkedin: https://www.linkedin.com/in/hardiksodhani/) (Email: sodhani.h@northeastern.edu)
- Rucha Chotalia (NUID: 002711888) (Linkedin: https://www.linkedin.com/in/ruchachotalia/) (Email: chotalia.r@northeastern.edu)
