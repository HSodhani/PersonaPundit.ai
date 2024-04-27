import streamlit as st
import boto3
import json
import bcrypt
from botocore.exceptions import NoCredentialsError, ClientError
from PersonaPundit_app import main as run_persona_pundit

# Initialize a boto3 client
s3_client = boto3.client('s3')
aws_access_key_id = st.secrets["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
def fetch_user_data(bucket, key):
    """ Fetch user data from an S3 bucket. """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        user_data = json.loads(response['Body'].read().decode('utf-8'))
        return user_data
    except NoCredentialsError:
        st.error("AWS credentials not configured properly.")
        return {}
    except ClientError as e:
        st.error(f"Failed to fetch user data: {str(e)}")
        return {}

def verify_password(stored_hash, provided_password):
    """ Verify the provided password against the stored hash. """
    return bcrypt.checkpw(provided_password.encode(), stored_hash.encode())

def show_login_page():
    """ Display the login form and handle authentication. """
    st.title("Login to PersonaPundit.ai")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")
        if submit_button:
            # Replace 'your-bucket-name' with your actual bucket name
            users = fetch_user_data('personapundit', 'users.json')
            user_info = users.get(username)
            if user_info and verify_password(user_info['password'], password):
                st.session_state['logged_in'] = True
                st.session_state['user_role'] = user_info['role']
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    run_persona_pundit()
else:
    show_login_page()