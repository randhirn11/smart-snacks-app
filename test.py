import os
import json
import uuid
from email import policy
from email.parser import BytesParser
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pinecone import Pinecone
import streamlit as st

openai = st.secrets['OA_API_KEY']
pc_pinecone = st.secrets['PC_API_KEY']

client = OpenAI(
    api_key=openai,
)

pc = Pinecone(api_key=pc_pinecone)


# Connect to the index
index_name = "appetite1"
index = pc.Index(index_name)


# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


# 1. Parse Email into Proper Formatted Text
def parse_email(email_file):
    email = BytesParser(policy=policy.default).parse(email_file)
    subject = email['subject'] or ''
    sender = email['from'] or ''
    recipient = email['to'] or ''
    body = ""
    for part in email.walk():
        if part.get_content_type() == 'text/plain':
            body += part.get_payload(decode=True).decode(part.get_content_charset(), errors="replace")
    email_text = f"Subject: {subject}\nFrom: {sender}\nTo: {recipient}\n\n{' '.join(body.split())}"
    return {'email_text': email_text.strip(), 'email_name': subject or 'Unnamed Email'}


# 3. Upload Email to Pinecone
def upload_to_pinecone(index, model, email_data):
    email_embedding = model.encode(email_data['email_text'])

    # Generate a unique ID for the record
    record_id = str(uuid.uuid4())

    # Upload the combined data as a single record
    index.upsert(vectors=[
        (record_id, email_embedding.tolist(), {
            "type": "email",
            "content": email_data['email_text']
        })
    ])
    print(f"Record uploaded with ID: {record_id}")
    return record_id


# 4. Query Directly by Record ID
def query_pinecone(index, record_id, question):
    # Fetch the record directly by its ID
    result = index.fetch(ids=[record_id])

    if not result or record_id not in result['vectors']:
        return {"error": "No relevant records found for the given ID."}

    content = result['vectors'][record_id]['metadata']['content']

    # Use OpenAI GPT to generate an answer
    input_prompt = (
        f"You are helping underwriters to make decision about the policy. Please provide answers in a precise manner with helpful insights.\n"
        f"Context: {content}\n\nQuestion: {question}\nAnswer:"
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. If no information is available for the question asked, respond with 'No.'"},
            {"role": "user", "content": input_prompt}
        ],
        max_tokens=200,
        temperature=0.0
    )
    answer = response.choices[0].message.content.strip()

    # Determine the source of the answer (email or attachment)
    source = "email"
    if "Attachment:" in content and answer in content:
        source = "attachment"

    return {
        "record_id": record_id,
        "question": question,
        "source": source,
        "answer": answer
    }


# Streamlit App
st.title("Project Smart Snacks!!")
st.sidebar.title("Uploaded Records")

# Show uploaded records
results = index.query(vector=[0] * 384, top_k=100, include_metadata=True)  # Dummy query vector
record_dict = {
    record['id']: record['metadata'].get('email_name', 'Unnamed Record')
    for record in results.get('matches', [])
}

selected_record = st.sidebar.selectbox("Select a Record", options=list(record_dict.keys()), format_func=lambda x: record_dict[x])


# File Upload Section
uploaded_file = st.file_uploader("Upload an Email File (.eml)", type=["eml"])
if uploaded_file is not None:
    # Parse the uploaded file
    email_data = parse_email(uploaded_file)

    # Upload the parsed email to Pinecone
    record_id = upload_to_pinecone(index, model, email_data)

    st.session_state.record_id = record_id

    # Show success message and record ID
    st.success(f"Email uploaded successfully! Record ID: {record_id}")


# Query Section
user_query = st.text_input("Ask a question about the uploaded email:")
if st.button("Submit Query"):
    if 'record_id' in st.session_state:  # Ensure the record_id is defined in session state
        record_id = st.session_state.record_id
        result = query_pinecone(index, record_id, user_query)
        if "error" in result:
            st.error(result["error"])
        else:
            st.write("Answer:", result)
    else:
        st.error("No record uploaded yet. Please upload an email first.")
