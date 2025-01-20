import os
import json
import uuid
from email import policy
from email.parser import BytesParser
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import streamlit as st

client = OpenAI(
    api_key="API_KEY",
)

pc = Pinecone(api_key="PC_API_KEY")


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

# 2. Process Attachments and Extract Text
def extract_text_from_attachments(attachments):
    extracted_texts = {}
    for attachment in attachments:
        file_path, file_type = attachment
        try:
            if file_type == 'pdf':
                reader = PdfReader(file_path)
                extracted_texts[file_path] = "\n".join(page.extract_text() for page in reader.pages)
            elif file_type == 'docx':
                doc = Document(file_path)
                extracted_texts[file_path] = "\n".join(para.text for para in doc.paragraphs)
            else:
                extracted_texts[file_path] = "Unsupported file type"
        except Exception as e:
            extracted_texts[file_path] = f"Error extracting text: {str(e)}"
    return extracted_texts

# 3. Upload Email and Attachments to Pinecone
def upload_to_pinecone(index, model, email_data, attachment_texts=None):
    # Combine email and attachment text for embedding
    if attachment_texts:
        combined_text = email_data['email_text'] + "\n\n" + "\n\n".join([
            f"Attachment: {attachment}\n\n{text}"
            for attachment, text in attachment_texts.items()
        ])
    else:
        combined_text = email_data['email_text']

    email_embedding = model.encode(combined_text)

    # Generate a unique ID for the record
    record_id = str(uuid.uuid4())

    # Upload the combined data as a single record
    index.upsert(vectors=[
        (record_id, email_embedding.tolist(), {
            "type": "email_with_attachments",
            "content": combined_text
        })
    ])

    return record_id

# 4. Query Against a Specific Record
def query_pinecone(index, model, record_id, question):
    query_embedding = model.encode(question).tolist()
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    # Retrieve the relevant record
    record = next((match for match in results['matches'] if match['id'] == record_id), None)

    if not record:
        return {"error": "No relevant records found for the given ID."}

    content = record['metadata']['content']

    # Use OpenAI GPT to generate an answer
    input_prompt = (
        f"You are helping underwriters to make decision about the policy. Please provide answers in a precise manner with helpful insights.\n"
        f"Context: {content}\n\nQuestion: {question}\nAnswer:"
    )
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. If no information is available for the question asked, respond with 'No."},
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

    # No attachments are processed in this version, so pass `None`
    attachment_texts = None

    # Upload the parsed email to Pinecone
    record_id = upload_to_pinecone(index, model, email_data, attachment_texts)

    # Show success message and record ID
    st.success(f"Email uploaded successfully! Record ID: {record_id}")

# Query Section
if selected_record:
    user_query = st.text_input("Ask a question about the selected email:")
    if st.button("Submit Query"):
        answer = query_pinecone(index, model, selected_record, user_query)
        st.write("Answer:", answer)