import json
import boto3
import uuid
import os
import requests
from pypdf import PdfReader
from io import BytesIO

lambda_client = boto3.client('lambda')
s3 = boto3.client('s3')


def invoke_embedding_lambda(text):
    event = {'body': json.dumps({'query': text})}
    response = lambda_client.invoke(
        FunctionName=os.environ['EMBEDDING_LAMBDA'],
        InvocationType='RequestResponse',
        Payload=json.dumps(event)
    )
    print(response)
    # Parse the response from bytes to string
    # Extract the payload from the response
    payload = response['Payload']

    # Read the payload and convert it from bytes to a string
    payload_string = payload.read().decode('utf-8')

    # Parse the payload string into a JSON object
    payload_json = json.loads(payload_string)

    # Now you can access the 'body' key from the payload JSON object
    body_json = json.loads(payload_json['body'])

    # Get the 'vector' from the 'body' JSON object
    vector = body_json['vector']

    return vector


def add_to_pinecone(input: str, metadata: dict, namespace: str='default'):
    url = os.environ['PINECONE_URL']+"/vectors/upsert"

    payload = {
        "vectors":[{
            "id": str(uuid.uuid4()),
            "values": invoke_embedding_lambda(input),
            "metadata": metadata,
        }],
        "namespace": namespace
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Api-Key": os.environ['PINECONE_KEY'],
    }

    response = requests.post(url, json=payload, headers=headers)
    return response

def pdf_splitter(file_contents: bytes):
    # split pdf into paragraphs
    reader = PdfReader(BytesIO(file_contents))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    sentences = text.split('.')
    paragraphs = []
    paragraph = ''
    for sentence in sentences:
        if len(paragraph.split(' ')) > 300:
            paragraphs.append(paragraph)
            paragraph = ''
        paragraph += sentence + '.'
    return paragraphs
    

def text_splitter(text: str):
    # split text by sentences, then join sentences until the length is ~300 words
    sentences = text.split('.')
    paragraphs = []
    paragraph = ''
    for sentence in sentences:
        if len(paragraph.split(' ')) > 300:
            paragraphs.append(paragraph)
            paragraph = ''
        paragraph += sentence + '.'
    return paragraphs

def lambda_handler(event, context):
    # Get s3 bucket&key, tenantid, twinid from event
    body = json.loads(event['body'])
    bucket = body['bucket']
    key = body['key']
    tenant_id = body['tenantId']
    twin_id = body['twinId']

    # Ensure file is .txt or .pdf
    if key[-4:] != '.txt' and key[-4:] != '.pdf':
        raise Exception('File must be .txt or .pdf')
    # get file from S3
    s3_object = s3.get_object(Bucket=bucket, Key=key)
    # get file contents, parse streamingBody
    file_contents = s3_object['Body'].read()
    # check file type, .pdf, .txt, .docx, .doc
    if key[-4:] == '.pdf':
        paragraphs = pdf_splitter(file_contents)
        print("Paragraphs: ", paragraphs)
    elif key[-4:] == '.txt' or key[-3:] == '.md':
        paragraphs = text_splitter(file_contents)
        print("Paragraphs: ", paragraphs)
    else: 
        raise Exception('File must be .pdf, .txt, .md')
    for paragraph in paragraphs:
        paragraph_id = str(uuid.uuid4())
        # get namespace
        namespace = f'{twin_id}'
        # add type and source
        _type = 'applicable_idea'
        source = key
        # add paragraph to pinecone index
        metadata = {
                'namespace': namespace,
                'type': _type,
                'source': source,
                'id': paragraph_id,
                'content': paragraph,
                }
        resp = add_to_pinecone(paragraph, metadata, namespace)
        if resp.status_code != 200:
            print(resp)
            raise Exception('Adding to pinecone failed')
    return {
        'statusCode': 200,
        'body': json.dumps(f'Successfully added audio file: {key} from tenant: {tenant_id} for twin: {twin_id} to pinecone index')
    }
