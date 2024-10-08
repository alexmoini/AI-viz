import json
import boto3
import uuid
import os
import requests

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


def transcribe(file_path: str):
    url = "https://api.deepgram.com/v1/listen?model=general&tier=nova&version=latest&punctuate=true&diarize=false&multichannel=false&paragraphs=true"

    payload = {"url": file_path}
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.json()

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

    # Ensure file is .wav or .mp3
    if key[-4:] != '.wav' and key[-4:] != '.mp3':
        raise Exception('File must be .wav or .mp3')
    # get presigned url
    url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': bucket,
                'Key': key
                }
            )
    # Transcribe file
    response = transcribe(url)
    print(response)
    # Split into paragraphs of ~300 words, overlapping by a sentence
    transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
    split_text = text_splitter(transcript)
    # Assign metadata to paragraphs and add to pinecone index
    for paragraph in split_text:
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
