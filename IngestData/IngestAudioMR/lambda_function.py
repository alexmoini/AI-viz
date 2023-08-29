import json
import boto3
import uuid
import os
import requests
import asyncio
import aiohttp

lambda_client = boto3.client('lambda')
s3 = boto3.client('s3')
ddb = boto3.resource('dynamodb')
prompt_table = ddb.Table(os.environ['PROMPT_TABLE'])

async def async_openai_completion(messages: list, model: str, max_tokens: int=500, temperature: float=0.0) -> dict:
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}" 
    }

    data = {
        "model": model, 
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as resp:
            result = await resp.json()
            try:
                return result['choices'][0]['message']
            except:
                raise Exception("OpenAI Completion failed: ", result)

async def create_topics(docs, prompt_template):
    tasks = []

    for doc in docs:
        doc_dict = {'document': doc}
        prompt = prompt_template.format(**doc_dict)
        tasks.append(async_openai_completion([{'role':'user', 'content':prompt}], "gpt-3.5-turbo", 250, 0.0))

    results = await asyncio.gather(*tasks)
    return results

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

def break_down_with_overlap(corpus: str, block_size: int, overlap_size: int):
    """
    corpus - the corpus to break down
    block_size - number of words per block
    overlap_size - number of sentences to overlap between blocks
    """
    # split corpus into sentences
    sentences = corpus.split('.')
    # split sentences into blocks
    blocks = []
    block = ''
    for i, sentence in enumerate(sentences):
        if len(block.split(' ')) > block_size:
            blocks.append(block)
            # set the block to the last overlap_size sentences
            if overlap_size > 0:
                if i+1 < overlap_size:
                    # do not include overlap if it is the first block
                    block = ''
                else:
                    block = ' '.join(sentences[i-overlap_size:i])
            
        block += sentence + '.'
    return blocks
def validate_input(input: dict, expected_input: list):
    for input_key in input.keys():
        if input_key not in expected_input:
            raise Exception(f"Invalid input key: {input_key}")

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
    blocks = break_down_with_overlap(transcript, int(os.environ['BLOCK_SIZE']), int(os.environ['OVERLAP_SIZE']))
    # Create topics for each block
    try:
        response = prompt_table.get_item(
            Key={
                'promptId': os.environ['PROMPT_TEMPLATE_ID']
            }
        )
        prompt_template = response['Item']['value']
        input_validation = response['Item']['inputValidation']
        # test input validation
        doc_dict = {'document': 'document'}
        validate_input(doc_dict, input_validation)
    except:
        raise Exception("Prompt template not found")
    topics = asyncio.run(create_topics(blocks, prompt_template))
    # Add topics to pinecone
    for topic in topics:
        topic_id = str(uuid.uuid4())
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
                'id': topic_id,
                'content': topic['content'],
                }
        resp = add_to_pinecone(topic['content'], metadata, namespace)
        if resp.status_code != 200:
            print(resp)
            raise Exception('Adding to pinecone failed')
    return {
        'statusCode': 200,
        'body': json.dumps(f'Successfully added audio file: {key} from tenant: {tenant_id} for twin: {twin_id} to pinecone index')
    }
