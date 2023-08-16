import requests
import json
import os
import boto3
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all
import uuid

patch_all()

TOPIC_PROMPT_TABLE = os.environ['TOPIC_PROMPT_TABLE']
dynamodb = boto3.resource('dynamodb')
lambda_client = boto3.client('lambda')

def invoke_embedding_lambda(text):
    with xray_recorder.in_subsegment('Invoke Embedding Lambda'):
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

def openai_completion(messages: list[dict], model: str, max_tokens: int=500, temperature: float=0.0) -> dict:
    with xray_recorder.in_subsegment('OpenAI Completion'):
        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}" 
        }
        print(messages)
        data = {
            "model": model, 
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = requests.post(url, headers=headers, json=data)
        print(response.json())
        return response.json()


def add_to_pinecone(input: str, metadata: dict, namespace: str='default'):
    with xray_recorder.in_subsegment('Add to Pinecone'):
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

def query_pinecone(text: str, metadata_filters: dict, top_n: int, namespace: str='default'):
    with xray_recorder.in_subsegment('Query Pinecone'):
        url = os.environ['PINECONE_URL']+"/query"

        payload = {
            "vector": invoke_embedding_lambda(text),
            "filter": metadata_filters,
            "topK": top_n,
            "namespace": namespace,
            "includeMetadata": True
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Api-Key": os.environ['PINECONE_KEY'],
        }
        print(payload)
        response = requests.post(url, json=payload, headers=headers)
        return response.json()

def lambda_handler(event, context):
    """
    1. Input is content, twinId
    2. Add concepts to content
    3. Loop thru topics & query pinecone to find similar concepts
    4. If similar concepts exist, switch out the concept with the topic and add the inital concept to the subconcepts list
        4a. If similar concepts don't exist, create a new concept and add to pinecone
    """
    body = json.loads(event['body'])
    content = body['content']
    twinId = body['twinId']
    try:
        prev_topics = body['prevTopics']
    except:
        prev_topics = ""

    topics_prompt = (dynamodb.Table(TOPIC_PROMPT_TABLE)).get_item(
        TableName=TOPIC_PROMPT_TABLE,
        Key={
            'promptId': str(os.environ['TOPIC_PROMPT_KEY']),
        }
    )
    topics_prompt = topics_prompt['Item']['value']

    # Add concepts to content
    message = [
        {
            "role": "user",
            "content": topics_prompt.format(prev_topics, content)
        }
    ]
    topics = openai_completion(message, os.environ['OPENAI_MODEL'], max_tokens=100, temperature=0.0)
    topics = topics['choices'][0]['message']['content'].split('\n')
    topics = [topic.strip() for topic in topics if topic.strip() != '']

    # Loop thru topics & query pinecone to find similar concepts
    final_topics = []
    for topic in topics:
        # Query pinecone to find similar concepts
        response = query_pinecone(topic, {'twinId':twinId}, 1)
        print(response)
        if len(response['matches']) > 0:
            # If similar concepts exist, switch out the concept with the topic and add the inital concept to the subconcepts list, check over a threshold
            if response['matches'][0]['score'] > float(os.environ['THRESHOLD']):
                final_topics.append(response['matches'][0]['metadata']['concept'])
            else:
                final_topics.append(topic)
        else:
            # If similar concepts don't exist, create a new concept and add to pinecone
            metadata = {
                'twinId': twinId,
                'concept': topic,
                'subconcepts': []
            }
            add_to_pinecone(topic, metadata)
            final_topics.append(topic)
    
    # ensure topics are unique
    final_topics = list(set(final_topics))
    print(final_topics)

    return {
        'statusCode': 200,
        'body': json.dumps(final_topics)
    }


    