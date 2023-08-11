import json
import boto3
import uuid
import os
import requests
import tiktoken
# import Key
from boto3.dynamodb.conditions import Key, Attr


encoding = tiktoken.get_encoding("cl100k_base")
lambda_client = boto3.client('lambda')
ddb = boto3.resource('dynamodb')
convo_table = ddb.Table(os.environ['BLOCKS_DDB_TABLE'])
twin_table = ddb.Table(os.environ['TWINS_DDB_TABLE'])
user_twin_table = ddb.Table(os.environ['USER_TWINS_DDB_TABLE'])

# Import AWS X-Ray SDK
import aws_xray_sdk
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

# Patch boto3 and requests to enable them for tracing with X-Ray
patch_all()

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
    with xray_recorder.in_subsegment('Lambda Handler'):

        print(event)
        body = json.loads(event['body'])
        new_messages = body['messages']
        conversationId = body['conversationId']
        userId = body['userId']
        twinId = body['twinId']
        # check block table for existing blocks, if none, create new block
        twin_response = twin_table.get_item(Key={'twinId': twinId})
        twin_system_messages = twin_response['Item']['systemMessages']
        print(twin_system_messages)
        # get user relationship with twin
        user_twin_response = user_twin_table.get_item(Key={'twinId': twinId, 'userId': userId})
        user_relationship = user_twin_response['Item']['userRelationship']
        print(user_relationship)
        # get summarization prompt
        twin_summarization_prompt = twin_response['Item']['summarizationPrompt']
        print(twin_summarization_prompt)
        system_messages = [{
                'role': 'system',
                'content': system_message
            } for system_message in twin_system_messages]
        system_messages.append({
                'role': 'system',
                'content': user_relationship
            })
        # get system_messages token length
        response = convo_table.query(
            KeyConditionExpression=Key('conversationId').eq(conversationId),
            ScanIndexForward=False,
            Limit=1
        )

        items = response['Items']
        if len(items) > 0:
            most_recent_block = items[0]
        else:
            most_recent_block = None
        if most_recent_block is None:
            # create new block
            messages = []
            system_tokens = len(encoding.encode(' '.join([message['content'] for message in system_messages])))
            block = {
                'conversationId': conversationId, # partition key
                'blockId': 0, # sort key
                'twinId': twinId,
                'messages': messages,
                'systemMessages': system_messages,
                'totalTokens': system_tokens,
            }
            convo_table.put_item(Item=block)
            return {
                'statusCode': 200,
                'body': json.dumps({'messages': system_messages})
            }
        else:
            # get most recent block
            convo_item = most_recent_block
            blockId = convo_item['blockId']
            messages = convo_item['messages']
            totalTokens = convo_item['totalTokens']
            # count tokens of new messages
            new_tokens = len(encoding.encode(' '.join([message['content'] for message in new_messages])))
            # if new messages + old messages > n, create new block
            if new_tokens + totalTokens > int(os.environ['MAX_TOKENS']):
                print("Summarizing block")
                # summarize the block with the exception of the first 2 system messages
                content_to_summarize = '\n'.join([f"{message['role']}: {message['content']}" for message in messages])
                summarization_messages = [
                    {
                    'role': 'user',
                    'content': content_to_summarize
                },
                {
                    'role': 'system',
                    'content': twin_summarization_prompt
                }
                ]
                summarization_response = openai_completion(summarization_messages, 'gpt-3.5-turbo-16k', max_tokens=300)
                summarization = summarization_response['choices'][0]['message']['content']
                print(summarization)
                # get first 2 messages
                # append summarization to first 2 messages
                summary_message = [{'role': 'system', 'content': summarization}]
                full_messages = system_messages + summary_message + new_messages
                new_block = {
                    'conversationId': conversationId, # partition key
                    'blockId': blockId+1, # sort key
                    'twinId': twinId,
                    'messages': summary_message + new_messages,
                    'totalTokens': len(encoding.encode(' '.join([full_messages['content'] for full_messages in full_messages]))),
                    'systemMessages': system_messages
                }
                convo_table.put_item(Item=new_block)
                return {
                    'statusCode': 200,
                    'body': json.dumps({'messages': full_messages})
                }
            else:
                # append new messages to existing block
                full_messages = system_messages + messages + new_messages
                new_block = {
                    'conversationId': conversationId, # partition key
                    'blockId': blockId+1, # sort key
                    'twinId': twinId,
                    'messages': messages + new_messages,
                    'totalTokens': totalTokens + new_tokens,
                    'systemMessages': system_messages
                }
                convo_table.put_item(Item=new_block)
                return {
                    'statusCode': 200,
                    'body': json.dumps({'messages': full_messages})
                }


            



