import json
import boto3
import os
import time
from sentence_transformers import SentenceTransformer

os.environ['XDG_CACHE_HOME'] = '/tmp/.cache/'
os.environ['HF_HOME'] = '/tmp/.cache/huggingface/'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/.cache/huggingface/'
os.environ['HF_DATASETS_CACHE'] = '/tmp/.cache/huggingface/'
# tst
EMBEDDING_MODEL = os.environ['EMBEDDING_MODEL']

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
instruction = os.environ['INSTRUCTION']

def lambda_handler(event, context):
    payload = json.loads(event['body'])
    query = payload['query']
    instruction_query = instruction + ': ' + query
    start = time.time()
    vector = embedding_model.encode(instruction_query)
    list_vector = vector.tolist()
    end = time.time()
    print(f'Embedding took ({(end-start)*1000}) milliseconds')
    response = {
        'vector': list_vector
    }
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }
