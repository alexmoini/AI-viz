import requests
import json
import os

"""
Example curl:

curl -i -X POST https://YOUR_INDEX-YOUR_PROJECT.svc.YOUR_ENVIRONMENT.pinecone.io/vectors/delete \
  -H 'Api-Key: YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{
    "filter": {"genre": {"$in": ["comedy", "documentary", "drama"]}}
  }'
"""

def delete_from_pinecone_using_metadata(filter: dict, namespace: str='default'):
    url = os.environ['PINECONE_URL']+"/vectors/delete"

    payload = {
        "filter": filter,
        "namespace": namespace
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Api-Key": os.environ['PINECONE_KEY'],
    }

    response = requests.post(url, json=payload, headers=headers)
    return response

def lambda_handler(event, context):
    # Parse the payload string into a JSON object
    payload_json = event

    # Now you can access the 'body' key from the payload JSON object
    body_json = json.loads(payload_json['body'])

    # Get the 'filter' from the 'body' JSON object
    document_key = body_json['documentKey']
    tenant_id = body_json['tenantId']
    twin_id = body_json['twinId']
    namespace = f'{tenant_id}-{twin_id}'
    filter = {
        "source": document_key
    }
    return delete_from_pinecone_using_metadata(filter, namespace)