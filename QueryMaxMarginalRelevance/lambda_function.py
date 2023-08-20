import json
import boto3
import os
import requests
import math

lambda_client = boto3.client('lambda')

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

def cosine_similarity(vec_a, vec_b):
    """
    Compute the cosine similarity between two vectors.
    """
    dot_product = sum(p*q for p,q in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum([val**2 for val in vec_a]))
    magnitude_b = math.sqrt(sum([val**2 for val in vec_b]))
    if not magnitude_a or not magnitude_b:
        return 0
    return dot_product / (magnitude_a * magnitude_b)

def mmr_score(candidate_score, candidate_vector, selected_vectors, lambda_param=0.5):
    """
    Compute the MMR score of a candidate vector.
    """
    # Relevance score from Pinecone
    rel_score = candidate_score

    # If no vectors are selected, similarity score is 0
    if not selected_vectors:
        max_sim = 0
    else:
        sim_scores = [cosine_similarity(candidate_vector, vec) for vec in selected_vectors]
        max_sim = max(sim_scores)

    mmr = lambda_param * rel_score - (1 - lambda_param) * max_sim
    return mmr

def max_marginal_relevance(pinecone_response, lambda_param=0.5):
    """
    Compute MMR for each vector in the set.
    """
    selected = []
    mmr_scores = []

    for match in pinecone_response["matches"]:
        score = mmr_score(match["score"], match["values"], selected, lambda_param)
        mmr_scores.append(score)
        selected.append(match["values"])

    return mmr_scores

def query_pinecone(text: str, metadata_filters: dict, top_n: int, namespace: str = 'default'):
    with xray_recorder.in_subsegment('Query Pinecone'):
        url = os.environ['PINECONE_URL'] + "/query"

        try:
            payload = {
                "vector": invoke_embedding_lambda(text),
                "filter": metadata_filters,
                "topK": top_n,
                "namespace": namespace,
                "includeMetadata": True,
                "includeValues": True,
            }
            
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "Api-Key": os.environ['PINECONE_KEY'],
            }

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            # Log the error and potentially send a custom error response or re-throw the exception.
            print(f"Error querying Pinecone: {e}")
            raise

def rerank_based_on_mmr(response, mmr_scores):
    # Pair each match with its MMR score
    paired = list(zip(response["matches"], mmr_scores))
    # Sort matches based on MMR scores (higher is better)
    sorted_paired = sorted(paired, key=lambda x: x[1], reverse=True)

    # Check how much reranking has happened
    rerank_changes = sum(1 for i, (match, _) in enumerate(sorted_paired) if response["matches"][i] != match)

    # Extract the sorted matches
    reranked_matches = [match for match, _ in sorted_paired]
    
    return reranked_matches, rerank_changes


def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        text = body['text']
        metadata_filters = body['metadata_filters']
        top_n = body['top_n']
        namespace = body['namespace']
        print(body)

        response = query_pinecone(text, metadata_filters, top_n, namespace)
        # print response w/o/ values keys
        print("Before MMR rerank:")
        print("{:<50} {:<10}".format("Content", "Score"))
        print("-" * 60)
        for match in response['matches']:
            print("{:<50} {:<10.2f}".format(match['metadata']['content'], match['score']))

        # Compute MMR scores
                # Compute MMR scores
        lambda_param = float(os.environ.get('LAMBDA_PARAM', 0.5))
        mmr_scores = max_marginal_relevance(response, lambda_param)

        # Rerank based on MMR and get the count of matches that changed their position
        reranked_matches, rerank_changes = rerank_based_on_mmr(response, mmr_scores)
        response["matches"] = reranked_matches

        print(f"Total matches that changed position due to MMR: {rerank_changes}")

        # After MMR rerank
        print("After MMR rerank:")
        print("{:<50} {:<10}".format("Content", "Score"))
        print("-" * 60)
        for match, score in zip(response["matches"], mmr_scores):
            print("{:<50} {:<10.2f}".format(match['metadata']['content'], score))
        # remove values from response
        for match in response['matches']:
            del match['values']

        return {
            'statusCode': 200,
            'body': json.dumps(response)
        }

    except Exception as e:
        # A general error handler; refine as needed.
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }

    