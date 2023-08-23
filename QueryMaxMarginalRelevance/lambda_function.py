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

def max_marginal_relevance(matches, lambda_param=0.5):
    selected = []
    reranked_matches = []

    remaining_matches = matches.copy()
    while remaining_matches:
        mmr_scores = [mmr_score(match["score"], match["values"], selected, lambda_param) for match in remaining_matches]
        # Select match with the highest MMR score
        best_idx = mmr_scores.index(max(mmr_scores))
        best_match = remaining_matches.pop(best_idx)

        reranked_matches.append(best_match)
        selected.append(best_match["values"])

    return reranked_matches


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
            raise e


def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        queries = body['queries']
        metadata_filters = body['metadata_filters']
        top_n = body['top_n']
        namespace = body['namespace']
        final_set_size = body['final_set_size']
        assert top_n >= final_set_size, "final_set_size must be smaller than the number of matches returned by queries * top_n"
        print(body)
        # Accumulate matches from all queries
        full_matches = []
        for query in queries:
            print(f"Querying Pinecone with: {query}")
            response = query_pinecone(query, metadata_filters, top_n, namespace)
            try:
                matches = response['matches']
                full_matches.extend(matches)
                print(f"Found {len(matches)} matches")
                print(f"Matches: {matches}")
            except:
                raise Exception(f"Error querying Pinecone: {response}")

        # Deduplicate matches
        print("Deduplicating matches...")
        deduplicated_matches = []
        seen_ids = set()
        for match in full_matches:
            if 'id' in match:
                match_id = match['id']
                if match_id not in seen_ids:
                    deduplicated_matches.append(match)
                    seen_ids.add(match_id)
            else:
                print(f"Warning: Match missing 'id' key in metadata: {match}")
        # Sort matches by score
        print("Sorting matches by score...")
        matches = sorted(deduplicated_matches, key=lambda k: k['score'], reverse=True)

        # print response w/o/ values keys
        print("Before MMR rerank:")
        print("{:<50} {:<10}".format("Content", "Score"))
        print("-" * 60)
        for match in matches:
            print("{:<50} {:<10.2f}".format(match['metadata']['content'], match['score']))
        print(f"Total matches: {len(matches)}")
        # Compute MMR scores
        print("Computing MMR scores...")
        lambda_param = float(os.environ.get('LAMBDA_PARAM', 0.5))
        reranked_matches = max_marginal_relevance(matches, lambda_param)
        print(f"MMR scores: {reranked_matches}")
        print(f"Total MMR scores: {len(reranked_matches)}")

        # After MMR rerank
        print("After MMR rerank:")
        print("{:<50} {:<10}".format("Content", "Score"))
        print("-" * 60)
        for match in reranked_matches:
            print("{:<50} {:<10.2f}".format(match['metadata']['content'], match['score']))
        # remove values from response
        for match in matches:
            del match['values']

        return {
            'statusCode': 200,
            'body': json.dumps(matches[:final_set_size])
        }

    except Exception as e:
        # A general error handler; refine as needed.
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }

    