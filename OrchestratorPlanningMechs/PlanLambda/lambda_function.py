import json
import os
import time
import boto3
import requests

MAX_RETRIES = 5
RATE_LIMIT_DURATION = 61  # seconds

STEPS_TABLE = os.environ['STEPS_TABLE']
TWIN_TABLE = os.environ['TWINS_TABLE']

step_table = boto3.resource('dynamodb').Table(STEPS_TABLE)
twin_table = boto3.resource('dynamodb').Table(TWIN_TABLE)
lambda_client = boto3.client('lambda')
prompt_table = boto3.resource('dynamodb').Table(os.environ['PROMPT_TABLE'])

def openai_completion(messages: list[dict], model: str, max_tokens: int=500, temperature: float=0.0) -> dict:
    """
    Make a completion request to OpenAI's GPT-3.

    Parameters:
    - messages (list[dict]): Messages for the conversation with the model.
    - model (str): The model name (e.g. "text-davinci-002")
    - max_tokens (int): The maximum number of tokens to be returned.
    - temperature (float): Sampling temperature.

    Returns:
    - dict: The model's response message.
    """
    
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
    
    retries = 0
    while retries < MAX_RETRIES:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]
        
        if response.status_code == 429 or response.status_code == 502:  # HTTP status for rate limit exceeded
            print(f"Rate limit exceeded, waiting for {RATE_LIMIT_DURATION//60} minutes")
            time.sleep(RATE_LIMIT_DURATION)
            retries += 1
        else:
            raise Exception("OpenAI Completion failed: ", response.json())
    
    raise Exception(f"Failed after {MAX_RETRIES} retries.")


def validate_input(input: dict, expected_input: list[str]):
    for input_key in input.keys():
        if input_key not in expected_input:
            raise Exception(f"Invalid input key: {input_key}")
def query_mmr(text: list[str], metadata_filters: dict, top_n: int, namespace: str='default'):

    # query lambda function
    event = {'body': json.dumps({'queries': text, 'metadata_filters': metadata_filters, 'top_n': top_n, 'namespace': namespace, 'final_set_size': top_n})}
    response = lambda_client.invoke(
        FunctionName=os.environ['MMR_LAMBDA'],
        InvocationType='RequestResponse',
        Payload=json.dumps(event)
    )
    # Parse the response from bytes to string
    # Extract the payload from the response
    payload = response['Payload']

    # Read the payload and convert it from bytes to a string
    payload_string = payload.read().decode('utf-8')

    # Parse the payload string into a JSON object
    payload_json = json.loads(payload_string)
    # Now you can access the 'body' key from the payload JSON object
    body_json = json.loads(payload_json['body'])
    return body_json # list of dict matches

def validate_input(input: dict, expected_input: list[str]):
    for input_key in input.keys():
        if input_key not in expected_input:
            raise Exception(f"Invalid input key: {input_key}")

def lambda_handler(event, context):
    # event is an SQS trigger message
    for record in event['Records']:
        body = json.loads(record['body'])
        twin_id = body['twinId']
        conversation_id = body['conversationId']
        steps = body['steps']
        # get twin definition and prompt template
        twin_definition = twin_table.get_item(Key={'twinId': twin_id})['Item']['twinDefinition']
        query_prompt_template = prompt_table.get_item(Key={'promptId': os.environ['QUERY_PROMPT_TEMPLATE']})['Item']['value']
        
        steps_str = ''
        for step in steps:
            step_definition = step['step_definition']
            step_observation = step['step_observation']
            step_is_finished = step['is_finished']
            step_index = step['step_index']
            step_str = f"Step {step_index}: {step_definition}\n"
            if len(step_observation) > 0:
                step_str += f"Observation: {step_observation}\n"
            if step_is_finished:
                step_str += "Step is finished\n"
            else:
                step_str += "Step is not finished\n"
            steps_str += step_str
        print("Steps: ", steps_str)

        query_prompt = query_prompt_template.format(**{'steps': steps_str})
        print("Query prompt: ", query_prompt)
        # get query response
        q_msgs = [
            {'role':'system', 'content':twin_definition},
            {'role':'user', 'content':query_prompt}
        ]
        query_response = openai_completion(q_msgs, os.environ['QUERY_MODEL'], int(os.environ['QUERY_MAX_TOK']), float(os.environ['QUERY_TEMPERATURE']))
        print("Query response: ", query_response)
        # get ideas
        queries_list = query_response['content'].split('\n')
        # query mmr
        results = query_mmr(queries_list, {'_type': "applicable_idea"}, int(os.environ['TOP_N']), namespace=twin_id)
        # get ideas
        ideas = [result['metadata']['content'] for result in results]
        ideas_str = '/n----/n'.join(ideas)
        print("Ideas: ", ideas_str)
        # get prompt 
        ideas_prompt_template = prompt_table.get_item(Key={'promptId': os.environ['IDEAS_PROMPT_TEMPLATE']})['Item']['value']
        ideas_prompt = ideas_prompt_template.format(**{'ideas': ideas_str, 'steps': steps_str, 'start_index': steps[-1]['step_index']})
        # get ideas response
        p_msgs = [
            {'role':'system', 'content':twin_definition},
            {'role':'user', 'content':ideas_prompt}
        ]
        steps_response = openai_completion(p_msgs, os.environ['IDEAS_MODEL'], int(os.environ['IDEAS_MAX_TOK']))
        print("Steps response: ", steps_response)

        try:
            new_steps = json.loads(steps_response['content'])
        except:
            raise Exception("Steps response not valid JSON")
        new_steps_list = []
        for i, new_step in enumerate(new_steps):
            step_definition = new_step
            step_index = len(steps) + i
            new_steps_list.append({
                'step_definition': step_definition,
                'step_observation': [],
                'is_finished': False,
                'step_index': step_index
            })
        # update steps item in ddb
        print("Updating steps item in ddb")
        start = time.time()
        steps_item = step_table.get_item(
            Key={'twinId': twin_id,
                'conversationId': conversation_id})['Item']
        steps_item['steps'] += new_steps_list
        step_table.put_item(Item=steps_item)
        print(f"Updated steps item in ddb in {1000*(time.time() - start)} milliseconds")
        




        
        