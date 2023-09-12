import requests
import boto3
import os
import json 
import time

STEPS_TABLE = os.environ['STEPS_TABLE']
TWIN_TABLE = os.environ['TWINS_TABLE']

step_table = boto3.resource('dynamodb').Table(STEPS_TABLE)
twin_table = boto3.resource('dynamodb').Table(TWIN_TABLE)

def lambda_handler(event, context):
    body = json.loads(event['body'])
    twin_id = body['twinId']
    conversation_id = body['conversationId']

    # get step item from ddb
    try:
        steps_item = step_table.get_item(
            Key={'twinId': twin_id,
                'conversationId': conversation_id})['Item']
    except:
        # New conversation, create new steps item
        steps_item = {
            'twinId': twin_id,
            'conversationId': conversation_id,
            'steps': []
        }
        initial_steps = twin_table.get_item(Key={'twinId': twin_id})['Item']['initialSteps']
        for i, step in enumerate(initial_steps):
            steps_item['steps'].append({
                'step_definition': step,
                'step_observation': [],
                'is_finished': False,
                'step_index': i
            })
        step_table.put_item(Item=steps_item)
        print("New conversation, created new steps item. Step: ", steps_item['steps'][0]['step_definition'])
        return {'statusCode': 200,
                'body': json.dumps(steps_item['steps'][0]['step_definition'])}
    # get steps
    steps = steps_item['steps']
    # get the first unfinished step
    for step in steps:
        if step['is_finished'] == False:
            print("Continuing conversation. Step: ", step['step_definition'])
            return {'statusCode': 200,
                    'body': json.dumps(step['step_definition'])}
    # all steps are finished
    # wait for steps table update
    print("All steps are finished, waiting for steps table update")
    while True:
        steps_item = step_table.get_item(
            Key={'twinId': twin_id,
                'conversationId': conversation_id})['Item']
        steps = steps_item['steps']
        for step in steps:
            if step['is_finished'] == False:
                print("Got new steps, continuing conversation. Step: ", step['step_definition'])
                return {'statusCode': 200,
                        'body': json.dumps(step['step_definition'])}
        time.sleep(1)
    
    return {
        'statusCode': 200,
        'body': json.dumps('Something really weird happened')}
