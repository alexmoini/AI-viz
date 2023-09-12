import requests
import boto3
import os
import json
import time

SQS_QUEUE_URL = os.environ['SQS_QUEUE_URL']
STEP_TABLE_NAME = os.environ['STEP_TABLE_NAME']
TWIN_TABLE = os.environ['TWINS_TABLE']

sqs = boto3.client('sqs')
step_table = boto3.resource('dynamodb').Table(STEP_TABLE_NAME)
twin_table = boto3.resource('dynamodb').Table(TWIN_TABLE)

def lambda_handler(event, context):
    body = json.loads(event['body'])
    twin_id = body['twinId']
    conversation_id = body['conversationId']
    step_index = body['stepIndex']
    step_observation = body['stepObservation']
    step_is_finished = body['stepIsFinished']

    # get step item from ddb
    try:
        steps_item = step_table.get_item(
            Key={'twinId': twin_id,
                'conversationId': conversation_id})['Item']
    except:
        raise Exception("Step item not found, be sure to get steps before updating them")
    # update step
    steps_item['steps'][step_index]['step_observation'].append(step_observation)
    if 'T' in step_is_finished:
        step_is_finished = True
    else:
        step_is_finished = False
    steps_item['steps'][step_index]['is_finished'] = step_is_finished
    # update step item in ddb
    steps = steps_item['steps']
    if step_is_finished:
        # check number of unfinished steps left
        unfinished_steps = 0
        if steps[-1]['is_finished'] == False and steps[-2]['is_finished'] == True:
            # trigger planning module
            payload = {
                "twinId": twin_id,
                "conversationId": conversation_id,
                "steps": steps_item['steps']
            }
            # send to sqs
            response = sqs.send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps(payload)
            )
            print(f"Sent message to SQS: {response['MessageId']}")
        elif steps[-1]['is_finished'] == True:
            # all steps are finished
            print("All steps are finished, waiting for steps table update")
        else:
            # there are unfinished steps left
            print("There are more than 1 unfinished steps left")
    else:
        # step is not finished
        print("Step is not finished")
    # update steps item in ddb
    step_table.put_item(Item=steps_item)
    return {
        'statusCode': 200,
        'body': json.dumps(f'Successfully updated step {step_index} for twin: {twin_id} and conversation: {conversation_id}')
    }


