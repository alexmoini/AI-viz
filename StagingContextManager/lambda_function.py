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
stage_blocks_table = ddb.Table(os.environ['STAGE_BLOCKS_DDB_TABLE'])
stage_twin_table = ddb.Table(os.environ['STAGE_TWINS_DDB_TABLE'])
user_twin_table = ddb.Table(os.environ['USER_TWINS_DDB_TABLE'])
prompt_template_table = ddb.Table(os.environ['PROMPT_TEMPLATE_DDB_TABLE'])

# Import AWS X-Ray SDK
import aws_xray_sdk
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

# Patch boto3 and requests to enable them for tracing with X-Ray
patch_all()

def query_mmr(text: str, metadata_filters: dict, top_n: int, namespace: str='default'):
    with xray_recorder.in_subsegment('Query MMR'):
        # query lambda function
        event = {'body': json.dumps({'query': text, 'metadata_filters': metadata_filters, 'top_n': top_n, 'namespace': namespace})}
        response = lambda_client.invoke(
            FunctionName=os.environ['MMR_LAMBDA'],
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

        return body_json # list of dict matches

def validate_input(input: dict, expected_input: list[str]):
    for input_key in input.keys():
        if input_key not in expected_input:
            raise Exception(f"Invalid input key: {input_key}")

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
        try:
            return json.loads(response.json()["choices"][0]["message"])
        except:
            raise Exception("OpenAI Completion failed: ", response.json())

def openai_functions_only_completion(messages: list[dict], functions: list[dict], function_name: str , model: str, max_tokens: int=500, temperature: float=0.0) -> dict:
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
            "functions": functions
            "function_calling": {"name": function_name}}
        response = requests.post(url, headers=headers, json=data)
        try:
            return json.loads(response.json()["choices"][0]["message"]["function_call"]["arguments"])
        except:
            raise Exception("OpenAI Function Completion failed", response.json())

def lambda_handler(event, context):
    """
    
    """
    body = json.loads(event['body'])
    print('Body: ', body)
    new_messages = body['messages']
    conversation_id = body['conversationId']
    user_id = body['userId']
    twin_id = body['twinId']
    # get conversation
    convo_state = convo_table.query(
        KeyConditionExpression=Key('conversationId').eq(conversationId),
        ScanIndexForward=False,
        Limit=1
        )['Item']
    items = convo_state['Items']
    if len(items) > 0:
        most_recent_block = items[0]
        if most_recent_block['blockId'] % int(os.environ['StageIdentificationFrequency']) == 0: # Re-identify stage
            # re-identify stage
            print("Reidentifying stage for conversation: ", conversationId)
            prompt_template = most_recent_block['stagePromptTemplate']
            stage_prompts = most_recent_block['stagePrompts']
            current_stage_prompts = stage_prompts[most_recent_block['currentStageId']]
            print("Current stage prompts: ", current_stage_prompts)
            # append new messages to messages
            new_messages = most_recent_block['messages'].extend(new_messages)
            # get stage identification prompt template from prompt template table
            stage_identification_prompt_template = prompt_template_table.get_item(Key={'promptId': os.environ['stageIdentificationPrompt']})['Item']
            print("Intro prompt template: ", intro_prompt_template)
            function = intro_prompt_template['function']
            template = intro_prompt_template['value']
            input_validation = intro_prompt_template['inputValidation']
            # format template
            stage_format_dict = {
                'stageGoal': stage_prompt['stageGoal'],
                'stageName': stage_prompt['stageName'],
                'stageInformationToGather': stage_prompt['stageInformationToGather'],
                'conversation': '\n'.join([f"{'User' if message['role'] == 'user' else 'You'}: {message['content']}" for message in new_messages]),
            }
            # validate input
            validate_input(stage_format_dict, input_validation)
            stage_identification_prompt = template.format(**stage_format_dict)
            print("Stage identification prompt: ", stage_identification_prompt)
            # call openai
            openai_response = openai_functions_only_completion(messages=[{'role': 'user', 'content': stage_identification_prompt}],
                                                               functions=[function],
                                                               function_name=function['name'],
                                                               model=os.environ['PROGRESSION_MODEL'],
                                                               max_tokens=int(os.environ['PROGRESSION_MAX_TOK']))
            print("Identification result: ", openai_response)
            # get stage identification result
            stage_summary = openai_response['gathered_information']
            progress_stage = openai_response['progress_stage']
            if progress_stage == False:
                # get prompt template for query questions
                query_questions_prompt_template = prompt_template_table.get_item(Key={'promptId': os.environ['queryQuestionsPromptNoProgression']})['Item']
                print("Query questions prompt template: ", query_questions_prompt_template)
                function = query_questions_prompt_template['function']
                template = query_questions_prompt_template['value']
                input_validation = query_questions_prompt_template['inputValidation']
                # format template
                query_questions_format_dict = {
                    'twinDefinition': most_recent_block['twinDefinition'],
                    'finalizedSummaries': '\n'.join([f"{summary['stageName']}: {summary['stageSummary']}" for summary in most_recent_block['finalizedSummaries']]),
                    'stageGoal': stage_prompt['stageGoal'],
                    'stageName': stage_prompt['stageName'],
                    'stageInformationToGather': stage_prompt['stageInformationToGather'],
                    'gatheredInformation': stage_summary,
                    'definition': stage_prompt['definition'],
                }
                # validate input
                validate_input(query_questions_format_dict, input_validation)
                query_questions_prompt = template.format(**query_questions_format_dict)
                print("Query questions prompt: ", query_questions_prompt)
                # call openai
                openai_response = openai_functions_only_completion(messages=[{'role': 'user', 'content': query_questions_prompt}],
                                                                   functions=[function],
                                                                   function_name=function['name'],
                                                                   model=os.environ['QUESTIONS_MODEL'],
                                                                   max_tokens=int(os.environ['QUESTIONS_MAX_TOK']))
                print("Query questions result: ", openai_response)
                # get query questions result
                query_questions = openai_response['query_questions']
                # retrieve content with MMR
                retrieved_content = query_mmr(query_questions, stage_summary)
                # create document set
                document_set = '\n'.join([match['content'] for match in retrieved_content])
                print("Document set: ", document_set)
                # get stage prompt tempalte
                stage_prompts = most_recent_block['stagePrompts'][most_recent_block['currentStageId']]
                # get prompt template for stage message
                stage_prompt_template = most_recent_block['stagePromptTemplate']
                format_dict = {
                    'stageGoal': stage_prompt['stageGoal'],
                    'stageName': stage_prompt['stageName'],
                    'stageInformationToGather': stage_summary,
                    'stageInteractionDefinition': stage_prompt['stageInteractionDefinition'],
                    'documentSet': document_set,
                }
                stage_prompt = stage_prompt_template.format(**format_dict)
                # get introduction prompt template
                intro_prompt_template = most_recent_block['introPromptTemplate']
                format_dict = {
                    'twinDefinition': most_recent_block['twinDefinition'],
                    'userTwinRelationship': stage_prompt['userTwinRelationship'],
                    'finalizedSummaries': '\n'.join([f"{summary['stageName']}: {summary['stageSummary']}" for summary in most_recent_block['finalizedSummaries']]),
                }
                intro_prompt = intro_prompt_template.format(**format_dict)
                # create response conditioning message set
                response_conditioning_messages = [
                    {'role': 'system', 'content': intro_prompt},
                    {'role': 'system', 'content': stage_prompt},
                ].extend(new_messages)
                # create block
                new_block = {
                    'conversationId': conversationId,
                    'blockId': most_recent_block['blockId']+1,
                    'twinId': twinId,
                    'userId': userId,
                    'messages': response_conditioning_messages,
                    'twinDefinition': most_recent_block['twinDefinition'],
                    'userTwinRelationship': most_recent_block['userTwinRelationship'],
                    'currentStage': most_recent_block['currentStage'],
                    'stageStateSummary': stage_summary,
                    'finalizedSummaries': most_recent_block['finalizedSummaries'],
                    'queryQuestions': query_questions,
                    'retrievedContent': retrieved_content,
                    'stageStep': most_recent_block['stageStep']+1,
                    'stagePrompts': stage_prompts,
                    'stageCurrentPrompt': stage_prompt,
                    'introPrompt': intro_prompt,
                    'stagePromptTemplate': stage_prompt_template,
                    'introPromptTemplate': intro_prompt_template,
                }
                # save block
                table.put_item(Item=new_block)
                # return response
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'messages': response_conditioning_messages,
                    })
                }
            else:
                # get next stage prompts
                stage_prompts = most_recent_block['stagePrompts'][most_recent_block['currentStageId']+1]
                # get prompt template for query questions 
                query_questions_prompt_template = prompt_template_table.get_item(Key={'promptId': os.environ['queryQuestionsPromptForProgression']})['Item']
                print("Query questions prompt template: ", query_questions_prompt_template)
                function = query_questions_prompt_template['function']
                template = query_questions_prompt_template['value']
                input_validation = query_questions_prompt_template['inputValidation']
                # format template
                query_questions_format_dict = {
                    'twinDefinition': most_recent_block['twinDefinition'],
                    'finalizedSummaries': '\n'.join([f"{summary['stageName']}: {summary['stageSummary']}" for summary in most_recent_block['finalizedSummaries']]),
                    'stageGoal': stage_prompt['stageGoal'],
                    'stageName': stage_prompt['stageName'],
                    'stageInformationToGather': stage_prompt['stageInformationToGather'],
                    'gatheredInformation': stage_summary,
                    'definition': stage_prompt['definition'],
                }
                # validate input
                validate_input(query_questions_format_dict, input_validation)
                query_questions_prompt = template.format(**query_questions_format_dict)
                print("Query questions prompt: ", query_questions_prompt)
                # call openai
                openai_response = openai_functions_only_completion(messages=[{'role': 'user', 'content': query_questions_prompt}],
                                                                   functions=[function],
                                                                   function_name=function['name'],
                                                                   model=os.environ['QUESTIONS_MODEL'],
                                                                   max_tokens=int(os.environ['QUESTIONS_MAX_TOK']))
                print("Query questions result: ", openai_response)
                # get query questions result
                query_questions = openai_response['query_questions']
                # retrieve content with MMR
                retrieved_content = query_mmr(query_questions, stage_summary)
                # create document set
                document_set = '\n'.join([match['content'] for match in retrieved_content])
                print("Document set: ", document_set)
                # get stage prompt template
                stage_prompt_template = most_recent_block['stagePromptTemplate']
                format_dict = {
                    'stageGoal': stage_prompt['stageGoal'],
                    'stageName': stage_prompt['stageName'],
                    'stageInformationToGather': stage_summary,
                    'stageInteractionDefinition': stage_prompt['stageInteractionDefinition'],
                    'documentSet': document_set,
                }
                stage_prompt = stage_prompt_template.format(**format_dict)
                # create intro prompt
                intro_prompt_template = most_recent_block['introPromptTemplate']
                # update finalized summaries
                if most_recent_block['finalizedSummaries'] == []:
                    most_recent_block['finalizedSummaries'] = [{
                        'stageName': most_recent_block['stageName'],
                        'stageSummary': stage_summary,
                    }]
                else:
                    most_recent_block['finalizedSummaries'].append({
                        'stageName': most_recent_block['stageName'],
                        'stageSummary': stage_summary,
                    })
                format_dict = {
                    'twinDefinition': most_recent_block['twinDefinition'],
                    'userTwinRelationship': stage_prompt['userTwinRelationship'],
                    'finalizedSummaries': '\n'.join([f"{summary['stageName']}: {summary['stageSummary']}" for summary in most_recent_block['finalizedSummaries']]),
                }
                intro_prompt = intro_prompt_template.format(**format_dict)
                # create response conditioning message set
                response_conditioning_messages = [
                    {'role': 'system', 'content': intro_prompt},
                    {'role': 'system', 'content': stage_prompt},
                ].extend(new_messages)
                # create block
                new_block = {
                    'conversationId': conversationId,
                    'blockId': most_recent_block['blockId']+1,
                    'twinId': twinId,
                    'userId': userId,
                    'messages': response_conditioning_messages,
                    'twinDefinition': most_recent_block['twinDefinition'],
                    'userTwinRelationship': most_recent_block['userTwinRelationship'],
                    'currentStage': most_recent_block['currentStage'],
                    'stageStateSummary': stage_summary,
                    'finalizedSummaries': most_recent_block['finalizedSummaries'],
                    'queryQuestions': query_questions,
                    'retrievedContent': retrieved_content,
                    'stageStep': most_recent_block['stageStep']+1,
                    'stagePrompts': most_recent_block['stagePrompts'],
                    'stageCurrentPrompt': most_recent_block['stageCurrentPrompt'],
                    'introPrompt': most_recent_block['introPrompt'],
                    'stagePromptTemplate': most_recent_block['stagePromptTemplate'],
                    'introPromptTemplate': most_recent_block['introPromptTemplate'],
                }
                # save block
                table.put_item(Item=new_block)
                # return response
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'messages': response_conditioning_messages,
                    })
                }
        else: # Continue within stage
            # continue within stage
            new_block = {
                'conversationId': conversationId,
                'blockId': most_recent_block['blockId']+1,
                'twinId': twinId,
                'userId': userId,
                'messages': most_recent_block['messages'].extend(new_messages),
                'userTwinRelationship': most_recent_block['userTwinRelationship'],
                'currentStage': most_recent_block['currentStage'],
                'stageStateSummary': most_recent_block['stageStateSummary'],
                'finalizedSummaries': most_recent_block['finalizedSummaries'],
                'queryQuestions': most_recent_block['queryQuestions'],
                'retrievedContent': most_recent_block['retrievedContent'],
                'stageStep': most_recent_block['stageStep']+1,
                'stagePrompts': most_recent_block['stagePrompts'],
                'stageCurrentPrompt': most_recent_block['stageCurrentPrompt'],
                'introPrompt': most_recent_block['introPrompt'],
                'stagePromptTemplate': most_recent_block['stagePromptTemplate'],
                'introPromptTemplate': most_recent_block['introPromptTemplate'],
            }
            messages = new_block['messages']
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'messages': messages,
                })
            }
    else: # New conversation
        twin = twin_table.get_item(Key={'twinId': twinId})['Item']
        # get user-twin relationship
        user_twin_response = user_twin_table.get_item(Key={'twinId': twinId, 'userId': userId})['Item']
        # retrieve from pinecone
        query_response = query_mmr(new_messages[1]['content'], {}, 3, namespace=f'{twin_id}')
        # format retrieved content
        retrieved_content = ''
        for item in query_response:
            retrieved_content+=item['content']
            retrieved_content+='\n'

        # get intro prompt template
        intro_prompt_template = prompt_template_table.get_item(Key={'promptId': os.environ['IntroPromptTemplateId']})['Item']
        template = intro_prompt_template['value']
        input_validation = intro_prompt_template['inputValidation']
        # create dict
        format_dict = {
            'twinDefinition': twin['twinDefinition'],
            'userTwinRelationship': user_twin_response['userTwinRelationship'],
            'finalizedSummaries': 'None yet, the conversation is just starting.',
        }
        validate_input(format_dict, expected_input)
        intro_prompt = template.format(**format_dict)
        # get stage prompt template
        stage_prompt_template = prompt_template_table.get_item(Key={'promptId': os.environ['StagePromptTemplateId']})['Item']
        template = stage_prompt_template['value']
        input_validation = stage_prompt_template['inputValidation']
        # get twin stage prompts
        stage_prompts = twin['stagePrompts']
        # create dict
        format_dict = {
            'stageName': stage_prompts[0]['stageName'],
            'stageGoal': stage_prompts[0]['stageGoal'],
            'stageInteractionDefinition': stage_prompts[0]['stageInteractionDefinition'],
            'stageInformationToGather': stage_prompts[0]['stageInformationToGather'],
            'documentSet': retrieved_content,
        }
        validate_input(format_dict, expected_input)
        stage_prompt = template.format(**format_dict)
        # create block
        block = {
            'conversationId': conversationId,
            'blockId': 0,
            'twinId': twinId,
            'userId': userId,
            'messages': new_messages,
            'userTwinRelationship': user_twin_response['userRelationship'],
            'twinDefinition': twin['twinDefinition'],
            'currentStage': 0,
            'stageStep': 0,
            'stageStateSummary': '',
            'finalizedSummaries': '',
            'retrievedContent': retrieved_content,
            'queryQuestions': new_messages[1]['content'],
            'stagePrompts': stage_prompts,
            'stageCurrentPrompt': stage_prompt,
            'introPrompt': intro_prompt,
            'introPromptTemplate': intro_prompt_template,
            'stagePromptTemplate': stage_prompt_template,
        }
        # format messages
        messages = [
            {'role':'system', 'content': intro_prompt},
            {'role':'system', 'content': stage_prompt},
        ]
        # extend with new messages
        messages.extend(new_messages)
        # return messages
        return {
            'statusCode': 200,
            'body': json.dumps({
                'messages': messages,
            })
        }