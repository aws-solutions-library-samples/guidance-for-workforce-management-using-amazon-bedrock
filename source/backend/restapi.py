import json
import os
import re
import logging
from datetime import datetime, timedelta 
from typing import Any, Dict, List, Optional
import boto3
from pydantic import BaseModel, Field, create_model
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import hashlib
from datetime import datetime
import inspect
from pydantic import BaseModel, Field, create_model
import uuid
from boto3.dynamodb.conditions import Key, Attr
from botocore.config import Config
import logging
import random
from botocore.exceptions import ClientError
from datetime import timezone
from dotenv import load_dotenv
import base64
import io
from PIL import Image
from auth import validate_token
import psutil  # For monitoring system resources
import gc  # For garbage collection

# Add retry and backoff imports
from botocore.exceptions import ClientError, BotoCoreError
from functools import wraps
import math

# Add imports for Flask/FastAPI
from fastapi import FastAPI, HTTPException, Security, Depends, UploadFile, File, Form, Response, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.status import HTTP_403_FORBIDDEN

# Retry and backoff utilities
def exponential_backoff(attempt, base_delay=1.0, max_delay=60.0, jitter=True):
    """
    Calculate exponential backoff delay with optional jitter.
    
    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add random jitter
    
    Returns:
        Delay in seconds
    """
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)  # Add 0-50% jitter
    return delay


def retry_with_backoff(max_retries=5, 
                      base_delay=1.0, 
                      max_delay=60.0,
                      retryable_exceptions=None,
                      logger=None):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        retryable_exceptions: Tuple of exception types to retry on
        logger: Logger instance for logging retry attempts
    """
    if retryable_exceptions is None:
        retryable_exceptions = (ClientError, BotoCoreError)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            total_throttling_wait_time = 0.0
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    # Check if this is a throttling error
                    is_throttling = False
                    if isinstance(e, ClientError):
                        error_code = e.response.get('Error', {}).get('Code', '')
                        is_throttling = error_code in ['ThrottlingException', 'TooManyRequestsException', 
                                                     'ProvisionedThroughputExceededException', 'RequestLimitExceeded']
                    
                    # Don't retry non-throttling errors on the last attempt
                    if attempt == max_retries:
                        if logger:
                            if total_throttling_wait_time > 0:
                                logger.error(f"Function {func.__name__} failed after {max_retries} retries and {total_throttling_wait_time:.2f}s total throttling wait time: {e}")
                            else:
                                logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise e
                    
                    # Calculate delay
                    delay = exponential_backoff(attempt, base_delay, max_delay)
                    
                    if is_throttling:
                        total_throttling_wait_time += delay
                    
                    if logger:
                        if is_throttling:
                            logger.warning(f"Throttling detected in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}). "
                                         f"Waiting {delay:.2f}s (total throttling wait: {total_throttling_wait_time:.2f}s): {e}")
                        else:
                            logger.warning(f"Retryable error in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}). "
                                         f"Retrying in {delay:.2f} seconds: {e}")
                    
                    time.sleep(delay)
                    
                except Exception as e:
                    # Non-retryable exception
                    if logger:
                        if total_throttling_wait_time > 0:
                            logger.error(f"Non-retryable error in {func.__name__} after {total_throttling_wait_time:.2f}s throttling wait: {e}")
                        else:
                            logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise e
            
            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


load_dotenv()
BEDROCK_MODEL_ID = os.getenv('BEDROCK_MODEL_ID', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')

DEFAUL_SYSTEM_PROMPT = os.getenv('DEFAUL_SYSTEM_PROMPT', '''You are a professional retail store assistant focused on efficiency and accuracy. Your responses will be formatted in markdown.

Context:
- Current User: {current_user} (userId and email)
- Current Session ID: {sessionId}
- Current Date: {current_date}

Available Tools:
1. Knowledge & Information:
  - search_knowledge_database (FAQ search across standard operating procedures)
  - get_products (product catalog that lists what is in the inventory)
  - get_product_details (specific item details) if no productId is provided, use an empty string ('')

2. Staff Management:
  - get_schedule (user schedules)
  - get_timeoff (time-off records)
  - add_timeoff (time-off requests)
  - list_tasks (view assigned/created tasks)
  - create_task (assign new tasks)

3. Analytics & Recommendations:
  - generate_store_recommendations (generate a daily store recommendations for a store manager)
  - create_daily_task_suggestions_for_staff (staff task planning)
  - customer_recommendation (personalized product suggestions based on past purchase history, customer details, and product catalog)
  - get_customer_details (customer details) if no customerId is provided, use an empty string ('')
  - get_image_description (based on the userId, sessionId and query, provide a description of the last uploaded image)

Operating Guidelines:
- Use tools only when necessary
- Multiple tool calls permitted per response
- Use current user's email as userId when required
- Avoid tool references in responses
- Provide complete product details without summarization

Output Formatting:
1. Schedules:
  - List format
  - Order: Monday through Sunday
  - Group by day when showing full store schedule

2. Daily Task Suggestions:
  Format as:
  ```
  1)Task: [task_name]
  Assigned to: [task_owner]
  ```

3. General Responses:
  - Concise and clear
  - Markdown formatted
  - Professional tone

Important:
- Request clarification when needed
- Don't make assumptions
- Maintain accuracy and completeness''')
GUARDRAIL_IDENTIFIER = os.getenv('BD_GUARDRAIL_IDENTIFIER', 'default_identifier')
guardrail_version = os.getenv('BD_GUARDRAIL_VERSION', 'DRAFT')
GUARDRAIL_VERSION = guardrail_version.split('|')[1] if '|' in guardrail_version else guardrail_version
BD_KB_ID = os.getenv('BD_KB_ID', 'default_kb_id')
API_KEY = os.getenv('API_KEY', 'default_api_key')
STACK_PREFIX = os.environ['STACK_NAME']
print(f"STACK_PREFIX: {STACK_PREFIX}")
STACK_SUFFIX = os.environ['STACK_ENVIRONMENT']
print(f"STACK_SUFFIX: {STACK_SUFFIX}")
DOMAIN_NAME = os.getenv('DOMAIN_NAME', '*')
print(f"DOMAIN_NAME: {DOMAIN_NAME}")

# Retry and rate limiting configuration
BEDROCK_MAX_RETRIES = int(os.getenv('BEDROCK_MAX_RETRIES', '3'))
BEDROCK_BASE_DELAY = float(os.getenv('BEDROCK_BASE_DELAY', '1.0'))
BEDROCK_MAX_DELAY = float(os.getenv('BEDROCK_MAX_DELAY', '30.0'))
BEDROCK_MIN_REQUEST_INTERVAL = float(os.getenv('BEDROCK_MIN_REQUEST_INTERVAL', '0.1'))

print(f"Bedrock retry configuration: max_retries={BEDROCK_MAX_RETRIES}, base_delay={BEDROCK_BASE_DELAY}s, max_delay={BEDROCK_MAX_DELAY}s")
print(f"Bedrock rate limiting: min_interval={BEDROCK_MIN_REQUEST_INTERVAL}s")

# S3 bucket name for storing images
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'ASSISTANT-WebsiteBucket')
print(f"S3_BUCKET_NAME: {S3_BUCKET_NAME}")

# Initialize S3 client
s3_client = boto3.client('s3')

# Initialize logger with service name
logger = logging.getLogger("api")
from decimal import Decimal
# Custom JSON encoder to handle decimal class
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)

class ConversationStore:
    def __init__(self,
                  stack_prefix,
                  stack_suffix,
                  profile_name
                  ):
        # Configure boto3 with retries
        config = Config(
            retries = {
                'max_attempts': 10,
                'mode': 'standard'
            }
        )
        
        # Use default credential provider chain - will automatically use pod's IAM role
        self.dynamodb = boto3.resource('dynamodb', config=config)

        if stack_prefix is None:
            self.stack_prefix = 'SPEECH-TO-SPEECH'
        else:
            self.stack_prefix = stack_prefix
        if stack_suffix is None:
            self.stack_suffix = 'DEV'
        else:
            self.stack_suffix = stack_suffix
              
        self.table = self.dynamodb.Table(f'{self.stack_prefix}-SESSION-HISTORY-{self.stack_suffix}')

    def store_message(self, user_id: str, session_id: str, role: str, content: str):
        """
        Store a single message in DynamoDB
        """
        try:
            current_time = datetime.now(timezone.utc)
            timestamp = int(current_time.timestamp() * 1000)
            expiration_time = int((current_time + timedelta(days=30)).timestamp())
            
            # Store content in the format expected by Converse API
            if isinstance(content, str):
                formatted_content = [{'text': content}]
            elif isinstance(content, list):
                formatted_content = content
            else:
                formatted_content = [{'text': str(content)}]
            
            item = {
                'userId_sessionId': f"{user_id}#{session_id}",  # partition key
                'timestamp': timestamp,  # sort key
                'userId': user_id,
                'sessionId': session_id,
                'role': role,
                'content': formatted_content,  # Store as list of content blocks
                'ttl': expiration_time,
                'status': 'active'
            }
            
            self.table.put_item(Item=item)
        except Exception as e:
            logger.error(f"Error storing message: {str(e)}")
            raise

    def get_conversation_history(self, user_id: str, session_id: str, limit: int = 10) -> List[Dict]:
        """
        Retrieve conversation history for a specific user and session where the status is 'active'
        """
        response = self.table.query(
            KeyConditionExpression='userId_sessionId = :uid_sid',
            FilterExpression='#status = :status',
            ExpressionAttributeValues={
                ':uid_sid': f"{user_id}#{session_id}",
                ':status': 'active'
            },
            ExpressionAttributeNames={
                '#status': 'status'
            },
            Limit=limit,
            ScanIndexForward=False  # Get most recent messages first
        )
        
        # Convert to format expected by Bedrock
        messages = []
        for item in response['Items']:
            messages.append({
                'role': item['role'],
                'content': item['content']
            })
        
        return messages
    
    def delete_conversation_history(self, user_id: str, session_id: str):
        """
        Mark all messages for a specific user and session as deleted/invalid
        """
        try:
            # First get all items for this user/session combination
            response = self.table.query(
                KeyConditionExpression='userId_sessionId = :uid_sid',
                ExpressionAttributeValues={
                    ':uid_sid': f"{user_id}#{session_id}"
                }
            )
            
            # Update the status of each item to 'inactive'
            for item in response['Items']:
                self.table.update_item(
                    Key={'userId_sessionId': item['userId_sessionId'], 'timestamp': item['timestamp']},
                    UpdateExpression='SET #status = :status',
                    ExpressionAttributeNames={'#status': 'status'},
                    ExpressionAttributeValues={':status': 'inactive'}
                )
        except Exception as e:
            logger.error(f"Error deleting conversation history: {str(e)}")
            raise
    
class KnowledgeBase():
    def __init__(self,        
        region: str = 'us-east-1',
        kb_id: str = ''

    ):
        # check if environment variable is set, if yes use it, otherwise set to ''
        if 'BD_KB_ID' in os.environ:
            self.kb_id = os.environ['BD_KB_ID']
        else:
            self.kb_id = ''

        config = Config(
            retries = {
                'max_attempts': 10,
                'mode': 'standard'
            }
        )
        # First ensure required environment variables are set
        if 'AWS_DEFAULT_REGION' not in os.environ:
            self.region='us-east-1'
            os.environ['AWS_DEFAULT_REGION'] = self.region
        else:
            self.region=region
            
        self.bedrock_agent_client = boto3.client(service_name="bedrock-agent-runtime", config=config, region_name=self.region)
        
    def kb_search(self,user_question):
        numberOfResults=3
        response = self.bedrock_agent_client.retrieve(
            retrievalQuery= {
                'text': user_question
            },
            knowledgeBaseId=self.kb_id,
            retrievalConfiguration= {
                'vectorSearchConfiguration': {
                    'numberOfResults': numberOfResults,
                    'overrideSearchType': "HYBRID"
                }
            }
        )
        
        contexts = []
        retrievalResults = response.get('retrievalResults')
        for retrievedResult in retrievalResults:
            print(type(retrievedResult))
            print(str(retrievedResult))
            
            text = retrievedResult.get('content').get('text')
            # Remove the "Document 1: " prefix if it exists
            if text.startswith("Document 1: "):
                text = text[len("Document 1: "):]
            contexts.append(text)
        contexts_string = ', '.join(contexts)
        return contexts_string

knowledgebase = KnowledgeBase()


class LocalDBService:
    def __init__(self,
                  stack_prefix,
                  stack_suffix,
                  profile_name
                  ):
          self.logger = logging.getLogger("LocalDatabaseService")
          
          # Configure boto3 with retries
          config = Config(
              retries = {
                  'max_attempts': 10,
                  'mode': 'standard'
              }
          )
          
          # Use default credential provider chain - will automatically use pod's IAM role
          self.dynamodb = boto3.resource('dynamodb', config=config)
          
          # if stack_prefix and stack_suffix are not provided, use default values
          if stack_prefix is None:
              self.stack_prefix = 'SPEECH-TO-SPEECH'
          else:
              self.stack_prefix = stack_prefix
          if stack_suffix is None:
              self.stack_suffix = 'DEV'
          else:
              self.stack_suffix = stack_suffix
              
          self.feedback_table = self.dynamodb.Table(f'{self.stack_prefix}-FEEDBACK-{self.stack_suffix}')
          self.customer_transactions_table = self.dynamodb.Table(f'{self.stack_prefix}-CUSTOMER_TRANSACTIONS-{self.stack_suffix}')
          self.customer_table = self.dynamodb.Table(f'{self.stack_prefix}-CUSTOMER-{self.stack_suffix}')
          self.task_table = self.dynamodb.Table(f'{self.stack_prefix}-TASKLIST-{self.stack_suffix}')
          self.user_table = self.dynamodb.Table(f'{self.stack_prefix}-USERROLE-{self.stack_suffix}')
          self.product_table = self.dynamodb.Table(f'{self.stack_prefix}-PRODUCTS-{self.stack_suffix}')
          self.timeoff_table = self.dynamodb.Table(f'{self.stack_prefix}-TIMEOFF-{self.stack_suffix}')
          self.schedule_table = self.dynamodb.Table(f'{self.stack_prefix}-SCHEDULE-{self.stack_suffix}')
          self.daily_tasks_by_day_table = self.dynamodb.Table(f'{self.stack_prefix}-DAILY_TASKS_BY_DAY-{self.stack_suffix}')
          self.image_table = self.dynamodb.Table(f'{self.stack_prefix}-IMAGES-{self.stack_suffix}')

        
    def get_users(self):
        response = self.user_table.scan()
        self.logger.info(f"Database Response: {response}")
        return response.get('Items', [])
    
    def add_user(self, userId, userRole, firstName, lastName, email):
        item = {
            'userId': userId,
            'userRole': userRole,
            'firstName': firstName,
            'lastName': lastName,
            'email': email
        }
        self.user_table.put_item(Item=item)
        return item
        
    def get_todos(self, user_id):
        # self.logger.info(f"Getting todos for user_id: {user_id}")

        # self.logger.info(f"User Table key schema: {self.user_table.key_schema}")

        # get role for user_id, if role is StoreManager, get all todos, if role is StoreAssociate, get todos where taskOwner equals userId
        user_id = str(user_id).strip().lower()  # Normalize the email format
        
        self.logger.info(f"Formatted user_id: {user_id}")

        role_response = self.user_table.scan(
                FilterExpression=Attr('userId').eq(user_id)
            )
        # self.logger.info(f"Role Response: {role_response}")

        role = role_response.get('Items')[0].get('userRole')

        self.logger.info(f"Role: {role}")
        if role == 'StoreManager':
            response = self.task_table.scan()
        else:
            response = self.task_table.scan(
                FilterExpression=Attr('taskOwner').eq(user_id)
            )
        self.logger.info(f"Database Response: {response}")
        return response.get('Items', [])  # Use .get() method with a default value
    
    def add_feedback(self, messageId, message, feedback, userId, sessionId, timestamp):
            
        item = {
            'messageId': messageId,
            'message': message,
            'feedback': feedback,
            'userId': userId,
            'sessionId': sessionId,
            'timestamp': timestamp
        }
        self.feedback_table.put_item(Item=item)
        return item

    def get_feedback(self, messageId):
        response = self.feedback_table.get_item(Key={'messageId': messageId})
        self.logger.info(f"Database Response: {response}")
        return response.get('Item', {})
    
    def get_feedbacks(self):
        response = self.feedback_table.scan()
        self.logger.info(f"Database Response: {response}")
        return response.get('Items', [])

    def add_todo(self, userId, taskOwner, text, description, status):
            
        taskId = str(uuid.uuid4())
        createdAt = str(int(time.time() * 1000))  # Convert to string
        item = {
            'taskId': str(taskId),
            'userId': userId,
            'taskOwner': taskOwner,
            'text': text,
            'description': description,
            'status': status,
            'createdAt': createdAt
        }
        try:
            self.task_table.put_item(Item=item)
            return item
        except Exception as e:
            print(f"Error adding todo: {str(e)}")
            raise
    
    def update_todo(self, userId, taskId, description=None, status=None):
            
        update_expression = []
        expression_values = {}
        expression_attribute_names = {}

        self.logger.info(f"Formatted userId: {userId}")
        self.logger.info(f"Formatted taskId: {taskId}")
        

        if description is not None:
            update_expression.append('#description = :description')
            expression_values[':description'] = str(description)
            expression_attribute_names['#description'] = 'description'
        if status is not None:
            update_expression.append('#status = :status')
            expression_values[':status'] = str(status)
            expression_attribute_names['#status'] = 'status'
            
        if not update_expression:
            return None  # Nothing to update
            
        try:
            response = self.task_table.update_item(
                Key={
                    'userId': str(userId),
                    'taskId': str(taskId)
                },
                UpdateExpression='SET ' + ', '.join(update_expression),
                ExpressionAttributeValues=expression_values,
                ExpressionAttributeNames=expression_attribute_names,
                ReturnValues='ALL_NEW'
            )
            return response.get('Attributes')
        except Exception as e:
            self.logger.error(f"Error updating todo: {str(e)}")
            raise
    
    def delete_todo(self, user_id, task_id):
            
        response = self.task_table.delete_item(
            Key={
                'userId': user_id,
                'taskId': task_id
            }
        )
        self.logger.info(f"Database Response: {response}")
        return response


    def get_products(self):
        response = self.product_table.scan()
        self.logger.info(f"Database Response: {response}")
        return response.get('Items', [])
    
    def get_product_details(self, productId, productName):
        if productId == None or productId == '':
            # then use productName
            print(f"Getting product details for productName: {productName}")
            response = self.product_table.scan(
                FilterExpression=Attr('productName').eq(str(productName))
            )
        else:
            print(f"Getting product details for productId: {productId}")
            response = self.product_table.scan(
                FilterExpression=Attr('productId').eq(str(productId))
            )

        self.logger.info(f"Database Response: {response}")
        return response.get('Items', {})

    def add_product(self, productId, productName, description, stock, location):
        createdAt = str(int(time.time() * 1000))  # Convert to string
        item = {
            'productId': str(productId),
            'productName': productName,
            'description': str(description),
            'stock': stock,
            'location': str(location), 
            'createdAt': createdAt
        }
        try:
            self.product_table.put_item(Item=item)
            return item
        except Exception as e:
            print(f"Error adding product: {str(e)}")
            raise

    def delete_product(self, product_id):
        response = self.product_table.delete_item(
            Key={
                'productId': str(product_id)
            }
        )
        self.logger.info(f"Database Response: {response}")
        return response

    def get_timeoff(self, user_id):
            
        response = self.timeoff_table.scan(
            FilterExpression=Attr('userId').eq(user_id)
        )
        self.logger.info(f"Database Response: {response}")
        return response.get('Items', [])

    def add_timeoff(self, userId, startDate, endDate, details):
            
        timeoffId = str(uuid.uuid4())
        createdAt = str(int(time.time() * 1000))  # Convert to string
        item = {
            'timeoffId': str(timeoffId),
            'userId': userId,
            'startDate': startDate,
            'endDate': endDate,
            'details': details,
            'createdAt': createdAt
        }
        self.timeoff_table.put_item(Item=item)
        return item
    
    def delete_timeoff(self, userId, timeoff_id):
            
        response = self.timeoff_table.delete_item(
            Key={
                'timeoffId': timeoff_id
            }
        )
        return {"message": "Timeoff deleted successfully"}

    def get_schedule(self, user_id):

        # get role for user_id, if role is StoreManager, get all todos, if role is StoreAssociate, get todos where taskOwner equals userId
        user_id = str(user_id).strip().lower()  # Normalize the email format
        
        self.logger.info(f"Formatted user_id: {user_id}")
        # self.logger.info(f"User Table: {self.user_table}")
        role_response = self.user_table.scan(
                FilterExpression=Attr('userId').eq(user_id)
            )
        self.logger.info(f"Role Response: {role_response}")

        # Check if user exists
        if not role_response.get('Items'):
            self.logger.error(f"No user found with ID: {user_id}")
            return {"error": f"User with ID {user_id} not found", "status": "not_found"}

        role = role_response.get('Items')[0].get('userRole')

        self.logger.info(f"Role: {role}")
        if role == 'StoreManager':
            response = self.schedule_table.scan()
        else:
            response = self.schedule_table.scan(
                FilterExpression=Attr('userId').eq(user_id)
            )
        
        self.logger.info(f"Database Response: {response}")
        return response.get('Items', [])
    
    def add_schedule(self, userId, monday, tuesday, wednesday, thursday, friday, saturday, sunday):
            
        scheduleId = str(uuid.uuid4())
        createdAt = str(int(time.time() * 1000))  # Convert to string
        item = {
            'scheduleId': str(scheduleId),
            'userId': userId,
            'monday': monday,
            'tuesday': tuesday,
            'wednesday': wednesday,
            'thursday': thursday,
            'friday': friday,
            'saturday': saturday,
            'sunday': sunday,
            'createdAt': createdAt
        }
        self.schedule_table.put_item(Item=item)
        return item

    def add_daily_task_by_day(self, day, priority, taskName, description):
        taskId = str(uuid.uuid4())
        createdAt = str(int(time.time() * 1000))  # Convert to string
        item = {
            'taskId': str(taskId),
            'day': day,
            'priority': priority,
            'taskName': taskName,
            'description': description,
            'createdAt': createdAt
        }
        try:
            self.daily_tasks_by_day_table.put_item(Item=item)
            return item
        except Exception as e:
            print(f"Error adding daily task by day: {str(e)}")
            raise

    def get_daily_tasks_by_day(self, day):
        response = self.daily_tasks_by_day_table.scan(
            FilterExpression=Attr('day').eq(day)
        )
        self.logger.info(f"Database Response: {response}")
        return response.get('Items', [])
    
    def get_customer(self, customerId, customerName):
        if customerId == None or customerId == '':
            # then use customerName
            print(f"Getting customer details for customerName: {customerName}")
            response = self.customer_table.scan(
                FilterExpression=Attr('customerName').eq(str(customerName))
            )
        else:
            print(f"Getting customer details for customerId: {customerId}")
            response = self.customer_table.scan(
                FilterExpression=Attr('customerId').eq(str(customerId))
            )

        self.logger.info(f"Database Response: {response}")
        return response.get('Items', {})
    
    def add_customer(self, customerId, customerName, email, phone, address):
        createdAt = str(int(time.time() * 1000))  # Convert to string
        item = {
            'customerId': customerId,
            'customerName': customerName,
            'email': email,
            'phone': phone,
            'address': address,
            'createdAt': createdAt
        }  
        try:
            self.customer_table.put_item(Item=item)
            return item
        except Exception as e:
            print(f"Error adding customer: {str(e)}")
            raise  
    
    def delete_customer(self, customerId):
        response = self.customer_table.delete_item(
            Key={
                'customerId': customerId
            }
        )
        self.logger.info(f"Database Response: {response}")
        return response
    
    def get_customers(self):
        response = self.customer_table.scan()
        self.logger.info(f"Database Response: {response}")
        return response.get('Items', [])
    
    def add_customer_transaction(self, customerId, transactionId, transactionDate, transactionAmount, transactionType):
        createdAt = str(int(time.time() * 1000))  # Convert to string
        item = {
            'customerId': customerId,
            'transactionId': transactionId,
            'transactionDate': transactionDate,
            'transactionAmount': transactionAmount,
            'transactionType': transactionType,
            'createdAt': createdAt
        }
        try:
            self.customer_transactions_table.put_item(Item=item)
            return item
        except Exception as e:
            print(f"Error adding customer transaction: {str(e)}")
            raise

    def get_customer_transactions(self, customerId):
        response = self.customer_transactions_table.scan(
            FilterExpression=Attr('customerId').eq(customerId)
        )
        self.logger.info(f"Database Response: {response}")
        return response.get('Items', [])    

    def save_image_reference(self, user_id, session_id, filename, s3_url, s3_key, thumbnail_base64):
        """
        Save a reference to the uploaded image in DynamoDB.
        
        Args:
            user_id: User ID
            session_id: Session ID
            filename: Original filename
            s3_url: S3 URL of the image
            s3_key: S3 key of the image
            thumbnail_base64: Base64 encoded thumbnail
            
        Returns:
            Image reference ID
        """
        try:
            # Generate a unique ID for the image reference
            image_id = str(uuid.uuid4())
            
            # Current timestamp
            timestamp = int(time.time() * 1000)
            
            # Create the item
            item = {
                'imageId': image_id,
                'userId': user_id,
                'sessionId': session_id,
                'filename': filename,
                's3Url': s3_url,
                's3Key': s3_key,
                'thumbnail': thumbnail_base64,
                'createdAt': timestamp,
                'ttl': int((datetime.now(timezone.utc) + timedelta(days=30)).timestamp())
            }
            
            # Put the item in the table
            self.image_table.put_item(Item=item)
            
            return image_id
        except Exception as e:
            self.logger.error(f"Error saving image reference: {e}")
            return None

    def get_image_by_id(self, image_id):
        """
        Retrieve an image reference from DynamoDB by its ID.
        
        Args:
            image_id: The ID of the image to retrieve
            
        Returns:
            Dictionary containing image details or None if not found
        """
        try:
            response = self.image_table.get_item(
                Key={'imageId': image_id}
            )
            
            if 'Item' not in response:
                self.logger.warning(f"Image not found with ID: {image_id}")
                return None
                
            return response['Item']
        except Exception as e:
            self.logger.error(f"Error retrieving image reference: {e}")
            return None
        
    def get_last_uploaded_image(self, userId, sessionId):
        # get the last uploaded image for a given userId and sessionId
        # sort by createdAt in descending order
        # return the first item
        print(f"Getting last uploaded image for userId: {userId} and sessionId: {sessionId}")
        try:
            response = self.image_table.scan(
                FilterExpression=Attr('userId').eq(userId) & Attr('sessionId').eq(sessionId)
            )
            print(f"Database Response: {response}")
            
            # Get items and sort by createdAt in descending order
            items = response.get('Items', [])
            if items:
                sorted_items = sorted(items, key=lambda x: x.get('createdAt', 0), reverse=True)
                return sorted_items[0]
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving last uploaded image: {e}")
            return None

db = LocalDBService(profile_name=None, stack_prefix=STACK_PREFIX, stack_suffix=STACK_SUFFIX)



class LocalBedrockService:
    def __init__(self,
                model_id: str = BEDROCK_MODEL_ID,
                system_prompt_template: str = DEFAUL_SYSTEM_PROMPT,
                temperature: float = 0,
                max_token_count: int = 4096,
                top_p: float = 1,
                top_k: int = 250,
                messages: list = [],
                tools_list = None
                ):
        self.logger = logging.getLogger("LocalBedrockService")
        logging.basicConfig(level=logging.INFO)
        
        # Configure boto3 client with enhanced retry configuration for throttling
        retry_config = Config(
            retries={
                'max_attempts': 10,
                'mode': 'adaptive'  # Use adaptive retry mode for better throttling handling
            }
        )
        
        self.bedrock_client = boto3.client(service_name='bedrock-runtime', config=retry_config)
        
        # Initialize conversation store once
        self.conv_store = ConversationStore(profile_name=None, stack_prefix=STACK_PREFIX, stack_suffix=STACK_SUFFIX)
        
        # Rate limiting variables
        self.last_request_time = 0
        self.min_request_interval = BEDROCK_MIN_REQUEST_INTERVAL  # Use configurable value
        self.consecutive_throttles = 0
        self.throttle_backoff_multiplier = 2.0
        self.last_throttle_time = 0  # Track when the last throttle occurred
        
        # Throttling metrics tracking
        self.total_rate_limit_wait_time = 0.0
        self.total_throttling_incidents = 0
        
        self.messages = messages

        self.tools_list = tools_list
        if self.tools_list is not None:
            try:
                # Get all tool methods with bedrock_schema
                tool_methods = []
                for method_name in dir(self.tools_list):
                    if (not method_name.startswith('_') and 
                        hasattr(getattr(self.tools_list, method_name), 'bedrock_schema')):
                        tool_methods.append(method_name)
                
                logger.info(f"Found {len(tool_methods)} tool methods: {tool_methods}")
                
                # Build tool configuration
                tools = []
                for method_name in tool_methods:
                    try:
                        method = getattr(self.tools_list, method_name)
                        tool_spec = method.bedrock_schema['toolSpec']
                        
                        # Validate tool spec structure
                        if not all(key in tool_spec for key in ['name', 'description', 'inputSchema']):
                            logger.error(f"Invalid tool spec for {method_name}: missing required fields")
                            continue
                            
                        if 'json' not in tool_spec['inputSchema']:
                            logger.error(f"Invalid tool spec for {method_name}: missing inputSchema.json")
                            continue
                            
                        # Validate that inputSchema.json is a dict
                        input_schema = tool_spec['inputSchema']['json']
                        if not isinstance(input_schema, dict):
                            logger.error(f"Invalid tool spec for {method_name}: inputSchema.json is not a dict")
                            continue
                            
                        # Test JSON serialization
                        json.dumps(input_schema)
                        
                        tools.append({'toolSpec': tool_spec})
                        logger.info(f"Successfully added tool: {tool_spec['name']}")
                        
                    except Exception as e:
                        logger.error(f"Error processing tool {method_name}: {e}")
                        continue
                
                if tools:
                    self.toolConfig = {
                        'tools': tools,
                        'toolChoice': {'auto': {}}
                    }
                    logger.info(f"Tool configuration created with {len(tools)} valid tools")
                    
                    # Final validation of the entire tool config
                    try:
                        json.dumps(self.toolConfig)
                        logger.info("Tool configuration is valid JSON")
                    except Exception as e:
                        logger.error(f"Tool configuration is not valid JSON: {e}")
                        self.toolConfig = None
                else:
                    logger.warning("No valid tools found, disabling tool configuration")
                    self.toolConfig = None
                    
            except Exception as e:
                logger.error(f"Error setting up tool configuration: {e}")
                self.toolConfig = None
        else:
            self.toolConfig = None

        self.model_id = model_id
        self.system_prompt_template = system_prompt_template
        self.temperature = temperature
        self.max_token_count = max_token_count
        self.top_p = top_p
        self.top_k = top_k
        # model specific inference parameters to use.
        if "anthropic" in self.model_id.lower():
            # Base inference parameters to use.
            self.inference_config = {
                                "temperature": self.temperature, 
                                "maxTokens": self.max_token_count,
                                "stopSequences": ["\n\nHuman:"],
                                "topP": self.top_p,
                            }
            self.additional_model_fields = {"top_k": self.top_k}
        elif "nova" in self.model_id.lower():
            # Nova models have different parameter requirements
            self.inference_config = {
                                "temperature": self.temperature, 
                                "maxTokens": self.max_token_count,
                                "topP": self.top_p,
                            }
            self.additional_model_fields = {}
        else:
            # Default configuration for other models
            self.inference_config = {
                                "temperature": self.temperature, 
                                "maxTokens": self.max_token_count,
                            }
            self.additional_model_fields = {}


    def _apply_rate_limiting(self):
        """Apply rate limiting to prevent overwhelming the API"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        # Calculate dynamic interval based on consecutive throttles
        dynamic_interval = self.min_request_interval * (self.throttle_backoff_multiplier ** self.consecutive_throttles)
        
        if time_since_last_request < dynamic_interval:
            sleep_time = dynamic_interval - time_since_last_request
            if sleep_time > 0:
                self.logger.info(f"Rate limiting: waiting {sleep_time:.3f}s (dynamic interval: {dynamic_interval:.3f}s, consecutive throttles: {self.consecutive_throttles})")
                self.total_rate_limit_wait_time += sleep_time
                time.sleep(sleep_time)
                self.logger.debug(f"Rate limiting wait completed. Total rate limit wait time this session: {self.total_rate_limit_wait_time:.2f}s")
        
        self.last_request_time = time.time()

    def _handle_throttling_success(self):
        """Reset throttling counters on successful request, but gradually"""
        if self.consecutive_throttles > 0:
            # Check if enough time has passed since last throttle event
            current_time = time.time()
            time_since_throttle = current_time - self.last_throttle_time
            
            # Only gradually decrease throttle count if some time has passed
            if time_since_throttle > 5:  # 5 seconds cooldown between throttle reductions
                # Reduce throttle count gradually - don't reset to zero immediately
                old_value = self.consecutive_throttles
                self.consecutive_throttles = max(0, self.consecutive_throttles - 1)
                self.logger.info(f"Throttling gradually clearing: {old_value} -> {self.consecutive_throttles} "
                                f"(Session totals: {self.total_throttling_incidents} incidents, "
                                f"{self.total_rate_limit_wait_time:.2f}s rate limit wait time)")
            
            # If we've completely cleared the throttling, log it
            if self.consecutive_throttles == 0:
                self.logger.info(f"Throttling completely cleared. "
                                f"Session totals: {self.total_throttling_incidents} incidents, "
                                f"{self.total_rate_limit_wait_time:.2f}s rate limit wait time")

    def _handle_throttling_error(self):
        """Increment throttling counters on throttling error"""
        # Record time of the throttle event first to ensure accurate timestamps
        current_time = time.time()
        self.last_throttle_time = current_time
        
        # Update counters
        self.consecutive_throttles += 1
        self.total_throttling_incidents += 1
        
        # Increase backoff for subsequent requests
        dynamic_interval = self.min_request_interval * (self.throttle_backoff_multiplier ** self.consecutive_throttles)
        
        self.logger.warning(f"Throttling error #{self.consecutive_throttles} (session total: {self.total_throttling_incidents} incidents)")
        self.logger.warning(f"Next request will be delayed by at least {dynamic_interval:.2f}s")

    def reset_throttling_state(self):
        """Reset throttling counters to recover from circuit breaker state"""
        old_value = self.consecutive_throttles
        self.consecutive_throttles = 0
        self.last_request_time = 0
        
        # Reset throttling state but maintain last_throttle_time for global cooldown
        # This ensures we don't immediately trigger throttling again
        current_time = time.time()
        if self.last_throttle_time == 0:
            # If no previous throttle event, set it to a long time ago
            self.last_throttle_time = current_time - 300  # 5 minutes ago
            
        if old_value > 0:
            self.logger.info(f"Throttling state reset - circuit breaker cleared (was {old_value})")
            
        return old_value > 0
        
    def get_throttling_metrics(self):
        """Get current throttling metrics for monitoring and debugging"""
        return {
            "consecutive_throttles": self.consecutive_throttles,
            "total_throttling_incidents": self.total_throttling_incidents,
            "total_rate_limit_wait_time": self.total_rate_limit_wait_time,
            "current_dynamic_interval": self.min_request_interval * (self.throttle_backoff_multiplier ** self.consecutive_throttles)
        }

    @retry_with_backoff(max_retries=BEDROCK_MAX_RETRIES, base_delay=BEDROCK_BASE_DELAY, max_delay=BEDROCK_MAX_DELAY, logger=logging.getLogger("LocalBedrockService"))
    def converse_with_tools(self, modelId, messages, system='', toolConfig=None):
        # self.logger.info(f'toolConfig: {toolConfig}')
        
        # Apply rate limiting before making the request
        self._apply_rate_limiting()
        
        # Enhanced circuit breaker with auto-reset after cooldown
        if self.consecutive_throttles > 5:
            current_time = time.time()
            
            # Add a global cooldown period based on the time since the last throttle event
            global_cooldown_period = 60  # 60 seconds global cooldown
            auto_reset_cooldown = 30     # 30 seconds for auto reset
            
            # Check if we have last_throttle_time and it's valid
            has_valid_throttle_time = hasattr(self, 'last_throttle_time') and self.last_throttle_time > 0
            
            if has_valid_throttle_time:
                time_since_throttle = current_time - self.last_throttle_time
                
                # Check for auto-reset condition
                if time_since_throttle > auto_reset_cooldown:
                    self.logger.info(f"Auto-resetting circuit breaker after {auto_reset_cooldown}s cooldown")
                    self.reset_throttling_state()
                    # Add extra delay to ensure we don't immediately trigger more throttling
                    extra_delay = 2.0
                    self.logger.info(f"Adding extra {extra_delay:.1f}s delay after circuit breaker reset")
                    time.sleep(extra_delay)
                    # Continue with the request after the delay
                    
                # If we're in global cooldown mode, enforce a waiting period
                elif time_since_throttle < global_cooldown_period:
                    remaining_cooldown = global_cooldown_period - time_since_throttle
                    metrics = self.get_throttling_metrics()
                    self.logger.error(f"Circuit breaker activated - {remaining_cooldown:.1f}s remaining in global cooldown period. "
                                    f"Session metrics: {metrics['total_throttling_incidents']} total incidents, "
                                    f"{metrics['total_rate_limit_wait_time']:.2f}s total rate limit wait time")
                    
                    # For very short remaining cooldowns, just wait it out instead of failing
                    if remaining_cooldown < 5:
                        self.logger.info(f"Short cooldown remaining ({remaining_cooldown:.1f}s) - waiting it out")
                        time.sleep(remaining_cooldown)
                        # Partially reset throttle count to allow the request
                        self.consecutive_throttles = 3  # Set to half the circuit breaker threshold
                    else:
                        # Otherwise fail the request with ServiceUnavailable
                        raise ClientError(
                            {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service temporarily unavailable due to rate limiting'}}, 
                            'converse'
                        )
                else:
                    # We're past the global cooldown but not yet at auto-reset, reduce throttle count
                    old_value = self.consecutive_throttles
                    self.consecutive_throttles = max(3, self.consecutive_throttles - 1)
                    self.logger.info(f"Reducing throttle count: {old_value} -> {self.consecutive_throttles} (past global cooldown)")
            else:
                # No valid last_throttle_time, initialize it and raise error
                self.last_throttle_time = current_time
                metrics = self.get_throttling_metrics()
                self.logger.error(f"Circuit breaker activated - consecutive throttles ({self.consecutive_throttles}) without valid timestamp. "
                                f"Session metrics: {metrics['total_throttling_incidents']} total incidents, "
                                f"{metrics['total_rate_limit_wait_time']:.2f}s total rate limit wait time")
                raise ClientError(
                    {'Error': {'Code': 'ServiceUnavailable', 'Message': 'Service temporarily unavailable due to rate limiting'}}, 
                    'converse'
                )
        
        try:
            response = self.bedrock_client.converse(
                modelId=modelId,
                system=system,
                messages=messages,
                toolConfig=toolConfig,
                inferenceConfig=self.inference_config,
                additionalModelRequestFields=self.additional_model_fields,
                guardrailConfig={
                    'guardrailIdentifier': GUARDRAIL_IDENTIFIER,
                    'guardrailVersion': GUARDRAIL_VERSION,
                    # 'trace': 'enabled'|'disabled'
                },
                performanceConfig={
                    'latency': 'optimized'
                }
            )
            
            # Success - reset throttling counters
            self._handle_throttling_success()
            return response
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            
            # Check if this is a throttling error
            if error_code in ['ThrottlingException', 'TooManyRequestsException', 
                            'ProvisionedThroughputExceededException', 'RequestLimitExceeded']:
                self._handle_throttling_error()
                self.logger.warning(f"Throttling error detected: {error_code} - {e}")
            
            # Log additional context for debugging but don't log full message content to save memory
            self.logger.error(f"Error in converse_with_tools: {error_code}")
            self.logger.error(f"Model ID: {modelId}")
            self.logger.error(f"Message count: {len(messages) if messages else 0}")
            raise
        except Exception as e:
            # Log additional context for debugging but limit details to prevent memory issues
            self.logger.error(f"Unexpected error in converse_with_tools: {type(e).__name__}")
            self.logger.error(f"Model ID: {modelId}")
            self.logger.error(f"Message count: {len(messages) if messages else 0}")
            # Reset throttling counter on non-throttling errors
            self.consecutive_throttles = 0
            raise

    def converse(self, tool_class, modelId, messages, system='', toolConfig=None):
        logger.info(f"{datetime.now():%H:%M:%S} - Invoking model...")
        MAX_LOOPS = 10  # Reduced from 15 to prevent excessive loops and memory buildup
        loop_count = 0
        continue_loop = True
        consecutive_failures = 0  # Track consecutive failures
        MAX_CONSECUTIVE_FAILURES = 3  # Max failures before giving up

        while continue_loop:
            loop_count = loop_count + 1
            logger.info(f"Loop count: {loop_count}")
            if loop_count >= MAX_LOOPS:
                logger.warning(f"Hit loop limit: {loop_count} - preventing infinite loop")
                break
                
            # Stop if too many consecutive failures
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(f"Too many consecutive failures ({consecutive_failures}) - stopping conversation")
                break
            
            # Memory management: limit message history size
            if len(messages) > 20:
                logger.warning(f"Message history too large ({len(messages)} messages), truncating to prevent memory issues")
                # Keep the system message and recent messages
                messages = messages[:1] + messages[-19:]
            
            logger.info(f"modelId: {modelId}")
            logger.info(f"Message count: {len(messages)}")
            # Don't log full message content to save memory and reduce log noise
            # logger.info(f"system: {system}")
            # logger.info(f"toolConfig: {toolConfig}")
            
            try:
                output = self.converse_with_tools(modelId, messages, system, toolConfig)
                
                # Check if output is valid
                if not output or 'output' not in output or 'message' not in output['output']:
                    logger.error("Invalid output structure from converse_with_tools")
                    consecutive_failures += 1
                    continue
                
                messages.append(output['output']['message'])
                logger.info(f"{datetime.now():%H:%M:%S} - Got output from model...")
                
                # Reset consecutive failures on success
                consecutive_failures = 0

                function_calling = [c['toolUse'] for c in output['output']['message']['content'] if 'toolUse' in c]
                logger.info(f'length of function_calling list: {len(function_calling)}')
                
                if function_calling:
                    tool_result_message = {"role": "user", "content": []}
                    for function in function_calling:
                        logger.info(f"{datetime.now():%H:%M:%S} - Function calling - Calling tool...")
                        tool_name = function['name']
                        tool_args = function['input'] or {}
                        
                        try:
                            tool_response = getattr(tool_class, tool_name)(**tool_args)
                            if not tool_response:
                                tool_response = "No response received from tool."
                        except Exception as tool_error:
                            logger.error(f"Tool {tool_name} failed: {type(tool_error).__name__}")
                            tool_response = f"Tool error: {type(tool_error).__name__}"
                        
                        logger.info(f"{datetime.now():%H:%M:%S} - Function calling - Got tool response...")
                        tool_result_message['content'].append({
                            'toolResult': {
                                'toolUseId': function['toolUseId'],
                                'content': [{"text": str(tool_response)[:2000]}]  # Limit response size
                            }
                        })
                    messages.append(tool_result_message)
                    logger.info(f"{datetime.now():%H:%M:%S} - Function calling - Calling model with result...")
                                
                else:
                    # check if further messages are required by going through content
                    response_content_blocks = output['output']['message'].get('content')
                    for content_block in response_content_blocks:
                        text = content_block.get('text')
                        if text is not None:
                            continue_loop = False
                    logger.info(f"{datetime.now():%H:%M:%S} - Function calling - Got final answer.")
            
            except Exception as e:
                consecutive_failures += 1
                logger.error(f"Error in conversation loop {loop_count}: {type(e).__name__}: {str(e)}")
                
                # For certain errors, break immediately
                if "ValidationException" in str(e) and "toolConfig" in str(e):
                    logger.error("Tool configuration validation error - disabling tools and trying without them")
                    # Try once without tools
                    if toolConfig is not None:
                        toolConfig = None
                        consecutive_failures = 0  # Reset failures since we're changing approach
                        continue
                    else:
                        logger.error("Still failing even without tools - stopping")
                        break
                
                # Break the loop on serious errors to prevent crash
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.error("Too many consecutive errors in conversation loop, stopping to prevent crash")
                    break

        # Ensure we return valid output even if conversation fails
        if 'output' not in locals() or output is None:
            logger.error("No valid output from conversation - creating fallback response")
            output = {
                'output': {
                    'message': {
                        'content': [{'text': "I apologize, but I encountered technical difficulties processing your request."}]
                    }
                }
            }

        return messages, output

    def call_bedrock(self, text, userId=None, sessionId=None):
        
        # Monitor system resources at the start of the call
        resource_info = monitor_system_resources()
        if resource_info and resource_info["memory_mb"] > 1200:
            logger.warning(f"Starting call_bedrock with high memory usage: {resource_info['memory_mb']:.1f}MB")
        
        # Record start time and initial throttling metrics
        start_time = time.time()
        initial_metrics = self.get_throttling_metrics()
        
        from datetime import date
        current_date = date.today()
        
            
        current_user = userId
        logger.info(f"current_user: {current_user}")
        logger.info(f"sessionId: {sessionId}")

        # # Use the instance conversation store instead of creating new one
        # if text == 'reset':
        #     # delete conversation history
        #     self.conv_store.delete_conversation_history(current_user, sessionId)
        #     return 'Conversation history deleted'
        
        system_prompt = [{"text": self.system_prompt_template.format(current_date=current_date, current_user=current_user, sessionId=sessionId)}]

        # Get conversation history using instance conversation store
        conversation_history = self.conv_store.get_conversation_history(current_user, sessionId)

        logger.info(f"conversation_history: {len(conversation_history)} messages")  # Log count instead of full content
        
        # Format conversation history properly for Converse API
        formatted_history = []
        for msg in conversation_history:
            # Handle both string content and list content formats
            if isinstance(msg['content'], list):
                content = msg['content']
            elif isinstance(msg['content'], str):
                content = [{'text': msg['content']}]
            else:
                content = [{'text': str(msg['content'])}]
            
            formatted_history.append({
                'role': msg['role'],
                'content': content
            })

        # Initialize messages with conversation history (excluding the current message)
        # Reverse the history since get_conversation_history returns most recent first
        self.messages = list(reversed(formatted_history))
        
        # Add the current user message
        self.messages.append({'role': 'user', 'content': [{'text': text}]})

        # Initialize output to None to prevent UnboundLocalError
        output = None
        response = None

        try:
            # Store the current user message
            self.conv_store.store_message(current_user, sessionId, 'user', text)

            messages, output = self.converse(ToolsList(), self.model_id, self.messages, system_prompt, self.toolConfig)
            
            # Check if output is valid
            if output and 'output' in output and 'message' in output['output'] and 'content' in output['output']['message']:
                content = output['output']['message']['content']
                if content and len(content) > 0 and 'text' in content[0]:
                    response = content[0]['text']
                else:
                    logger.error("Invalid output structure - no text content found")
                    response = "I apologize, but I encountered an issue processing your request. Please try again."
            else:
                logger.error("Invalid output structure from conversation")
                response = "I apologize, but I encountered an issue processing your request. Please try again."

            # Store the assistant response
            self.conv_store.store_message(current_user, sessionId, 'assistant', response)
        
            # Calculate request duration and throttling metrics
            request_duration = time.time() - start_time
            final_metrics = self.get_throttling_metrics()
            
            # Calculate metrics for this request
            throttling_incidents_this_request = final_metrics['total_throttling_incidents'] - initial_metrics['total_throttling_incidents']
            rate_limit_wait_this_request = final_metrics['total_rate_limit_wait_time'] - initial_metrics['total_rate_limit_wait_time']
            
            logger.info(f"Request completed in {request_duration:.2f}s. Output length: {len(response)} characters")
            
            # Log throttling metrics if there was any throttling activity
            if throttling_incidents_this_request > 0 or rate_limit_wait_this_request > 0:
                logger.info(f"Throttling activity this request: {throttling_incidents_this_request} incidents, "
                          f"{rate_limit_wait_this_request:.2f}s wait time. "
                          f"Session totals: {final_metrics['total_throttling_incidents']} incidents, "
                          f"{final_metrics['total_rate_limit_wait_time']:.2f}s total wait time")
            
            # Clean up memory after processing
            cleanup_memory()
            
            return response
            
        except Exception as e:
            # Calculate metrics even on error
            request_duration = time.time() - start_time
            final_metrics = self.get_throttling_metrics()
            throttling_incidents_this_request = final_metrics['total_throttling_incidents'] - initial_metrics['total_throttling_incidents']
            rate_limit_wait_this_request = final_metrics['total_rate_limit_wait_time'] - initial_metrics['total_rate_limit_wait_time']
            
            logger.error(f"Error in call_bedrock after {request_duration:.2f}s: {type(e).__name__}: {str(e)}")
            if throttling_incidents_this_request > 0 or rate_limit_wait_this_request > 0:
                logger.error(f"Throttling during failed request: {throttling_incidents_this_request} incidents, "
                           f"{rate_limit_wait_this_request:.2f}s wait time")
            
            # Clean up memory even on error
            cleanup_memory()
            
            # Return a user-friendly error message instead of raising the exception
            return "I apologize, but I'm currently experiencing technical difficulties. Please try again in a moment."

    @retry_with_backoff(max_retries=2, base_delay=BEDROCK_BASE_DELAY, max_delay=min(20.0, BEDROCK_MAX_DELAY), logger=logging.getLogger("LocalBedrockService"))
    def generate(self, prompt):
        message = {
            "role": "user",
            "content": [{"text": prompt}]
        }
        messages = [message]

        system_prompt = [{"text": "You are a helpful retail store assistant."}]

        # Apply rate limiting before making the request
        self._apply_rate_limiting()

        try:
            # Send the message.
            response = self.bedrock_client.converse(
                modelId=self.model_id,
                messages=messages,
                system=system_prompt,
                inferenceConfig=self.inference_config,
                additionalModelRequestFields=self.additional_model_fields,
                # performanceConfig={
                #     'latency': 'optimized'
                # }
            )

            # Success - reset throttling counters
            self._handle_throttling_success()

            # Log token usage.
            text = response['output'].get('message').get('content')[0].get('text')
            usage = response['usage']
            latency = response['metrics'].get('latencyMs')

            return [text, usage, latency]
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            
            # Check if this is a throttling error
            if error_code in ['ThrottlingException', 'TooManyRequestsException', 
                            'ProvisionedThroughputExceededException', 'RequestLimitExceeded']:
                self._handle_throttling_error()
                self.logger.warning(f"Throttling error in generate: {error_code} - {e}")
            
            # Log additional context for debugging
            self.logger.error(f"Error in generate method: {e}")
            self.logger.error(f"Model ID: {self.model_id}")
            self.logger.error(f"Error code: {error_code}")
            raise
        except Exception as e:
            # Log additional context for debugging
            self.logger.error(f"Unexpected error in generate method: {e}")
            self.logger.error(f"Model ID: {self.model_id}")
            raise

    # Threaded function for queue processing.
    def thread_request(self, q, results):
        while True:
            try:
                index, prompt = q.get(block=False)
                data = self.generate(prompt)
                results[index] = data
            except Queue.Empty:
                break
            except Exception as e:
                print(f'Error with prompt: {str(e)}')
                results[index] = str(e)
            finally:
                q.task_done()

    def generate_threaded(self, prompts, max_workers=15):
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(self.generate, prompt): i for i, prompt in enumerate(prompts)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
                    results[index] = str(exc)
        
        return results

class WorkforceService:
    def __init__(self):
        self.logger = logging.getLogger("WorkforceService")
        # with no tools
        self.bedrock = LocalBedrockService(model_id=BEDROCK_MODEL_ID)

        # Define KPI ranges
        self.kpis = {
            'Sales': (1000, 5000),  # Range of daily sales ($)
            'Waste': (50, 300),     # Range of waste in $ (unsellable or expired items)
            'Shrinkage': (30, 200), # Range of shrinkage in $ (theft, loss)
            'Tasks Completed': (50, 100) # Number of daily tasks completed by staff
        }

        self.kpi_data = {
          "Total Sales Revenue": {"value": 87542, "target": 90000, "unit": "$"},
          "Foot Traffic": {"value": 5217, "target": 5500, "unit": "visitors"},
          "Conversion Rate": {"value": 18.6, "target": 20, "unit": "%"},
          "Average Basket Size": {"value": 42.75, "target": 45, "unit": "$"},
          "Customer Satisfaction": {"value": 4.2, "target": 4.5, "unit": "stars"}
        }

        self.department_data = {
            "Electronics": {"aisle": 12, "bestsellers": ["Samsung 4K TV - 32 units", "Apple AirPods Pro - 58 units"], "revenue": "$33,890"},
            "Apparel": {"aisle": "3-8", "bestsellers": ["Men's Nike Dri-FIT shirts - 124 units", "Women's Levi's jeans - 87 units"], "revenue": "$10,755"},
            "Home Goods": {"aisle": "15-18", "bestsellers": ["Keurig Coffee Maker - 43 units", "Casper Queen Mattress - 12 units"], "revenue": "$17,375"}
        }

        self.inventory_alerts = {
            "Electronics": {"item": "Sony WH-1000XM4 Headphones", "units_left": 7, "status": "Restock"},
            "Apparel": {"item": "Women's Nike Running Shorts (size S)", "units_left": 3, "status": "Urgent reorder"},
            "Home Goods": {"item": "Instant Pot 6qt Pressure Cooker", "units_left": 5, "status": "Restock"}
        }

        self.recommendations = [
                {"department": "Electronics", "aisle": 12, "action": "Set up demo stations for popular TVs and audio equipment.", "expected_impact": "Increase customer engagement and potential sales"},
                {"department": "Apparel", "aisle": "3-8", "action": "Implement end-cap displays with trendy items.", "expected_impact": "Increase impulse buys and customer interest"},
                {"department": "Home Goods", "aisle": "15-18", "action": "Run promotions for home office items.", "expected_impact": "Drive foot traffic and increase sales of home office supplies"}
        ]

    
    def generate_store_recommendations(self, num_recommendations=3):
        """Generates a set of recommendations for store managers"""
        
        from datetime import datetime
        date = datetime.today().strftime('%Y-%m-%d')
        

        prompt='''As a retail analytics assistant, generate {num_recommendations} DAILY actionable recommendations based on these metrics:

                  Date: {date}
                  Metrics: {kpis}

                  Format response as JSON:
                  [
                    {{
                      "department": "<specific dept name>",
                      "location": "<aisle/section>",
                      "action": "<specific action/task to complete TODAY>",
                      "time_to_complete": "<duration in hours/minutes>",
                      "staff_needed": "<number and type of staff>",
                      "priority": "<urgent/high/medium/low>",
                      "expected_impact": "<same-day measurable outcome>"
                    }}
                  ]
                '''.format(date=date, kpis=self.kpi_data, num_recommendations=num_recommendations)

        result = self.bedrock.generate(prompt)

        return result[0]

    def get_daily_store_data(self, store_name):
        
        
        from datetime import datetime
        date = datetime.today().strftime('%Y-%m-%d')

        

        return {
            "store_name": store_name,
            "date": date,
            "kpi_data": self.kpi_data,
            "department_data": self.department_data,
            "inventory_alerts": self.inventory_alerts,
            "recommendations": self.recommendations
        }
    
    def generate_daily_staff_tasks_suggestions(self, schedule, daily_tasks):
        # Use the instance bedrock service
        schedule = json.dumps(schedule)
        daily_tasks = json.dumps(daily_tasks)

        prompt ='''Given the below schedule and daily tasks, assign the tasks to all available employees evenly.
        Schedule:
        {schedule}

        Daily Tasks:
        {daily_tasks}

        Follow the below steps to assign the tasks:
        1. Identify the number of employees available for the day
        2. Assign the tasks to the employees evenly
        3. Return the assigned tasks in the same format as the input daily tasks
        4. Format the assigned tasks as a JSON object with the following keys: taskName, description, taskOwner
        5. Return the assigned tasks as a JSON object and nothing else!
        '''.format(schedule=schedule, daily_tasks=daily_tasks)

        result = self.bedrock.generate(prompt)

        return result[0]
    
    def generate_customer_recommendations(self, customer_details, past_purchase_history, products):
        # Use the instance bedrock service
        customer_details = json.dumps(customer_details)
        past_purchase_history = json.dumps(past_purchase_history)
        products = json.dumps(products,cls=DecimalEncoder)

        prompt = '''
                    Given the following customer profile, past purchase history, and products, recommend the top 5 products that this customer might like. 
                    Customer Profile:
                    {customer_details}
                    Past Purchase History:
                    {past_purchase_history}
                    Products:
                    {products}

                    Recommendation: 
                    Think creatively and provide 5 product ID and description.
                    Also add any promotions about these product purchase. 
                    Generate the response in following format:
                    Recommended product 1 : product ID and decription, promotion available, in store inventory or stock?
                    Recommended product 2 : product ID and decription, promotion available, in store inventory or stock?
                    Recommended product 3: product ID and decription, promotion available, in store inventory or stock?
                    '''.format(customer_details=customer_details, past_purchase_history=past_purchase_history, products=products)

        result = self.bedrock.generate(prompt)

        return result[0]


# connect to workforce wrapper class
workforce = WorkforceService()


def bedrock_tool(name, description):
    def decorator(func):
        # Get function signature
        sig = inspect.signature(func)
        
        # Create model fields properly handling Field() parameters
        model_fields = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':  # Skip self parameter
                continue
                
            # Handle Field() parameters properly
            if hasattr(param.default, '__class__') and param.default.__class__.__name__ == 'FieldInfo':
                # This is a Field() parameter
                field_info = param.default
                model_fields[param_name] = (param.annotation, field_info)
            elif param.default is not inspect.Parameter.empty:
                # This is a regular parameter with a default value
                model_fields[param_name] = (param.annotation, param.default)
            else:
                # This is a required parameter without default
                model_fields[param_name] = (param.annotation, ...)

        # Create the input model
        input_model = create_model(
            func.__name__ + "_input",
            **model_fields
        )

        # Get the schema as a dictionary and ensure it's properly formatted
        try:
            schema_dict = input_model.model_json_schema()
            
            # Ensure the schema is a proper dictionary and clean up any issues
            if not isinstance(schema_dict, dict):
                logger.error(f"Schema for {name} is not a dict: {type(schema_dict)}")
                schema_dict = {"type": "object", "properties": {}, "required": []}
            
            # Clean up the schema to ensure compatibility with Bedrock
            if 'title' in schema_dict:
                del schema_dict['title']
            if '$defs' in schema_dict:
                del schema_dict['$defs']
                
            # Ensure required fields
            if 'type' not in schema_dict:
                schema_dict['type'] = 'object'
            if 'properties' not in schema_dict:
                schema_dict['properties'] = {}
                
            logger.info(f"Tool {name} schema created successfully with {len(schema_dict.get('properties', {}))} properties")
            
        except Exception as e:
            logger.error(f"Error creating schema for tool {name}: {e}")
            # Fallback to basic schema
            schema_dict = {
                "type": "object",
                "properties": {},
                "required": []
            }
        
        # Ensure the schema is properly formatted for Bedrock
        func.bedrock_schema = {
            'toolSpec': {
                'name': name,
                'description': description,
                'inputSchema': {
                    'json': schema_dict  # This must be a dict, not a string
                }
            }
        }
        
        # Validate that the schema is properly structured
        try:
            json.dumps(func.bedrock_schema)  # Test if it's JSON serializable
        except Exception as e:
            logger.error(f"Schema for {name} is not JSON serializable: {e}")
            # Create a minimal valid schema
            func.bedrock_schema = {
                'toolSpec': {
                    'name': name,
                    'description': description,
                    'inputSchema': {
                        'json': {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
            }
        
        return func

    return decorator

class ToolsList:
    @bedrock_tool(
        name="search_knowledge_database",
        description="search knowledge database to find answers to frequently asked questions (FAQ)"
    )
    def search_knowledge_database(self, user_question: str = Field(..., description="customer question")):
        kb_search_response = knowledgebase.kb_search(user_question)
        return kb_search_response
    
    @bedrock_tool(
        name="create_task",
        description="Create a task / to do list item"
    )
    def create_task(self, 
                    userId: str = Field(..., description="userId that created the task"),
                    taskOwner: str = Field(..., description="userId of the person who is assigned the task"),
                    text: str = Field(..., description="task / to do list item description"),
                    description: str = Field(..., description="description of the task"),
                    status: str = Field(..., description="status of the task; this is either open, in_progress or completed. Unless otherwise specified, set status to open."),
                  ):
        try:
            logger.info(f"{datetime.now():%H:%M:%S} - BEG create_task")
            
                
            response = db.add_todo(userId=userId, taskOwner=taskOwner, text=text, description=description, status=status)
            logger.info(f"{datetime.now():%H:%M:%S} - create_task response {response}")
            logger.info(f"{datetime.now():%H:%M:%S} - END create_task")
            # convert response to string
            response_string = json.dumps(response)
            return response_string
        except Exception as e:
            logger.error(f"{datetime.now():%H:%M:%S} - create_task error {e}")
            raise e
        

    @bedrock_tool(
        name="list_tasks",
        description="List all assigned tasks and tasks that they have created."
    )
    def list_tasks(self, 
                    taskOwner: str = Field(..., description="userId of the person who is assigned the task"),
                  ):
        try:
            logger.info(f"{datetime.now():%H:%M:%S} - BEG list_tasks")
            response = db.get_todos(taskOwner)
            logger.info(f"{datetime.now():%H:%M:%S} - list_tasks response {response}")
            logger.info(f"{datetime.now():%H:%M:%S} - END list_tasks")
            # convert response to string
            response_string = json.dumps(response)
            return response_string
        except Exception as e:
            logger.error(f"{datetime.now():%H:%M:%S} - list_tasks error {e}")
            raise e

    @bedrock_tool(
        name="generate_store_recommendations",
        description="Generate a daily store recommendations for a store manager."
    )
    def generate_store_recommendations(self):
        
        report = workforce.generate_store_recommendations()
        # convert report to string
        report_string = json.dumps(report)
        return report_string

    @bedrock_tool(
        name="get_customer_details",
        description="Get product details for a given product name, productId or product category."
    )
    def get_customer_details(self,
                            customerId: str = Field(..., description="customerId"),
                            customerName: str = Field(..., description="name of the customer")
                  ):
        try:
            logger.info(f"{datetime.now():%H:%M:%S} - BEG get_customer_details")
            logger.info(f"{datetime.now():%H:%M:%S} - get_customer_details customerId {customerId}")
            logger.info(f"{datetime.now():%H:%M:%S} - get_customer_details customerName {customerName}")
            response = db.get_customer(customerId, customerName)
            logger.info(f"{datetime.now():%H:%M:%S} - get_customer_details response {response}")
            logger.info(f"{datetime.now():%H:%M:%S} - END get_customer_details")
            # convert response to string
            response_string = json.dumps(response, cls=DecimalEncoder)
            logger.info(f"{datetime.now():%H:%M:%S} - get_customer_details response_string {response_string}")
            return response_string
        except Exception as e:
            logger.error(f"{datetime.now():%H:%M:%S} - get_customer_details error {e}")
            raise e
    

    @bedrock_tool(
        name="customer_recommendation",
        description="Provide personalized customer recommendations for a given customerId"
    )
    def customer_recommendation(self,  
                          customerId: str = Field(..., description="customer Id"),            
                  ):
        past_purchase_history = db.get_customer_transactions(customerId)
        cutomer_details = db.get_customer(customerId,customerName='')
        products = db.get_products()
        customer_product_recommendations = workforce.generate_customer_recommendations(cutomer_details,past_purchase_history,products)

        customer_recommendation_string = json.dumps(customer_product_recommendations)
        return customer_recommendation_string

    @bedrock_tool(
        name="list_products",
        description="List all products in the store."
    )
    def list_products(self
                  ):
        try:
            logger.info(f"{datetime.now():%H:%M:%S} - BEG list_products")
            response = db.get_products()
            logger.info(f"{datetime.now():%H:%M:%S} - list_products response {response}")
            logger.info(f"{datetime.now():%H:%M:%S} - END list_products")
            # convert response to string
            response_string = json.dumps(response, cls=DecimalEncoder)
            return response_string
        except Exception as e:
            logger.error(f"{datetime.now():%H:%M:%S} - list_products error {e}")
            raise e
        
    @bedrock_tool(
        name="get_product_details",
        description="Get product details for a given product name, productId or product category."
    )
    def get_product_details(self,
                            productId: str = Field(..., description="productId of the product"),
                            productName: str = Field(..., description="name of the product")
                  ):
        try:
            logger.info(f"{datetime.now():%H:%M:%S} - BEG get_product_details")
            logger.info(f"{datetime.now():%H:%M:%S} - get_product_details productId {productId}")
            logger.info(f"{datetime.now():%H:%M:%S} - get_product_details productName {productName}")
            response = db.get_product_details(productId, productName)
            logger.info(f"{datetime.now():%H:%M:%S} - get_product_details response {response}")
            logger.info(f"{datetime.now():%H:%M:%S} - END get_product_details")
            # convert response to string
            response_string = json.dumps(response, cls=DecimalEncoder)
            logger.info(f"{datetime.now():%H:%M:%S} - get_product_details response_string {response_string}")
            return response_string
        except Exception as e:
            logger.error(f"{datetime.now():%H:%M:%S} - get_product_details error {e}")
            raise e
        
    @bedrock_tool(
        name="get_timeoff",
        description="Get time off for a given userId."
    )
    def get_timeoff(self,
                            userId: str = Field(..., description="UserId to get timeoff for")
                            
                  ):
        try:
            logger.info(f"{datetime.now():%H:%M:%S} - BEG get_timeoff")
            
                
            response = db.get_timeoff(userId)
            logger.info(f"{datetime.now():%H:%M:%S} - get_timeoff response {response}")
            logger.info(f"{datetime.now():%H:%M:%S} - END get_timeoff")
            # convert response to string
            response_string = json.dumps(response)
            return response_string
        except Exception as e:
            logger.error(f"{datetime.now():%H:%M:%S} - get_timeoff error {e}")
            return f"Error getting timeoff: {e}"
            
    @bedrock_tool(
        name="add_timeoff",
        description="create a timeoff entry"
    )
    def add_timeoff(self,
                            userId: str = Field(..., description="UserId to get timeoff for"),
                            startDate: str = Field(..., description="start date of timeoff"),
                            endDate: str = Field(..., description="end date of timeoff"),
                            details: str = Field(..., description="details of timeoff")
                            
                  ):
        try:
            logger.info(f"{datetime.now():%H:%M:%S} - BEG add_timeoff")
            
                
            response = db.add_timeoff(userId=userId, startDate=startDate, endDate=endDate, details=details)
            logger.info(f"{datetime.now():%H:%M:%S} - add_timeoff response {response}")
            logger.info(f"{datetime.now():%H:%M:%S} - END add_timeoff")
            # convert response to string
            response_string = json.dumps(response)
            return response_string
        except Exception as e:
            logger.error(f"{datetime.now():%H:%M:%S} - add_timeoff error {e}")
            return f"Error adding timeoff: {e}"
            
    @bedrock_tool(
        name="delete_timeoff",
        description="delete a timeoff entry"
    )
    def delete_timeoff(self,
                            timeoffId: str = Field(..., description="timeoffId to delete timeoff for"),
                            userId: str = Field(..., description="userId to delete timeoff for"),
                            
                  ):
        try:
            logger.info(f"{datetime.now():%H:%M:%S} - BEG delete_timeoff")
            
                
            response = db.delete_timeoff(userId=userId, timeoff_id=timeoffId)
            logger.info(f"{datetime.now():%H:%M:%S} - delete_timeoff response {response}")
            logger.info(f"{datetime.now():%H:%M:%S} - END delete_timeoff")
            # convert response to string
            response_string = json.dumps(response)
            return response_string
        except Exception as e:
            logger.error(f"{datetime.now():%H:%M:%S} - delete_timeoff error {e}")
            return f"Error deleting timeoff: {e}"


    @bedrock_tool(
        name="get_schedule",
        description="get schedule for a given email address"
    )
    def get_schedule(self,
                    userId: str = Field(..., description="userId / email address for which to get schedule")            
                  ):
        try:
            logger.info(f"{datetime.now():%H:%M:%S} - BEG get_schedule")
            logger.info(f"{datetime.now():%H:%M:%S} - BEG get schedule for user with email: {userId}")
            response = db.get_schedule(userId)
            logger.info(f"{datetime.now():%H:%M:%S} - get_schedule response {response}")
            logger.info(f"{datetime.now():%H:%M:%S} - END get_schedule")
            
            # Check if response contains an error
            if isinstance(response, dict) and response.get('error'):
                return json.dumps({"status": "error", "message": response.get('error')})
                
            # convert response to string
            response_string = json.dumps(response)
            return response_string
        except Exception as e:
            logger.error(f"{datetime.now():%H:%M:%S} - get_schedule error {e}")
            error_message = f"Error retrieving schedule: {str(e)}"
            return json.dumps({"status": "error", "message": error_message})

    @bedrock_tool(
        name="create_daily_task_suggestions_for_staff",
        description="create daily task suggestions for available staff"
    )
    def create_daily_task_suggestions_for_staff(self,
                    email_address: str = Field(..., description="email address for which to get schedule")            
                  ):
        try:
            logger.info(f"{datetime.now():%H:%M:%S} - BEG create_daily_task_suggestions_for_staff")
            schedule = db.get_schedule(email_address)
            
            #get current day
            current_day = datetime.now().strftime("%A")
            print(f"{datetime.now():%H:%M:%S} - current_day {current_day}")

            # get daily tasks for current day
            daily_tasks = db.get_daily_tasks_by_day(current_day)

            # assign daily tasks to available staff
            response = workforce.generate_daily_staff_tasks_suggestions(schedule, daily_tasks)

            logger.info(f"{datetime.now():%H:%M:%S} - create_daily_task_suggestions_for_staff response {response}")
            logger.info(f"{datetime.now():%H:%M:%S} - END create_daily_task_suggestions_for_staff")
            # convert response to string
            response_string = json.dumps(response)
            return response_string
        except Exception as e:
            logger.error(f"{datetime.now():%H:%M:%S} - create_daily_task_suggestions_for_staff error {e}")
            raise e
    
    @bedrock_tool(
        name="get_image_description",
        description="get a description of the last uploaded image that the user has uploaded. If a user refers to an image use this tool to find out more."
    )
    def get_image_description(self,
        userId: str = Field(..., description="userId"),
        sessionId: str = Field(..., description="sessionId"),
        query: str = Field(..., description="user query for the image")
    ):
        # check if userPrompt is empty
        if not query:
            prompt = "describe the image"
        else:
            prompt = query

        last_uploaded_image = db.get_last_uploaded_image(userId, sessionId)
        if not last_uploaded_image:
            return "No image uploaded for this user session"
        
        s3Key = last_uploaded_image.get('s3Key')
        s3Url = last_uploaded_image.get('s3Url')

        # get the image from s3
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3Key)
        # convert the image to bytes
        image_data = response['Body'].read()
        image_data = base64.b64encode(image_data).decode('utf-8')

        file_extension = s3Key.split('.')[-1]
        print(f"file_extension: {file_extension}")
        
        IMAGE_MODEL_ID = "us.amazon.nova-lite-v1:0"
        
        # Configure the system prompt
        system_list = [{
            "text": "You are an expert artist. When the user provides you with an image, review the image and use that information to answer the user's question."
        }]
        
        # Prepare the message list
        message_list = []
        
        message_list.append(
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": file_extension,
                            "source": {"bytes": image_data},
                        }
                    },
                    {
                        "text": prompt
                    }
                ],
            }
        )
        
        # Configure the inference parameters
        inf_params = {"maxTokens": 300, "topP": 0.1, "topK": 20, "temperature": 0}

        # Prepare the request
        native_request = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "system": system_list,
            "inferenceConfig": inf_params,
        }
        
        # Initialize Bedrock client
        bedrock = boto3.client(service_name='bedrock-runtime')
        
        # Invoke the model and extract the response body
        response = bedrock.invoke_model(modelId=IMAGE_MODEL_ID, body=json.dumps(native_request))
        model_response = json.loads(response["body"].read())
        print(f"model_response: {model_response}")
        # Extract the result
        result = []
        for r in model_response["output"]["message"]["content"]:
            result.append(r["text"])
        # if result is empty, return "No response from model"
        if not result:
            return "No response from model"
        result = ' .'.join(result)
        print(f"result: {result}")
        return result


# Initialize Bedrock with tools
bedrock_service = LocalBedrockService(model_id=BEDROCK_MODEL_ID, tools_list=ToolsList())

# Initialize FastAPI app
app = FastAPI(title="API")

# Add startup event to log authentication configuration
@app.on_event("startup")
async def startup_event():
    logger.info("=== REST API Authentication Configuration ===")
    logger.info("✅ REST API JWT token authentication enabled")
    logger.info("   Authorization: Bearer <token> required for all protected endpoints")
    logger.info("===============================================")

# Add CORS middleware with specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3001", 
        "http://localhost:8000",
        f"https://{DOMAIN_NAME}",
        f"https://backend.{DOMAIN_NAME}",
        f"https://backend.{DOMAIN_NAME}/api",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# JWT Token security setup
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Extract and validate JWT token from Authorization header.
    
    Args:
        credentials: HTTP Bearer credentials from the Authorization header
        
    Returns:
        dict: User information if valid
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    
    if not credentials:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, 
            detail="Authorization token required"
        )
    
    try:
        # Validate the token using the same function as WebSocket
        # logger.info(f"credentials: {credentials}")
        user_info = validate_token(credentials.credentials)
        return user_info
    except HTTPException:
        # Re-raise HTTP exceptions from validate_token
        raise
    except Exception as e:
        logger.error(f"Unexpected error during REST API token validation: {e}")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Could not validate authentication token"
        )


@app.get("/api/users")
async def get_users(current_user: dict = Depends(get_current_user)):
    users = db.get_users()
    return {"users": users}

@app.get("/api/chat")
async def chat(query: str, userId: str, sessionId: str, current_user: dict = Depends(get_current_user)):
    try:
        # Validate input parameters to prevent issues
        if not query or not userId or not sessionId:
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        if len(query) > 10000:  # Limit query size to prevent memory issues
            raise HTTPException(status_code=400, detail="Query too long")
        
        # Add timeout protection
        import asyncio
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(bedrock_service.call_bedrock, query, userId, sessionId),
                timeout=300.0  # 5 minute timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Chat request timeout for user {userId}")
            raise HTTPException(status_code=504, detail="Request timeout")
        
        return {"chat_response": results}
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/reset_chat")
async def reset_chat(userId: str, sessionId: str, current_user: dict = Depends(get_current_user)):
    
    # Delete conversation history
    results = bedrock_service.conv_store.delete_conversation_history(userId, sessionId)
    
    # Reset throttling state
    was_throttled = bedrock_service.reset_throttling_state()
    if was_throttled:
        logger.info(f"Reset throttling state for user {userId}, session {sessionId}")
    
    return {"response": results, "throttling_reset": was_throttled}

@app.get("/api/todos")
async def get_todos(userId: str, current_user: dict = Depends(get_current_user)):
    
    
    todos = db.get_todos(userId)
    return {"todos": todos}

@app.post("/api/todos")
async def add_todo(todo: dict, current_user: dict = Depends(get_current_user)):
    userId = todo.get('userId')

    
    return db.add_todo(
        userId=todo.get('userId'),
        taskOwner=todo.get('taskOwner'),
        text=todo.get('text'),
        description=todo.get('description'),
        status=todo.get('status')
    )


@app.get("/api/store")
async def get_store(current_user: dict = Depends(get_current_user)):
    
    store_data = workforce.get_daily_store_data('retail')
    return store_data

@app.post("/api/feedback")
async def add_feedback(feedback: dict, current_user: dict = Depends(get_current_user)):
    db.add_feedback(
        messageId=feedback.get('messageId'),
        message=feedback.get('message'),
        feedback=feedback.get('feedback'),
        userId=feedback.get('userId'),
        sessionId=feedback.get('sessionId'),
        timestamp=feedback.get('timestamp')
    )

@app.get("/api/feedbacks")
async def get_feedbacks(current_user: dict = Depends(get_current_user)):
    feedbacks = db.get_feedbacks()
    return {"feedbacks": feedbacks}

@app.patch("/api/todos/{userId}/{taskId}")
async def update_todo(userId: str, taskId: str, todo: dict, current_user: dict = Depends(get_current_user)):
    
    return db.update_todo(
        userId=userId,
        taskId=taskId,
        description=todo.get('description'),
        status=todo.get('status')
    )

@app.delete("/api/todos/{userId}/{taskId}")
async def delete_todo(userId: str, taskId: str, current_user: dict = Depends(get_current_user)):
    
    return db.delete_todo(userId, taskId)



# Function to generate a thumbnail from an image
def generate_thumbnail(image_data, max_size=(200, 200)):
    """
    Generate a thumbnail from the provided image data.
    
    Args:
        image_data: Binary image data
        max_size: Maximum dimensions for the thumbnail (width, height)
        
    Returns:
        Base64 encoded thumbnail image
    """
    try:
        # Open the image from binary data
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary (handles PNG with transparency)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
            img = background
        
        # Resize the image while maintaining aspect ratio
        img.thumbnail(max_size, Image.LANCZOS)
        
        # Save the thumbnail to a bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        
        # Convert to base64
        thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return thumbnail_base64
    except Exception as e:
        logger.error(f"Error generating thumbnail: {e}")
        return None

# Function to upload image to S3
def upload_to_s3(image_data, user_id, session_id, filename, content_type):
    """
    Upload an image to S3.
    
    Args:
        image_data: Binary image data
        user_id: User ID
        session_id: Session ID
        filename: Original filename
        
    Returns:
        S3 URL of the uploaded image
    """
    try:
        # Generate a unique key for the S3 object
        timestamp = int(time.time() * 1000)
        unique_id = str(uuid.uuid4())[:8]
        s3_key = f"images/{user_id}/{session_id}/{timestamp}_{unique_id}_{filename}"
        
        print(f"S3_BUCKET_NAME: {S3_BUCKET_NAME}")
        print(f"s3_key: {s3_key}")
        # print(f"image_data: {image_data}")
        print(f"content_type: {content_type}")
        
        # Upload the image to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=image_data,
            ContentType=content_type  # Adjust based on actual image type
        )
        
        # Generate the S3 URL
        s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
        
        return s3_url, s3_key
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        return None, None

# Function to save image reference to DynamoDB
def save_image_reference(user_id, session_id, filename, s3_url, s3_key, thumbnail_base64):
    """
    Save a reference to the uploaded image in DynamoDB.
    
    Args:
        user_id: User ID
        session_id: Session ID
        filename: Original filename
        s3_url: S3 URL of the image
        s3_key: S3 key of the image
        thumbnail_base64: Base64 encoded thumbnail
        
    Returns:
        Image reference ID
    """
    return db.save_image_reference(user_id, session_id, filename, s3_url, s3_key, thumbnail_base64)

@app.post("/api/uploadimage")
async def upload_image(
    file: UploadFile = File(None),
    userId: str = Form(None),
    sessionId: str = Form(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Upload an image, save it to S3, store a reference in DynamoDB,
    generate a thumbnail, and return the thumbnail data.
    """
    print(f"userId: {userId}")
    print(f"sessionId: {sessionId}")
    print(f"file: {file}")
    
    
    
    try:
        # Read the image file
        print(f"file: {file}")
        image_data = await file.read()
        
        
        # Generate a thumbnail
        thumbnail_base64 = generate_thumbnail(image_data)
        if not thumbnail_base64:
            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
        
        # print(f"thumbnail_base64: {thumbnail_base64}")
        # Upload the image to S3
        s3_url, s3_key = upload_to_s3(image_data, userId, sessionId, file.filename, file.content_type)
        if not s3_url:
            raise HTTPException(status_code=500, detail="Failed to upload image to S3")
        
        print(f"s3_url: {s3_url}")
        print(f"s3_key: {s3_key}")
        # Save the image reference to DynamoDB
        image_id = save_image_reference(userId, sessionId, file.filename, s3_url, s3_key, thumbnail_base64)
        if not image_id:
            raise HTTPException(status_code=500, detail="Failed to save image reference")
        
        print(f"image_id: {image_id}")
        # Return the thumbnail and image ID
        return {
            "imageId": image_id,
            "thumbnail": thumbnail_base64,
            "s3_key": s3_key
        }
    
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")

@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    try:
        # Get image reference from DynamoDB
        image_data = db.get_image_by_id(image_id)
        
        if not image_data:
            raise HTTPException(status_code=404, detail="Image not found")
            
        s3_key = image_data['s3Key']
        
        # Get image from S3
        s3_response = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key
        )
        
        # Read image data
        image_data = s3_response['Body'].read()
        
        # Return image data with appropriate content type
        content_type = s3_response['ContentType']
        return Response(content=image_data, media_type=content_type)
        
    except Exception as e:
        logger.error(f"Error retrieving image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/api/metrics/throttling")
async def get_throttling_metrics(current_user: dict = Depends(get_current_user)):
    """Get current throttling metrics for monitoring and debugging"""
    try:
        metrics = bedrock_service.get_throttling_metrics()
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error retrieving throttling metrics: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving metrics")

def monitor_system_resources():
    """Monitor system resources and log warnings if usage is high"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        # Log warnings if resource usage is high
        if memory_mb > 1000:  # More than 1GB
            logger.warning(f"High memory usage: {memory_mb:.1f}MB")
            
        if cpu_percent > 80:
            logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
            
        # Force garbage collection if memory is very high
        if memory_mb > 1500:  # More than 1.5GB
            logger.warning("Forcing garbage collection due to high memory usage")
            gc.collect()
            
        return {"memory_mb": memory_mb, "cpu_percent": cpu_percent}
    except Exception as e:
        logger.error(f"Error monitoring resources: {e}")
        return None

def cleanup_memory():
    """Force garbage collection and clean up resources"""
    try:
        collected = gc.collect()
        # logger.info(f"Garbage collection freed {collected} objects")
    except Exception as e:
        logger.error(f"Error during garbage collection: {e}")
