import * as cdk from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as iam from "aws-cdk-lib/aws-iam";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import { Construct } from "constructs";

export interface SyntheticDataStackProps extends cdk.StackProps {
  resourcePrefix: string;
  dataBucket: s3.Bucket;
  environment: string;
  emailAddress: string;
  tables?: { [key: string]: dynamodb.Table };
}

export class SyntheticDataStack extends cdk.Stack {

  constructor(scope: Construct, id: string, props: SyntheticDataStackProps) {
    super(scope, id, props);

    const resourcePrefix = props.resourcePrefix;
    const dataBucket = props.dataBucket;
    const tables = props.tables || {};

    // Create Lambda function to create synthetic data
    const syntheticDataFunction = new lambda.Function(this, `${resourcePrefix}-SyntheticDataFunction`, {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      code: lambda.Code.fromInline(`
import json
import os
import logging
import boto3
import uuid
from boto3.dynamodb.conditions import Key, Attr
from botocore.config import Config
import time
import cfnresponse

# Update logger import
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.typing import LambdaContext
from aws_lambda_powertools.logging import correlation_paths

# Initialize logger with service name
logger = Logger(service="SyntheticDataCreationFunction")
from decimal import Decimal

def sendResponseCfn(event, context, responseStatus):
    responseData = {}
    responseData['Data'] = {}
    cfnresponse.send(event, context, responseStatus, responseData, "CustomResourcePhysicalID")


# Custom JSON encoder to handle decimal class
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)
        return super(DecimalEncoder, self).default(obj)


class LocalDBService:
    def __init__(self,
                stack_prefix,
                stack_suffix,
                profile_name
                ):
        self.logger = logging.getLogger('LocalDatabaseService')
        # if profile_name is provided, use it to create the session
        if profile_name is not None:
            self.session = boto3.Session(profile_name=profile_name)
        else:
            self.session = boto3.Session()
        self.dynamodb = self.session.resource('dynamodb')
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
    
    
    def get_todos(self,user_id):
        self.logger.info(f"Getting todos for user_id: {user_id}")

        self.logger.info(f"User Table key schema: {self.user_table.key_schema}")

        # get role for user_id, if role is StoreManager, get all todos, if role is StoreAssociate, get todos where taskOwner equals userId
        user_id = str(user_id).strip().lower()  # Normalize the email format
        
        self.logger.info(f"Formatted user_id: {user_id}")

        role_response = self.user_table.scan(
                FilterExpression=Attr('userId').eq(user_id)
            )
        self.logger.info(f"Role Response: {role_response}")

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
            'messageId': str(messageId),  # Ensure userId is a string
            'message': str(message),
            'feedback': str(feedback),
            'userId': str(userId),
            'sessionId': str(sessionId),
            'timestamp': str(timestamp), 

        }
        try:
            self.feedback_table.put_item(Item=item)
            return item
        except Exception as e:
            print(f"Error adding feedback: {str(e)}")
            raise

    def get_feedback(self, messageId):
        response = self.feedback_table.get_item(Key={'messageId': messageId})
        self.logger.info(f"Database Response: {response}")
        return response.get('Item', {})
    
    def get_feedbacks(self):
        response = self.feedback_table.scan()
        self.logger.info(f"Database Response: {response}")
        return response.get('Items', [])

    def add_todo(self, userId, taskOwner, text, description,status):
        taskId = str(uuid.uuid4())
        createdAt = str(int(time.time() * 1000))  # Convert to string
        item = {
            'userId': str(userId),  # Ensure userId is a string
            'taskId': taskId,
            'taskOwner': str(taskOwner),
            'text': str(text),
            'description': str(description),
            'status': str(status), 
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
            'startDate':startDate,
            'endDate': endDate,
            'details': str(details), 
            'createdAt': createdAt
        }
        try:
            self.timeoff_table.put_item(Item=item)
            return item
        except Exception as e:
            print(f"Error adding timeoff: {str(e)}")
            raise
    
    def delete_timeoff(self, userId, timeoff_id):
        response = self.timeoff_table.delete_item(
            Key={
                'userId': str(userId),
                'timeoffId': str(timeoff_id)
                
            }
        )
        self.logger.info(f"Database Response: {response}")
        return response

    def get_schedule(self, user_id):

        # get role for user_id, if role is StoreManager, get all todos, if role is StoreAssociate, get todos where taskOwner equals userId
        user_id = str(user_id).strip().lower()  # Normalize the email format
        
        self.logger.info(f"Formatted user_id: {user_id}")

        role_response = self.user_table.scan(
                FilterExpression=Attr('userId').eq(user_id)
            )
        self.logger.info(f"Role Response: {role_response}")

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
            'Monday': monday,
            'Tuesday': tuesday,
            'Wednesday': wednesday,
            'Thursday': thursday,
            'Friday': friday,
            'Saturday': saturday,
            'Sunday': sunday,
            'createdAt': createdAt
        }
        try:
            self.schedule_table.put_item(Item=item)
            return item
        except Exception as e:
            print(f"Error adding schedule: {str(e)}")
            raise

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


# initialize database service class
stack_prefix = os.environ['STACK_NAME']
stack_suffix = os.environ['STACK_ENVIRONMENT']
db_service = LocalDBService(profile_name=None, stack_prefix=stack_prefix, stack_suffix=stack_suffix)

@logger.inject_lambda_context(log_event=True)
def handler(event, context):
    logger.info(f"Event: {event}")
    logger.info(f"Context: {context}")
    email = os.environ['EMAIL']
    logger.info(f"Email: {email}")
    # if event.RequestType is 'Delete', return success as resoures will be deleted with table delete
    if event.get('RequestType') == 'Delete':
        return sendResponseCfn(event, context, 'SUCCESS')

    # add users
    db_service.add_user(email, 'StoreManager', 'Retail', 'Demo', email)
    # split email at @
    email_parts = email.split('@')
    email_part1 = email_parts[0]
    email_part2 = email_parts[1]
    db_service.add_user(email_part1 + '+1@' + email_part2, 'StoreAssociate', 'Mike', 'Brown', email_part1 + '+1@' + email_part2)
    db_service.add_user(email_part1 + '+2@' + email_part2, 'StoreAssociate', 'Sarah', 'Johnson', email_part1 + '+2@' + email_part2)
    db_service.add_user(email_part1 + '+3@' + email_part2, 'StoreAssociate', 'John', 'Smith', email_part1 + '+3@' + email_part2)
    db_service.add_user(email_part1 + '+4@' + email_part2, 'StoreAssociate', 'Emily', 'Davis', email_part1 + '+4@' + email_part2)
    db_service.add_user(email_part1 + '+5@' + email_part2, 'StoreAssociate', 'David', 'Lee', email_part1 + '+5@' + email_part2)
    db_service.add_user(email_part1 + '+6@' + email_part2, 'StoreAssociate', 'Lisa', 'Wilson', email_part1 + '+6@' + email_part2)

    # add schedule
    schedule_list = [
        {
        "userId": email,
        "Monday": "OFF",
        "Tuesday": "9-5",
        "Wednesday": "9-5",
        "Thursday": "1-9",
        "Friday": "1-9",
        "Saturday": "10-6",
        "Sunday": "OFF"
        },
        { "userId": email_part1 + '+1@' + email_part2,
        "Monday": "9-5",
        "Tuesday": "1-9",
        "Wednesday": "OFF",
        "Thursday": "9-5",
        "Friday": "9-5",
        "Saturday": "OFF",
        "Sunday": "11-7"
        },
        {
        "userId": email_part1 + '+2@' + email_part2,
        "Monday": "1-9",
        "Tuesday": "OFF",
        "Wednesday": "1-9",
        "Thursday": "OFF",
        "Friday": "9-5",
        "Saturday": "10-6",
        "Sunday": "11-7"
        },
        {
        "userId": email_part1 + '+3@' + email_part2,
        "Monday": "9-5",
        "Tuesday": "9-5",
        "Wednesday": "1-9",
        "Thursday": "1-9",
        "Friday": "OFF",
        "Saturday": "10-6",
        "Sunday": "OFF"
        },
        {
        "userId": email_part1 + '+4@' + email_part2,
        "Monday": "OFF",
        "Tuesday": "1-9",
        "Wednesday": "9-5",
        "Thursday": "9-5",
        "Friday": "1-9",
        "Saturday": "OFF",
        "Sunday": "11-7"
        },
        {
        "userId": email_part1 + '+5@' + email_part2,
        "Monday": "1-9",
        "Tuesday": "9-5",
        "Wednesday": "OFF",
        "Thursday": "1-9",
        "Friday": "9-5",
        "Saturday": "10-6",
        "Sunday": "OFF"
        },
        {
        "userId": email_part1 + '+6@' + email_part2,
        "Monday": "9-5",
        "Tuesday": "OFF",
        "Wednesday": "9-5",
        "Thursday": "9-5",
        "Friday": "1-9",
        "Saturday": "OFF",
        "Sunday": "11-7"
        }]

    for schedule in schedule_list:
        result = db_service.add_schedule(schedule['userId'], schedule['Monday'], schedule['Tuesday'], schedule['Wednesday'], schedule['Thursday'], schedule['Friday'], schedule['Saturday'], schedule['Sunday'])
        print(result)


    # add daily tasks by day
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    daily_tasks = [
        {
            'priority': '1',
            'taskName': 'Security and safety inspection',
            'description': '''Before entering the store, make sure nothing (from the outside) looks suspicious.
                            Verify the following: 
                                Are there any broken windows? 
                                Do you see signs of a break-in? 
                                Are there unusual cars or people loitering around the area? 
                                If the answer to any of these questions is yes, don?t enter the store. Go back to your car or a safe place and call the police. 

                                If everything looks normal:
                                Enter the store and lock the door behind you.
                                Disable the security alarm if necessary. 
                                Unlock and lock the door again as more staff members arrive. 
                                Only leave the door unlocked once the store is open for business. 
                                Make sure the alarm is working properly and security cameras are on and functioning.'''
        },
        {
            'priority': '2',
            'taskName': 'Opening inspection and housekeeping',
            'description': '''Turn on all the lights and do an inspection to make sure there are no issues and that the closing staff completed all the checklist items from the night before.
                                Follow these steps: 
                                Look for wet spots on the floor, ceiling, and walls. These could be signs of issues with plumbing, heating, or cooling. 
                                Check for signs of vermin or pests. If there?s an issue, call an exterminator. 
                                Perform general cleaning, including sweeping the floor, making sure the fitting rooms are ready for customers, and making sure your windows are clean. 
                                Make a note of areas that need improvement and share it with your cleaning staff (or closing staff from the night before).'''
        },
        {
            'priority': '3',
            'taskName': 'Turn on electronics',
            'description': '''Before opening to the public, you?ll want to make sure the following electronics are turned on and working properly: 
                            Point of sale (POS) system. Are your hardware and software working properly? 
                            Cash registers/tills. Make sure your cash register is balanced and that you have enough cash on hand to give shoppers change when they pay using cash.
                            TVs, sound systems, and air conditioning. Are they all turned on and functioning? 
                            Make sure people-counting software to track foot traffic is turned on and working.'''
        },
        {
            'priority': '4',
            'taskName': 'Check visual merchandising',
            'description': '''Make sure the store is visually appealing and customers can find what they?re looking for.
                                Check that the appropriate amount of stock is out on the floor.
                                Dress mannequins appropriately.
                                Make sure all products are tagged with size, price, and special pricing, if applicable. 
                                Ensure the stockroom is organized and that any new deliveries are properly sorted and stored.'''
        },
        {
            'priority': '5',
            'taskName': 'Signage and storefront',
            'description': '''Regardless of whether your retail store is in a stand-alone building or in a shopping mall, you?ll want to make sure your storefront attracts passersby.
                            Do this by completing the following: 
                                Check to make sure signage is correct, and take down signage from the day before if it?s no longer applicable. 
                                Clean and tidy up your storefront by wiping down the windows and sweeping the sidewalk.
                                Set the store temperature to a comfortable level.
                                Put up your sidewalk sign and refresh the messaging. 
                                Make sure your window displays are attractive and reflect the latest merchandise you have in-store.'''
        },
        {
            'priority': '6',
            'taskName': 'Team huddle',
            'description': '''Hold a daily team huddle to excite your staff, motivate them to reach their sales goals, and make sure everyone has the information they need to do their best.
                                During your daily staff meeting: 
                                Review shifts and individual and team duties to make sure each associate knows what they?re responsible for. 
                                Review daily and weekly sales goals and discuss what each associate can do to help reach those targets. 
                                Recap performance from the day before, including sales results and positive and negative customer interactions. 
                                Discuss how you can improve today, and ask your team to provide feedback. 
                                Review best practices for health and safety. 
                                Review active promotions, ways to boost awareness of them, and suggestive selling strategies your team can try.'''
        },
        {
            'priority': '7',
            'taskName': 'Register and POS system',
            'description': '''Cover POS system and cash registers/tills.'''
        },
        {
            'priority': '8',
            'taskName': 'Unloading new delivery and restocking',
            'description': '''Unload new deliveries and restock shelves.
                                Survey the checkout counter and fitting rooms for merchandise and put items back where they belong.
                                Straighten up shelves and fixtures. 
                                Replace sold inventory by taking more stock from the back and adding it to the appropriate product displays.'''
        },
        {
            'priority': '9',
            'taskName': 'Daily close housekeeping',
            'description': '''Make sure the store is clean before you leave.  
                                Then, when closing time comes: 
                                Check the dressing rooms, bathrooms, and other areas of your store to make sure everyone has left. 
                                Bring in outdoor signage and anything else that was outside. 
                                Lock the doors and station a staff member at the door to let last-minute customers and employees out. 
                                Check customers and employees who are leaving to make sure they?re not taking products that aren?t paid for.'''
        },
        {
            'priority': '10',
            'taskName': 'Close registers and POS systems',
            'description': '''Before closing up for the night, it?s important to review and reconcile sales.
                            Follow theese steps:
                            Set a specific time to close out registers and count cash, noting any discrepancies. This should be done away from the checkout counter and lingering customers, and out of sight from the front door. 
                            Ensure any carts saved in your POS system are cleared. 
                            Place tills or cash register drawers in the safe. 
                            Settle credit card machines. 
                            Shut down the POS system.'''
        },
        {
            'priority': '11',
            'taskName': 'Lock up and final checks',
            'description': '''Before locking the doors and leaving for the night, do a final walkthrough to make sure all the tasks on your closing checklist have been completed.
                            Make a note of incomplete duties to discuss during your team huddle the next day. 
                            Ensure employees clock out and know the next time they?re scheduled to work. 
                            Turn off the lights, turn on the security alarm, double-check that all electronics have been turned off, and lock up.'''
        }
    ]   

    for day in days:
        for task in daily_tasks:
            result = db_service.add_daily_task_by_day(day, task['priority'], task['taskName'], task['description'])
            print(result)


    # add products
    
    products = [
        {
            "name": "Cozy Comfort Throw Blanket",
            "description": "Ultra-soft microfiber blanket, 50x60, machine washable",
            "stock": 35,
            "location": "Home Goods, Aisle 3"
        },
        {
            "name": "Stainless Steel Water Bottle",
            "description": "24 oz, vacuum-insulated, BPA-free, leak-proof lid",
            "stock": 50,
            "location": "Sporting Goods, Aisle 7"
        },
        {
            "name": "Wireless Bluetooth Earbuds",
            "description": "True wireless, 20-hour battery life, water-resistant",
            "stock": 25,
            "location": "Electronics, Aisle 2"
        },
        {
            "name": "Organic Lavender Essential Oil",
            "description": "100% pure, steam-distilled, 1 fl oz glass bottle",
            "stock": 40,
            "location": "Health & Beauty, Aisle 5"
        },
        {
            "name": "Ceramic Plant Pot Set",
            "description": "Set of 3 succulent pots with drainage holes, assorted colors",
            "stock": 20,
            "location": "Garden Center, Aisle 1"
        },
        {
            "name": "Gourmet Dark Chocolate Bar",
            "description": "70% cacao, fair trade, organic, 3.5 oz bar",
            "stock": 75,
            "location": "Grocery, Aisle 9"
        },
        {
            "name": "Yoga Mat with Carrying Strap",
            "description": "Non-slip surface, 1/4 thick, 72 long, eco-friendly materials",
            "stock": 30,
            "location": "Sporting Goods, Aisle 6"
        },
        {
            "name": "Stainless Steel Mixing Bowl Set",
            "description": "Set of 3 nesting bowls, dishwasher safe, non-slip bottom",
            "stock": 15,
            "location": "Kitchenware, Aisle 4"
        },
        {
            "name": "LED Desk Lamp with USB Charging Port",
            "description": "Adjustable arm, 3 color modes, 5 brightness levels",
            "stock": 22,
            "location": "Home Office, Aisle 8"
        },
        {
            "name": "Natural Bamboo Cutting Board",
            "description": "18x12, juice groove, reversible, antimicrobial",
            "stock": 28,
            "location": "Kitchenware, Aisle 4"
        },
        {
            "name": "The Mindful Bookworm",
            "description": "Relaxing read, thought-provoking insights, sustainable materials",
            "stock": 20,
            "location": "Books, Aisle 2"
        },
        {
            "name": "Wireless Noise-Cancelling Headphones",
            "description": "Immersive sound, long-lasting battery, comfortable design, Bluetooth connectivity.",
            "stock": 15,
            "location": "Electronics, Aisle 5"
        },
        {
            "name": "Rustic Oak Dining Table",
            "description": "Solid oak construction, distressed finish, seats 6-8 people",
            "stock": 5,
            "location": "Furniture, Aisle 7"
        },
        {
            "name": "Stainless Steel Mixing Bowls",
            "description": "Durable, easy-to-clean, nesting design, versatile for cooking and baking",
            "stock": 30,
            "location": "Kitchenware, Aisle 4"
        },
        {
            "name": "Magnetic Building Tiles",
            "description": "Colorful, durable tiles, easy to connect, spark creativity",
            "stock": 25,
            "location": "Toys, Aisle 8"
        },
        {
            "name": "Comfort Stride Athletic Shoes",
            "description": "Breathable mesh, cushioned sole, flexible design, durable construction",
            "stock": 18,
            "location": "Footwear, Aisle 6"
        },
        {
            "name": "Chic Floral Midi Dress",
            "description": "Elegant floral print, flattering A-line silhouette, breathable cotton blend, machine washable",
            "stock": 12,
            "location": "Apparel, Aisle 3"
        },
        {
            "name": "Modern Farmhouse Dining Table",
            "description": "Solid wood construction, distressed finish, seats 6-8 people",
            "stock": 4,
            "location": "Furniture, Aisle 7"
        },
        {
            "name": "Lightweight Hiking Backpack",
            "description": "Durable, water-resistant, breathable mesh back, adjustable straps",
            "stock": 22,
            "location": "Outdoor Gear, Aisle 9"
        },
        {
            "name": "Modern Leather Recliner Chair",
            "description": "Adjustable backrest, plush cushions, durable leather upholstery, swivel base",
            "stock": 3,
            "location": "Furniture, Aisle 10"
        },
        {
            "name": "Timeless Silk Scarf",
            "description": "Luxurious silk, vibrant colors, versatile design, hand-rolled edges",
            "stock": 15,
            "location": "Accessories, Aisle 2"
        },
        {
            "name": "Leather Keychain Organizer",
            "description": "Durable leather, compact design, multiple key rings, secure clasp",
            "stock": 40,
            "location": "Accessories, Aisle 1"
        },
        {
            "name": "Comfort Stride Sneakers",
            "description": "Breathable mesh, cushioned sole, flexible design, durable construction",
            "stock": 20,
            "location": "Footwear, Aisle 6"
        },
        {
            "name": "Chic Floral Maxi Dress",
            "description": "Flowing fabric, adjustable straps, vibrant floral print, machine washable",
            "stock": 10,
            "location": "Apparel, Aisle 3"
        },
        {
            "name": "Ultralight Hiking Backpack",
            "description": "Durable, waterproof, breathable, adjustable straps",
            "stock": 25,
            "location": "Outdoor Gear, Aisle 9"
        }

    ]
    productId=0
    for product in products:
        productId = productId+1
        result = db_service.add_product(productId, product['name'], product['description'], product['stock'], product['location'])
        print(result)


    # add customers
    import random
    import time
    import string
    # Lists for generating random names
    first_names = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Mary', 'Patricia', 'Jennifer', 'Linda', 
                'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen', 'Nancy', 'Lisa', 'Margaret', 'Betty']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez', 
                'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin']

    # Lists for generating addresses
    streets = ['Main St', 'Oak Ave', 'Maple Dr', 'Cedar Ln', 'Pine Rd', 'Elm St', 'Washington Ave', 'Park Rd', 
            'Lake Dr', 'River Rd']
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 
            'Dallas', 'San Jose']
    states = ['NY', 'CA', 'IL', 'TX', 'AZ', 'PA', 'FL', 'OH', 'GA', 'NC']

    def generate_random_string(length):
        """Generate a random string of specified length."""
        return ''.join(random.choices(string.ascii_lowercase, k=length))

    def generate_phone():
        """Generate a random phone number."""
        area_code = random.randint(200, 999)
        prefix = random.randint(200, 999)
        line = random.randint(1000, 9999)
        return f"{area_code}-{prefix}-{line}"

    def generate_email(name):
        """Generate an email from a name."""
        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        cleaned_name = name.lower().replace(' ', '.')
        return f"{cleaned_name}@{random.choice(domains)}"

    def generate_address():
        """Generate a random address."""
        house_number = random.randint(100, 9999)
        street = random.choice(streets)
        city = random.choice(cities)
        state = random.choice(states)
        zip_code = random.randint(10000, 99999)
        return f"{house_number} {street}, {city}, {state} {zip_code}"

    def generate_customer_data(customer_id):
        """Generates a synthetic customer data record."""
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        full_name = f"{first_name} {last_name}"
        
        return {
            "customerId": str(customer_id),
            "customerName": full_name,
            "email": generate_email(full_name),
            "phone": generate_phone(),
            "address": generate_address()
        }

    # Add 100 synthetic customer records to DynamoDB
    for i in range(2, 102):  # Customer IDs from 2 to 101
        customer_data = generate_customer_data(i)
        try:
            db_service.add_customer(customer_data["customerId"], 
                                    customer_data["customerName"], 
                                    customer_data["email"], 
                                    customer_data["phone"], 
                                    customer_data["address"])
            
        except Exception as e:
            print(f"Error adding customer {i}: {e}")

    print("Finished adding 100 synthetic customers.")

    # add customer transactions
    import random
    import time
    from datetime import datetime, timedelta

    def generate_random_date(start_date='-1y'):
        """Generate a random date between start_date and today."""
        today = datetime.now()
        if start_date == '-1y':
            start = today - timedelta(days=365)
        else:
            start = datetime.strptime(start_date, '%Y-%m-%d')
        
        days_between = (today - start).days
        random_days = random.randint(0, days_between)
        random_date = start + timedelta(days=random_days)
        return random_date.strftime("%Y-%m-%d")

    def add_synthetic_transactions():
        """Add synthetic transactions for 100 customers."""
        transaction_id = 1
        for customer_id in range(1, 101):  # Customer IDs from 1 to 100
            num_transactions = random.randint(1, 5)  # Each customer has 1 to 5 transactions
            for _ in range(num_transactions):
                # Generate transaction data
                transaction_date = generate_random_date()
                transaction_amount = f"{random.uniform(10.0, 500.0):.2f}"
                transaction_type = random.choice(["purchase", "refund"])
                
                try:
                    db_service.add_customer_transaction(
                        str(customer_id),
                        str(transaction_id),
                        transaction_date,
                        transaction_amount,
                        transaction_type
                    )
                    print(f"Added transaction {transaction_id} for customer {customer_id}.")
                except Exception as e:
                    print(f"Error adding transaction {transaction_id}: {e}")
                transaction_id += 1
        
        print("Finished adding transactions for 100 customers.")

    # Run the function to generate and add transactions
    add_synthetic_transactions()

    sendResponseCfn(event, context, cfnresponse.SUCCESS)
    return {'message': 'Synthetic data creation function executed successfully'}
`),
      timeout: cdk.Duration.minutes(15),
      memorySize: 128,
      ephemeralStorageSize: cdk.Size.mebibytes(512),
      environment: {
        BUCKET: dataBucket.bucketName,
        STACK_NAME: props.resourcePrefix,
        STACK_ENVIRONMENT: props.environment,
        EMAIL: props.emailAddress
      },
      layers: [
        lambda.LayerVersion.fromLayerVersionArn(this, `${resourcePrefix}-Boto3Layer`, `arn:aws:lambda:${cdk.Stack.of(this).region}:770693421928:layer:Klayers-p312-boto3:22`),
        lambda.LayerVersion.fromLayerVersionArn(this, `${resourcePrefix}-PydanticLayer`, `arn:aws:lambda:${cdk.Stack.of(this).region}:558258168256:layer:pydantic:1`),
        lambda.LayerVersion.fromLayerVersionArn(this, `${resourcePrefix}-PowertoolsLayer`, `arn:aws:lambda:${cdk.Stack.of(this).region}:017000801446:layer:AWSLambdaPowertoolsPythonV3-python312-x86_64:4`)
      ],
      initialPolicy: [
        new iam.PolicyStatement({
          actions: [
            'dynamodb:GetItem',
            'dynamodb:Scan',
            'dynamodb:Query',
            'dynamodb:PutItem',
            'dynamodb:DeleteItem',
            'dynamodb:UpdateItem',
            'dynamodb:DescribeTable',
            'dynamodb:BatchWriteItem'
          ],
          resources: Object.values(tables).map(table => table.tableArn)
        })
      ]
    });

    // Create custom resource to enable layers
    const enableSyntheticData = new cdk.CustomResource(this, `${resourcePrefix}-EnableSyntheticData`, {
      serviceToken: syntheticDataFunction.functionArn,
    });

  }
}