import * as cdk from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as iam from "aws-cdk-lib/aws-iam";
import * as opensearchserverless from "aws-cdk-lib/aws-opensearchserverless";
import * as bedrock from "aws-cdk-lib/aws-bedrock";
import { Construct } from "constructs";

export interface OpenSearchStackProps extends cdk.StackProps {
  resourcePrefix: string;
  environment: string;
  dataBucket: s3.Bucket;
}

export class OpenSearchStack extends cdk.Stack {
  public readonly collection: opensearchserverless.CfnCollection;
  public readonly knowledgeBase: bedrock.CfnKnowledgeBase;
  public readonly dataSource: bedrock.CfnDataSource;

  constructor(scope: Construct, id: string, props: OpenSearchStackProps) {
    super(scope, id, props);

    const resourcePrefix = props.resourcePrefix;
    const dataBucket = props.dataBucket;

    // Create Lambda function to set up layers
    const layersSetupFunction = new lambda.Function(this, `${resourcePrefix}-LayersSetupFunction`, {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.lambda_handler',
      code: lambda.Code.fromInline(`
import json
import boto3
import os
import cfnresponse
import string
import random
import urllib3
import shutil
import time
import botocore
from botocore.exceptions import ClientError
        
def download_public_files(src,tgt):
    http = urllib3.PoolManager()
    with open("/tmp/"+tgt, 'wb') as out:
      r = http.request('GET', src, preload_content=False)
      shutil.copyfileobj(r, out)
    return "Files Downloaded Locally"

def empty_bucket(bucket_name,region_name):
    try:
      s3 = boto3.resource('s3')
      bucket = s3.Bucket(bucket_name)
      bucket.objects.all().delete()  
      
    except Exception as e:
      print(str(e))
    return "Bucket {} Emptied ".format(bucket_name)                            

def provision_s3_dirs(bucket_name,region_name,ret_dict):
    print("BUCKET NAME IS "+bucket_name)
    
    s3 = boto3.client('s3')
    try:
      s3.put_object(Bucket=bucket_name, Key=("code/"))

    except Exception as e:
      print(str(e))
    
    try:
        assets3 = boto3.resource('s3')
        if assets3.Bucket(bucket_name).creation_date is None:
          if region_name == 'us-east-1' or region_name == 'us-west-2':
              print('trying to create bucket')
              assets3.create_bucket(Bucket=bucket_name)
          else:
              print('other region')
              assets3.create_bucket(Bucket=bucket_name,CreateBucketConfiguration={'LocationConstraint':region_name})
          print("Asset bucket {} doesn't exists, created".format(bucket_name))
          time.sleep(20)
          print("End Timed wait after Asset bucket created")
    except Exception as e:
        print(str(e))
    
    ret_dict["WorkshopBucket"]=bucket_name
    return ret_dict

def deploy_assets(bucket_name,region_name, ret_dict):
    print("deploy assets to bucket: "+bucket_name)
    try:
        s3_client = boto3.client('s3')

        download_public_files("https://d3q8adh3y5sxpk.cloudfront.net/retail-app/opensearch-lib.zip","opensearch-lib.zip")
        s3_client.upload_file('/tmp/opensearch-lib.zip', bucket_name, 'code/opensearch-lib.zip')

        download_public_files("https://d3q8adh3y5sxpk.cloudfront.net/retail-app/requests-aws4auth-lib.zip","awsauth-lib.zip")
        s3_client.upload_file('/tmp/awsauth-lib.zip', bucket_name, 'code/awsauth-lib.zip')

        download_public_files("https://d3q8adh3y5sxpk.cloudfront.net/retail-app/anycompany-sop.txt","anycompany-sop.txt")
        s3_client.upload_file('/tmp/anycompany-sop.txt', bucket_name, 'app/anycompany-sop.txt')

    except Exception as e:
      print("Failed provisioning assets "+str(e))
    
    return ret_dict      

def handle_delete(bucket_name,region_name):
    dict_return={}
    dict_return["Data"]="delete"
    empty_bucket(bucket_name,region_name)  
    
    return dict_return  

def handle_create(bucket_name,region_name):
    print('start handle create')
    dict_return={}
    dict_return=provision_s3_dirs(bucket_name,region_name,dict_return)
    dict_return=deploy_assets(bucket_name,region_name, dict_return)
    
    return dict_return

def lambda_handler(event, context):
    response_ = cfnresponse.SUCCESS
    print(str(event))
    return_dict={}
    physical_resourceId = ''.join(random.choices(string.ascii_lowercase +string.digits, k=7))
    
    try:
        account_id = context.invoked_function_arn.split(":")[4]
        region_name = context.invoked_function_arn.split(":")[3]
        
        bucket_arg = str(os.environ['BUCKET'])
        
        request_type = str(event.get("RequestType",""))
        print('picked up event: '+ str(request_type))
        if request_type=='Create':
            return_dict = handle_create(bucket_arg,region_name)
        elif request_type =='Delete':
            return_dict = handle_delete(bucket_arg,region_name)
        else:
            return_dict = {}
            return_dict["Data"] = "testupdate"
    except Exception as e:
      return_dict['Data'] = str(e)
      response_ = cfnresponse.FAILED
    cfnresponse.send(event,context,response_,return_dict,physical_resourceId)
`),
      timeout: cdk.Duration.minutes(15),
      memorySize: 128,
      ephemeralStorageSize: cdk.Size.mebibytes(512),
      environment: {
        BUCKET: dataBucket.bucketName,
      },
      initialPolicy: [
        new iam.PolicyStatement({
          actions: ['s3:*'],
          resources: [dataBucket.arnForObjects('*'), dataBucket.bucketArn]
        })
      ]
    });

    // Create custom resource to enable layers
    const enableLayers = new cdk.CustomResource(this, `${resourcePrefix}-EnableLayers`, {
      serviceToken: layersSetupFunction.functionArn,
    });

    // Create Lambda layers
    const openSearchLayer = new lambda.LayerVersion(this, `${resourcePrefix}-OpenSearchLayer`, {
      code: lambda.Code.fromBucket(dataBucket, 'code/opensearch-lib.zip'),
      description: 'opensearch-py layer',
      compatibleRuntimes: [
        lambda.Runtime.PYTHON_3_12,
      ],
      layerVersionName: 'OpenSearchLayer',
    });

    const authLayer = new lambda.LayerVersion(this, `${resourcePrefix}-AuthLayer`, {
      code: lambda.Code.fromBucket(dataBucket, 'code/awsauth-lib.zip'),
      description: 'awsauthlayer',
      compatibleRuntimes: [
        lambda.Runtime.PYTHON_3_12,
      ],
      layerVersionName: 'AuthLayer',
    });

    // Add dependencies to ensure layers are created after files are uploaded
    openSearchLayer.node.addDependency(enableLayers);
    authLayer.node.addDependency(enableLayers);

    // Create OpenSearch Serverless collection
    this.collection = new opensearchserverless.CfnCollection(this, `${resourcePrefix}-Collection`, {
      name: `${resourcePrefix}-collection`,
      type: 'VECTORSEARCH',
      description: `Collection to hold ${resourcePrefix} data`,
    });

    // Create OpenSearch Serverless encryption policy
    const encryptionPolicy = new opensearchserverless.CfnSecurityPolicy(this, `${resourcePrefix}-EncryptionPolicy`, {
      name: `${resourcePrefix}-enc-policy`,
      type: 'encryption',
      description: 'Encryption policy for AOSS collection',
      policy: JSON.stringify({
        Rules: [{
          ResourceType: 'collection',
          Resource: [`collection/${this.collection.name}`]
        }],
        AWSOwnedKey: true
      })
    });

    // Create OpenSearch Serverless network policy
    const networkPolicy = new opensearchserverless.CfnSecurityPolicy(this, `${resourcePrefix}-NetworkPolicy`, {
      name: `${resourcePrefix}-net-policy`,
      type: 'network',
      description: 'Network policy for AOSS collection',
      policy: JSON.stringify([{
        Rules: [
          {
            ResourceType: 'collection',
            Resource: [`collection/${this.collection.name}`]
          },
          {
            ResourceType: 'dashboard',
            Resource: [`collection/${this.collection.name}`]
          }
        ],
        AllowFromPublic: true
      }])
    });
    this.collection.addDependency(networkPolicy);
    this.collection.addDependency(encryptionPolicy);

    // Create a dedicated IAM role for Bedrock Knowledge Base
    const bedrockKnowledgeBaseRole = new iam.Role(this, `${resourcePrefix}-BedrockKnowledgeBaseRole`, {
      roleName: `AmazonBedrockExecutionRoleForKnowledgeBase-${this.collection.name}`,
      assumedBy: new iam.ServicePrincipal('bedrock.amazonaws.com'),
      path: '/',
    });

    // Add condition to the trust policy
    const cfnBedrockRole = bedrockKnowledgeBaseRole.node.defaultChild as iam.CfnRole;
    cfnBedrockRole.addPropertyOverride('AssumeRolePolicyDocument.Statement.0.Condition', {
      StringEquals: {
        "aws:SourceAccount": cdk.Stack.of(this).account
      },
      ArnLike: {
        "AWS:SourceArn": `arn:aws:bedrock:${cdk.Stack.of(this).region}:${cdk.Stack.of(this).account}:knowledge-base/*`
      }
    });

    // Add S3 read-only access policy
    bedrockKnowledgeBaseRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        's3:Get*',
        's3:List*',
        's3:Describe*',
        's3-object-lambda:Get*',
        's3-object-lambda:List*'
      ],
      resources: [
        dataBucket.bucketArn,
        `${dataBucket.bucketArn}/*`
      ]
    }));

    // Add AOSS API access policy
    bedrockKnowledgeBaseRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'aoss:APIAccessAll',
        'aoss:DashboardsAccessAll',
        'aoss:CreateIndex',
        'aoss:DeleteIndex',
        'aoss:UpdateIndex',
        'aoss:DescribeIndex',
        'aoss:ReadDocument',
        'aoss:WriteDocument',
        'aoss:CreateCollectionItems',
        'aoss:DeleteCollectionItems',
        'aoss:UpdateCollectionItems',
        'aoss:DescribeCollectionItems'
      ],
      resources: [
        `arn:aws:aoss:${cdk.Stack.of(this).region}:${cdk.Stack.of(this).account}:collection/*`,
        `arn:aws:aoss:${cdk.Stack.of(this).region}:${cdk.Stack.of(this).account}:index/*/*`
      ]
    }));

    // Add Bedrock model access policy
    bedrockKnowledgeBaseRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:ListCustomModels'
      ],
      resources: ['*']
    }));

    bedrockKnowledgeBaseRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:InvokeModel'
      ],
      resources: [`arn:aws:bedrock:${cdk.Stack.of(this).region}::foundation-model/*`]
    }));

    // Create data access policy for OpenSearch Serverless
    const dataAccessPolicy = new opensearchserverless.CfnAccessPolicy(this, `${resourcePrefix}-DataAccessPolicy`, {
      name: `${resourcePrefix}-data-policy`,
      type: 'data',
      description: 'Access policy for AOSS collection',
      policy: JSON.stringify([{
        Description: 'Access for Bedrock Knowledge Base and other services',
        Rules: [
          {
            ResourceType: 'collection',
            Resource: [`collection/${this.collection.name}`],
            Permission: ['aoss:*']
          },
          {
            ResourceType: 'index',
            Resource: [`index/${this.collection.name}/*`],
            Permission: ['aoss:*']
          }
        ],
        Principal: [
          bedrockKnowledgeBaseRole.roleArn,
          layersSetupFunction.role!.roleArn,
          `arn:aws:sts::${cdk.Stack.of(this).account}:assumed-role/Admin/*`
        ]
      }])
    });

    // Create administrative data access policy
    const adminDataAccessPolicy = new opensearchserverless.CfnAccessPolicy(this, `${resourcePrefix}-AdminDataAccessPolicy`, {
      name: `${resourcePrefix}-admin-data-policy`,
      type: 'data',
      description: 'Administrative access policy for AOSS collection',
      policy: JSON.stringify([{
        Description: 'Administrative access for deployment and management',
        Rules: [
          {
            ResourceType: 'collection',
            Resource: [`collection/${this.collection.name}`],
            Permission: ['aoss:*']
          },
          {
            ResourceType: 'index',
            Resource: [`index/${this.collection.name}/*`],
            Permission: ['aoss:*']
          }
        ],
        Principal: [
          `arn:aws:iam::${cdk.Stack.of(this).account}:role/Admin`,
          `arn:aws:sts::${cdk.Stack.of(this).account}:assumed-role/Admin/*`,
          `arn:aws:iam::${cdk.Stack.of(this).account}:root`
        ]
      }])
    });

    // Create OpenSearch Serverless index using native CDK resource
    const opensearchIndex = new opensearchserverless.CfnIndex(this, `${resourcePrefix}-Index`, {
      collectionEndpoint: this.collection.attrCollectionEndpoint,
      indexName: `${resourcePrefix}-sop`,
      mappings: {
        properties: {
          vector: {
            type: 'knn_vector',
            dimension: 1024,
            method: {
              name: 'hnsw',
              engine: 'faiss',
              spaceType: 'l2',
              parameters: {
                efConstruction: 512,
                m: 16
              }
            }
          },
          text: {
            type: 'text'
          },
          metadata: {
            type: 'text'
          }
        }
      }
    });

    // Add dependencies for the index
    opensearchIndex.addDependency(this.collection);
    opensearchIndex.addDependency(dataAccessPolicy);
    opensearchIndex.addDependency(adminDataAccessPolicy);
    opensearchIndex.addDependency(networkPolicy);
    opensearchIndex.addDependency(encryptionPolicy);

    // Create a Lambda function to add delay between index creation and knowledge base creation
    const delayFunction = new lambda.Function(this, `${resourcePrefix}-DelayFunction`, {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      code: lambda.Code.fromInline(`
import cfnresponse
import time

def handler(event, context):
    try:
        if event['RequestType'] == 'Delete':
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
            return
        
        print("Waiting 60 seconds for OpenSearch index to be fully ready...")
        time.sleep(60)
        print("Delay completed, proceeding with Knowledge Base creation")
        
        cfnresponse.send(event, context, cfnresponse.SUCCESS, {
            'Message': 'Delay completed successfully'
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        cfnresponse.send(event, context, cfnresponse.FAILED, {'Error': str(e)})
      `),
      timeout: cdk.Duration.minutes(2),
      memorySize: 128
    });

    // Create a custom resource to trigger the delay
    const delayResource = new cdk.CustomResource(this, `${resourcePrefix}-DelayResource`, {
      serviceToken: delayFunction.functionArn,
      properties: {
        IndexName: opensearchIndex.indexName,
        Timestamp: Date.now().toString() // Force update on each deployment
      }
    });

    // Add dependency to ensure delay happens after index creation
    delayResource.node.addDependency(opensearchIndex);

    // Create Bedrock Knowledge Base
    this.knowledgeBase = new bedrock.CfnKnowledgeBase(this, `${resourcePrefix}-KnowledgeBase`, {
      name: this.collection.name,
      description: `Knowledge base for ${resourcePrefix}`,
      roleArn: bedrockKnowledgeBaseRole.roleArn,
      knowledgeBaseConfiguration: {
        type: 'VECTOR',
        vectorKnowledgeBaseConfiguration: {
          embeddingModelArn: `arn:${cdk.Stack.of(this).partition}:bedrock:${cdk.Stack.of(this).region}::foundation-model/amazon.titan-embed-text-v2:0`,
          // the properties below are optional
        embeddingModelConfiguration: {
          bedrockEmbeddingModelConfiguration: {
            dimensions: 1024
          },
      },
        }
      },
      storageConfiguration: {
        type: 'OPENSEARCH_SERVERLESS',
        opensearchServerlessConfiguration: {
          collectionArn: this.collection.attrArn,
          vectorIndexName: `${resourcePrefix}-sop`,
          fieldMapping: {
            vectorField: 'vector',
            textField: 'text',
            metadataField: 'metadata'
          }
        }
      }
    });
    
    // Add dependency to ensure the delay is completed before creating the knowledge base
    this.knowledgeBase.node.addDependency(delayResource);
    this.knowledgeBase.node.addDependency(dataAccessPolicy);
    this.knowledgeBase.node.addDependency(adminDataAccessPolicy);
    this.knowledgeBase.node.addDependency(networkPolicy);
    this.knowledgeBase.node.addDependency(encryptionPolicy);

    // Create Bedrock Data Source
    this.dataSource = new bedrock.CfnDataSource(this, `${resourcePrefix}-DataSource`, {
      knowledgeBaseId: this.knowledgeBase.ref,
      dataDeletionPolicy: 'RETAIN',
      name: this.collection.name,
      dataSourceConfiguration: {
        type: 'S3',
        s3Configuration: {
          bucketArn: dataBucket.bucketArn,
          inclusionPrefixes: [`app/`]
        }
      }
    });

    // Create a dedicated IAM role for the trigger sync Lambda function
    const triggerSyncRole = new iam.Role(this, `${resourcePrefix}-TriggerSyncRole`, {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      description: 'IAM role for the trigger sync Lambda function',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSLambdaBasicExecutionRole')
      ]
    });
    
    // Add permissions to start ingestion jobs - scoped to account knowledge bases
    triggerSyncRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:StartIngestionJob',
        'bedrock:GetIngestionJob',
        'bedrock:ListIngestionJobs'
      ],
      resources: [
        `arn:aws:bedrock:${cdk.Stack.of(this).region}:${cdk.Stack.of(this).account}:knowledge-base/*`
      ]
    }));

    // Create a Lambda function to trigger the data source sync job
    const triggerSyncFunction = new lambda.Function(this, `${resourcePrefix}-TriggerSyncFunction`, {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      code: lambda.Code.fromInline(`
import boto3
import cfnresponse
import os
import time

def handler(event, context):
    try:
        if event['RequestType'] == 'Delete':
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
            return

        # Get the data source ID from the event
        knowledge_base_id = event['ResourceProperties']['KnowledgeBaseId']
        data_source_id = event['ResourceProperties']['DataSourceId']
        region = os.environ['AWS_REGION']
        
        # Remove the knowledge base ID from the data source ID
        data_source_id = data_source_id.replace(knowledge_base_id, '').replace('|', '')

        print(f"Starting sync job for knowledge base {knowledge_base_id}, data source {data_source_id}")

        # Create Bedrock Agent client
        bedrock_agent = boto3.client('bedrock-agent', region_name=region)
        
        # Start the sync job
        response = bedrock_agent.start_ingestion_job(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id
        )
        print(f"Start Sync Job Response: {response}")
        job_id = response.get('ingestionJob', {}).get('ingestionJobId')
        print(f"Started ingestion job with ID: {job_id}")
        if job_id is not None:
          # Wait for the job to start (optional)
          time.sleep(5)
          
          # Get the job status
          job_response = bedrock_agent.get_ingestion_job(
              knowledgeBaseId=knowledge_base_id,
              dataSourceId=data_source_id,
              ingestionJobId=job_id
          )
          print(f"Job response: {job_response}")
          status = job_response.get('status')
          print(f"Ingestion job status: {status}")
        
        cfnresponse.send(event, context, cfnresponse.SUCCESS, {
            'IngestionJobId': job_id,
            'Status': status
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        cfnresponse.send(event, context, cfnresponse.FAILED, {'Error': str(e)})
      `),
      timeout: cdk.Duration.minutes(5),
      memorySize: 256,
      role: triggerSyncRole
    });
    
    // Create a custom resource to trigger the sync job
    const triggerSync = new cdk.CustomResource(this, `${resourcePrefix}-TriggerSync`, {
      serviceToken: triggerSyncFunction.functionArn,
      properties: {
        KnowledgeBaseId: this.knowledgeBase.ref,
        DataSourceId: this.dataSource.ref,
        Timestamp: Date.now().toString() // Force update on each deployment
      }
    });
    
    // Add dependencies
    triggerSync.node.addDependency(this.dataSource);
    triggerSync.node.addDependency(this.knowledgeBase);

    // Set up dependencies
    this.dataSource.addDependency(encryptionPolicy);
    this.dataSource.addDependency(networkPolicy);
    this.dataSource.addDependency(dataAccessPolicy);
    this.dataSource.addDependency(adminDataAccessPolicy);
    this.dataSource.addDependency(this.collection);
    this.dataSource.addDependency(this.knowledgeBase);
    
    // Add knowledgebaseid to output
    new cdk.CfnOutput(this, `${resourcePrefix}-KnowledgeBaseId`, {
      value: this.knowledgeBase.ref,
      description: 'The ID of the Bedrock Knowledge Base',
      exportName: `${resourcePrefix}-KnowledgeBaseId`
    });
  }
} 