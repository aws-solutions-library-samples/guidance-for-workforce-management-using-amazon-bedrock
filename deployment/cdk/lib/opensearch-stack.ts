import * as cdk from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as s3deploy from "aws-cdk-lib/aws-s3-deployment";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as iam from "aws-cdk-lib/aws-iam";
import * as opensearchserverless from "aws-cdk-lib/aws-opensearchserverless";
import * as bedrock from "aws-cdk-lib/aws-bedrock";
import { Construct } from "constructs";
import * as path from "path";

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

    // Deploy SOP document from local file to S3
    const sopDeployment = new s3deploy.BucketDeployment(this, `${resourcePrefix}-SOPDeployment`, {
      sources: [s3deploy.Source.asset(path.join(__dirname, '../../../source/data'))],
      destinationBucket: dataBucket,
      destinationKeyPrefix: 'app',
      prune: false, // Don't delete files that aren't in the source
    });

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
      settings: {
        index: {
          knn: true  // Enable KNN capability at the index level
        }
      },
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
    
    // Ensure data source is created after SOP file is deployed to S3
    this.dataSource.node.addDependency(sopDeployment);

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
