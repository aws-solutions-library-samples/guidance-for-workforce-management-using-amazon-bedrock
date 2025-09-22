import * as cdk from "aws-cdk-lib";
import * as cognito from "aws-cdk-lib/aws-cognito";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import { Construct } from "constructs";

export interface AuthStackProps extends cdk.StackProps {
  resourcePrefix: string;
  environment: string;
  emailAddress: string;
  initialPassword: string;
}

export class AuthStack extends cdk.Stack {
  public readonly userPool: cognito.UserPool;
  public readonly userPoolClient: cognito.UserPoolClient;
  public readonly identityPool: cognito.CfnIdentityPool;
  public readonly authenticatedRole: iam.Role;

  constructor(scope: Construct, id: string, props: AuthStackProps) {
    super(scope, id, props);

    const resourcePrefix = props.resourcePrefix;

    // A Cognito User Pool
    this.userPool = new cognito.UserPool(this, `${resourcePrefix}-UserPool`, {
      selfSignUpEnabled: true,
      signInAliases: {
        email: true
      },
      autoVerify: {
        email: true
      },
      standardAttributes: {
        email: {
          required: true,
          mutable: true
        }
      },
    });

    // Create Admin User Group
    const adminGroup = new cognito.CfnUserPoolGroup(this, `${resourcePrefix}-AdminGroup`, {
      userPoolId: this.userPool.userPoolId,
      groupName: `${resourcePrefix}-admin`,
      precedence: 1,
    });

    // A Cognito User Pool Client
    this.userPoolClient = new cognito.UserPoolClient(this, `${resourcePrefix}-UserPoolClient`, {
      userPool: this.userPool,
      generateSecret: false,
      preventUserExistenceErrors: true,
    });

    // Create a user with the provided email address
    const initialUser = new cognito.CfnUserPoolUser(this, `${resourcePrefix}-InitialUser`, {
      userPoolId: this.userPool.userPoolId,
      username: props.emailAddress,
      userAttributes: [
        {
          name: 'email',
          value: props.emailAddress
        },
        {
          name: 'email_verified',
          value: 'true'
        }
      ],
      messageAction: 'SUPPRESS', // Don't send welcome email
    });

    // Create Lambda function to set up the user properly
    const userSetupFunction = new lambda.Function(this, `${resourcePrefix}-UserSetupFunction`, {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      environment: {
        COGNITO_PASSWORD: props.initialPassword || 'TempPassword123!'
      },
      code: lambda.Code.fromInline(`
import json
import boto3
import cfnresponse
import os

def handler(event, context):
    try:
        if event['RequestType'] == 'Delete':
            cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
            return
            
        cognito = boto3.client('cognito-idp')
        user_pool_id = event['ResourceProperties']['UserPoolId']
        username = event['ResourceProperties']['Username']
        group_name = event['ResourceProperties']['GroupName']
        
        # Get password from environment variable or use default
        password = os.environ.get('COGNITO_PASSWORD', 'TempPassword123!')
        
        # Set user password to confirmed state
        try:
            cognito.admin_set_user_password(
                UserPoolId=user_pool_id,
                Username=username,
                Password=password,
                Permanent=True  # Make the password permanent so users can sign in directly
            )
        except Exception as e:
            print(f"Password setting failed (user might not exist yet): {e}")
        
        # Add user to group
        try:
            cognito.admin_add_user_to_group(
                UserPoolId=user_pool_id,
                Username=username,
                GroupName=group_name
            )
            print(f"Successfully added user {username} to group {group_name}")
        except Exception as e:
            print(f"Group assignment failed: {e}")
            
        cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
        
    except Exception as e:
        print(f"Error: {e}")
        cfnresponse.send(event, context, cfnresponse.FAILED, {})
`),
      timeout: cdk.Duration.minutes(5),
      initialPolicy: [
        new iam.PolicyStatement({
          actions: [
            'cognito-idp:AdminSetUserPassword',
            'cognito-idp:AdminAddUserToGroup',
            'cognito-idp:AdminGetUser'
          ],
          resources: [this.userPool.userPoolArn]
        })
      ]
    });

    // Create custom resource to trigger user setup
    const userSetup = new cdk.CustomResource(this, `${resourcePrefix}-UserSetup`, {
      serviceToken: userSetupFunction.functionArn,
      properties: {
        UserPoolId: this.userPool.userPoolId,
        Username: props.emailAddress,
        GroupName: `${resourcePrefix}-admin`
      }
    });

    // Ensure user setup happens after user and group creation
    userSetup.node.addDependency(initialUser);
    userSetup.node.addDependency(adminGroup);

    // Output the Cognito User Pool Id
    new cdk.CfnOutput(this, `${resourcePrefix}-UserPoolId`, {
      value: this.userPool.userPoolId
    });

    // Output the Cognito User Pool Client Id
    new cdk.CfnOutput(this, `${resourcePrefix}-UserPoolClientId`, {
      value: this.userPoolClient.userPoolClientId
    });

    // Create an Identity Pool
    this.identityPool = new cognito.CfnIdentityPool(this, `${resourcePrefix}-IdentityPool`, {
      allowUnauthenticatedIdentities: false,
      cognitoIdentityProviders: [
        {
          clientId: this.userPoolClient.userPoolClientId,
          providerName: this.userPool.userPoolProviderName,
        },
      ],
    });

    // Output the Cognito Identity Pool Id
    new cdk.CfnOutput(this, `${resourcePrefix}-IdentityPoolId`, {
      value: this.identityPool.ref,
    });

    // Create roles for authenticated users
    this.authenticatedRole = new iam.Role(this, `${resourcePrefix}-AuthenticatedRole`, {
      assumedBy: new iam.FederatedPrincipal(
        'cognito-identity.amazonaws.com',
        {
          'StringEquals': {
            'cognito-identity.amazonaws.com:aud': this.identityPool.ref
          },
          'ForAnyValue:StringLike': {
            'cognito-identity.amazonaws.com:amr': 'authenticated'
          }
        },
        'sts:AssumeRoleWithWebIdentity'
      ),
      description: 'IAM role for authenticated Cognito users',
    });

    // Only allow STS permissions for identity verification with specific condition
    this.authenticatedRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'sts:GetCallerIdentity'
      ],
      resources: ['*'],
      conditions: {
        'StringEquals': {
          'aws:RequestedRegion': this.region
        }
      }
    }));

    // Minimal Cognito permissions for user management
    this.authenticatedRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'cognito-identity:GetId',
        'cognito-identity:GetCredentialsForIdentity'
      ],
      resources: [
        `arn:aws:cognito-identity:${this.region}:${this.account}:identitypool/${this.identityPool.ref}`
      ]
    }));
    
    // Add specific EKS permissions for cluster access
    // These permissions are needed for the EKS access entry in the EKS stack
    this.authenticatedRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'eks:DescribeCluster',
        'eks:AccessKubernetesApi'
      ],
      resources: [
        `arn:aws:eks:${this.region}:${this.account}:cluster/${resourcePrefix}-cluster`
      ],
      conditions: {
        'StringEquals': {
          'aws:RequestedRegion': this.region
        }
      }
    }));
    
    // Add specific permissions for ECR to allow pulling images
    this.authenticatedRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'ecr:GetDownloadUrlForLayer',
        'ecr:BatchGetImage',
        'ecr:BatchCheckLayerAvailability',
        'ecr:GetAuthorizationToken'
      ],
      resources: [
        `arn:aws:ecr:${this.region}:${this.account}:repository/${resourcePrefix}-backend`
      ],
      conditions: {
        'StringEquals': {
          'aws:RequestedRegion': this.region
        }
      }
    }));
    
    // GetAuthorizationToken requires a wildcard resource
    this.authenticatedRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'ecr:GetAuthorizationToken'
      ],
      resources: ['*'],
      conditions: {
        'StringEquals': {
          'aws:RequestedRegion': this.region
        }
      }
    }));
    
    // Add specific permissions for S3 buckets - including write permissions for image uploads
    this.authenticatedRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        's3:GetObject',
        's3:PutObject',
        's3:DeleteObject',
        's3:ListBucket',
        's3:GetBucketLocation',
        's3:HeadBucket'
      ],
      resources: [
        `arn:aws:s3:::${resourcePrefix}-databucket*`,
        `arn:aws:s3:::${resourcePrefix}-databucket*/*`,
        // Also allow access to the actual bucket name pattern used by CDK
        `arn:aws:s3:::*${resourcePrefix.toLowerCase()}*databucket*`,
        `arn:aws:s3:::*${resourcePrefix.toLowerCase()}*databucket*/*`
      ],
      conditions: {
        'StringEquals': {
          'aws:RequestedRegion': this.region
        }
      }
    }));
    
    // Add specific permissions for Bedrock
    this.authenticatedRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:InvokeModel',
        'bedrock:InvokeModelWithResponseStream'
      ],
      resources: [
        // Foundation models - specific to commonly used models
        `arn:aws:bedrock:${this.region}::foundation-model/anthropic.claude-*`,
        `arn:aws:bedrock:${this.region}::foundation-model/amazon.nova-*`,
        `arn:aws:bedrock:${this.region}::foundation-model/amazon.titan-*`
      ],
      conditions: {
        'StringEquals': {
          'aws:RequestedRegion': this.region
        }
      }
    }));
    
    // Add specific permissions for Bedrock Guardrails
    this.authenticatedRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:ApplyGuardrail'
      ],
      resources: [
        `arn:aws:bedrock:${this.region}:${this.account}:guardrail/*`,
        `arn:aws:bedrock:${this.region}:${this.account}:guardrail-profile/*`
      ],
      conditions: {
        'StringEquals': {
          'aws:RequestedRegion': this.region
        }
      }
    }));
    
    // Add specific permissions for Knowledge Base retrieval
    this.authenticatedRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:Retrieve'
      ],
      resources: [
        `arn:aws:bedrock:${this.region}:${this.account}:knowledge-base/*`
      ],
      conditions: {
        'StringEquals': {
          'aws:RequestedRegion': this.region
        }
      }
    }));
    
    // Add specific permissions for DynamoDB tables
    const environment = props.environment || 'DEV';
    this.authenticatedRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'dynamodb:GetItem',
        'dynamodb:Query',
        'dynamodb:Scan'
      ],
      resources: [
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${resourcePrefix}-USERROLE-${environment}`,
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${resourcePrefix}-SESSION-HISTORY-${environment}`,
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${resourcePrefix}-TASKLIST-${environment}`,
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${resourcePrefix}-CUSTOMER-${environment}`,
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${resourcePrefix}-DAILY_TASKS_BY_DAY-${environment}`,
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${resourcePrefix}-SCHEDULE-${environment}`,
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${resourcePrefix}-TIMEOFF-${environment}`,
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${resourcePrefix}-PRODUCTS-${environment}`,
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${resourcePrefix}-CUSTOMER_TRANSACTIONS-${environment}`,
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${resourcePrefix}-FEEDBACK-${environment}`,
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${resourcePrefix}-IMAGES-${environment}`
      ],
      conditions: {
        'StringEquals': {
          'aws:RequestedRegion': this.region
        }
      }
    }));

    // Attach roles to the Identity Pool
    new cognito.CfnIdentityPoolRoleAttachment(this, 'IdentityPoolRoleAttachment', {
      identityPoolId: this.identityPool.ref,
      roles: {
        authenticated: this.authenticatedRole.roleArn,
      },
    });
  }
}
