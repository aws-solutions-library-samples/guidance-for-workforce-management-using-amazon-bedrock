import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as eks from "aws-cdk-lib/aws-eks";
import * as iam from "aws-cdk-lib/aws-iam";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as logs from "aws-cdk-lib/aws-logs";
import { CfnJson } from "aws-cdk-lib";
import { Construct } from "constructs";
import * as s3 from "aws-cdk-lib/aws-s3";
import { KubectlV33Layer } from '@aws-cdk/lambda-layer-kubectl-v33';

export interface EksStackProps extends cdk.StackProps {
  resourcePrefix: string;
  environment: string;
  vpc: ec2.Vpc;
  authenticatedRole: iam.Role;
  dataBucket: s3.Bucket;
}

export class EksStack extends cdk.Stack {
  public readonly cluster: eks.Cluster;
  public readonly backendRepo: ecr.IRepository;
  public readonly albControllerRole: iam.IRole;
  public readonly albSecurityGroup: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: EksStackProps) {
    super(scope, id, props);

    const resourcePrefix = props.resourcePrefix;
    const environment = props.environment;
    const vpc = props.vpc;
    const authenticatedRole = props.authenticatedRole;
    const dataBucket = props.dataBucket;


    // Create or import log groups with retention policy
    const clusterLogGroupName = `/aws/eks/${resourcePrefix}-cluster`;
    let clusterLogGroup: logs.ILogGroup;

    try {
      // Try to import existing log group
      clusterLogGroup = logs.LogGroup.fromLogGroupName(
        this,
        `${resourcePrefix}-ImportedClusterLogGroup`,
        clusterLogGroupName
      );
      console.log(`Imported existing log group: ${clusterLogGroupName}`);
    } catch (e) {
      // Create new log group if import fails
      clusterLogGroup = new logs.LogGroup(this, `${resourcePrefix}-ClusterLogGroup`, {
        logGroupName: clusterLogGroupName,
        removalPolicy: cdk.RemovalPolicy.DESTROY,
      });
    }

    // Create or import LB controller log group
    const lbControllerLogGroupName = `/aws/eks/${resourcePrefix}-cluster/lb-controller`;
    let lbControllerLogGroup: logs.ILogGroup;

    try {
      // Try to import existing log group
      lbControllerLogGroup = logs.LogGroup.fromLogGroupName(
        this,
        `${resourcePrefix}-ImportedLBControllerLogGroup`,
        lbControllerLogGroupName
      );
      console.log(`Imported existing log group: ${lbControllerLogGroupName}`);
    } catch (e) {
      // Create new log group if import fails
      lbControllerLogGroup = new logs.LogGroup(this, `${resourcePrefix}-LBControllerLogGroup`, {
        logGroupName: lbControllerLogGroupName,
        removalPolicy: cdk.RemovalPolicy.DESTROY,
      });
    }

    // Handle the Bedrock AgentCore observability log group creation safely
    const bedrockObservabilityLogGroupName = 'bedrock-agentcore-observability';
    
    // First, import the log group reference (this doesn't check if it exists)
    const bedrockObservabilityLogGroup = logs.LogGroup.fromLogGroupName(
      this,
      `${resourcePrefix}-ImportedBedrockObservabilityLogGroup`,
      bedrockObservabilityLogGroupName
    );
    
    // Use AWS SDK via a custom resource to check if the log group exists
    // and create it if it doesn't exist
    const checkAndCreateLogGroup = new cdk.custom_resources.AwsCustomResource(this, `${resourcePrefix}-CheckAndCreateLogGroup`, {
      onCreate: {
        service: 'CloudWatchLogs',
        action: 'describeLogGroups',
        parameters: {
          logGroupNamePrefix: bedrockObservabilityLogGroupName
        },
        physicalResourceId: cdk.custom_resources.PhysicalResourceId.of(`${bedrockObservabilityLogGroupName}-check`),
      },
      onUpdate: {
        service: 'CloudWatchLogs',
        action: 'describeLogGroups',
        parameters: {
          logGroupNamePrefix: bedrockObservabilityLogGroupName
        },
        physicalResourceId: cdk.custom_resources.PhysicalResourceId.of(`${bedrockObservabilityLogGroupName}-check`),
      },
      policy: cdk.custom_resources.AwsCustomResourcePolicy.fromSdkCalls({
        resources: cdk.custom_resources.AwsCustomResourcePolicy.ANY_RESOURCE
      })
    });
    
    // Create a Lambda function to process the result and create the log group if needed
    const createLogGroupIfNeeded = new cdk.custom_resources.AwsCustomResource(this, `${resourcePrefix}-CreateLogGroupIfNeeded`, {
      onCreate: {
        service: 'CloudWatchLogs',
        action: 'createLogGroup',
        parameters: {
          logGroupName: bedrockObservabilityLogGroupName,
          // Remove tags to avoid needing logs:TagResource permission
          // tags: {
          //   'ManagedBy': 'CDK',
          //   'Purpose': 'Bedrock AgentCore Observability'
          // }
        },
        physicalResourceId: cdk.custom_resources.PhysicalResourceId.of(`${bedrockObservabilityLogGroupName}-create`),
        ignoreErrorCodesMatching: 'ResourceAlreadyExistsException' // Ignore if already exists
      },
      onUpdate: {
        service: 'CloudWatchLogs',
        action: 'putRetentionPolicy',
        parameters: {
          logGroupName: bedrockObservabilityLogGroupName,
          retentionInDays: 7 // ONE_WEEK
        },
        physicalResourceId: cdk.custom_resources.PhysicalResourceId.of(`${bedrockObservabilityLogGroupName}-update`),
      },
      policy: cdk.custom_resources.AwsCustomResourcePolicy.fromStatements([
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            'logs:CreateLogGroup',
            'logs:DescribeLogGroups',
            'logs:PutRetentionPolicy',
            'logs:TagResource'  // Add explicit permission for tagging
          ],
          resources: ['*']
        })
      ])
    });
    
    // Ensure the check happens before the create
    createLogGroupIfNeeded.node.addDependency(checkAndCreateLogGroup);
    
    // Create the default log stream in the log group
    const createLogStream = new cdk.custom_resources.AwsCustomResource(this, `${resourcePrefix}-CreateLogStream`, {
      onCreate: {
        service: 'CloudWatchLogs',
        action: 'createLogStream',
        parameters: {
          logGroupName: bedrockObservabilityLogGroupName,
          logStreamName: 'default'
        },
        physicalResourceId: cdk.custom_resources.PhysicalResourceId.of(`${bedrockObservabilityLogGroupName}-default-stream`),
        ignoreErrorCodesMatching: 'ResourceAlreadyExistsException' // Ignore if already exists
      },
      policy: cdk.custom_resources.AwsCustomResourcePolicy.fromStatements([
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            'logs:CreateLogStream',
            'logs:DescribeLogStreams'
          ],
          resources: ['*']
        })
      ])
    });
    
    // Ensure the log group is created before the log stream
    createLogStream.node.addDependency(createLogGroupIfNeeded);
    
    console.log(`Ensured log group and default stream exist: ${bedrockObservabilityLogGroupName}`);

    // Create a dedicated security group for the ALB
    this.albSecurityGroup = new ec2.SecurityGroup(this, `${resourcePrefix}-AlbSecurityGroup`, {
      vpc,
      description: 'Security group for the Application Load Balancer',
      allowAllOutbound: false, // Restrict outbound traffic
    });

    // Add ingress rule for HTTPS traffic (port 443)
    this.albSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(443),
      'Allow HTTPS traffic from anywhere'
    );

    // Add ingress rule for HTTP traffic (port 80) - for redirect to HTTPS
    this.albSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(80),
      'Allow HTTP traffic from anywhere (for redirect to HTTPS)'
    );

    // Add egress rule to allow traffic to the EKS cluster nodes
    this.albSecurityGroup.addEgressRule(
      ec2.Peer.ipv4(vpc.vpcCidrBlock),
      ec2.Port.tcpRange(30000, 32767),
      'Allow traffic to NodePort range in the EKS cluster'
    );

    // Add egress rule to allow traffic to the backend service on port 8000
    this.albSecurityGroup.addEgressRule(
      ec2.Peer.ipv4(vpc.vpcCidrBlock),
      ec2.Port.tcp(8000),
      'Allow traffic to backend service on port 8000'
    );


    // Create the EKS cluster
    this.cluster = new eks.Cluster(this, `${resourcePrefix}-Cluster`, {
      version: eks.KubernetesVersion.V1_33,
      vpc,
      defaultCapacity: 0, // We'll create our own node group instead of using the default
      clusterName: `${resourcePrefix}-cluster`,
      outputClusterName: true,
      outputConfigCommand: true,
      clusterLogging: [
        eks.ClusterLoggingTypes.API,
        eks.ClusterLoggingTypes.AUDIT,
        eks.ClusterLoggingTypes.AUTHENTICATOR,
        eks.ClusterLoggingTypes.CONTROLLER_MANAGER,
        eks.ClusterLoggingTypes.SCHEDULER,
      ],
      authenticationMode: eks.AuthenticationMode.API,
      kubectlLayer: new KubectlV33Layer(this, `${resourcePrefix}-KubectlLayer`)
    });

    // Allow traffic from ALB security group to EKS cluster security group on port 8000
    this.cluster.clusterSecurityGroup.addIngressRule(
      ec2.Peer.securityGroupId(this.albSecurityGroup.securityGroupId),
      ec2.Port.tcp(8000),
      'Allow traffic from ALB to backend service on port 8000'
    );

    // Add the EKS Pod Identity Agent add-on
    const podIdentityAddon = new eks.CfnAddon(this, 'EksPodIdentityAgent', {
      addonName: 'eks-pod-identity-agent',
      clusterName: this.cluster.clusterName,
      addonVersion: 'v1.3.8-eksbuild.2',
      resolveConflicts: 'OVERWRITE',
    });

    // Create IAM role for EKS node group
    const nodeGroupRole = new iam.Role(this, `${resourcePrefix}-NodeGroupRole`, {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      description: 'IAM role for EKS node group',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKSWorkerNodePolicy'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKS_CNI_Policy'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryReadOnly'),
      ],
    });

    // Create a managed node group with optimized configuration
    const nodeGroup = this.cluster.addNodegroupCapacity(`${resourcePrefix}-NodeGroup`, {
      instanceTypes: [
        // Use a mix of instance types for better cost optimization and availability
        // ec2.InstanceType.of(ec2.InstanceClass.T3A, ec2.InstanceSize.LARGE), // AMD-based, cost-effective
        // ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.LARGE),  // Intel-based
        // ec2.InstanceType.of(ec2.InstanceClass.M5A, ec2.InstanceSize.LARGE), // AMD-based, better performance
        // ec2.InstanceType.of(ec2.InstanceClass.M5, ec2.InstanceSize.LARGE),  // Intel-based, better performance
        ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM), // Smaller instance for burstable workloads
      ],
      minSize: 1,
      desiredSize: 2,
      maxSize: 4,     // Increased to allow for more scaling headroom

      nodeRole: nodeGroupRole,
      subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      amiType: eks.NodegroupAmiType.AL2023_X86_64_STANDARD, // AL2023_X86_64_STANDARD
      capacityType: eks.CapacityType.ON_DEMAND,
      diskSize: 20,
      labels: {
        'role': 'application',
        'workload-type': 'general',
        'environment': environment,
      },
      tags: {
        'Name': `${resourcePrefix}-worker-node`,
        'Environment': environment,
        'ManagedBy': 'CDK',
        'Purpose': 'Application workloads',
      },

    });

    // Create IAM role for EKS admin access
    const eksAdminRole = new iam.Role(this, `${resourcePrefix}-EksAdminRole`, {
      assumedBy: new iam.ServicePrincipal('eks.amazonaws.com'),
      description: 'IAM role for EKS admin access',
    });

    // Attach necessary policies to the admin role
    eksAdminRole.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEKSClusterPolicy')
    );

    // Add custom policy for admin access
    eksAdminRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'eks:*',
        'cloudwatch:*',
        'ecr:*',
        'logs:*',
        'secretsmanager:*',
      ],
      resources: ['*']
    }));

    // Add scoped KMS permissions for admin role
    eksAdminRole.addToPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'kms:Decrypt',
        'kms:GenerateDataKey',
        'kms:DescribeKey',
        'kms:ListKeys',
        'kms:ListAliases'
      ],
      resources: [
        // S3 bucket encryption key
        `arn:aws:kms:${this.region}:${this.account}:key/alias/aws/s3`,
        // DynamoDB encryption key
        `arn:aws:kms:${this.region}:${this.account}:key/alias/aws/dynamodb`,
        // EBS encryption key (for EKS volumes)
        `arn:aws:kms:${this.region}:${this.account}:key/alias/aws/ebs`,
        // CloudWatch Logs encryption key
        `arn:aws:kms:${this.region}:${this.account}:key/alias/aws/logs`,
        // ECR encryption key
        `arn:aws:kms:${this.region}:${this.account}:key/alias/aws/ecr`,
        // Secrets Manager encryption key
        `arn:aws:kms:${this.region}:${this.account}:key/alias/aws/secretsmanager`,
        // Any customer-managed keys
        `arn:aws:kms:${this.region}:${this.account}:key/*`
      ]
    }));

    // Create access entry for admin role
    const accessEntry = new eks.AccessEntry(this, `${resourcePrefix}-AdminAccessEntry`, {
      cluster: this.cluster,
      principal: eksAdminRole.roleArn,
      accessPolicies: [
        {
          policy: 'arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy',
          accessScope: {
            type: eks.AccessScopeType.CLUSTER
          }
        }
      ]
    });

    // Create access entry for the authenticated user (CDK deployer)
    const userAccessEntry = new eks.AccessEntry(this, `${resourcePrefix}-UserAccessEntry`, {
      cluster: this.cluster,
      principal: authenticatedRole.roleArn,
      accessPolicies: [
        {
          policy: 'arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy',
          accessScope: {
            type: eks.AccessScopeType.CLUSTER
          }
        }
      ]
    });


    // Add aws-auth ConfigMap to handle assumed roles
    const awsAuth = new eks.KubernetesManifest(this, 'AwsAuth', {
      cluster: this.cluster,
      manifest: [{
        apiVersion: 'v1',
        kind: 'ConfigMap',
        metadata: {
          name: 'aws-auth',
          namespace: 'kube-system'
        },
        data: {
          mapRoles: `- rolearn: arn:aws:iam::${cdk.Stack.of(this).account}:role/Admin
  username: admin:{{SessionName}}
  groups:
    - system:masters
- rolearn: arn:aws:sts::${cdk.Stack.of(this).account}:assumed-role/Admin/*
  username: admin:{{SessionName}}
  groups:
    - system:masters`
        }
      }],
      overwrite: true,
    });

    // Create a namespace for our application
    const appNamespace = new eks.KubernetesManifest(this, 'AppNamespace', {
      cluster: this.cluster,
      manifest: [{
        apiVersion: 'v1',
        kind: 'Namespace',
        metadata: {
          name: `${resourcePrefix}-app`,
          labels: {
            name: `${resourcePrefix}-app`
          }
        }
      }],
      overwrite: true, // Allow overwriting if namespace already exists
    });

    // Install the AWS Load Balancer Controller
    const albController = new eks.AlbController(this, 'AlbController', {
      cluster: this.cluster,
      version: eks.AlbControllerVersion.V2_8_2
    });
    albController.node.addDependency(this.cluster);
    albController.node.addDependency(nodeGroup);
    albController.node.addDependency(appNamespace);
    albController.node.addDependency(podIdentityAddon); // Add dependency on the Pod Identity Agent

    // Create ECR repositories with RETAIN removal policy
    const backendRepoName = `${resourcePrefix}-backend`;
    const backendRepo = new ecr.Repository(this, `${resourcePrefix}-BackendRepo`, {
      repositoryName: backendRepoName,
      lifecycleRules: [
        {
          maxImageCount: 2,
          description: 'Keep only the 2 most recent images',
        },
      ],
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    this.backendRepo = backendRepo;

    // Output the retail-backend repository URI
    new cdk.CfnOutput(this, 'BackendRepositoryUri', {
      value: backendRepo.repositoryUri,
      description: 'The URI of the backend ECR repository',
      exportName: `${resourcePrefix}-BackendRepositoryUri`,
    });

    // Output the retail-backend repository name
    new cdk.CfnOutput(this, 'BackendRepositoryName', {
      value: backendRepoName,
      description: 'The name of the backend ECR repository',
      exportName: `${resourcePrefix}-BackendRepositoryName`,
    });

    // Output the EKS cluster name
    new cdk.CfnOutput(this, `${resourcePrefix}-ClusterName`, {
      value: this.cluster.clusterName,
    });

    // Output the node group name
    new cdk.CfnOutput(this, `${resourcePrefix}-NodeGroupName`, {
      value: nodeGroup.nodegroupName,
    });

    // Create service account with the custom role
    const backendServiceAccount = new eks.ServiceAccount(this, `${resourcePrefix}-backend-sa`, {
      cluster: this.cluster,
      name: `${resourcePrefix}-backend-sa`,
      namespace: `${resourcePrefix}-app`,
      annotations: {
        "eks.amazonaws.com/token-expiration": "86400"  // 24 hours in seconds
      }
    });
    backendServiceAccount.node.addDependency(appNamespace);

    // Enhance the trust policy to include both the OIDC provider (for IRSA) and pods.eks.amazonaws.com (for Pod Identity)
    // Get the underlying CloudFormation role resource
    const cfnRole = backendServiceAccount.role.node.defaultChild as iam.CfnRole;

    // Create a CfnJson resource for the condition to handle dynamic keys
    const oidcCondition = new CfnJson(this, `${resourcePrefix}-OidcCondition`, {
      value: {
        [`${this.cluster.openIdConnectProvider.openIdConnectProviderIssuer}:sub`]: `system:serviceaccount:${resourcePrefix}-app:${resourcePrefix}-backend-sa`,
        [`${this.cluster.openIdConnectProvider.openIdConnectProviderIssuer}:aud`]: 'sts.amazonaws.com'
      }
    });

    // Create a new trust policy that includes both principals
    cfnRole.assumeRolePolicyDocument = {
      Version: '2012-10-17',
      Statement: [
        // Keep the existing OIDC provider trust relationship (for IRSA)
        {
          Effect: 'Allow',
          Principal: {
            Federated: this.cluster.openIdConnectProvider.openIdConnectProviderArn
          },
          Action: 'sts:AssumeRoleWithWebIdentity',
          Condition: {
            StringEquals: oidcCondition
          }
        },
        // Add the Pod Identity trust relationship
        {
          Sid: 'AllowEksAuthToAssumeRoleForPodIdentity',
          Effect: 'Allow',
          Principal: {
            Service: 'pods.eks.amazonaws.com'
          },
          Action: [
            'sts:AssumeRole',
            'sts:TagSession'
          ]
        }
      ]
    };

    // Add granular permissions for Bedrock models
    backendServiceAccount.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:InvokeModel',
        'bedrock:InvokeModelWithResponseStream',
      ],
      resources: [
      // Foundation models without account (global models)
        `arn:aws:bedrock:::foundation-model/anthropic.claude-*`,
        `arn:aws:bedrock:::foundation-model/amazon.nova-*`,
        `arn:aws:bedrock:::foundation-model/amazon.titan-*`,
        `arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-*`,
        `arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-*`,
        `arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-*`,
        `arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-*`,
        `arn:aws:bedrock:us-east-2::foundation-model/amazon.nova-*`,
        `arn:aws:bedrock:us-east-2::foundation-model/amazon.titan-*`,
        `arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-*`,
        `arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-*`,
        `arn:aws:bedrock:us-west-2::foundation-model/amazon.titan-*`,
        `arn:aws:bedrock:${this.region}::foundation-model/anthropic.claude-*`,
        `arn:aws:bedrock:${this.region}::foundation-model/amazon.nova-*`,
        `arn:aws:bedrock:${this.region}::foundation-model/amazon.titan-*`,
        // Inference profiles (cross-region and account-specific)
        `arn:aws:bedrock:${this.region}:${this.account}:inference-profile/*`,
        `arn:aws:bedrock:us-east-1:${this.account}:inference-profile/*`,
        `arn:aws:bedrock:us-east-2:${this.account}:inference-profile/*`,
        `arn:aws:bedrock:us-west-2:${this.account}:inference-profile/*`,
        `arn:aws:bedrock:us-east-1:${this.account}:guardrail-profile/*`,
        `arn:aws:bedrock:us-east-2:${this.account}:guardrail-profile/*`,
        `arn:aws:bedrock:us-west-1:${this.account}:guardrail-profile/*`,
        `arn:aws:bedrock:us-west-2:${this.account}:guardrail-profile/*`
      ]
    }));

    // Add permissions for Bedrock Guardrails
    backendServiceAccount.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:ApplyGuardrail',
        'bedrock:GetGuardrail',
        'bedrock:ListGuardrails',
      ],
      resources: [
        `arn:aws:bedrock:${this.region}:${this.account}:guardrail/*`,
        `arn:aws:bedrock:us-east-1:${this.account}:guardrail/*`,
        `arn:aws:bedrock:us-east-2:${this.account}:guardrail/*`,
        `arn:aws:bedrock:us-west-2:${this.account}:guardrail/*`,
        `arn:aws:bedrock:us-east-1:${this.account}:guardrail-profile/*`,
        `arn:aws:bedrock:us-east-2:${this.account}:guardrail-profile/*`,
        `arn:aws:bedrock:us-west-1:${this.account}:guardrail-profile/*`,
        `arn:aws:bedrock:us-west-2:${this.account}:guardrail-profile/*`
      ]
    }));

    // Add permissions for Bedrock Knowledge Base
    backendServiceAccount.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:Retrieve',
        'bedrock:RetrieveAndGenerate',
        'bedrock:GetKnowledgeBase',
        'bedrock:ListKnowledgeBases',
      ],
      resources: [
        `arn:aws:bedrock:${this.region}:${this.account}:knowledge-base/*`,
      ]
    }));

    // Add permissions for Bedrock Agents
    backendServiceAccount.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:InvokeAgent',
        'bedrock:GetAgent',
        'bedrock:ListAgents',
      ],
      resources: [
        `arn:aws:bedrock:${this.region}:${this.account}:agent/*`,
      ]
    }));

    // Add explicit permissions for DynamoDB tables
    backendServiceAccount.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'dynamodb:GetItem',
        'dynamodb:PutItem',
        'dynamodb:UpdateItem',
        'dynamodb:DeleteItem',
        'dynamodb:Query',
        'dynamodb:Scan',
        'dynamodb:BatchGetItem',
        'dynamodb:BatchWriteItem',
        'dynamodb:DescribeTable',
        'dynamodb:ListTables'
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
      ]
    }));

    // Add permissions for KMS with specific key ARNs
    backendServiceAccount.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'kms:Decrypt',
        'kms:GenerateDataKey'
      ],
      resources: [
        // S3 bucket encryption key
        `arn:aws:kms:${this.region}:${this.account}:key/alias/aws/s3`,
        // DynamoDB encryption key
        `arn:aws:kms:${this.region}:${this.account}:key/alias/aws/dynamodb`,
        // EBS encryption key (for EKS volumes)
        `arn:aws:kms:${this.region}:${this.account}:key/alias/aws/ebs`,
        // CloudWatch Logs encryption key
        `arn:aws:kms:${this.region}:${this.account}:key/alias/aws/logs`,
        // Any customer-managed keys for the data bucket
        `arn:aws:kms:${this.region}:${this.account}:key/*`
      ]
    }));

    // Add permissions for s3 - including HeadBucket for bucket access validation
    backendServiceAccount.addToPrincipalPolicy(new iam.PolicyStatement({
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
        `arn:aws:s3:::${dataBucket.bucketName}`,
        `arn:aws:s3:::${dataBucket.bucketName}/*`
      ]
    }));

    // Add permissions for CloudWatch Logs (OpenTelemetry observability)
    backendServiceAccount.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents',
        'logs:DescribeLogGroups',
        'logs:DescribeLogStreams',
      ],
      resources: [
        `arn:aws:logs:${this.region}:${this.account}:log-group:bedrock-agentcore-observability`,
        `arn:aws:logs:${this.region}:${this.account}:log-group:bedrock-agentcore-observability:*`,
        `arn:aws:logs:${this.region}:${this.account}:log-group:/aws/eks/${resourcePrefix}-cluster:*`,
      ]
    }));

    // Add permissions for X-Ray tracing (OpenTelemetry traces)
    backendServiceAccount.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'xray:PutTraceSegments',
        'xray:PutTelemetryRecords',
        'xray:GetSamplingRules',
        'xray:GetSamplingTargets',
      ],
      resources: ['*']
    }));

    // Create the Pod Identity Association
    // Let CloudFormation handle updates if this resource already exists
    console.log(`Creating Pod Identity Association for ${resourcePrefix}-app:${resourcePrefix}-backend-sa`);

    const podIdentityAssociation = new eks.CfnPodIdentityAssociation(this, `${resourcePrefix}-PodIdentityAssociation`, {
      clusterName: this.cluster.clusterName,
      namespace: `${resourcePrefix}-app`,
      serviceAccount: `${resourcePrefix}-backend-sa`,
      roleArn: backendServiceAccount.role.roleArn
    });

    // Add dependencies
    podIdentityAssociation.node.addDependency(backendServiceAccount);
    podIdentityAssociation.node.addDependency(podIdentityAddon);

    // Output the Pod Identity Association ID
    new cdk.CfnOutput(this, 'PodIdentityAssociationId', {
      value: `${this.cluster.clusterName}:${resourcePrefix}-app:${resourcePrefix}-backend-sa`,
      description: 'The ID of the Pod Identity Association',
      exportName: `${resourcePrefix}-PodIdentityAssociationId`
    });

    // Output the service account role ARN
    new cdk.CfnOutput(this, 'BackendRoleArn', {
      value: backendServiceAccount.role.roleArn,
      description: 'The ARN of the IAM role for backend service account',
      exportName: `${resourcePrefix}-BackendRoleArn`
    });

    // Output the ALB security group ID
    new cdk.CfnOutput(this, 'AlbSecurityGroupId', {
      value: this.albSecurityGroup.securityGroupId,
      description: 'The ID of the ALB security group',
      exportName: `${resourcePrefix}-AlbSecurityGroupId`
    });

  }
}
