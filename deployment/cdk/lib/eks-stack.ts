import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as eks from "aws-cdk-lib/aws-eks";
import * as iam from "aws-cdk-lib/aws-iam";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as logs from "aws-cdk-lib/aws-logs";
import { Construct } from "constructs";
import { CfnParameter } from "aws-cdk-lib";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as s3 from "aws-cdk-lib/aws-s3";
import { KubectlV30Layer } from '@aws-cdk/lambda-layer-kubectl-v30';

export interface EksStackProps extends cdk.StackProps {
  resourcePrefix: string;
  environment: string;
  vpc: ec2.Vpc;
  domainName: string;
  certificateArn: string;
  dataBucket: s3.Bucket;
  description: string;
}

export class EksStack extends cdk.Stack {
  public readonly cluster: eks.Cluster;
  public readonly backendRepo: ecr.IRepository;

  constructor(scope: Construct, id: string, props: EksStackProps) {
    super(scope, id, props);

    const resourcePrefix = props.resourcePrefix;
    const environment = props.environment;
    const vpc = props.vpc;
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

    // Create the EKS cluster
    this.cluster = new eks.Cluster(this, `${resourcePrefix}-Cluster`, {
      version: eks.KubernetesVersion.V1_30,
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
      kubectlLayer: new KubectlV30Layer(this, `${resourcePrefix}-KubectlLayer`)
    });

    // Create security group for ALB only
    const albSecurityGroup = new ec2.SecurityGroup(this, `${resourcePrefix}-AlbSecurityGroup`, {
      vpc: vpc,
      description: 'Security group for Application Load Balancer',
      allowAllOutbound: true, // ALB needs to reach targets
    });

    // Allow HTTPS traffic from internet
    albSecurityGroup.addIngressRule(
      ec2.Peer.anyIpv4(),
      ec2.Port.tcp(443),
      'Allow HTTPS traffic from internet'
    );

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

    // Create a managed node group - let EKS handle security groups
    const nodeGroup = this.cluster.addNodegroupCapacity(`${resourcePrefix}-NodeGroup`, {
      instanceTypes: [
        ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM)
      ],
      minSize: 2,
      desiredSize: 2,
      maxSize: 4,
      nodeRole: nodeGroupRole,
      subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      amiType: eks.NodegroupAmiType.AL2_X86_64,
      capacityType: eks.CapacityType.ON_DEMAND,
      diskSize: 20,
      labels: {
        'role': 'application',
      },
      tags: {
        'Name': `${resourcePrefix}-worker-node`,
        'Environment': 'dev',
      },
    });

    // Add ALB access to the cluster's default node security group after node group is created
    const clusterNodeSecurityGroup = this.cluster.clusterSecurityGroup;
    clusterNodeSecurityGroup.addIngressRule(
      ec2.Peer.securityGroupId(albSecurityGroup.securityGroupId),
      ec2.Port.tcp(8000),
      'Allow traffic from ALB to backend pods'
    );

    // Create a namespace for our application
    const appNamespace = new eks.KubernetesManifest(this, 'AppNamespace', {
      cluster: this.cluster,
      manifest: [{
        apiVersion: 'v1',
        kind: 'Namespace',
        metadata: {
          name:  `${resourcePrefix}-app`,
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
      version: eks.AlbControllerVersion.V2_4_1
    });
    albController.node.addDependency(appNamespace);

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

    // Add permissions for Bedrock - include both foundation models and inference profiles
    backendServiceAccount.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:InvokeModel',
        'bedrock:InvokeModelWithResponseStream'        
      ],
      resources: [
        // Foundation models
        `arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-*`,
        `arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-*`,
        `arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-*`,
        `arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-*`,
        `arn:aws:bedrock:us-east-2::foundation-model/amazon.nova-*`,
        `arn:aws:bedrock:us-east-2::foundation-model/amazon.titan-*`,
        `arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-*`,
        `arn:aws:bedrock:us-west-2::foundation-model/amazon.nova-*`,
        `arn:aws:bedrock:us-west-2::foundation-model/amazon.titan-*`,
        `arn:aws:bedrock:eu-west-1::foundation-model/anthropic.claude-*`,
        `arn:aws:bedrock:eu-west-1::foundation-model/amazon.nova-*`,
        `arn:aws:bedrock:eu-west-1::foundation-model/amazon.titan-*`,
        `arn:aws:bedrock:ap-southeast-1::foundation-model/anthropic.claude-*`,
        `arn:aws:bedrock:ap-southeast-1::foundation-model/amazon.nova-*`,
        `arn:aws:bedrock:ap-southeast-1::foundation-model/amazon.titan-*`,
        `arn:aws:bedrock:ap-northeast-1::foundation-model/anthropic.claude-*`,
        `arn:aws:bedrock:ap-northeast-1::foundation-model/amazon.nova-*`,
        `arn:aws:bedrock:ap-northeast-1::foundation-model/amazon.titan-*`,
        // Inference profiles (cross-region and account-specific)
        `arn:aws:bedrock:${this.region}:${this.account}:inference-profile/*`,
        `arn:aws:bedrock:us-east-1:${this.account}:inference-profile/*`,
        `arn:aws:bedrock:us-east-2:${this.account}:inference-profile/*`,
        `arn:aws:bedrock:us-west-2:${this.account}:inference-profile/*`
      ]
    }));

    // Add permissions for Bedrock Guardrails
    backendServiceAccount.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:ApplyGuardrail'
      ],
      resources: [
        `arn:aws:bedrock:${this.region}:${this.account}:guardrail/*`,
        `arn:aws:bedrock:us-east-1:${this.account}:guardrail/*`,
        `arn:aws:bedrock:us-east-2:${this.account}:guardrail/*`,
        `arn:aws:bedrock:us-west-2:${this.account}:guardrail/*`
      ]
    }));

    // Add permissions for Knowledge Base retrieval - scoped to account
    backendServiceAccount.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'bedrock:Retrieve'
      ],
      resources: [
        `arn:aws:bedrock:${this.region}:${this.account}:knowledge-base/*`
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
        `arn:aws:dynamodb:${this.region}:${this.account}:table/${resourcePrefix}-IMAGES-${environment}`,
      ]
    }));
    
    // Add permissions for KMS - scoped to AWS managed keys and account keys
    backendServiceAccount.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        'kms:Decrypt',
        'kms:GenerateDataKey'
      ],
      resources: [
        `arn:aws:kms:${this.region}:${this.account}:key/*`,
        `arn:aws:kms:${this.region}:${this.account}:alias/aws/*`
      ]
    }));
    
    // Add permissions for s3
    backendServiceAccount.addToPrincipalPolicy(new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      actions: [
        's3:GetObject',
        's3:PutObject',
        's3:DeleteObject',
        's3:ListBucket',
      ],
      resources: [
        `arn:aws:s3:::${dataBucket.bucketName}`,
        `arn:aws:s3:::${dataBucket.bucketName}/*`
      ]
    }));

    // Output the service account role ARN
    new cdk.CfnOutput(this, 'BackendRoleArn', {
      value: backendServiceAccount.role.roleArn,
      description: 'The ARN of the IAM role for backend service account',
      exportName: `${resourcePrefix}-BackendRoleArn`
    });

    // Output the ALB security group ID
    new cdk.CfnOutput(this, 'AlbSecurityGroupId', {
      value: albSecurityGroup.securityGroupId,
      description: 'The ID of the ALB security group',
      exportName: `${resourcePrefix}-AlbSecurityGroupId`
    });

  }
}