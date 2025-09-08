#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { InfrastructureStack } from '../lib/infrastructure-stack';
import { StorageStack } from '../lib/storage-stack';
import { AuthStack } from '../lib/auth-stack';
import { EksStack } from '../lib/eks-stack';
import { OpenSearchStack } from '../lib/opensearch-stack';
import { SyntheticDataStack } from '../lib/synthetic-data-stack';
import { GuardrailsStack } from '../lib/guardrails-stack';
const app = new cdk.App();

// Define environment
const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION || 'us-east-1'
};

// Define common properties
const stackEnvironment = process.env.STACK_ENVIRONMENT;
const resourcePrefix = process.env.STACK_NAME;
const parentDomainName = process.env.PARENT_DOMAIN_NAME;
const domainName = process.env.DOMAIN_NAME;
const certificateArn = process.env.CERTIFICATE_ARN;
const webCertificateArn = process.env.WEB_CERTIFICATE_ARN;
const emailAddress = process.env.EMAIL;
if (!resourcePrefix) {
  throw new Error('STACK_NAME environment variable must be set');
}

if (!stackEnvironment) {
  throw new Error('STACK_ENVIRONMENT environment variable must be set');
}

if (!domainName || !certificateArn || !parentDomainName || !webCertificateArn) {
  throw new Error('DOMAIN_NAME, CERTIFICATE_ARN, PARENT_DOMAIN_NAME, and WEB_CERTIFICATE_ARN environment variables must be set');
}

if (!emailAddress) {
  throw new Error('EMAIL environment variable must be set');
}

// Type assertions since we've checked they're defined
const stackName = resourcePrefix as string;
const envName = stackEnvironment as string;
const domain = domainName as string;
const certArn = certificateArn as string;
const webCertArn = webCertificateArn as string;
const parentDomain = parentDomainName as string;
const email = emailAddress as string;

// Common tags for all stacks
const commonTags = {
  Project: stackName,
  Environment: envName
};

// Create infrastructure stack (VPC)
const infraStack = new InfrastructureStack(app, `${stackName}InfraStack`, { 
  env,
  resourcePrefix: stackName,
  terminationProtection: false,
  description: 'Guidance for Workforce Management using Amazon Bedrock (SO9595) - Core infrastructure resources including VPC and networking'
});
// Add tags
cdk.Tags.of(infraStack).add('Component', 'Infrastructure');
Object.entries(commonTags).forEach(([key, value]) => {
  cdk.Tags.of(infraStack).add(key, value);
});

// Create storage stack (S3, DynamoDB)
const storageStack = new StorageStack(app, `${stackName}StorageStack`, { 
  env,
  resourcePrefix: stackName,
  environment: envName,
  domainName: domain,
  certificateArn: webCertArn,
  parentDomainName: parentDomain,
  terminationProtection: false,
  description: 'Guidance for Workforce Management using Amazon Bedrock (SO9595) - Storage resources including S3 buckets and DynamoDB tables'
});
// Add tags
cdk.Tags.of(storageStack).add('Component', 'Storage');
Object.entries(commonTags).forEach(([key, value]) => {
  cdk.Tags.of(storageStack).add(key, value);
});

// Create auth stack (Cognito)
const authStack = new AuthStack(app, `${stackName}AuthStack`, {
  env,
  resourcePrefix: stackName,
  environment: envName,
  emailAddress: email,
  terminationProtection: false, 
  description: 'Guidance for Workforce Management using Amazon Bedrock (SO9595) - Authentication resources including Cognito user pools'
});
// Add tags
cdk.Tags.of(authStack).add('Component', 'Authentication');
Object.entries(commonTags).forEach(([key, value]) => {
  cdk.Tags.of(authStack).add(key, value);
});

// Create EKS stack
const eksStack = new EksStack(app, `${stackName}EksStack`, {
  env,
  resourcePrefix: stackName,
  environment: envName,
  vpc: infraStack.vpc,
  authenticatedRole: authStack.authenticatedRole,
  domainName: domain,
  certificateArn: certArn,
  terminationProtection: false,
  dataBucket: storageStack.dataBucket,
  description: 'Guidance for Workforce Management using Amazon Bedrock (SO9595) - EKS cluster and backend resources'
});
// Add tags
cdk.Tags.of(eksStack).add('Component', 'EKS');
Object.entries(commonTags).forEach(([key, value]) => {
  cdk.Tags.of(eksStack).add(key, value);
});

// Create OpenSearch stack
const openSearchStack = new OpenSearchStack(app, `${stackName}OpenSearchStack`, {
  env,
  resourcePrefix: stackName,
  environment: envName,
  dataBucket: storageStack.dataBucket,
  description: 'Guidance for Workforce Management using Amazon Bedrock (SO9595) - OpenSearch and Bedrock knowledge base resources'
});
// Add tags
cdk.Tags.of(openSearchStack).add('Component', 'Search');
Object.entries(commonTags).forEach(([key, value]) => {
  cdk.Tags.of(openSearchStack).add(key, value);
});

// Create synthetic data stack
const syntheticDataStack = new SyntheticDataStack(app, `${stackName}SyntheticDataStack`, {
  env,
  resourcePrefix: stackName,
  environment: envName,
  dataBucket: storageStack.dataBucket,
  emailAddress: email,
  tables: storageStack.tables
});
// Add tags
cdk.Tags.of(syntheticDataStack).add('Component', 'Data');
Object.entries(commonTags).forEach(([key, value]) => {
  cdk.Tags.of(syntheticDataStack).add(key, value);
});  

// Create guardrails stack
const guardrailsStack = new GuardrailsStack(app, `${stackName}GuardrailsStack`, {
  env,
  resourcePrefix: stackName,
  environment: envName
});
// Add tags
cdk.Tags.of(guardrailsStack).add('Component', 'Guardrails');
Object.entries(commonTags).forEach(([key, value]) => {
  cdk.Tags.of(guardrailsStack).add(key, value);
});  



// Add dependencies between stacks
storageStack.addDependency(infraStack);
authStack.addDependency(infraStack);
eksStack.addDependency(infraStack);
openSearchStack.addDependency(storageStack);
syntheticDataStack.addDependency(storageStack);
guardrailsStack.addDependency(storageStack);