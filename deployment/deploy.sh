#!/bin/bash

# Exit on error
set -e

# Disable AWS CLI pager to prevent pausing for user input
export AWS_PAGER=""

# AWS OpenTelemetry Configuration 
export OTEL_PYTHON_DISTRO="aws_distro"
export OTEL_PYTHON_CONFIGURATOR="aws_configurator"

# Service Identification
export OTEL_RESOURCE_ATTRIBUTES="service.name=\"retail-agent\""
export AGENT_OBSERVABILITY_ENABLED="true"
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="https://xray.us-east-1.amazonaws.com/v1/traces"

# CloudWatch Integration
export OTEL_EXPORTER_OTLP_LOGS_HEADERS="x-aws-log-group=bedrock-agentcore-observability,x-aws-log-stream=default,x-aws-metric-namespace=bedrock-agentcore"

# Instrumentation Exclusions - Extended to include all patterns from otel_config.py
export OTEL_PYTHON_FASTAPI_EXCLUDED_URLS="/ws/.*|/health|/metrics|.*websocket.*|/api/.*|.*\.amazonaws\.com.*|.*bedrock-runtime\..*|.*dynamodb\..*|.*cognito-identity\..*|.*s3\..*"
export OTEL_PYTHON_REQUESTS_EXCLUDED_URLS="/ws/.*|/health|/metrics|.*websocket.*|/api/.*|.*\.amazonaws\.com.*|.*bedrock-runtime\..*|.*dynamodb\..*|.*cognito-identity\..*|.*s3\..*"
export OTEL_PYTHON_URLLIB3_EXCLUDED_URLS="/ws/.*|/health|/metrics|.*websocket.*|/api/.*|.*\.amazonaws\.com.*|.*bedrock-runtime\..*|.*dynamodb\..*|.*cognito-identity\..*|.*s3\..*"
export OTEL_PYTHON_HTTPX_EXCLUDED_URLS="/ws/.*|/health|/metrics|.*websocket.*|/api/.*|.*\.amazonaws\.com.*|.*bedrock-runtime\..*|.*dynamodb\..*|.*cognito-identity\..*|.*s3\..*"
export OTEL_PYTHON_AIOHTTP_CLIENT_EXCLUDED_URLS="/ws/.*|/health|/metrics|.*websocket.*|/api/.*|.*\.amazonaws\.com.*|.*bedrock-runtime\..*|.*dynamodb\..*|.*cognito-identity\..*|.*s3\..*"
export OTEL_PYTHON_BOTO3SQS_EXCLUDED_URLS="/ws/.*|/health|/metrics|.*websocket.*|/api/.*|.*\.amazonaws\.com.*|.*bedrock-runtime\..*|.*dynamodb\..*|.*cognito-identity\..*|.*s3\..*"
export OTEL_PYTHON_BOTOCORE_EXCLUDED_URLS="/ws/.*|/health|/metrics|.*websocket.*|/api/.*|.*\.amazonaws\.com.*|.*bedrock-runtime\..*|.*dynamodb\..*|.*cognito-identity\..*|.*s3\..*"

# Disable unwanted instrumentations (from custom_otel_setup.py)
export OTEL_PYTHON_DISABLED_INSTRUMENTATIONS="boto3sqs,botocore,requests,urllib3,httpx,aiohttp-client,asyncio,threading,logging,system_metrics,psutil,sqlite3,redis,pymongo,sqlalchemy,django,flask,tornado,pyramid,falcon,starlette,fastapi,websockets"

# Propagation and Sampling
export OTEL_PROPAGATORS="tracecontext,baggage,xray"
export OTEL_TRACES_SAMPLER="always_on"
export OTEL_BSP_SCHEDULE_DELAY="1000"
export OTEL_BSP_MAX_EXPORT_BATCH_SIZE="512"
export OTEL_BSP_EXPORT_TIMEOUT="30000"

# AWS X-Ray specific
export AWS_XRAY_TRACING_NAME="retailagent"
export AWS_XRAY_CONTEXT_MISSING="LOG_ERROR"

# Debug Configuration (optional - enable when needed)
export OTEL_LOG_LEVEL="DEBUG"
export OTEL_PYTHON_LOG_LEVEL="DEBUG"
export AWS_XRAY_DEBUG_MODE="true"
export OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="true"
export OTEL_PYTHON_LOGGING_LEVEL="DEBUG"

# Load environment variables
if [ -f .env ]; then
  echo "Loading environment variables from .env file"
  set -a
  source .env
  set +a
else
  echo "No .env file found. Please create one with the required variables."
  exit 1
fi

# Function to update or add environment variable in .env file
update_env_var() {
  local var_name="$1"
  local var_value="$2"
  local env_file=".env"
  
  # Create .env file if it doesn't exist
  if [ ! -f "$env_file" ]; then
    touch "$env_file"
    echo "Created new .env file"
  fi
  
  if grep -q "^${var_name}=" "$env_file"; then
    # Variable exists, update it
    # Handle different sed syntax for macOS vs Linux
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS version
      sed -i '' "s/^${var_name}=.*$/${var_name}=${var_value}/" "$env_file"
    else
      # Linux version
      sed -i "s/^${var_name}=.*$/${var_name}=${var_value}/" "$env_file"
    fi
    echo "Updated ${var_name} in .env file"
  else
    # Variable doesn't exist, add it
    echo "${var_name}=${var_value}" >> "$env_file"
    echo "Added ${var_name} to .env file"
  fi
}

# Set AWS region if not already set
if [ -z "$AWS_REGION" ]; then
  export AWS_REGION="us-east-1"
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
  echo "AWS CLI is not installed. Please install it first."
  exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
  echo "AWS credentials are not configured or are invalid. Please run 'aws configure'."
  exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
  echo "Docker is not running. Please start Docker."
  exit 1
fi

# Build and deploy the CDK stack
echo "Building and deploying the CDK stack..."
cd cdk

# Install dependencies
echo "Installing dependencies..."
npm install

# Compile TypeScript
echo "Compiling TypeScript..."
npx tsc || {
  echo "TypeScript compilation failed. Please check the errors above."
  exit 1
}

# Deploy the CDK stack
echo "Deploying CDK stack..."
npx cdk deploy --all --require-approval never --continue-on-error || {
  echo "Warning: Some CDK stacks failed to deploy. Continuing with available resources..."
  echo "You may need to manually address failed stacks later."
}

# Check which stacks actually deployed successfully
echo "Checking deployment status of individual stacks..."
STACK_STATUSES=""
for stack in "${STACK_NAME}EksStack" "${STACK_NAME}AuthStack" "${STACK_NAME}StorageStack" "${STACK_NAME}GuardrailsStack" "${STACK_NAME}OpenSearchStack"; do
  STATUS=$(aws cloudformation describe-stacks --stack-name "$stack" --query "Stacks[0].StackStatus" --output text 2>/dev/null || echo "NOT_FOUND")
  echo "Stack $stack: $STATUS"
  if [[ "$STATUS" == *"COMPLETE"* ]] && [[ "$STATUS" != *"ROLLBACK"* ]]; then
    echo "✓ $stack deployed successfully"
  else
    echo "✗ $stack failed or not found"
    if [[ "$stack" == *"OpenSearchStack" ]]; then
      echo "Warning: OpenSearch stack failed - Knowledge Base features may not be available"
    fi
  fi
done

cd ..

# Get the ECR repository URIs from the CDK output
BACKEND_REPO_URI=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME}EksStack --query "Stacks[0].Outputs[?OutputKey=='BackendRepositoryUri'].OutputValue" --output text)

echo "Backend Repository URI: $BACKEND_REPO_URI"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
echo "AWS Account ID: $AWS_ACCOUNT_ID"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push the backend Docker image
echo "Building and pushing the retail-backend Docker image..."

# Build for the platform that matches EKS nodes (typically amd64/x86_64)
docker build --platform=linux/amd64 -t $BACKEND_REPO_URI:latest ../source/backend/.
docker push $BACKEND_REPO_URI:latest


# Temporarily disable exit on error for kubectl installation
set +e

# install kubectl if not present
if ! command -v kubectl &> /dev/null; then
  echo "kubectl not found, installing..."
  
  # Create a bin directory in the current directory if it doesn't exist
  mkdir -p ./bin
  
  # Determine OS and architecture
  OS=$(uname -s | tr '[:upper:]' '[:lower:]')
  ARCH=$(uname -m)
  
  # Map architecture names to kubectl naming convention
  case ${ARCH} in
    x86_64)
      KUBECTL_ARCH="amd64"
      ;;
    aarch64|arm64)
      KUBECTL_ARCH="arm64"
      ;;
    *)
      KUBECTL_ARCH=${ARCH}
      ;;
  esac
  
  echo "Installing kubectl v1.31.0 for ${OS}/${KUBECTL_ARCH}..."
  if ! curl -L -o ./bin/kubectl "https://dl.k8s.io/release/v1.31.0/bin/${OS}/${KUBECTL_ARCH}/kubectl"; then
    echo "Failed to download kubectl. Exiting."
    exit 1
  fi
  
  # Make kubectl executable
  chmod +x ./bin/kubectl
  
  # Add to PATH
  export PATH="$(pwd)/bin:$PATH"
  
  # Define kubectl function instead of alias (works better in non-interactive shells)
  kubectl() {
    $(pwd)/bin/kubectl "$@"
  }
  export -f kubectl
  
  # Check if kubectl is working
  if kubectl version --client 2>/dev/null; then
    echo "kubectl installed successfully"
    KUBECTL_VERSION=$(kubectl version --client --short 2>/dev/null | cut -d " " -f 3)
    echo "kubectl version: ${KUBECTL_VERSION}"
  else
    echo "kubectl installation verification failed. Continuing anyway..."
  fi
else
  KUBECTL_VERSION=$(kubectl version --client --short 2>/dev/null | cut -d " " -f 3)
  echo "kubectl is already installed with version: ${KUBECTL_VERSION}"
fi

# Re-enable exit on error
set -e


echo "Get EKS cluster details and configure kubectl..."
# Get the EKS cluster name from the CDK output
CLUSTER_NAME=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME}EksStack --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}ClusterName'].OutputValue" --output text)

echo "EKS Cluster Name: $CLUSTER_NAME"

# Configure kubectl to use the EKS cluster
echo "Configuring kubectl to use the EKS cluster... $CLUSTER_NAME"
aws eks update-kubeconfig --name $CLUSTER_NAME --region $AWS_REGION
echo "kubectl configured successfully to use cluster: $CLUSTER_NAME"

# Get current user ARN for RBAC configuration
USER_ARN=$(aws sts get-caller-identity --query "Arn" --output text)
echo "Current user ARN: $USER_ARN"

# Extract the role name from the ARN (format: arn:aws:sts::ACCOUNT_ID:assumed-role/ROLE_NAME/SESSION_NAME)
ROLE_NAME=$(echo $USER_ARN | sed -n 's/.*assumed-role\/\([^/]*\)\/.*/\1/p')
ROLE_ARN="arn:aws:iam::$AWS_ACCOUNT_ID:role/$ROLE_NAME"
echo "IAM Role ARN: $ROLE_ARN"

# Check if access entry already exists
echo "Checking if EKS access entry already exists..."
ACCESS_ENTRY_EXISTS=$(aws eks list-access-entries --cluster-name $CLUSTER_NAME --region $AWS_REGION | grep -c "$ROLE_ARN" || true)

# Use AWS CLI to create an access entry for the current role if it doesn't exist
if [ "$ACCESS_ENTRY_EXISTS" -eq "0" ]; then
  echo "Creating EKS access entry for your IAM role..."
  aws eks create-access-entry \
    --cluster-name $CLUSTER_NAME \
    --principal-arn $ROLE_ARN \
    --username admin \
    --region $AWS_REGION
else
  echo "EKS access entry already exists for $ROLE_ARN"
fi

# Check if policy association already exists
echo "Checking if policy association already exists..."
POLICY_ASSOCIATION_EXISTS=$(aws eks list-associated-access-policies \
  --cluster-name $CLUSTER_NAME \
  --principal-arn $ROLE_ARN \
  --region $AWS_REGION | grep -c "AmazonEKSClusterAdminPolicy" || true)

# Associate the role with the system:masters Kubernetes group if not already associated
if [ "$POLICY_ASSOCIATION_EXISTS" -eq "0" ]; then
  echo "Associating role with system:masters group..."
  aws eks associate-access-policy \
    --cluster-name $CLUSTER_NAME \
    --principal-arn $ROLE_ARN \
    --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy \
    --access-scope type=cluster \
    --region $AWS_REGION

  # Wait for permissions to propagate
  echo "Waiting for permissions to propagate (10 seconds)..."
  sleep 10
else
  echo "Policy already associated with $ROLE_ARN"
fi

# Fix ALB Controller permissions
echo "Fixing ALB Controller permissions..."

# Dynamically retrieve the ALB Controller role name
echo "Retrieving ALB Controller role name..."

# Try to find the ALB Controller role using AWS CLI by searching for roles with the ALB Controller name pattern
echo "Searching for ALB Controller role in IAM roles..."
ALB_CONTROLLER_ROLE_NAME=$(aws iam list-roles --query "Roles[?contains(RoleName, 'AlbController') && contains(RoleName, '${STACK_NAME}EksStack')].RoleName" --output text)

# If that doesn't work, try a more generic search
if [ -z "$ALB_CONTROLLER_ROLE_NAME" ]; then
  echo "Could not find ALB Controller role with specific name pattern, trying broader search..."
  ALB_CONTROLLER_ROLE_NAME=$(aws iam list-roles --query "Roles[?contains(RoleName, '${STACK_NAME}EksStack') && contains(RoleName, 'albsa')].RoleName" --output text | head -n 1)
fi

echo "Using ALB Controller role: $ALB_CONTROLLER_ROLE_NAME"

# Create a policy document for the ALB Controller
cat > alb-controller-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "elasticloadbalancing:CreateLoadBalancer",
        "elasticloadbalancing:CreateTargetGroup",
        "elasticloadbalancing:CreateListener",
        "elasticloadbalancing:CreateRule",
        "elasticloadbalancing:DeleteLoadBalancer",
        "elasticloadbalancing:DeleteTargetGroup",
        "elasticloadbalancing:DeleteListener",
        "elasticloadbalancing:DeleteRule",
        "elasticloadbalancing:ModifyLoadBalancerAttributes",
        "elasticloadbalancing:ModifyTargetGroup",
        "elasticloadbalancing:ModifyTargetGroupAttributes",
        "elasticloadbalancing:ModifyListener",
        "elasticloadbalancing:ModifyRule",
        "elasticloadbalancing:RegisterTargets",
        "elasticloadbalancing:DeregisterTargets",
        "elasticloadbalancing:DescribeLoadBalancers",
        "elasticloadbalancing:DescribeTargetGroups",
        "elasticloadbalancing:DescribeListeners",
        "elasticloadbalancing:DescribeRules",
        "elasticloadbalancing:DescribeTargetHealth",
        "elasticloadbalancing:DescribeLoadBalancerAttributes",
        "elasticloadbalancing:DescribeTargetGroupAttributes",
        "elasticloadbalancing:DescribeListenerAttributes",
        "elasticloadbalancing:DescribeSSLPolicies",
        "elasticloadbalancing:AddTags",
        "elasticloadbalancing:RemoveTags",
        "elasticloadbalancing:SetSecurityGroups",
        "elasticloadbalancing:SetSubnets",
        "elasticloadbalancing:SetIpAddressType",
        "elasticloadbalancing:SetWebAcl",
        "elasticloadbalancing:SetRulePriorities",
        "elasticloadbalancing:ModifyListenerAttributes",
        "elasticloadbalancing:DescribeCapacityReservation",
        "elasticloadbalancing:ModifyCapacityReservation",
        "ec2:DescribeVpcs",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeInstances",
        "ec2:DescribeNetworkInterfaces",
        "ec2:DescribeAvailabilityZones",
        "ec2:DescribeInternetGateways",
        "ec2:DescribeRouteTables",
        "ec2:DescribeTags",
        "ec2:CreateSecurityGroup",
        "ec2:CreateTags",
        "ec2:AuthorizeSecurityGroupIngress",
        "ec2:RevokeSecurityGroupIngress",
        "ec2:AuthorizeSecurityGroupEgress",
        "ec2:RevokeSecurityGroupEgress",
        "ec2:DeleteSecurityGroup",
        "ec2:GetSecurityGroupsForVpc",
        "cognito-idp:DescribeUserPoolClient",
        "acm:ListCertificates",
        "acm:DescribeCertificate",
        "iam:ListServerCertificates",
        "iam:GetServerCertificate",
        "waf-regional:GetWebACL",
        "waf-regional:GetWebACLForResource",
        "waf-regional:AssociateWebACL",
        "waf-regional:DisassociateWebACL",
        "wafv2:GetWebACL",
        "wafv2:GetWebACLForResource",
        "wafv2:AssociateWebACL",
        "wafv2:DisassociateWebACL",
        "shield:GetSubscriptionState",
        "shield:DescribeProtection",
        "shield:CreateProtection",
        "shield:DeleteProtection"
      ],
      "Resource": "*"
    }
  ]
}
EOF

# Create the policy with better error handling and flexible search
echo "Creating or finding ALB Controller policy..."

# Debug: Print the policy document
echo "Policy document contents:"
cat alb-controller-policy.json

# First check if the policy already exists
echo "Checking if policy already exists..."
POLICY_NAME="${STACK_NAME}EksAlbControllerPolicy-Direct"
TEMP_OUTPUT=$(mktemp)

# Check for existing policy
if aws iam get-policy --policy-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${POLICY_NAME}" > "$TEMP_OUTPUT" 2>&1; then
  echo "Policy already exists, retrieving ARN..."
  ALB_POLICY_ARN=$(aws iam get-policy --policy-arn "arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${POLICY_NAME}" --query 'Policy.Arn' --output text)
  echo "Found existing policy: $ALB_POLICY_ARN"
else
  echo "Policy does not exist, creating new policy..."
  if ! aws iam create-policy \
    --policy-name "${POLICY_NAME}" \
    --policy-document file://alb-controller-policy.json \
    --query 'Policy.Arn' \
    --output text > "$TEMP_OUTPUT" 2>&1; then
    echo "Policy creation failed. Error output:"
    cat "$TEMP_OUTPUT"
    ALB_POLICY_ARN=""
  else
    ALB_POLICY_ARN=$(cat "$TEMP_OUTPUT")
    echo "Policy created successfully: $ALB_POLICY_ARN"
  fi
fi
rm -f "$TEMP_OUTPUT"

# If creation fails, try to find existing policy with flexible search patterns
if [ -z "$ALB_POLICY_ARN" ]; then
  echo "Policy creation failed, searching for existing policy with alternative patterns..."
  
  # Try different search patterns
  SEARCH_PATTERNS=(
    "${STACK_NAME}EksAlbControllerPolicy-Direct"
    "${STACK_NAME}AlbControllerPolicy"
  )
  
  for pattern in "${SEARCH_PATTERNS[@]}"; do
    echo "Searching for policy with pattern: $pattern"
    TEMP_OUTPUT=$(mktemp)
    if ! aws iam list-policies \
      --scope Local \
      --query "Policies[?contains(PolicyName, '$pattern')].Arn" \
      --output text > "$TEMP_OUTPUT" 2>&1; then
      echo "Error searching for policy with pattern '$pattern':"
      cat "$TEMP_OUTPUT"
      rm -f "$TEMP_OUTPUT"
      continue
    fi
    
    ALB_POLICY_ARN=$(cat "$TEMP_OUTPUT")
    rm -f "$TEMP_OUTPUT"
    
    if [ -n "$ALB_POLICY_ARN" ]; then
      echo "Found existing policy with pattern '$pattern': $ALB_POLICY_ARN"
      break
    else
      echo "No policy found with pattern '$pattern'"
    fi
  done
fi

if [ -z "$ALB_POLICY_ARN" ]; then
  echo "Error: Failed to create or find ALB Controller policy after multiple attempts."
  echo "Please check your IAM permissions and try again."
  echo "Debug information:"
  echo "STACK_NAME: $STACK_NAME"
  echo "ALB_CONTROLLER_ROLE_NAME: $ALB_CONTROLLER_ROLE_NAME"
  echo "Checking IAM permissions..."
  aws iam get-user 2>&1 || echo "Failed to get IAM user info"
  echo "Checking if role exists..."
  aws iam get-role --role-name "$ALB_CONTROLLER_ROLE_NAME" 2>&1 || echo "Failed to get role info"
  exit 1
else
  echo "Successfully found or created ALB Controller policy: $ALB_POLICY_ARN"
  
  # Attach the policy directly to the ALB Controller role
  echo "Attaching policy to ALB Controller role..."
  TEMP_OUTPUT=$(mktemp)
  if ! aws iam attach-role-policy --role-name "$ALB_CONTROLLER_ROLE_NAME" --policy-arn "$ALB_POLICY_ARN" > "$TEMP_OUTPUT" 2>&1; then
    echo "Warning: Failed to attach policy to $ALB_CONTROLLER_ROLE_NAME"
    echo "Error output:"
    cat "$TEMP_OUTPUT"
    echo "The policy exists but may not be attached. You may need to attach it manually."
  else
    echo "Successfully attached policy to $ALB_CONTROLLER_ROLE_NAME"
  fi
  rm -f "$TEMP_OUTPUT"
  
  echo "ALB Controller permissions updated."
fi

# Continue with the original ALB Controller permissions code as a fallback
echo "Checking for original ALB Controller policy..."
TEMP_OUTPUT=$(mktemp)
if ! aws cloudformation describe-stacks --stack-name ${STACK_NAME}EksStack --query "Stacks[0].Outputs[?OutputKey=='AlbControllerPolicyArn'].OutputValue" --output text > "$TEMP_OUTPUT" 2>&1; then
  echo "Error retrieving original policy ARN:"
  cat "$TEMP_OUTPUT"
  ALB_POLICY_ARN_ORIGINAL=""
else
  ALB_POLICY_ARN_ORIGINAL=$(cat "$TEMP_OUTPUT")
fi
rm -f "$TEMP_OUTPUT"

if [ -n "$ALB_POLICY_ARN_ORIGINAL" ]; then
  echo "Found original ALB Controller policy ARN: $ALB_POLICY_ARN_ORIGINAL"
  echo "Attaching original policy to ALB Controller role as well..."
  TEMP_OUTPUT=$(mktemp)
  if ! aws iam attach-role-policy --role-name "$ALB_CONTROLLER_ROLE_NAME" --policy-arn "$ALB_POLICY_ARN_ORIGINAL" > "$TEMP_OUTPUT" 2>&1; then
    echo "Failed to attach original policy:"
    cat "$TEMP_OUTPUT"
  else
    echo "Successfully attached original policy"
  fi
  rm -f "$TEMP_OUTPUT"
fi

# Add a debug checkpoint
echo "ALB Controller policy section completed."

# Create the app namespace if it doesn't exist
echo "Creating retail-app namespace if it doesn't exist..."
kubectl create namespace ${STACK_NAME}-app --dry-run=client -o yaml | kubectl apply -f -

# Apply Pod Security Standards to the namespace
echo "Applying Pod Security Standards to namespace..."
kubectl label namespace ${STACK_NAME}-app \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted \
  --overwrite

# Create ConfigMap for environment variables
echo "Creating ConfigMap for environment variables..."

# Get the KnowledgeBaseId directly from the OpenSearchStack output
echo "Retrieving KnowledgeBaseId from CloudFormation..."
KB_ID=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME}OpenSearchStack --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}KnowledgeBaseId'].OutputValue" --output text 2>/dev/null || echo "")

if [ -z "$KB_ID" ] || [ "$KB_ID" = "None" ]; then
  echo "Warning: Could not retrieve KnowledgeBaseId from CloudFormation OpenSearchStack."
else
  echo "Found KnowledgeBaseId: $KB_ID"
  export KB_ID="$KB_ID"
fi

# Get the GUARDRAIL_IDENTIFIER and GUARDRAIL_VERSION directly from the GuardrailsStack output
echo "Retrieving GUARDRAIL_IDENTIFIER and GUARDRAIL_VERSION from CloudFormation..."
GUARDRAIL_IDENTIFIER_FULL=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME}GuardrailsStack --query "Stacks[0].Outputs[?OutputKey=='GuardrailIdentifier'].OutputValue" --output text)
# Extract only the last part after "guardrail/"
GUARDRAIL_IDENTIFIER=$(echo $GUARDRAIL_IDENTIFIER_FULL | sed 's/.*guardrail\///')
GUARDRAIL_VERSION=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME}GuardrailsStack --query "Stacks[0].Outputs[?OutputKey=='GuardrailVersion'].OutputValue" --output text)
if [ -z "$GUARDRAIL_IDENTIFIER" ]; then
  echo "Warning: Could not retrieve GUARDRAIL_IDENTIFIER from CloudFormation stack."

else
  echo "Found GUARDRAIL_IDENTIFIER: $GUARDRAIL_IDENTIFIER"
  export GUARDRAIL_IDENTIFIER="$GUARDRAIL_IDENTIFIER"

fi

if [ -z "$GUARDRAIL_VERSION" ]; then
  echo "Warning: Could not retrieve GUARDRAIL_VERSION from CloudFormation."
else
  echo "Found GUARDRAIL_VERSION: $GUARDRAIL_VERSION"
  export GUARDRAIL_VERSION="$GUARDRAIL_VERSION"
fi
# Get the S3 bucket name from the StorageStack using the exact output key
STORAGE_STACK_NAME="${STACK_NAME}StorageStack"
S3_BUCKET_NAME=$(aws cloudformation describe-stacks --stack-name $STORAGE_STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}DataBucketName'].OutputValue" --output text)
echo "S3 data bucket name: $S3_BUCKET_NAME"

# Get the CloudFront distribution domain name using the exact output key
echo "Retrieving CloudFront distribution domain from CloudFormation..."
CLOUDFRONT_DOMAIN=$(aws cloudformation describe-stacks --stack-name $STORAGE_STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}CloudFrontDistributionDomainName'].OutputValue" --output text)
if [ -z "$CLOUDFRONT_DOMAIN" ]; then
  echo "Error: Could not retrieve CloudFront distribution domain from CloudFormation."
  echo "Checking all available outputs from the storage stack..."
  aws cloudformation describe-stacks --stack-name $STORAGE_STACK_NAME --query "Stacks[0].Outputs" --output table
  exit 1
fi
echo "Found CloudFront Distribution Domain: $CLOUDFRONT_DOMAIN"

# Get the Cognito User Pool ID and Client ID using the exact output keys
AUTH_STACK_NAME="${STACK_NAME}AuthStack"
echo "Getting User Pool ID and Client ID from $AUTH_STACK_NAME..."
USER_POOL_ID=$(aws cloudformation describe-stacks --stack-name $AUTH_STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}UserPoolId'].OutputValue" --output text)
USER_POOL_CLIENT_ID=$(aws cloudformation describe-stacks --stack-name $AUTH_STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}UserPoolClientId'].OutputValue" --output text)
IDENTITY_POOL_ID=$(aws cloudformation describe-stacks --stack-name $AUTH_STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}IdentityPoolId'].OutputValue" --output text)
echo "User Pool ID: $USER_POOL_ID"
echo "User Pool Client ID: $USER_POOL_CLIENT_ID"
echo "Identity Pool ID: $IDENTITY_POOL_ID"


# Create a temporary file for the ConfigMap YAML
CONFIG_MAP_FILE=$(mktemp)
echo "Creating ConfigMap in temporary file: $CONFIG_MAP_FILE"

# Generate the ConfigMap YAML with proper escaping
cat > "$CONFIG_MAP_FILE" << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: ${STACK_NAME}-app-config
  namespace: ${STACK_NAME}-app
data:
  BD_GUARDRAIL_IDENTIFIER: "${GUARDRAIL_IDENTIFIER}"
  BD_GUARDRAIL_VERSION: "${GUARDRAIL_VERSION}"
  BD_KB_ID: "${KB_ID}"
  BEDROCK_MODEL_ID: "${BEDROCK_MODEL_ID}"
  AWS_DEFAULT_REGION: "${AWS_REGION}"
  AWS_REGION: "${AWS_REGION}"
  STACK_NAME: "${STACK_NAME}"
  STACK_ENVIRONMENT: "${STACK_ENVIRONMENT}"
  S3_BUCKET_NAME: "${S3_BUCKET_NAME}"
  DOMAIN_NAME: "${CLOUDFRONT_DOMAIN}"
  COGNITO_USER_POOL_ID: "${USER_POOL_ID}"
  COGNITO_APP_CLIENT_ID: "${USER_POOL_CLIENT_ID}"
  COGNITO_IDENTITY_POOL_ID: "${IDENTITY_POOL_ID}"
EOF

# Add the OpenTelemetry variables with proper YAML escaping
for var in OTEL_PYTHON_DISTRO OTEL_PYTHON_CONFIGURATOR OTEL_RESOURCE_ATTRIBUTES AGENT_OBSERVABILITY_ENABLED \
           OTEL_EXPORTER_OTLP_PROTOCOL OTEL_EXPORTER_OTLP_TRACES_ENDPOINT OTEL_EXPORTER_OTLP_LOGS_HEADERS \
           OTEL_PYTHON_FASTAPI_EXCLUDED_URLS OTEL_PYTHON_REQUESTS_EXCLUDED_URLS OTEL_PYTHON_URLLIB3_EXCLUDED_URLS \
           OTEL_PYTHON_HTTPX_EXCLUDED_URLS OTEL_LOG_LEVEL OTEL_PYTHON_LOG_LEVEL AWS_XRAY_DEBUG_MODE \
           OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED OTEL_PYTHON_DISABLED_INSTRUMENTATIONS \
           OTEL_PROPAGATORS OTEL_TRACES_SAMPLER OTEL_BSP_SCHEDULE_DELAY OTEL_BSP_MAX_EXPORT_BATCH_SIZE \
           OTEL_BSP_EXPORT_TIMEOUT AWS_XRAY_TRACING_NAME AWS_XRAY_CONTEXT_MISSING OTEL_PYTHON_LOGGING_LEVEL; do
    # Get the value with a fallback to empty string
    value="${!var:-}"
    
    # Properly escape the value for YAML
    # Replace newlines with literal \n, backslashes with double backslashes, and quotes with escaped quotes
    # Then trim any trailing spaces to avoid issues with string comparison in the application
    escaped_value=$(echo "$value" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | sed 's/\t/\\t/g' | tr '\n' ' ' | sed 's/ *$//')
    
    # Add to the ConfigMap file
    echo "  $var: \"$escaped_value\"" >> "$CONFIG_MAP_FILE"
done

# Apply the ConfigMap
echo "Applying ConfigMap from file..."
kubectl apply -f "$CONFIG_MAP_FILE"

# Clean up the temporary file
rm -f "$CONFIG_MAP_FILE"

# Deploy the backend service
echo "Deploying backend service..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${STACK_NAME}-backend
  namespace: ${STACK_NAME}-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${STACK_NAME}-backend
  template:
    metadata:
      labels:
        app: ${STACK_NAME}-backend
    spec:
      serviceAccountName: ${STACK_NAME}-backend-sa
      terminationGracePeriodSeconds: 30
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: ${STACK_NAME}-backend
        image: ${BACKEND_REPO_URI}:latest
        ports:
        - containerPort: 8000
          name: backend
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          runAsGroup: 1000
          allowPrivilegeEscalation: false
          capabilities:
            drop:
              - ALL
          readOnlyRootFilesystem: false
          seccompProfile:
            type: RuntimeDefault
        envFrom:
        - configMapRef:
            name: ${STACK_NAME}-app-config
        env:
        - name: HOST
          value: "0.0.0.0"
        - name: PORT
          value: "8000"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
          timeoutSeconds: 10
          failureThreshold: 3
          successThreshold: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 15
          failureThreshold: 5
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 10"]
---
apiVersion: v1
kind: Service
metadata:
  name: ${STACK_NAME}-backend-service
  namespace: ${STACK_NAME}-app
  annotations:
    # Ensure proper WebSocket support
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "http"
    service.beta.kubernetes.io/aws-load-balancer-connection-idle-timeout: "600"
spec:
  selector:
    app: ${STACK_NAME}-backend
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
EOF


# Get the ALB security group ID from CDK output
ALB_SECURITY_GROUP_ID=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME}EksStack --query "Stacks[0].Outputs[?OutputKey=='AlbSecurityGroupId'].OutputValue" --output text)
echo "ALB Security Group ID: $ALB_SECURITY_GROUP_ID"

# Get the access logs bucket name
ACCESS_LOGS_BUCKET_NAME=$(aws cloudformation describe-stacks --stack-name $STORAGE_STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}AccessLogsBucketName'].OutputValue" --output text)
echo "Access Logs Bucket Name: $ACCESS_LOGS_BUCKET_NAME"


# Create Ingress
echo "Creating Ingress resources with ..."
# Add debug statements to help pinpoint the error
echo "DEBUG: About to create Ingress YAML"
echo "DEBUG: ACCESS_LOGS_BUCKET_NAME = $ACCESS_LOGS_BUCKET_NAME"

# Apply the ingress resources and capture the output
echo "DEBUG: Starting kubectl apply for Ingress"
INGRESS_OUTPUT=$(kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ${STACK_NAME}-ingress
  namespace: ${STACK_NAME}-app
  annotations:
    alb.ingress.kubernetes.io/scheme: "internet-facing"
    alb.ingress.kubernetes.io/target-type: "ip"
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP":80}]'
    alb.ingress.kubernetes.io/security-groups: "${ALB_SECURITY_GROUP_ID}"
    # WebSocket support configuration
    alb.ingress.kubernetes.io/backend-protocol-version: "HTTP1"
    alb.ingress.kubernetes.io/target-group-attributes: >-
      stickiness.enabled=true,
      stickiness.type=lb_cookie,
      stickiness.lb_cookie.duration_seconds=3600,
      deregistration_delay.timeout_seconds=30
    alb.ingress.kubernetes.io/load-balancer-attributes: >-
      routing.http2.enabled=false,
      idle_timeout.timeout_seconds=600,
      access_logs.s3.enabled=true,
      access_logs.s3.bucket=${ACCESS_LOGS_BUCKET_NAME}
    alb.ingress.kubernetes.io/healthcheck-protocol: HTTP
    alb.ingress.kubernetes.io/healthcheck-port: "8000"
    alb.ingress.kubernetes.io/healthcheck-path: "/health"
    alb.ingress.kubernetes.io/success-codes: "200"
spec:
  ingressClassName: alb
  rules:
  - http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: ${STACK_NAME}-backend-service
            port:
              number: 80
      - path: /ws
        pathType: Prefix
        backend:
          service:
            name: ${STACK_NAME}-backend-service
            port:
              number: 80
      - path: /health
        pathType: Prefix
        backend:
          service:
            name: ${STACK_NAME}-backend-service
            port:
              number: 80
EOF
)

echo "DEBUG: kubectl apply for Ingress completed"
echo "DEBUG: INGRESS_OUTPUT = $INGRESS_OUTPUT"

# Only wait for ingress provisioning if the resources are being created or configured
if echo "$INGRESS_OUTPUT" | grep -q "created\|configured"; then
  echo "Waiting for ingress to be provisioned (this may take up to 5 minutes)..."
  kubectl wait --namespace ${STACK_NAME}-app \
    --for=condition=ready ingress \
    --field-selector metadata.name=${STACK_NAME}-ingress \
    --timeout=5m || echo "Ingress may still be provisioning. This is normal as ALB provisioning can take 5-10 minutes."
else
  echo "Ingress resources are unchanged, skipping wait for provisioning."
fi

# Get the required values from CloudFormation outputs
echo "Retrieving values from CloudFormation for the frontend app..."

# Get the S3 bucket name from the StorageStack
echo "Using storage stack: $STORAGE_STACK_NAME"

# Get the S3 bucket name from the StorageStack using the exact output key
WEBSITE_BUCKET_NAME=$(aws cloudformation describe-stacks --stack-name $STORAGE_STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}WebsiteBucketName'].OutputValue" --output text)
echo "Website Bucket Name: $WEBSITE_BUCKET_NAME"

# update the data bucket name in the .env file
DATA_BUCKET_NAME=$(aws cloudformation describe-stacks --stack-name $STORAGE_STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}DataBucketName'].OutputValue" --output text)
echo "Data Bucket Name: $DATA_BUCKET_NAME"

# Get the API endpoints
echo "Retrieving ALB endpoint..."
BACKEND_ALB=$(kubectl get ingress ${STACK_NAME}-ingress -n ${STACK_NAME}-app -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
echo "Current BACKEND_ALB value: $BACKEND_ALB"

# Wait for ALB to be ready with more verbose logging
echo "Waiting for ALB to be ready (this may take several minutes)..."
attempt=1
max_attempts=12
while [ $attempt -le $max_attempts ]; do
  echo "Attempt $attempt of $max_attempts: Checking ALB status..."
  
  # Find the ALB by DNS name instead of using the name parameter
  ALB_STATUS=$(aws elbv2 describe-load-balancers --query "LoadBalancers[?DNSName=='$BACKEND_ALB'].State.Code" --output text)
  echo "Current ALB status: $ALB_STATUS"
  
  if [ -n "$BACKEND_ALB" ] && [ "$ALB_STATUS" = "active" ]; then
    echo "ALB is ready! Hostname: $BACKEND_ALB (Status: $ALB_STATUS)"
    break
  fi
  
  echo "ALB not ready yet (Status: ${ALB_STATUS:-unknown}), waiting 30 seconds..."
  sleep 30
  attempt=$((attempt + 1))
done

if [ -z "$BACKEND_ALB" ] || [ "$ALB_STATUS" != "active" ]; then
  echo "Warning: ALB not ready after $max_attempts attempts. Current status: ${ALB_STATUS:-unknown}"
  echo "You can check ALB status later with: kubectl get ingress ${STACK_NAME}-ingress -n ${STACK_NAME}-app"
  echo "Continuing with deployment as ALB may still become available..."
fi


# Add ALB as a second origin to CloudFront distribution
echo "Adding ALB as a second origin to CloudFront distribution..."
DISTRIBUTION_ID=$(aws cloudfront list-distributions --query "DistributionList.Items[?DomainName=='$CLOUDFRONT_DOMAIN'].Id" --output text)


if [ -n "$DISTRIBUTION_ID" ] && [ -n "$BACKEND_ALB" ]; then
  echo "Found CloudFront distribution ID: $DISTRIBUTION_ID"
  echo "Found ALB endpoint: $BACKEND_ALB"
  
  # Get the current CloudFront configuration
  echo "Getting current CloudFront configuration..."
  aws cloudfront get-distribution-config --id $DISTRIBUTION_ID > cloudfront-config.json
  
  # Check if the distribution already has more than one origin
  ORIGIN_COUNT=$(jq '.DistributionConfig.Origins.Quantity' cloudfront-config.json)
  echo "Current number of origins: $ORIGIN_COUNT"
  
  if [ "$ORIGIN_COUNT" -gt 1 ]; then
    echo "CloudFront distribution already has multiple origins. Skipping update to avoid adding duplicate origins."
  else
    echo "CloudFront distribution has only one origin. Proceeding with update..."
  
    # Parameters
    CONFIG_FILE=cloudfront-config.json
    ALB_DOMAIN=$BACKEND_ALB

    # Make a backup of the original config
    cp "$CONFIG_FILE" "${CONFIG_FILE}.original"

    # Remove the ETag from the config file
    sed -i.bak '/"ETag"/d' "$CONFIG_FILE"

    # Create a temporary file for processing
    TMP_FILE=$(mktemp)

    # Step 1: Update Origins - Add ALB origin and update quantity
    jq --arg domain "$ALB_DOMAIN" '.DistributionConfig.Origins.Quantity = 2 | 
        .DistributionConfig.Origins.Items += [{
            "Id": "ALBOrigin",
            "DomainName": $domain,
            "OriginPath": "",
            "CustomOriginConfig": {
                "HTTPPort": 80,
                "HTTPSPort": 443,
                "OriginProtocolPolicy": "http-only",
                "OriginKeepaliveTimeout": 60,
                "OriginSslProtocols": {
                    "Quantity": 1,
                    "Items": ["TLSv1.2"]
                },
                "OriginReadTimeout": 30,
                "OriginKeepaliveTimeout": 5
            },
            "CustomHeaders": {
                "Quantity": 0,
                "Items": []
            },
            "ConnectionAttempts": 3,
            "ConnectionTimeout": 10,
            "OriginShield": {
                "Enabled": false
            },
            "OriginAccessControlId": ""
        }]' "$CONFIG_FILE" > "$TMP_FILE"

    # Step 2: Update CacheBehaviors - Add API and WebSocket behaviors
    jq '.DistributionConfig.CacheBehaviors.Quantity = 2 | 
        .DistributionConfig.CacheBehaviors.Items = [
            {
                "PathPattern": "/api/*",
                "TargetOriginId": "ALBOrigin",
                "TrustedSigners": {
                    "Enabled": false,
                    "Quantity": 0
                },
                "TrustedKeyGroups": {
                    "Enabled": false,
                    "Quantity": 0
                },
                "ViewerProtocolPolicy": "redirect-to-https",
                "AllowedMethods": {
                    "Quantity": 7,
                    "Items": ["GET", "HEAD", "POST", "PUT", "PATCH", "OPTIONS", "DELETE"],
                    "CachedMethods": {
                        "Quantity": 2,
                        "Items": ["GET", "HEAD"]
                    }
                },
                "ForwardedValues": {
                    "QueryString": true,
                    "Cookies": {
                        "Forward": "all"
                    },
                    "Headers": {
                        "Quantity": 7,
                        "Items": ["Host", "Authorization", "Origin", "Referer", "Accept", "Accept-Language", "Content-Type"]
                    },
                    "QueryStringCacheKeys": {
                        "Quantity": 0,
                        "Items": []
                    },
                    "QueryString": true
                },
                "MinTTL": 0,
                "DefaultTTL": 0,
                "MaxTTL": 0,
                "Compress": true,
                "SmoothStreaming": false,
                "FieldLevelEncryptionId": "",
                "LambdaFunctionAssociations": {
                    "Quantity": 0,
                    "Items": []
                },
                "FunctionAssociations": {
                    "Quantity": 0,
                    "Items": []
                },
                "GrpcConfig": {
                    "Enabled": false
                }
            },
            {
                "PathPattern": "/ws/*",
                "TargetOriginId": "ALBOrigin",
                "TrustedSigners": {
                    "Enabled": false,
                    "Quantity": 0
                },
                "TrustedKeyGroups": {
                    "Enabled": false,
                    "Quantity": 0
                },
                "ViewerProtocolPolicy": "redirect-to-https",
                "AllowedMethods": {
                    "Quantity": 7,
                    "Items": ["GET", "HEAD", "POST", "PUT", "PATCH", "OPTIONS", "DELETE"],
                    "CachedMethods": {
                        "Quantity": 2,
                        "Items": ["GET", "HEAD"]
                    }
                },
                "ForwardedValues": {
                    "QueryString": true,
                    "Cookies": {
                        "Forward": "all"
                    },
                    "Headers": {
                        "Quantity": 7,
                        "Items": ["Host", "Authorization", "Origin", "Referer", "Accept", "Accept-Language", "Content-Type"]
                    },
                    "QueryStringCacheKeys": {
                        "Quantity": 0,
                        "Items": []
                    }
                },
                "MinTTL": 0,
                "DefaultTTL": 0,
                "MaxTTL": 0,
                "Compress": true,
                "SmoothStreaming": false,
                "FieldLevelEncryptionId": "",
                "LambdaFunctionAssociations": {
                    "Quantity": 0,
                    "Items": []
                },
                "FunctionAssociations": {
                    "Quantity": 0,
                    "Items": []
                },
                "GrpcConfig": {
                    "Enabled": false
                }
            }
        ]' "$TMP_FILE" > "updated-config.json"

    # Step 3: Ensure DefaultRootObject is set
    jq '.DistributionConfig.DefaultRootObject = "index.html"' "updated-config.json" > "$TMP_FILE"
    mv "$TMP_FILE" "updated-config.json"

    echo "Updated CloudFront configuration saved to updated-config.json"
    echo "Original configuration backed up to ${CONFIG_FILE}.original"

    # Get the ETag for the distribution
    ETAG=$(aws cloudfront get-distribution --id "$DISTRIBUTION_ID" --query 'ETag' --output text)
    if [ -z "$ETAG" ]; then
      echo "Error: Could not retrieve ETag for distribution $DISTRIBUTION_ID"
      exit 1
    fi

    echo "Updating CloudFront distribution $DISTRIBUTION_ID with ETag $ETAG"

    # Update the CloudFront distribution
    aws cloudfront update-distribution --id "$DISTRIBUTION_ID" --if-match "$ETAG" --cli-input-json file://updated-config.json

    if [ $? -eq 0 ]; then
      echo "CloudFront distribution updated successfully"
    else
      echo "Failed to update CloudFront distribution"
      exit 1
    fi
    
    # Clean up temporary files
    rm -f cloudfront-config.json updated-config.json cloudfront-config.json cloudfront-config.json.original cloudfront-config.json.bak alb-controller-policy.json
    
    echo "CloudFront distribution updated with ALB origin for /api/* and /ws/* paths"
  fi
else
  echo "Skipping CloudFront update: Either distribution ID or ALB endpoint not found."
fi

# Create a .env file for the backend app
echo "Creating .env file for the backend app..."
cat > ../source/backend/.env << EOF
AWS_REGION=${AWS_REGION}
COGNITO_USER_POOL_ID=${USER_POOL_ID}
COGNITO_APP_CLIENT_ID=${USER_POOL_CLIENT_ID}
COGNITO_IDENTITY_POOL_ID=${IDENTITY_POOL_ID}
S3_BUCKET_NAME=${DATA_BUCKET_NAME}
WEBSITE_BUCKET_NAME=${WEBSITE_BUCKET_NAME}
STACK_NAME=${STACK_NAME}
STACK_ENVIRONMENT=${STACK_ENVIRONMENT}
BD_GUARDRAIL_IDENTIFIER=${GUARDRAIL_IDENTIFIER}
BD_GUARDRAIL_VERSION=${GUARDRAIL_VERSION}
BD_KB_ID=${KB_ID}
BEDROCK_MODEL_ID=${BEDROCK_MODEL_ID}
DOMAIN_NAME=${CLOUDFRONT_DOMAIN}
OTEL_PYTHON_DISTRO=${OTEL_PYTHON_DISTRO}
OTEL_PYTHON_CONFIGURATOR=${OTEL_PYTHON_CONFIGURATOR}
OTEL_RESOURCE_ATTRIBUTES=${OTEL_RESOURCE_ATTRIBUTES}
AGENT_OBSERVABILITY_ENABLED=${AGENT_OBSERVABILITY_ENABLED}
OTEL_EXPORTER_OTLP_PROTOCOL=${OTEL_EXPORTER_OTLP_PROTOCOL}
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}
OTEL_EXPORTER_OTLP_LOGS_HEADERS=${OTEL_EXPORTER_OTLP_LOGS_HEADERS}
OTEL_PYTHON_FASTAPI_EXCLUDED_URLS=${OTEL_PYTHON_FASTAPI_EXCLUDED_URLS}
OTEL_PYTHON_REQUESTS_EXCLUDED_URLS=${OTEL_PYTHON_REQUESTS_EXCLUDED_URLS}
OTEL_PYTHON_URLLIB3_EXCLUDED_URLS=${OTEL_PYTHON_URLLIB3_EXCLUDED_URLS}
OTEL_PYTHON_HTTPX_EXCLUDED_URLS=${OTEL_PYTHON_HTTPX_EXCLUDED_URLS}
OTEL_PYTHON_AIOHTTP_CLIENT_EXCLUDED_URLS=${OTEL_PYTHON_AIOHTTP_CLIENT_EXCLUDED_URLS}
OTEL_PYTHON_BOTO3SQS_EXCLUDED_URLS=${OTEL_PYTHON_BOTO3SQS_EXCLUDED_URLS}
OTEL_PYTHON_BOTOCORE_EXCLUDED_URLS=${OTEL_PYTHON_BOTOCORE_EXCLUDED_URLS}
OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=${OTEL_PYTHON_DISABLED_INSTRUMENTATIONS}
EOF


# Create a .env file for the frontend app
echo "Creating .env file for the frontend app..."
cat > ../source/frontend/.env << EOF
VITE_AWS_REGION=${AWS_REGION}
VITE_USER_POOL_ID=${USER_POOL_ID}
VITE_USER_POOL_CLIENT_ID=${USER_POOL_CLIENT_ID}
VITE_IDENTITY_POOL_ID=${IDENTITY_POOL_ID}
VITE_RESTAPI_URL=https://${CLOUDFRONT_DOMAIN}/api
VITE_WEBSOCKET_URL=https://${CLOUDFRONT_DOMAIN}/ws
EOF

# Build the frontend app
echo "Building the frontend app..."
cd ../source/frontend

# Install dependencies
echo "Installing frontend dependencies..."
npm install

# Build the app
echo "Building the frontend app..."
npm run build

# Deploy to S3
echo "Deploying frontend app to S3 bucket: $WEBSITE_BUCKET_NAME"
aws s3 sync dist/ s3://$WEBSITE_BUCKET_NAME/ --delete

 
# Invalidate CloudFront cache
echo "Invalidating CloudFront cache..."
aws cloudfront create-invalidation --distribution-id $DISTRIBUTION_ID --paths "/*" | cat
echo "CloudFront cache invalidation created for distribution: $DISTRIBUTION_ID"


# Display kubectl commands for monitoring
echo ""
echo "To monitor your deployments, use the following commands:"
echo "kubectl get pods -n ${STACK_NAME}-app"
echo "kubectl get services -n ${STACK_NAME}-app"
echo "kubectl get ingress -n ${STACK_NAME}-app"
echo ""
echo "To view logs:"  
echo "kubectl logs -n ${STACK_NAME}-app -l app=${STACK_NAME}-backend"

# Display the CloudFront distribution domain name
echo "--------------------------------"
echo "Frontend URL: https://$CLOUDFRONT_DOMAIN"
echo "--------------------------------"
echo "Login EMAIL: $EMAIL"
echo "--------------------------------"