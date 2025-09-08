#!/bin/bash

# Exit on error
set -e

# Disable AWS CLI pager to prevent pausing for user input
export AWS_PAGER=""

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

# Check for required environment variables
required_vars=(
  "AWS_REGION"
  "STACK_NAME"
  "STACK_ENVIRONMENT"
  "COGNITO_USER_POOL_ID"
  "COGNITO_APP_CLIENT_ID"
  "COGNITO_IDENTITY_POOL_ID"
  "BD_GUARDRAIL_IDENTIFIER"
  "BD_GUARDRAIL_VERSION"
  "BD_KB_ID"
  "BEDROCK_MODEL_ID"
  "DOMAIN_NAME"
  "PARENT_DOMAIN_NAME"
  "WEB_CERTIFICATE_ARN"
  "CERTIFICATE_ARN"
  "EMAIL"
  "COGNITO_PASSWORD"
)

for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "Error: Required environment variable $var is not set."
    exit 1
  fi
done

# Function to update or add environment variable in .env file
update_env_var() {
  local var_name="$1"
  local var_value="$2"
  local env_file=".env"
  
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


# Get the EKS cluster name from the CDK output
CLUSTER_NAME=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME}EksStack --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}ClusterName'].OutputValue" --output text)

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
echo "ALB Controller policy section completed. Continuing with deployment..."

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
  echo "This may be because the OpenSearchStack failed to deploy."
  echo "Using the value from .env file: ${BD_KB_ID}"
  if [ -z "$BD_KB_ID" ]; then
    echo "Error: No KnowledgeBaseId available in .env file either."
    echo "Knowledge Base features will not work until this is resolved."
    echo "You may need to deploy the OpenSearchStack manually or provide a valid KB_ID in .env"
  fi
else
  echo "Found KnowledgeBaseId: $KB_ID"
  # Only override the BD_KB_ID from .env with the one from CloudFormation if it is of value 'XXX'
  if [ "$BD_KB_ID" = "XXX" ]; then
    BD_KB_ID=$KB_ID

    # Update the .env file with the new KB_ID
    echo "Updating .env file with the new KnowledgeBaseId..."
    if grep -q "^BD_KB_ID=" .env; then
      # If BD_KB_ID exists in .env, update it
      # Handle different sed syntax for macOS vs Linux
      if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS version
        sed -i '' "s/^BD_KB_ID=.*$/BD_KB_ID=$KB_ID/" .env
      else
        # Linux version
        sed -i "s/^BD_KB_ID=.*$/BD_KB_ID=$KB_ID/" .env
      fi
    else
      # If BD_KB_ID doesn't exist in .env, add it
      echo "BD_KB_ID=$KB_ID" >> .env
    fi
    echo "Updated .env file with BD_KB_ID=$KB_ID"
  fi
fi
 

# Get the GUARDRAIL_IDENTIFIER and GUARDRAIL_VERSION directly from the GuardrailsStack output
echo "Retrieving GUARDRAIL_IDENTIFIER and GUARDRAIL_VERSION from CloudFormation..."
GUARDRAIL_IDENTIFIER_FULL=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME}GuardrailsStack --query "Stacks[0].Outputs[?OutputKey=='GuardrailIdentifier'].OutputValue" --output text)
# Extract only the last part after "guardrail/"
GUARDRAIL_IDENTIFIER=$(echo $GUARDRAIL_IDENTIFIER_FULL | sed 's/.*guardrail\///')
GUARDRAIL_VERSION=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME}GuardrailsStack --query "Stacks[0].Outputs[?OutputKey=='GuardrailVersion'].OutputValue" --output text)
if [ -z "$GUARDRAIL_IDENTIFIER" ]; then
  echo "Warning: Could not retrieve GUARDRAIL_IDENTIFIER from CloudFormation. Using the value from .env file: ${BD_GUARDRAIL_IDENTIFIER}"
else
  echo "Found GUARDRAIL_IDENTIFIER: $GUARDRAIL_IDENTIFIER"
  # Override the GUARDRAIL_IDENTIFIER from .env with the one from CloudFormation
  BD_GUARDRAIL_IDENTIFIER=$GUARDRAIL_IDENTIFIER
  
  # Update the .env file with the new KB_ID
  echo "Updating .env file with the new GUARDRAIL_IDENTIFIER..."
  if grep -q "^BD_GUARDRAIL_IDENTIFIER=" .env; then
    # If GUARDRAIL_IDENTIFIER exists in .env, update it
    # Handle different sed syntax for macOS vs Linux
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS version
      sed -i '' "s/^BD_GUARDRAIL_IDENTIFIER=.*$/BD_GUARDRAIL_IDENTIFIER=$GUARDRAIL_IDENTIFIER/" .env
    else
      # Linux version
      sed -i "s/^BD_GUARDRAIL_IDENTIFIER=.*$/BD_GUARDRAIL_IDENTIFIER=$GUARDRAIL_IDENTIFIER/" .env
    fi
  else
    # If GUARDRAIL_IDENTIFIER doesn't exist in .env, add it
    echo "BD_GUARDRAIL_IDENTIFIER=$GUARDRAIL_IDENTIFIER" >> .env
  fi
  echo "Updated .env file with BD_GUARDRAIL_IDENTIFIER=$GUARDRAIL_IDENTIFIER"
fi

if [ -z "$GUARDRAIL_VERSION" ]; then
  echo "Warning: Could not retrieve GUARDRAIL_VERSION from CloudFormation. Using the value from .env file: ${BD_GUARDRAIL_VERSION}"
else
  echo "Found GUARDRAIL_VERSION: $GUARDRAIL_VERSION"
  # Override the   from .env with the one from CloudFormation
  BD_GUARDRAIL_VERSION=$GUARDRAIL_VERSION
  
  # Update the .env file with the new KB_ID
  echo "Updating .env file with the new GUARDRAIL_VERSION..."
  if grep -q "^BD_GUARDRAIL_VERSION=" .env; then
    # If GUARGUARDRAIL_VERSION exists in .env, update it
    # Handle different sed syntax for macOS vs Linux
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS version
      sed -i '' "s/^BD_GUARDRAIL_VERSION=.*$/BD_GUARDRAIL_VERSION=$GUARDRAIL_VERSION/" .env
    else
      # Linux version
      sed -i "s/^BD_GUARDRAIL_VERSION=.*$/BD_GUARDRAIL_VERSION=$GUARDRAIL_VERSION/" .env
    fi
  else
    # If GUARDRAIL_VERSION doesn't exist in .env, add it
    echo "BD_GUARDRAIL_VERSION=$GUARDRAIL_VERSION" >> .env
  fi
  echo "Updated .env file with BD_GUARDRAIL_VERSION=$GUARDRAIL_VERSION"
fi


# Get the Cognito User Pool ID and Client ID using the exact output keys
AUTH_STACK_NAME="${STACK_NAME}AuthStack"
echo "Getting User Pool ID and Client ID from $AUTH_STACK_NAME..."
USER_POOL_ID=$(aws cloudformation describe-stacks --stack-name $AUTH_STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}UserPoolId'].OutputValue" --output text)
USER_POOL_CLIENT_ID=$(aws cloudformation describe-stacks --stack-name $AUTH_STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}UserPoolClientId'].OutputValue" --output text)
IDENTITY_POOL_ID=$(aws cloudformation describe-stacks --stack-name $AUTH_STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}IdentityPoolId'].OutputValue" --output text)
echo "User Pool ID: $USER_POOL_ID"
echo "User Pool Client ID: $USER_POOL_CLIENT_ID"
echo "Identity Pool ID: $IDENTITY_POOL_ID"

# Get the S3 bucket name from the StorageStack using the exact output key
STORAGE_STACK_NAME="${STACK_NAME}StorageStack"
S3_BUCKET_NAME=$(aws cloudformation describe-stacks --stack-name $STORAGE_STACK_NAME --query "Stacks[0].Outputs[?OutputKey=='${STACK_NAME}DataBucketName'].OutputValue" --output text)
echo "S3 data bucket name: $S3_BUCKET_NAME"
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
  BD_GUARDRAIL_IDENTIFIER: "${BD_GUARDRAIL_IDENTIFIER}"
  BD_GUARDRAIL_VERSION: "${BD_GUARDRAIL_VERSION}"
  BD_KB_ID: "${BD_KB_ID}"
  BEDROCK_MODEL_ID: "${BEDROCK_MODEL_ID}"
  AWS_DEFAULT_REGION: "${AWS_REGION}"
  AWS_REGION: "${AWS_REGION}"
  STACK_NAME: "${STACK_NAME}"
  STACK_ENVIRONMENT: "${STACK_ENVIRONMENT}"
  S3_BUCKET_NAME: "${S3_BUCKET_NAME}"
  DOMAIN_NAME: "${DOMAIN_NAME}"
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


# Create Ingress with SSL
echo "Creating Ingress resources with SSL..."
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
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS":443}]'
    alb.ingress.kubernetes.io/certificate-arn: "${CERTIFICATE_ARN}"
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
    alb.ingress.kubernetes.io/ssl-policy: "ELBSecurityPolicy-TLS-1-2-Ext-2018-06"
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

# Update the website bucket name in the .env file
update_env_var "WEBSITE_BUCKET_NAME" "$WEBSITE_BUCKET_NAME"

# Update the data bucket name in the .env file
update_env_var "S3_BUCKET_NAME" "$DATA_BUCKET_NAME"

# Update the user pool id in the .env file
update_env_var "COGNITO_USER_POOL_ID" "$USER_POOL_ID"

# Update the user pool client id in the .env file
update_env_var "COGNITO_APP_CLIENT_ID" "$USER_POOL_CLIENT_ID"

# Update the identity pool id in the .env file
update_env_var "COGNITO_IDENTITY_POOL_ID" "$IDENTITY_POOL_ID"




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

# Get the hosted zone ID with more verbose logging
echo "Retrieving hosted zone ID..."
HOSTED_ZONE_ID=$(aws route53 list-hosted-zones-by-name --dns-name ${PARENT_DOMAIN_NAME} --query 'HostedZones[0].Id' --output text | cut -d'/' -f3)
if [ -z "$HOSTED_ZONE_ID" ]; then
  echo "Error: Could not retrieve hosted zone ID. Please check your Route53 configuration."
  exit 1
fi
echo "Found Hosted Zone ID: $HOSTED_ZONE_ID"

# Create Route53 records for API and WebSocket ALBs with more verbose logging
echo "Creating Route53 records for API and WebSocket endpoints..."

# Get the ALB hosted zone ID using the DNS name
echo "Retrieving ALB hosted zone ID..."
ALB_HOSTED_ZONE_ID=$(aws elbv2 describe-load-balancers --query "LoadBalancers[?DNSName=='$BACKEND_ALB'].CanonicalHostedZoneId" --output text)
if [ -z "$ALB_HOSTED_ZONE_ID" ]; then
  echo "Error: Could not retrieve ALB hosted zone ID. Please check your ALB configuration."
  exit 1
fi
echo "Found ALB Hosted Zone ID: $ALB_HOSTED_ZONE_ID"

# Create API record
echo "Creating API record..."
aws route53 change-resource-record-sets \
  --hosted-zone-id $HOSTED_ZONE_ID \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "backend.'${DOMAIN_NAME}'",
        "Type": "A",
        "AliasTarget": {
          "HostedZoneId": "'${ALB_HOSTED_ZONE_ID}'",
          "DNSName": "'${BACKEND_ALB}'",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'

echo "Route53 records created successfully."

# Create record for CloudFront
echo "Creating record for CloudFront..."
echo "Debug - DOMAIN_NAME: ${DOMAIN_NAME}"
echo "Debug - CLOUDFRONT_DOMAIN: ${CLOUDFRONT_DOMAIN}"
echo "Debug - HOSTED_ZONE_ID: ${HOSTED_ZONE_ID}"

if [ -z "$DOMAIN_NAME" ]; then
  echo "Error: DOMAIN_NAME is empty. Please check your .env file."
  exit 1
fi

if [ -z "$CLOUDFRONT_DOMAIN" ]; then
  echo "Error: CLOUDFRONT_DOMAIN is empty. Please check your CloudFormation stack outputs."
  exit 1
fi

if [ -z "$HOSTED_ZONE_ID" ]; then
  echo "Error: HOSTED_ZONE_ID is empty. Please check your Route53 configuration."
  exit 1
fi

aws route53 change-resource-record-sets \
  --hosted-zone-id $HOSTED_ZONE_ID \
  --change-batch "{
    \"Changes\": [{
      \"Action\": \"UPSERT\",
      \"ResourceRecordSet\": {
        \"Name\": \"${DOMAIN_NAME}\",
        \"Type\": \"A\",
        \"AliasTarget\": {
          \"HostedZoneId\": \"Z2FDTNDATAQYW2\",
          \"DNSName\": \"${CLOUDFRONT_DOMAIN}\",
          \"EvaluateTargetHealth\": false
        }
      }
    }]
  }"

echo "Route53 records created successfully."


# Create a .env file for the frontend app
echo "Creating .env file for the frontend app..."
cat > ../source/frontend/.env << EOF
VITE_AWS_REGION=${AWS_REGION}
VITE_USER_POOL_ID=${USER_POOL_ID}
VITE_USER_POOL_CLIENT_ID=${USER_POOL_CLIENT_ID}
VITE_IDENTITY_POOL_ID=${IDENTITY_POOL_ID}
VITE_RESTAPI_URL=https://backend.${DOMAIN_NAME}/api
VITE_WEBSOCKET_URL=https://backend.${DOMAIN_NAME}/ws
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
DISTRIBUTION_ID=$(aws cloudfront list-distributions --query "DistributionList.Items[?DomainName=='$CLOUDFRONT_DOMAIN'].Id" --output text)
if [ -n "$DISTRIBUTION_ID" ]; then
  aws cloudfront create-invalidation --distribution-id $DISTRIBUTION_ID --paths "/*" | cat
  echo "CloudFront cache invalidation created for distribution: $DISTRIBUTION_ID"
else
  echo "Could not find CloudFront distribution ID for domain: $CLOUDFRONT_DOMAIN"
fi

cd ../../deployment

# Display the CloudFront distribution domain name
echo "--------------------------------"
echo "Frontend URL: https://$DOMAIN_NAME"
echo "--------------------------------"
echo "Login EMAIL: $EMAIL"
echo "--------------------------------"
echo "Temporary Login PASSWORD: $COGNITO_PASSWORD"
echo "--------------------------------"
echo "CloudFront URL: https://$CLOUDFRONT_DOMAIN"
echo "--------------------------------"
# Get and display the ALB endpoint
echo "Retrieving ALB endpoint (may take a few minutes to be available)..."
sleep 30

if [ -n "$BACKEND_ALB" ]; then
  echo "BACKEND Endpoint: https://api.${DOMAIN_NAME} (ALB: $BACKEND_ALB)"
else
  echo "BACKEND ALB not yet available. Check status with: kubectl get ingress ${STACK_NAME}-ingress -n ${STACK_NAME}-app"
fi
echo "--------------------------------"


# Display kubectl commands for monitoring
echo ""
echo "To monitor your deployments, use the following commands:"
echo "kubectl get pods -n ${STACK_NAME}-app"
echo "kubectl get services -n ${STACK_NAME}-app"
echo "kubectl get ingress -n ${STACK_NAME}-app"
echo ""
echo "To view logs:"  
echo "kubectl logs -n ${STACK_NAME}-app -l app=${STACK_NAME}-backend"
