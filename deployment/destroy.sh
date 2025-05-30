#!/bin/bash

# Exit on error
set -e

# Disable AWS CLI pager to prevent pausing for user input
export AWS_PAGER=""

# Helper function to check if a stack exists
stack_exists() {
  local stack_name=$1
  aws cloudformation describe-stacks --stack-name "$stack_name" &>/dev/null
  return $?
}

# Helper function to get stack output safely
get_stack_output() {
  local stack_name=$1
  local output_key=$2
  if stack_exists "$stack_name"; then
    aws cloudformation describe-stacks --stack-name "$stack_name" --query "Stacks[0].Outputs[?OutputKey=='$output_key'].OutputValue" --output text 2>/dev/null || true
  else
    echo ""
  fi
}

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
  "DOMAIN_NAME"
  "PARENT_DOMAIN_NAME"
)

for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "Error: Required environment variable $var is not set."
    exit 1
  fi
done

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

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
echo "AWS Account ID: $AWS_ACCOUNT_ID"

# Delete Route53 records
echo "Deleting Route53 records..."
HOSTED_ZONE_ID=$(aws route53 list-hosted-zones-by-name --dns-name ${PARENT_DOMAIN_NAME} --query 'HostedZones[0].Id' --output text 2>/dev/null | cut -d'/' -f3 || true)

if [ -n "$HOSTED_ZONE_ID" ]; then
  echo "Found hosted zone: $HOSTED_ZONE_ID"
  
  # Get the ALB hosted zone ID and DNS name
  CLUSTER_NAME=$(get_stack_output ${STACK_NAME}EksStack ${STACK_NAME}ClusterName)
  if [ -n "$CLUSTER_NAME" ]; then
    echo "Found EKS cluster: $CLUSTER_NAME"
    aws eks update-kubeconfig --name $CLUSTER_NAME --region $AWS_REGION || {
      echo "Warning: Could not update kubeconfig for cluster $CLUSTER_NAME. It may have been deleted."
      echo "Continuing with other cleanup tasks..."
    }
    
    if kubectl cluster-info &>/dev/null; then
      BACKEND_ALB=$(kubectl get ingress ${STACK_NAME}-ingress -n ${STACK_NAME}-app -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || true)
      if [ -n "$BACKEND_ALB" ]; then
        echo "Found ALB: $BACKEND_ALB"
        ALB_HOSTED_ZONE_ID=$(aws elbv2 describe-load-balancers --query "LoadBalancers[?DNSName=='$BACKEND_ALB'].CanonicalHostedZoneId" --output text 2>/dev/null || true)
        
        if [ -n "$ALB_HOSTED_ZONE_ID" ]; then
          echo "Deleting API record for ALB..."
          aws route53 change-resource-record-sets \
            --hosted-zone-id $HOSTED_ZONE_ID \
            --change-batch '{
              "Changes": [{
                "Action": "DELETE",
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
            }' || echo "Warning: Failed to delete API record. Continuing..."
        fi
      fi
    fi
  fi

  # Get CloudFront distribution ID
  CLOUDFRONT_DOMAIN=$(get_stack_output ${STACK_NAME}StorageStack ${STACK_NAME}CloudFrontDistributionDomainName)
  if [ -n "$CLOUDFRONT_DOMAIN" ]; then
    echo "Found CloudFront domain: $CLOUDFRONT_DOMAIN"
    echo "Deleting CloudFront record..."
    aws route53 change-resource-record-sets \
      --hosted-zone-id $HOSTED_ZONE_ID \
      --change-batch '{
        "Changes": [{
          "Action": "DELETE",
          "ResourceRecordSet": {
            "Name": "'${DOMAIN_NAME}'",
            "Type": "A",
            "AliasTarget": {
              "HostedZoneId": "Z2FDTNDATAQYW2",
              "DNSName": "'${CLOUDFRONT_DOMAIN}'",
              "EvaluateTargetHealth": false
            }
          }
        }]
      }' || echo "Warning: Failed to delete CloudFront record. Continuing..."
  fi
else
  echo "No hosted zone found. Skipping Route53 record deletion."
fi

# Delete Kubernetes resources
echo "Deleting Kubernetes resources..."
if command -v kubectl &> /dev/null; then
  # Get EKS cluster name
  CLUSTER_NAME=$(get_stack_output ${STACK_NAME}EksStack ${STACK_NAME}ClusterName)
  
  if [ -n "$CLUSTER_NAME" ]; then
    echo "Found EKS cluster: $CLUSTER_NAME"
    # Configure kubectl
    aws eks update-kubeconfig --name $CLUSTER_NAME --region $AWS_REGION || {
      echo "Warning: Could not update kubeconfig for cluster $CLUSTER_NAME. It may have been deleted."
      echo "Continuing with other cleanup tasks..."
    }
    
    # Only try to delete Kubernetes resources if we can access the cluster
    if kubectl cluster-info &>/dev/null; then
      echo "Deleting Kubernetes resources in cluster $CLUSTER_NAME..."
      # Delete ingress first to trigger ALB deletion
      kubectl delete ingress ${STACK_NAME}-ingress -n ${STACK_NAME}-app --ignore-not-found
      
      # Wait for ALB to be deleted
      echo "Waiting for Application Load Balancer to be deleted..."
      sleep 30
      
      # Delete service
      kubectl delete service ${STACK_NAME}-backend-service -n ${STACK_NAME}-app --ignore-not-found
      
      # Delete deployment
      kubectl delete deployment ${STACK_NAME}-backend -n ${STACK_NAME}-app --ignore-not-found
      
      # Delete configmap
      kubectl delete configmap ${STACK_NAME}-app-config -n ${STACK_NAME}-app --ignore-not-found
      
      # Delete service account
      kubectl delete serviceaccount ${STACK_NAME}-backend-sa -n ${STACK_NAME}-app --ignore-not-found
      
      # Delete namespace
      kubectl delete namespace ${STACK_NAME}-app --ignore-not-found
      
      # Wait for all resources to be fully deleted
      echo "Waiting for all Kubernetes resources to be fully deleted..."
      sleep 60
    else
      echo "Warning: Could not access cluster $CLUSTER_NAME. Skipping Kubernetes resource deletion."
    fi
  else
    echo "No EKS cluster found. Skipping Kubernetes resource deletion."
  fi
else
  echo "kubectl not found. Skipping Kubernetes resource deletion."
fi

# Delete ECR repositories
echo "Deleting ECR repositories..."
BACKEND_REPO_URI=$(get_stack_output ${STACK_NAME}EksStack BackendRepositoryUri)

if [ -n "$BACKEND_REPO_URI" ]; then
  echo "Found ECR repository URI: $BACKEND_REPO_URI"
  # Extract repository name from URI
  REPO_NAME=$(echo $BACKEND_REPO_URI | cut -d'/' -f2)
  
  # Check if repository exists before trying to delete
  if aws ecr describe-repositories --repository-names $REPO_NAME --region $AWS_REGION &>/dev/null; then
    echo "Deleting ECR repository: $REPO_NAME"
    aws ecr delete-repository --repository-name $REPO_NAME --force --region $AWS_REGION || {
      echo "Warning: Failed to delete ECR repository $REPO_NAME. Continuing..."
    }
  else
    echo "ECR repository $REPO_NAME not found. Skipping deletion."
  fi
else
  echo "No ECR repository found in CloudFormation outputs. Skipping ECR deletion."
fi

# Delete CloudFront distribution
echo "Deleting CloudFront distribution..."
CLOUDFRONT_DOMAIN=$(get_stack_output ${STACK_NAME}StorageStack ${STACK_NAME}CloudFrontDistributionDomainName)

if [ -n "$CLOUDFRONT_DOMAIN" ]; then
  echo "Found CloudFront domain: $CLOUDFRONT_DOMAIN"
  DISTRIBUTION_ID=$(aws cloudfront list-distributions --query "DistributionList.Items[?DomainName=='$CLOUDFRONT_DOMAIN'].Id" --output text 2>/dev/null || true)
  
  if [ -n "$DISTRIBUTION_ID" ]; then
    echo "Found CloudFront distribution ID: $DISTRIBUTION_ID"
    # Get the current distribution config
    DIST_CONFIG=$(aws cloudfront get-distribution-config --id $DISTRIBUTION_ID 2>/dev/null || true)
    if [ -n "$DIST_CONFIG" ]; then
      ETAG=$(echo "$DIST_CONFIG" | jq -r '.ETag')
      DIST_CONFIG=$(echo "$DIST_CONFIG" | jq -r '.DistributionConfig')
      
      # Update the config to disable the distribution
      DIST_CONFIG=$(echo "$DIST_CONFIG" | jq '.Enabled = false')
      
      # Disable the distribution
      echo "Disabling CloudFront distribution..."
      aws cloudfront update-distribution \
        --id $DISTRIBUTION_ID \
        --distribution-config "$DIST_CONFIG" \
        --if-match "$ETAG" || {
        echo "Warning: Failed to disable CloudFront distribution. Continuing..."
      }
      
      # Wait for distribution to be disabled
      echo "Waiting for CloudFront distribution to be disabled..."
      aws cloudfront wait distribution-deployed --id $DISTRIBUTION_ID || {
        echo "Warning: Failed to wait for CloudFront distribution to be disabled. Continuing..."
      }
      
      # Get the latest ETag
      ETAG=$(aws cloudfront get-distribution-config --id $DISTRIBUTION_ID | jq -r '.ETag' 2>/dev/null || true)
      
      if [ -n "$ETAG" ]; then
        # Delete the distribution
        echo "Deleting CloudFront distribution..."
        aws cloudfront delete-distribution --id $DISTRIBUTION_ID --if-match "$ETAG" || {
          echo "Warning: Failed to delete CloudFront distribution. Continuing..."
        }
      else
        echo "Warning: Could not get ETag for CloudFront distribution. Skipping deletion."
      fi
    else
      echo "Warning: Could not get CloudFront distribution config. Skipping deletion."
    fi
  else
    echo "No CloudFront distribution found. Skipping deletion."
  fi
else
  echo "No CloudFront domain found in CloudFormation outputs. Skipping CloudFront deletion."
fi

# Delete S3 buckets
echo "Deleting S3 buckets..."
# Get bucket names from CloudFormation
WEBSITE_BUCKET_NAME=$(get_stack_output ${STACK_NAME}StorageStack ${STACK_NAME}WebsiteBucketName)
DATA_BUCKET_NAME=$(get_stack_output ${STACK_NAME}StorageStack ${STACK_NAME}DataBucketName)

# Delete website bucket
if [ -n "$WEBSITE_BUCKET_NAME" ]; then
  echo "Deleting website bucket: $WEBSITE_BUCKET_NAME"
  aws s3 rm s3://$WEBSITE_BUCKET_NAME --recursive || echo "Warning: Failed to empty website bucket. Continuing..."
  aws s3 rb s3://$WEBSITE_BUCKET_NAME --force || echo "Warning: Failed to delete website bucket. Continuing..."
else
  echo "Website bucket name not found. Skipping website bucket deletion."
fi

# Delete data bucket
if [ -n "$DATA_BUCKET_NAME" ]; then
  echo "Deleting data bucket: $DATA_BUCKET_NAME"
  aws s3 rm s3://$DATA_BUCKET_NAME --recursive || echo "Warning: Failed to empty data bucket. Continuing..."
  aws s3 rb s3://$DATA_BUCKET_NAME --force || echo "Warning: Failed to delete data bucket. Continuing..."
else
  echo "Data bucket name not found. Skipping data bucket deletion."
fi

# Delete Cognito resources
echo "Deleting Cognito resources..."
USER_POOL_ID=$(get_stack_output ${STACK_NAME}AuthStack ${STACK_NAME}UserPoolId)

if [ -n "$USER_POOL_ID" ]; then
  echo "Deleting user pool: $USER_POOL_ID"
  aws cognito-idp delete-user-pool --user-pool-id $USER_POOL_ID || echo "Warning: Failed to delete user pool. Continuing..."
else
  echo "User pool ID not found. Skipping Cognito deletion."
fi

# Delete CDK stacks in correct order
echo "Deleting CDK stacks in dependency order..."
cd cdk

# Delete stacks in reverse dependency order
echo "Checking GuardrailsStack..."
if stack_exists ${STACK_NAME}GuardrailsStack; then
  echo "Deleting GuardrailsStack..."
  npx cdk destroy ${STACK_NAME}GuardrailsStack --force || echo "Warning: Failed to delete GuardrailsStack. Continuing..."
else
  echo "GuardrailsStack does not exist. Skipping..."
fi

echo "Checking SyntheticDataStack..."
if stack_exists ${STACK_NAME}SyntheticDataStack; then
  echo "Deleting SyntheticDataStack..."
  npx cdk destroy ${STACK_NAME}SyntheticDataStack --force || echo "Warning: Failed to delete SyntheticDataStack. Continuing..."
else
  echo "SyntheticDataStack does not exist. Skipping..."
fi

echo "Checking OpenSearchStack..."
if stack_exists ${STACK_NAME}OpenSearchStack; then
  echo "Deleting OpenSearchStack..."
  npx cdk destroy ${STACK_NAME}OpenSearchStack --force || echo "Warning: Failed to delete OpenSearchStack. Continuing..."
else
  echo "OpenSearchStack does not exist. Skipping..."
fi

echo "Checking EksStack..."
if stack_exists ${STACK_NAME}EksStack; then
  echo "Deleting EksStack..."
  npx cdk destroy ${STACK_NAME}EksStack --force || echo "Warning: Failed to delete EksStack. Continuing..."
else
  echo "EksStack does not exist. Skipping..."
fi

# Wait for EKS cluster to be fully deleted to ensure network interfaces are cleaned up
echo "Waiting for EKS cluster to be fully deleted..."
CLUSTER_NAME=$(get_stack_output ${STACK_NAME}EksStack ${STACK_NAME}ClusterName)
if [ -n "$CLUSTER_NAME" ]; then
  echo "Waiting for EKS cluster $CLUSTER_NAME to be deleted..."
  while aws eks describe-cluster --name $CLUSTER_NAME --region $AWS_REGION &>/dev/null; do
    echo "EKS cluster still exists, waiting 30 seconds..."
    sleep 30
  done
  echo "EKS cluster has been deleted."
else
  echo "EKS cluster name not found, assuming it's already deleted."
fi

# Additional wait to ensure network interfaces are cleaned up
echo "Waiting additional 60 seconds for network interfaces to be cleaned up..."
sleep 60

# Manual cleanup of any remaining network interfaces
echo "Checking for remaining network interfaces in VPC..."
VPC_ID=$(get_stack_output ${STACK_NAME}InfraStack VpcId)
if [ -n "$VPC_ID" ]; then
  echo "Found VPC ID: $VPC_ID"
  # Get all network interfaces in the VPC
  ENI_IDS=$(aws ec2 describe-network-interfaces --filters "Name=vpc-id,Values=$VPC_ID" --query "NetworkInterfaces[?Status=='available'].NetworkInterfaceId" --output text 2>/dev/null || true)
  
  if [ -n "$ENI_IDS" ]; then
    echo "Found network interfaces to clean up: $ENI_IDS"
    for ENI_ID in $ENI_IDS; do
      echo "Deleting network interface: $ENI_ID"
      aws ec2 delete-network-interface --network-interface-id $ENI_ID || echo "Warning: Failed to delete network interface $ENI_ID"
    done
    echo "Waiting 30 seconds after network interface cleanup..."
    sleep 30
  else
    echo "No available network interfaces found in VPC."
  fi
else
  echo "VPC ID not found, skipping network interface cleanup."
fi

echo "Checking AuthStack..."
if stack_exists ${STACK_NAME}AuthStack; then
  echo "Deleting AuthStack..."
  npx cdk destroy ${STACK_NAME}AuthStack --force || echo "Warning: Failed to delete AuthStack. Continuing..."
else
  echo "AuthStack does not exist. Skipping..."
fi

echo "Checking StorageStack..."
if stack_exists ${STACK_NAME}StorageStack; then
  echo "Deleting StorageStack..."
  npx cdk destroy ${STACK_NAME}StorageStack --force || echo "Warning: Failed to delete StorageStack. Continuing..."
else
  echo "StorageStack does not exist. Skipping..."
fi

echo "Checking InfraStack..."
if stack_exists ${STACK_NAME}InfraStack; then
  echo "Deleting InfraStack..."
  npx cdk destroy ${STACK_NAME}InfraStack --force || echo "Warning: Failed to delete InfraStack. Continuing..."
else
  echo "InfraStack does not exist. Skipping..."
fi

cd ..

echo "Cleanup completed successfully!" 