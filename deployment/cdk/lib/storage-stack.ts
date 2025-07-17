import * as cdk from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as cloudfront from "aws-cdk-lib/aws-cloudfront";
import * as origins from "aws-cdk-lib/aws-cloudfront-origins";
import * as iam from "aws-cdk-lib/aws-iam";
import * as route53 from "aws-cdk-lib/aws-route53";
import * as route53targets from "aws-cdk-lib/aws-route53-targets";
import * as acm from "aws-cdk-lib/aws-certificatemanager";
import * as cr from "aws-cdk-lib/custom-resources";
import { Construct } from "constructs";

export interface StorageStackProps extends cdk.StackProps {
  resourcePrefix: string;
  environment: string;
  domainName: string;
  certificateArn: string;
  parentDomainName: string;
}

export class StorageStack extends cdk.Stack {
  public readonly websiteBucket: s3.Bucket;
  public readonly dataBucket: s3.Bucket;
  public readonly tables: { [key: string]: dynamodb.Table } = {};
  public readonly cloudFrontDistribution: cloudfront.Distribution;

  constructor(scope: Construct, id: string, props: StorageStackProps) {
    super(scope, id, props);

    const resourcePrefix = props.resourcePrefix;
    const domainName = props.domainName;
    const certificateArn = props.certificateArn;
    const parentDomainName = props.parentDomainName;

    if (!domainName || !certificateArn || !parentDomainName) {
      throw new Error('domainName, certificateArn, and parentDomainName must be provided when creating the StorageStack');
    }

    // Create a dedicated S3 access logs bucket
    const accessLogsBucket = new s3.Bucket(this, `${resourcePrefix}-AccessLogsBucket`, {
      publicReadAccess: false,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      enforceSSL: true, // Enforce SSL to comply with security requirements
      lifecycleRules: [
        {
          id: 'DeleteOldAccessLogs',
          enabled: true,
          expiration: cdk.Duration.days(90), // Keep access logs for 90 days
        },
      ],
    });

    // Add bucket policy to allow ALB to write access logs
    // ALB service account varies by region - this is for us-east-1
    const albServiceAccountId = this.region === 'us-east-1' ? '127311923021' : 
                                this.region === 'us-west-2' ? '797873946194' :
                                this.region === 'eu-west-1' ? '156460612806' :
                                this.region === 'ap-southeast-1' ? '114774131450' :
                                '127311923021'; // Default to us-east-1

    accessLogsBucket.addToResourcePolicy(
      new iam.PolicyStatement({
        sid: 'AWSLogDeliveryWrite',
        effect: iam.Effect.ALLOW,
        principals: [new iam.AccountPrincipal(albServiceAccountId)],
        actions: ['s3:PutObject'],
        resources: [`${accessLogsBucket.bucketArn}/AWSLogs/${this.account}/*`],
        conditions: {
          StringEquals: {
            's3:x-amz-acl': 'bucket-owner-full-control'
          }
        }
      })
    );

    accessLogsBucket.addToResourcePolicy(
      new iam.PolicyStatement({
        sid: 'AWSLogDeliveryAclCheck',
        effect: iam.Effect.ALLOW,
        principals: [new iam.AccountPrincipal(albServiceAccountId)],
        actions: ['s3:GetBucketAcl'],
        resources: [accessLogsBucket.bucketArn]
      })
    );

    // Create a CloudFront distribution and S3 bucket for hosting the web page
    this.websiteBucket = new s3.Bucket(this, `${resourcePrefix}-WebsiteBucket`, {
      publicReadAccess: false,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.DESTROY, // NOT recommended for production
      autoDeleteObjects: true, // NOT recommended for production
      enforceSSL: true, // Enforce SSL to comply with security requirements
      serverAccessLogsBucket: accessLogsBucket,
      serverAccessLogsPrefix: 'website-access-logs/',
    });

    const CacheDisabledPolicy = new cloudfront.CachePolicy(this, `${resourcePrefix}-CachePolicy`, {
      cachePolicyName: `${resourcePrefix}-cache-disabled-policy`,
      comment: 'Cache policy with caching disabled',
      defaultTtl: cdk.Duration.days(0),
      minTtl: cdk.Duration.minutes(0),
      maxTtl: cdk.Duration.days(0),
    });
    
    // Create the CloudFront distribution
    this.cloudFrontDistribution = new cloudfront.Distribution(this, `${resourcePrefix}-Distribution`, {
      defaultBehavior: {
        origin: new origins.S3Origin(this.websiteBucket),
        cachePolicy: CacheDisabledPolicy,
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS, // Force HTTPS
      },
      defaultRootObject: 'index.html',
      domainNames: [`${domainName}`],
      certificate: acm.Certificate.fromCertificateArn(this, `${resourcePrefix}-ImportedCertificate`, certificateArn),
      minimumProtocolVersion: cloudfront.SecurityPolicyProtocol.TLS_V1_2_2021, // Set minimum TLS version
    });

    // Create the S3 bucket policy after the CloudFront distribution
    this.websiteBucket.addToResourcePolicy(
      new iam.PolicyStatement({
        actions: ['s3:GetObject'],
        resources: [this.websiteBucket.arnForObjects('*')],
        principals: [new iam.ServicePrincipal('cloudfront.amazonaws.com')],
        conditions: {
          StringEquals: {
            'AWS:SourceArn': `arn:aws:cloudfront::${cdk.Stack.of(this).account}:distribution/${this.cloudFrontDistribution.distributionId}`
          }
        }
      })
    );

    // Output the S3 bucket name
    new cdk.CfnOutput(this, `${resourcePrefix}-WebsiteBucketName`, {
      value: this.websiteBucket.bucketName,
    });
    
    // Output the CloudFront distribution domain name
    new cdk.CfnOutput(this, `${resourcePrefix}-CloudFrontDistributionDomainName`, {
      value: this.cloudFrontDistribution.distributionDomainName,
    });

    // Import the existing hosted zone
    const hostedZone = route53.HostedZone.fromLookup(this, `${resourcePrefix}-ExistingHostedZone`, {
      domainName: parentDomainName,
    });

    // Create a custom resource to check if the record exists
    const checkRecordExists = new cr.AwsCustomResource(this, `${resourcePrefix}-CheckRecordExists`, {
      onCreate: {
        service: 'Route53',
        action: 'listResourceRecordSets',
        parameters: {
          HostedZoneId: hostedZone.hostedZoneId,
          StartRecordName: domainName,
          StartRecordType: 'A',
          MaxItems: '1'
        },
        physicalResourceId: cr.PhysicalResourceId.of('RecordCheck')
      },
      policy: cr.AwsCustomResourcePolicy.fromSdkCalls({
        resources: cr.AwsCustomResourcePolicy.ANY_RESOURCE
      })
    });

    // Create A record only if it doesn't exist
    const aliasRecord = new route53.ARecord(this, `${resourcePrefix}-AliasRecord`, {
      zone: hostedZone,
      target: route53.RecordTarget.fromAlias(
        new route53targets.CloudFrontTarget(this.cloudFrontDistribution)
      ),
      ttl: cdk.Duration.minutes(5),
      recordName: domainName,
      // Only create if the record doesn't exist
      deleteExisting: false,
    });

    // Add dependency to ensure the check happens first
    aliasRecord.node.addDependency(checkRecordExists);

    // Create a data bucket for storing assets
    this.dataBucket = new s3.Bucket(this, `${resourcePrefix}-DataBucket`, {
      publicReadAccess: false,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      enforceSSL: true, // Enforce SSL to comply with security requirements
      serverAccessLogsBucket: accessLogsBucket,
      serverAccessLogsPrefix: 'data-access-logs/',
    });

    // Output the S3 bucket name
    new cdk.CfnOutput(this, `${resourcePrefix}-DataBucketName`, {
      value: this.dataBucket.bucketName,
    });

    // Output the access logs bucket name
    new cdk.CfnOutput(this, `${resourcePrefix}-AccessLogsBucketName`, {
      value: accessLogsBucket.bucketName,
      description: 'The name of the S3 access logs bucket',
    });

    // Create DynamoDB tables for synthetic data
    this.tables['sessionHistory'] = new dynamodb.Table(this, `${resourcePrefix}-SessionHistoryTable`, {
      tableName: `${resourcePrefix}-SESSION-HISTORY-${props.environment || 'DEV'}`,
      partitionKey: {
        name: 'userId_sessionId',
        type: dynamodb.AttributeType.STRING
      },
      sortKey: {
        name: 'timestamp',
        type: dynamodb.AttributeType.NUMBER
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: true
      },
    });

    this.tables['userTasks'] = new dynamodb.Table(this, `${resourcePrefix}-UserTasksTable`, {
      tableName: `${resourcePrefix}-TASKLIST-${props.environment || 'DEV'}`,
      partitionKey: {
        name: 'userId',
        type: dynamodb.AttributeType.STRING
      },
      sortKey: {
        name: 'taskId',
        type: dynamodb.AttributeType.STRING
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: true
      },
    });

    this.tables['userRole'] = new dynamodb.Table(this, `${resourcePrefix}-UserRoleTable`, {
      tableName: `${resourcePrefix}-USERROLE-${props.environment || 'DEV'}`,
      partitionKey: {
        name: 'userId',
        type: dynamodb.AttributeType.STRING
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: true
      },
    });

    this.tables['customer'] = new dynamodb.Table(this, `${resourcePrefix}-CustomerTable`, {
      tableName: `${resourcePrefix}-CUSTOMER-${props.environment || 'DEV'}`,
      partitionKey: {
        name: 'customerId',
        type: dynamodb.AttributeType.STRING
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: true
      },
    });

    this.tables['dailyTasksByDay'] = new dynamodb.Table(this, `${resourcePrefix}-DailyTasksByDayTable`, {
      tableName: `${resourcePrefix}-DAILY_TASKS_BY_DAY-${props.environment || 'DEV'}`,
      partitionKey: {
        name: 'taskId',
        type: dynamodb.AttributeType.STRING
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: true
      },
    });

    this.tables['schedule'] = new dynamodb.Table(this, `${resourcePrefix}-ScheduleTable`, {
      tableName: `${resourcePrefix}-SCHEDULE-${props.environment || 'DEV'}`,
      partitionKey: {
        name: 'userId',
        type: dynamodb.AttributeType.STRING
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: true
      },
    });

    this.tables['timeoff'] = new dynamodb.Table(this, `${resourcePrefix}-TimeoffTable`, {
      tableName: `${resourcePrefix}-TIMEOFF-${props.environment || 'DEV'}`,
      partitionKey: {
        name: 'userId',
        type: dynamodb.AttributeType.STRING
      },
      sortKey: {
        name: 'timeoffId',
        type: dynamodb.AttributeType.STRING
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: true
      },
    });

    this.tables['products'] = new dynamodb.Table(this, `${resourcePrefix}-ProductsTable`, {
      tableName: `${resourcePrefix}-PRODUCTS-${props.environment || 'DEV'}`,
      partitionKey: {
        name: 'productId',
        type: dynamodb.AttributeType.STRING
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: true
      },
    });

    this.tables['customerTransactions'] = new dynamodb.Table(this, `${resourcePrefix}-CustomerTransactionsTable`, {
      tableName: `${resourcePrefix}-CUSTOMER_TRANSACTIONS-${props.environment || 'DEV'}`,
      partitionKey: {
        name: 'customerId',
        type: dynamodb.AttributeType.STRING
      },
      sortKey: {
        name: 'transactionId',
        type: dynamodb.AttributeType.STRING
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: true
      },
    });

    this.tables['feedback'] = new dynamodb.Table(this, `${resourcePrefix}-FeedbackTable`, {
      tableName: `${resourcePrefix}-FEEDBACK-${props.environment || 'DEV'}`,
      partitionKey: {
        name: 'messageId',
        type: dynamodb.AttributeType.STRING
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: true
      },
    });

    this.tables['images'] = new dynamodb.Table(this, `${resourcePrefix}-ImagesTable`, {
      tableName: `${resourcePrefix}-IMAGES-${props.environment || 'DEV'}`,
      partitionKey: {
        name: 'imageId',
        type: dynamodb.AttributeType.STRING
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      pointInTimeRecoverySpecification: {
        pointInTimeRecoveryEnabled: true
      },
    });
  }
}
