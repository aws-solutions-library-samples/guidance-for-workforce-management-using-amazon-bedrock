import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import { Construct } from "constructs";

export interface InfrastructureStackProps extends cdk.StackProps {
  resourcePrefix: string;
}

export class InfrastructureStack extends cdk.Stack {
  public readonly vpc: ec2.Vpc;
  public readonly resourcePrefix: string;

  constructor(scope: Construct, id: string, props: InfrastructureStackProps) {
    super(scope, id, props);

    // get the resource prefix from the props
    this.resourcePrefix = props.resourcePrefix;

    // Create a VPC for the application
    this.vpc = new ec2.Vpc(this, `${this.resourcePrefix}-Vpc`, {
      maxAzs: 3,
      natGateways: 1,
    });

    // Output the VPC ID
    new cdk.CfnOutput(this, `${this.resourcePrefix}-VpcId`, {
      value: this.vpc.vpcId,
    });
  }
} 