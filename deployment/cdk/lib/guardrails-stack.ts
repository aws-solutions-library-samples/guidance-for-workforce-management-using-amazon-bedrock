import * as cdk from "aws-cdk-lib";
import { CfnGuardrail } from "aws-cdk-lib/aws-bedrock";
import { Construct } from "constructs";

export interface GuardrailsStackProps extends cdk.StackProps {
  resourcePrefix: string;
  environment: string;

}

export class GuardrailsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: GuardrailsStackProps) {
    super(scope, id, props);

    const resourcePrefix = props.resourcePrefix;
    const environment = props.environment;
    // Create a Bedrock Guardrail using CfnGuardrail (L1 construct)
    const guardrail = new CfnGuardrail(this, 'BedrockGuardrail', {
      name: `${resourcePrefix}-guardrail-${environment}`,
      description: 'Legal ethical guardrails for retail assistant.',
      blockedInputMessaging: 'Your input contains content that violates our usage policies.',
      blockedOutputsMessaging: 'The response contains content that violates our usage policies.',
      
      // Content policy configuration
      contentPolicyConfig: {
        filtersConfig: [
          {
            type: 'SEXUAL',
            inputStrength: 'HIGH',
            outputStrength: 'HIGH'
          },
          {
            type: 'VIOLENCE',
            inputStrength: 'HIGH',
            outputStrength: 'HIGH'
          },
          {
            type: 'HATE',
            inputStrength: 'HIGH',
            outputStrength: 'HIGH'
          },
          {
            type: 'INSULTS',
            inputStrength: 'HIGH',
            outputStrength: 'HIGH'
          },
          {
            type: 'MISCONDUCT',
            inputStrength: 'HIGH',
            outputStrength: 'HIGH'
          }
        ]
      },
      
      // Sensitive information policy configuration
      sensitiveInformationPolicyConfig: {
        piiEntitiesConfig: [
          {
            type: 'ADDRESS',
            action: 'ANONYMIZE'
          },
          // {
          //   type: 'EMAIL',
          //   action: 'BLOCK'
          // }
        ],
        regexesConfig: [
          {
            name: 'CustomerIDPattern',
            description: 'Pattern for customer IDs',
            pattern: '^[A-Z]{2}\\d{6}$',
            action: 'ANONYMIZE'
          }
        ]
      },
      
      // Contextual grounding policy configuration
      contextualGroundingPolicyConfig: {
        filtersConfig: [
          {
            type: 'GROUNDING',
            threshold: 0.5
          },
          {
            type: 'RELEVANCE',
            threshold: 0.5
          }
        ]
      },
      
      // Topic policy configuration
      topicPolicyConfig: {
        topicsConfig: [
          {
            name: 'FINANCIAL_ADVICE',
            type: 'DENY',
            definition: 'Offering guidance or suggestions on financial investments, financial planning, or financial decisions.'
          },
          {
            name: 'LEGAL_ADVICE',
            type: 'DENY',
            definition: 'Offering guidance or suggestions on legal matters, legal actions, interpretation of laws, or legal rights and responsibilities.',
            examples: [
              'Can I sue someone for this?',
              'What are my legal rights in this situation?',
              'Is this action against the law?',
              'What should I do to file a legal complaint?',
              'Can you explain this law to me?'
            ]
          }
        ]
      },
      
      // Word policy configuration
      wordPolicyConfig: {
        wordsConfig: [
          {
            text: 'drugs'
          }
        ],
        managedWordListsConfig: [
          {
            type: 'PROFANITY'
          }
        ]
      }
    });

    // Export the guardrail ARN
    new cdk.CfnOutput(this, 'GuardrailArn', {
      value: guardrail.attrGuardrailArn,
      description: 'The ARN of the Bedrock Guardrail',
      exportName: `${resourcePrefix}-guardrail-arn`
    });

    // Export the guardrail identifier
    new cdk.CfnOutput(this, 'GuardrailIdentifier', {
      value: guardrail.ref,
      description: 'The identifier of the Bedrock Guardrail',
      exportName: `${resourcePrefix}-guardrail-identifier`
    });

    // Export the guardrail version
    new cdk.CfnOutput(this, 'GuardrailVersion', {
      value: guardrail.attrVersion,
      description: 'The version of the Bedrock Guardrail',
      exportName: `${resourcePrefix}-guardrail-version`
    });
    
  }
}