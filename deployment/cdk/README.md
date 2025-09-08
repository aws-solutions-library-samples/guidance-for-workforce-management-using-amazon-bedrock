# Retail App Backend

## Deployment

### Install dependencies
```
cd cdk
npm install
```

### Review environment variables in .env
Ensure EMAIL, COGNITO_PASSWORD, PARENT_DOMAIN_NAME, DOMAIN_NAME, CERTIFICATE_ARN, WEB_CERTIFICATE_ARN is set with respective values for your environment, keep all other variables as placeholders as-is. 

### Run the deployment script
```
./deploy.sh
```