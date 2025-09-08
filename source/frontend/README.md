# Retail Store Assistant - Frontend App


## Configure

Add an `.env` file with the following fields
```
VITE_AWS_REGION=us-east-1
VITE_USER_POOL_ID=<details from your CDK deployment>
VITE_USER_POOL_CLIENT_ID=<details from your CDK deployment>
VITE_IDENTITY_POOL_ID=<details from your CDK deployment>
VITE_RESTAPI_URL=<details from your CDK deployment>
VITE_WEBSOCKET_URL=<details from your CDK deployment>
```

## Install

```bash
$ npm install
```

## Start locally

```bash
$ npm run dev
```


## Deploying an updated version of the frontend

```bash
$ npm run build
```

then load the dist/ folder contents into the website S3 bucket

```bash
$ aws s3 sync dist/ s3://<your-bucket-name>
```