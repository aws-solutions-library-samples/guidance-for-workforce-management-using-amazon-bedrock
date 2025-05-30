import { Auth } from 'aws-amplify';
import axios from 'axios';


interface AppConfig {
  auth: {
    region: string;
    userPoolId: string;
    userPoolClientId: string;
    identityPoolId: string;
    scopes: string[];
  };
  api: {
    baseUrl: string;
  };
  websocket: {
    baseUrl: string;
  }
}

// Load environment variables with defaults from vite.config.ts
const config: AppConfig = {
  auth: {
    region: import.meta.env.VITE_AWS_REGION || 'us-east-1',
    userPoolId: import.meta.env.VITE_USER_POOL_ID || '',
    userPoolClientId: import.meta.env.VITE_USER_POOL_CLIENT_ID || '',
    identityPoolId: import.meta.env.VITE_IDENTITY_POOL_ID || '',
    scopes: ['openid', 'profile', 'email']
  },
  api: {
    baseUrl: import.meta.env.VITE_RESTAPI_URL || 'http://localhost:8000'
  },
  websocket: {
    baseUrl: import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:8000'
  }
};

// Validate required config values
const requiredVars = [
  ['auth.userPoolId', config.auth.userPoolId],
  ['auth.userPoolClientId', config.auth.userPoolClientId],
  ['auth.identityPoolId', config.auth.identityPoolId],
  ['api.baseUrl', config.api.baseUrl],
  ['websocket.baseUrl', config.websocket.baseUrl]
];

for (const [name, value] of requiredVars) {
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
}

// Create axios instance with default config
const axiosInstance = axios.create({
  baseURL: config.api.baseUrl,
  timeout: 600000  // 10 minutes in milliseconds
  // Removed default Content-Type header to allow proper FormData handling
});

// Add request interceptor for auth
axiosInstance.interceptors.request.use(async (config) => {
  try {
    const session = await Auth.currentSession();
    const token = session.getIdToken().getJwtToken();
    config.headers.Authorization = `Bearer ${token}`;
  } catch (error) {
    console.error('Error setting auth header:', error);
  }
  return config;
});


export const fetchWithAuth = async (path = '', options: any = {}) => {
  const method = options.method?.toLowerCase() || 'get';
  
  // Handle request data/body
  let requestData;
  let headers = { ...options.headers }; // Don't duplicate auth headers - let interceptor handle it
  
  if (options.data instanceof FormData) {
    requestData = options.data;
    // For FormData, remove any Content-Type header to let axios set it properly
    delete headers['Content-Type'];
    console.log('🔄 Detected FormData - letting axios handle Content-Type automatically');
  } else if (options.data) {
    requestData = options.data;
    // For non-FormData, ensure JSON content type if not specified
    if (!headers['Content-Type']) {
      headers['Content-Type'] = 'application/json';
    }
  } else if (options.body) {
    try {
      requestData = JSON.parse(options.body);
    } catch (e) {
      requestData = options.body;
    }
    // For body data, ensure JSON content type if not specified
    if (!headers['Content-Type']) {
      headers['Content-Type'] = 'application/json';
    }
  }

  // Exclude data, method, and headers from options to avoid conflicts
  const { data, method: _, headers: __, ...restOptions } = options;
  
  const config = {
    url: path,
    method,
    headers,
    ...restOptions,  // Now safely spread without data, method, or headers
    data: requestData
  };

  try {
    console.log(`🌐 Initiating ${method.toUpperCase()} request to: ${path}`);
    console.log('📋 Request config:', {
      url: path,
      method: method.toUpperCase(),
      headers: config.headers,
      dataType: options.data instanceof FormData ? 'FormData' : typeof requestData
    });
    const response = await axiosInstance(config);
    console.log('✅ Request completed successfully', {
      status: response.status,
      url: path,
      method: method.toUpperCase(),
      data: response
    });
    return response;
  } catch (error) {
    console.error('🚨 Request failed:', {
      error: error instanceof Error ? {
        name: error.name,
        message: error.message,
        stack: error.stack
      } : error,
      url: path,
      method: method.toUpperCase()
    });
    throw error;
  }
};

export default config;

// Helper functions for common config access

// Update these constants to use import.meta.env
const API_URL = import.meta.env.VITE_APP_RESTAPI_URL || 'http://localhost:8000';
export const WEBSOCKET_URL = import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:8000';

export const getAuthConfig = () => {
  return {
    Auth: {
      region: import.meta.env.VITE_AWS_REGION,
      userPoolId: import.meta.env.VITE_USER_POOL_ID,
      userPoolWebClientId: import.meta.env.VITE_USER_POOL_CLIENT_ID,
      identityPoolId: import.meta.env.VITE_IDENTITY_POOL_ID,
      oauth: {
        domain: `${import.meta.env.VITE_USER_POOL_DOMAIN}.auth.${import.meta.env.VITE_AWS_REGION}.amazoncognito.com`,
        scope: ['email', 'openid', 'profile'],
        redirectSignIn: import.meta.env.VITE_REDIRECT_SIGN_IN,
        redirectSignOut: import.meta.env.VITE_REDIRECT_SIGN_OUT,
        responseType: 'code'
      }
    }
  };
};