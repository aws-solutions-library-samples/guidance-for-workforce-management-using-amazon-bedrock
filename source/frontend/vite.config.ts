import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";
import path from 'path';


export default defineConfig(({ mode }) => { 

  const env = loadEnv(mode, process.cwd(), 'VITE_');

  console.log('Raw env loaded:', env);

  // Set defaults if not found in .env
  const envDefaults = {
    VITE_AWS_REGION: 'us-east-1',
    VITE_USER_POOL_ID: 'dummy-pool-id',
    VITE_USER_POOL_CLIENT_ID: 'dummy-client-id',
    VITE_RESTAPI_URL: 'http://localhost:8000',
    VITE_WEBSOCKET_URL: 'ws://localhost:8000',
  };

  // Merge defaults with loaded env
  const finalEnv = { ...envDefaults, ...env };

  // Validate required env variables
  [
    'VITE_AWS_REGION',
    'VITE_USER_POOL_ID',
    'VITE_USER_POOL_CLIENT_ID',
    'VITE_RESTAPI_URL',
    'VITE_WEBSOCKET_URL',
  ].forEach((key) => {
    if (!finalEnv[key]) {
      throw new Error(`Environment variable ${key} must be set. See README.md`);
    }
  });

  // Debug log to see what's being loaded
  console.log('Loaded environment variables:', {
    defaults: envDefaults,
    loaded: env,
    final: finalEnv
  });

  return {
    build: {
      rollupOptions: {
        onwarn(warning, warn) {
          // Suppress "Module level directives cause errors when bundled" warnings
          if (warning.code === "MODULE_LEVEL_DIRECTIVE") {
            return;
          }
          warn(warning);
        },
        output: {
          assetFileNames: (assetInfo) => {
            if (assetInfo.name?.endsWith('.worklet.js')) {
              return 'assets/[name].[hash].worklet.js';
            }
            return 'assets/[name].[hash][extname]';
          }
        }
      },
    },
    plugins: [react()],
    resolve: {
      alias: {
        './runtimeConfig': './runtimeConfig.browser',
        '@': '/src',
        'http': 'stream-http'
      },
    },
    define: {
      'import.meta.env': JSON.stringify(finalEnv),
      'import.meta.env.VITE_RESTAPI_URL': JSON.stringify(finalEnv.VITE_RESTAPI_URL),
      'import.meta.env.VITE_WEBSOCKET_URL': JSON.stringify(finalEnv.VITE_WEBSOCKET_URL),
      'import.meta.env.VITE_USER_POOL_ID': JSON.stringify(finalEnv.VITE_USER_POOL_ID),
      'import.meta.env.VITE_USER_POOL_CLIENT_ID': JSON.stringify(finalEnv.VITE_USER_POOL_CLIENT_ID),
      'import.meta.env.VITE_AWS_REGION': JSON.stringify(finalEnv.VITE_AWS_REGION),
      'import.meta.env.MODE': JSON.stringify(mode)
    },
    server: {
      port: process.env.PORT || 3001
    },
    preview: {
      port: process.env.PORT || 3001
    },
    optimizeDeps: {
      exclude: ['*.worklet.js']
    }
  };
});
