import { Amplify } from 'aws-amplify';
import { Authenticator } from '@aws-amplify/ui-react';
import '@aws-amplify/ui-react/styles.css';
import './styles/authenticator.css';
import './styles/amplify-override.css';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { ChatProvider, AuthProvider } from './context';
import { lightTheme } from './theme';
import { getAuthConfig } from './config';
import AppRoutes from './components/Routes';
import { Box, Container, Typography } from '@mui/material';
import BottomNavigation from './components/BottomNavigation';
import { useState, useEffect } from 'react';

// Configure Amplify
Amplify.configure(getAuthConfig());

// Custom styles to override dark theme
const authStyles = {
  container: {
    backgroundColor: '#FFFFFF',
  },
  form: {
    backgroundColor: '#FFFFFF',
  },
  input: {
    backgroundColor: '#FFFFFF',
    color: '#000000',
  },
  label: {
    backgroundColor: '#FFFFFF',
    color: '#000000',
  },
  button: {
    backgroundColor: '#4285F4',
    color: '#FFFFFF',
  },
};

function App() {
  // Add effect to inject styles to override dark theme
  useEffect(() => {
    // Create a style element
    const style = document.createElement('style');
    
    // Add CSS rules to override dark backgrounds
    style.textContent = `
      .amplify-authenticator * {
        background-color: #FFFFFF !important;
      }
      .amplify-button[data-variation='primary'] {
        background-color: #4285F4 !important;
      }
      .amplify-button[data-variation='primary'] * {
        background-color: transparent !important;
        color: #FFFFFF !important;
      }
      .amplify-field-group__control {
        background-color: #FFFFFF !important;
      }
      .amplify-input {
        background-color: #FFFFFF !important;
        color: #000000 !important;
      }
      .amplify-field__label {
        background-color: #FFFFFF !important;
        color: #000000 !important;
      }
    `;
    
    // Append the style element to the head
    document.head.appendChild(style);
    
    // Clean up function
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  return (
    <ThemeProvider theme={lightTheme}>
      <CssBaseline />
      <Authenticator
        hideSignUp={true}
        variation="modal"
        loginMechanisms={['username']}
        components={{
          SignUp: { Header: () => null, Footer: () => null },
          Footer: () => null,
          Header() {
            return (
              <Box sx={{ 
                textAlign: 'center', 
                mb: 4,
                mt: 2,
                bgcolor: '#FFFFFF'
              }}>
                <Typography 
                  variant="h4" 
                  component="h1"
                  sx={{
                    color: '#000000',
                    mb: 2,
                    fontWeight: 500,
                    fontSize: { xs: '1.75rem', sm: '2rem' },
                    bgcolor: '#FFFFFF'
                  }}
                >
                  Welcome
                </Typography>
                <Typography 
                  variant="body1" 
                  sx={{
                    fontSize: '1rem',
                    color: '#666666',
                    bgcolor: '#FFFFFF'
                  }}
                >
                  Sign in to continue to AnyCompany Assist
                </Typography>
              </Box>
            );
          },
          SignIn: {
            Footer() {
              return (
                <Box sx={{ 
                  textAlign: 'center', 
                  mt: 2,
                  bgcolor: '#FFFFFF'
                }}>
                  <Typography 
                    variant="body2" 
                    component="a" 
                    href="#" 
                    sx={{ 
                      color: '#4285F4', 
                      textDecoration: 'none',
                      fontSize: '14px',
                      bgcolor: '#FFFFFF',
                      '&:hover': {
                        textDecoration: 'underline',
                      }
                    }}
                  >
                    Forgot your password?
                  </Typography>
                </Box>
              );
            }
          }
        }}
        formFields={{
          signIn: {
            username: {
              placeholder: 'Enter your username',
              isRequired: true,
            },
            password: {
              placeholder: 'Enter your password',
              isRequired: true,
            },
          }
        }}
      >
        {({ signOut, user }) => {
          return (
            <AuthProvider user={user} signOut={async () => {
              if (signOut) {
                signOut();
              }
              return Promise.resolve();
            }}>
              <ChatProvider>
                {user ? (
                  <Box sx={{ 
                    display: 'flex', 
                    flexDirection: 'column', 
                    minHeight: '100vh',
                    height: '100vh',
                    overflow: 'hidden',
                    bgcolor: 'background.default',
                    color: 'text.primary'
                  }}>
                    <Container 
                      disableGutters 
                      maxWidth={false} 
                      sx={{ 
                        display: 'flex',
                        flexDirection: 'column',
                        height: '100vh',
                        overflow: 'hidden',
                        pb: '56px' // Height of bottom navigation
                      }}
                    >
                      <Box component="main" sx={{ 
                        flex: 1,
                        display: 'flex', 
                        flexDirection: 'column',
                        overflow: 'auto',
                        WebkitOverflowScrolling: 'touch'
                      }}>
                        <AppRoutes />
                      </Box>
                    </Container>
                    <BottomNavigation />
                  </Box>
                ) : (
                  <AppRoutes />
                )}
              </ChatProvider>
            </AuthProvider>
          );
        }}
      </Authenticator>
    </ThemeProvider>
  );
}

export default App;