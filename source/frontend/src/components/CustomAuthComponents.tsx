import React from 'react';
import { 
  Box, 
  Typography, 
  TextField, 
  Button, 
  InputAdornment, 
  IconButton 
} from '@mui/material';
import { Visibility, VisibilityOff } from '@mui/icons-material';

// Custom Sign In component for Amplify Authenticator
export const CustomSignIn = ({ validationErrors, handleSubmit, handleChange }: any) => {
  const [showPassword, setShowPassword] = React.useState(false);

  const handleClickShowPassword = () => {
    setShowPassword(!showPassword);
  };

  return (
    <Box 
      component="form" 
      onSubmit={handleSubmit} 
      sx={{ 
        width: '100%',
        maxWidth: '400px',
        margin: '0 auto',
        padding: { xs: '16px', sm: '24px' },
        backgroundColor: '#FFFFFF',
        borderRadius: '8px',
        boxShadow: { xs: 'none', sm: '0 2px 10px rgba(0,0,0,0.1)' }
      }}
    >
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <Typography 
          variant="h4" 
          component="h1"
          sx={{
            color: 'text.primary',
            mb: 2,
            fontWeight: 500
          }}
        >
          Welcome
        </Typography>
        <Typography 
          variant="body1" 
          color="text.secondary"
        >
          Sign in to continue to AnyCompany Assist
        </Typography>
      </Box>

      <Box sx={{ mb: 3 }}>
        <Typography 
          variant="subtitle2" 
          component="label" 
          htmlFor="username"
          sx={{ 
            display: 'block', 
            mb: 1, 
            color: 'text.secondary',
            fontSize: '14px'
          }}
        >
          Username
        </Typography>
        <TextField
          id="username"
          name="username"
          fullWidth
          variant="outlined"
          size="small"
          onChange={handleChange}
          error={!!validationErrors.username}
          helperText={validationErrors.username}
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: '4px',
              '& fieldset': {
                borderColor: '#E0E0E0',
              },
              '&:hover fieldset': {
                borderColor: '#BDBDBD',
              },
              '&.Mui-focused fieldset': {
                borderColor: '#4285F4',
              },
            },
          }}
        />
      </Box>

      <Box sx={{ mb: 4 }}>
        <Typography 
          variant="subtitle2" 
          component="label" 
          htmlFor="password"
          sx={{ 
            display: 'block', 
            mb: 1, 
            color: 'text.secondary',
            fontSize: '14px'
          }}
        >
          Password
        </Typography>
        <TextField
          id="password"
          name="password"
          type={showPassword ? 'text' : 'password'}
          fullWidth
          variant="outlined"
          size="small"
          onChange={handleChange}
          error={!!validationErrors.password}
          helperText={validationErrors.password}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton
                  aria-label="toggle password visibility"
                  onClick={handleClickShowPassword}
                  edge="end"
                  size="small"
                >
                  {showPassword ? <VisibilityOff /> : <Visibility />}
                </IconButton>
              </InputAdornment>
            ),
          }}
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: '4px',
              '& fieldset': {
                borderColor: '#E0E0E0',
              },
              '&:hover fieldset': {
                borderColor: '#BDBDBD',
              },
              '&.Mui-focused fieldset': {
                borderColor: '#4285F4',
              },
            },
          }}
        />
      </Box>

      <Button
        type="submit"
        fullWidth
        variant="contained"
        sx={{
          backgroundColor: '#4285F4',
          color: 'white',
          textTransform: 'none',
          fontWeight: 'normal',
          padding: '10px 0',
          borderRadius: '4px',
          fontSize: '16px',
          '&:hover': {
            backgroundColor: '#3367D6',
          },
        }}
      >
        Sign in
      </Button>

      <Box sx={{ mt: 2, textAlign: 'center' }}>
        <Typography 
          variant="body2" 
          component="a" 
          href="#" 
          sx={{ 
            color: '#4285F4', 
            textDecoration: 'none',
            fontSize: '14px',
            '&:hover': {
              textDecoration: 'underline',
            }
          }}
        >
          Forgot your password?
        </Typography>
      </Box>
    </Box>
  );
};

// Custom Header component (empty to override default)
export const CustomHeader = () => null;

// Custom Footer component (empty to override default)
export const CustomFooter = () => null; 