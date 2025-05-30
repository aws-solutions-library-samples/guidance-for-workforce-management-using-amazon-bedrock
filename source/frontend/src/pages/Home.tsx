import React, { useState, useEffect } from 'react';
import { Box, Typography, Button, Paper, CircularProgress, Alert, Snackbar } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { fetchWithAuth } from '../config';
import { Link as RouterLink } from 'react-router-dom';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { useAuth } from '../context/AuthContext';
import logo from '../assets/logo.png';
import MobileHeader from '../components/MobileHeader';


const Home: React.FC = () => {
  const navigate = useNavigate();
  const { userId } = useAuth();

  const handleLogin = () => {
    navigate('/login');
  };

  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerateTasks = async (userId: string) => {

  const taskText = `Draft daily tasks for available staff but do not create tasks. 
  Skip any preamble, only include actual response. 
  Instead return the task list for review to the user in a markdown list, grouped by task_owner and focus on high-level tasks, following this structure:
  
  ### Task List:

  - Assigned to [task_owner]:
      - Task 1: [task_name]
      - Task 2: [task_name]
      - Task 3: [task_name]
  - Assigned to [task_owner]:
      - Task 1: [task_name]
      - Task 2: [task_name]
      - Task 3: [task_name]

Ensure that the task list is assigned to actual users.`;

      navigate('/assistant', { 
        replace: false, 
        state: { 
          query: taskText,
          isInitialQuery: true
        } 
      });
  }

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column',
      height: '100vh',
      width: '100%',
      position: 'relative',
      bgcolor: '#FFFFFF',
      padding: 0,
      margin: 0,
      overflow: 'hidden',
      mt: '56px' // Add top margin to account for fixed header
    }}>
      {/* Mobile Header */}
      <MobileHeader 
        title="AnyCompany Home" 
        showBackButton={false}
        showRefreshButton={false}
      />
      
      <Box sx={{ 
        flex: 1,
        overflowY: 'auto',
        p: 0,
        display: 'flex',
        flexDirection: 'column',
        width: '100%'
      }}>
        <Paper
          elevation={3}
          sx={{
            p: { xs: 2, sm: 4 },
            width: '100%',
            textAlign: 'center',
            borderRadius: { xs: 0, sm: 1 }
          }}
        >
          <Typography 
            variant="h3" 
            component="h1" 
            gutterBottom
            sx={{
              mt: {
                xs: 2,     // 16px on mobile
                sm: 1,     // 8px on tablet
                md: 0      // 0px on desktop
              },
              fontSize: {  // Optional: you might also want to adjust font size
                xs: '2rem',
                sm: '2.5rem',
                md: '3rem'
              }
            }}
          >
            Welcome to AnyCompany Assist
          </Typography>

          
          <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
            <img src={logo} alt="AnyCompany Logo" style={{ width: '100px' }} />
          </Box>
          <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
            Your AI-powered assistant for efficient task management
          </Typography>

          {userId ? (
            <Box sx={{ mt: 3 }}>
              <Typography variant="body1" sx={{ mb: 2 }}>
                Logged in as:
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                {userId}
              </Typography>
              <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>

                <Button 
                  variant="contained"
                  color="primary"
                  onClick={() => handleGenerateTasks(userId)}
                  disabled={isGenerating}
                  startIcon={isGenerating ? <CircularProgress size={20} /> : <AutoAwesomeIcon />}
                >
                  {isGenerating ? 'Generating Tasks...' : 'Generate Daily Tasks'}
                </Button>
                <Button 
                  variant="outlined"
                  disabled={isGenerating} 
                  onClick={() => navigate('/todolist')}
                >
                  Go to Tasks
                </Button>
                
              </Box>
            </Box>
          ) : (
            <Box sx={{ mt: 3 }}>
              <Typography variant="body1" sx={{ mb: 3 }}>
                Please log in to access the application
              </Typography>
              <Button 
                variant="contained" 
                color="primary"
                onClick={handleLogin}
                sx={{
                  px: 4,
                  py: 1.5,
                }}
              >
                Login
              </Button>
            </Box>
          )}

          
        </Paper>
        <Snackbar 
          open={!!error} 
          autoHideDuration={6000} 
          onClose={() => setError(null)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert onClose={() => setError(null)} severity="error" sx={{ width: '100%' }}>
            {error}
          </Alert>
        </Snackbar>
      </Box>
    </Box>
  );
};

export default Home;