import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  Box, 
  BottomNavigation as MuiBottomNavigation, 
  BottomNavigationAction,
  Paper
} from '@mui/material';
import HomeIcon from '@mui/icons-material/Home';
import AssignmentIcon from '@mui/icons-material/Assignment';
import BarChartIcon from '@mui/icons-material/BarChart';
import SupportAgentIcon from '@mui/icons-material/SupportAgent';
import LoginIcon from '@mui/icons-material/Login';
import LogoutIcon from '@mui/icons-material/Logout';
import { useAuth } from '../context/AuthContext';

const BottomNavigation: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated, signOut } = useAuth();
  
  // Determine which tab is active based on the current path
  const getActiveTab = () => {
    const path = location.pathname;
    if (path === '/') return 0;
    if (path === '/todolist') return 1;
    if (path === '/analytics') return 2;
    if (path === '/assistant' || path === '/chat') return 3;
    return 0; // Default to home
  };

  const handleLoginLogout = () => {
    if (isAuthenticated) {
      signOut();
    } else {
      navigate('/');
    }
  };

  return (
    <Paper 
      sx={{ 
        position: 'fixed', 
        bottom: 0, 
        left: 0, 
        right: 0,
        zIndex: 1100,
        borderTop: 1,
        borderColor: 'divider',
      }} 
      elevation={3}
    >
      <MuiBottomNavigation
        showLabels
        value={getActiveTab()}
        onChange={(event, newValue) => {
          if (newValue === 4) {
            // Handle login/logout separately
            handleLoginLogout();
            return;
          }
          
          switch(newValue) {
            case 0:
              navigate('/');
              break;
            case 1:
              navigate('/todolist');
              break;
            case 2:
              navigate('/analytics');
              break;
            case 3:
              navigate('/assistant');
              break;
            default:
              navigate('/');
          }
        }}
      >
        <BottomNavigationAction label="Home" icon={<HomeIcon />} />
        <BottomNavigationAction label="Tasks" icon={<AssignmentIcon />} />
        <BottomNavigationAction label="Analytics" icon={<BarChartIcon />} />
        <BottomNavigationAction label="Assistant" icon={<SupportAgentIcon />} />
        <BottomNavigationAction 
          label={isAuthenticated ? "Logout" : "Login"} 
          icon={isAuthenticated ? <LogoutIcon /> : <LoginIcon />} 
        />
      </MuiBottomNavigation>
    </Paper>
  );
};

export default BottomNavigation; 