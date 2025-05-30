import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { Link as MuiLink, Stack } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const Nav = () => {
  const { user, signOut } = useAuth();
  const navigate = useNavigate();

  const handleLogin = () => navigate('/login');
  const handleLogout = async () => {
    try {
      await signOut();
      navigate('/');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };
  
  return (
    <nav>
      <Stack direction="row" spacing={2}>
        <MuiLink
          component={RouterLink}
          to="/"
          id="home-nav-link"
          underline="none"
          sx={{ '&:hover': { color: '#4285F4' } }}
        >
          HOME
        </MuiLink>
        <MuiLink
          component={RouterLink}
          to="/analytics"
          id="protected-nav-link"
          underline="none"
          sx={{ '&:hover': { color: '#4285F4' } }}
        >
          ANALYTICS
        </MuiLink>
        <MuiLink
          component={RouterLink}
          to="/todolist"
          id="protected-nav-link"
          underline="none"
          sx={{ '&:hover': { color: '#4285F4' } }}
        >
          TASKS
        </MuiLink>
        <MuiLink
          component={RouterLink}
          to="/chat"
          id="protected-nav-link"
          underline="none"
          sx={{ '&:hover': { color: '#4285F4' } }}
        >
          CHAT
        </MuiLink>
        {
          !user ? (
            <MuiLink
              onClick={handleLogin}
              underline="none"
              sx={{ '&:hover': { color: '#4285F4' } }}
            >
              LOGIN
            </MuiLink>
          ) : (          
            <MuiLink
              onClick={handleLogout}
              underline="none"
              sx={{ '&:hover': { color: '#4285F4' } }}
            >
              LOGOUT
            </MuiLink>
          )
        }
      </Stack>
    </nav>
  );
};

export default Nav;