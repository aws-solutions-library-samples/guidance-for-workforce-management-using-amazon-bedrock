import React from 'react';
import {
  Box,
  Typography,
} from '@mui/material';

function Footer() {
  return (
    <Box 
      component="footer" 
      sx={{
        py: 2,
        borderTop: 1,
        borderColor: 'divider',
        position: 'fixed',
        bottom: 0,
        left: 0,
        width: '100%',
        backgroundColor: 'background.paper',
        zIndex: 1100,
        height: '56px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}
    >
      <Typography variant="body2" color="text.secondary">
        Version 1.0 - Copyright 2025
      </Typography>
    </Box>
  );
}

export default Footer;
