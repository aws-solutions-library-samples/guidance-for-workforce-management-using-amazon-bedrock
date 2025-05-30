import React from 'react';
import { Box, Typography, IconButton, Paper } from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useNavigate } from 'react-router-dom';

interface MobileHeaderProps {
  title: string;
  showBackButton?: boolean;
  showRefreshButton?: boolean;
  onRefresh?: () => void;
}

const MobileHeader: React.FC<MobileHeaderProps> = ({ 
  title, 
  showBackButton = true,
  showRefreshButton = true,
  onRefresh
}) => {
  const navigate = useNavigate();

  const handleBack = () => {
    navigate(-1);
  };

  const handleRefresh = () => {
    if (onRefresh) {
      onRefresh();
    }
  };

  return (
    <Paper 
      sx={{ 
        position: 'fixed', 
        top: 0, 
        left: 0, 
        right: 0,
        zIndex: 1100,
        borderBottom: 1,
        borderColor: 'divider',
        width: '100vw',
        margin: 0,
        padding: 0,
        borderRadius: 0
      }} 
      elevation={3}
    >
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center',
        justifyContent: 'space-between',
        width: '100%',
        height: '56px',
        px: 2
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          {showBackButton && (
            <IconButton 
              edge="start" 
              onClick={handleBack}
              sx={{ mr: 1 }}
            >
              <ArrowBackIcon />
            </IconButton>
          )}
          <Typography variant="h6" component="h1" sx={{ fontWeight: 500 }}>
            {title}
          </Typography>
        </Box>
        
        {showRefreshButton && (
          <IconButton edge="end" onClick={handleRefresh}>
            <RefreshIcon />
          </IconButton>
        )}
      </Box>
    </Paper>
  );
};

export default MobileHeader; 