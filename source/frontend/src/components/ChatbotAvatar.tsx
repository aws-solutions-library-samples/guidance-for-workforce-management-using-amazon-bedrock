import React from 'react';
import { Box, Typography } from '@mui/material';

interface ChatbotAvatarProps {
  message?: string;
}

const ChatbotAvatar: React.FC<ChatbotAvatarProps> = ({ message = "Ask me anything!" }) => {
  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      my: 4
    }}>
      {/* Robot Avatar */}
      <Box sx={{ 
        position: 'relative',
        width: 120,
        height: 120,
        mb: 2
      }}>
        {/* Robot Head */}
        <Box sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: 80,
          height: 60,
          backgroundColor: '#f0f0f0',
          borderRadius: '10px',
          border: '2px solid #ccc',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 2
        }}>
          {/* Robot Eyes */}
          <Box sx={{
            display: 'flex',
            justifyContent: 'space-around',
            width: '60%',
            mb: 1
          }}>
            <Box sx={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              backgroundColor: '#a8e6ff',
              border: '1px solid #88d8ff'
            }} />
            <Box sx={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              backgroundColor: '#a8e6ff',
              border: '1px solid #88d8ff'
            }} />
          </Box>
          
          {/* Robot Mouth */}
          <Box sx={{
            width: '60%',
            height: 10,
            borderRadius: '10px',
            border: '1px solid #ccc',
            backgroundColor: 'transparent',
            position: 'relative',
            overflow: 'hidden'
          }}>
            <Box sx={{
              position: 'absolute',
              bottom: 0,
              left: '10%',
              width: '80%',
              height: '50%',
              borderTopLeftRadius: '10px',
              borderTopRightRadius: '10px',
              backgroundColor: '#ccc'
            }} />
          </Box>
        </Box>
        
        {/* Robot Antenna */}
        <Box sx={{
          position: 'absolute',
          top: 10,
          left: '50%',
          transform: 'translateX(-50%)',
          width: 2,
          height: 20,
          backgroundColor: '#ccc',
          zIndex: 1
        }}>
          <Box sx={{
            position: 'absolute',
            top: 0,
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: 6,
            height: 6,
            borderRadius: '50%',
            backgroundColor: '#ccc'
          }} />
        </Box>
        
        {/* Robot Ears/Headphones */}
        <Box sx={{
          position: 'absolute',
          top: '50%',
          left: 20,
          transform: 'translateY(-50%)',
          width: 15,
          height: 30,
          borderRadius: '50%',
          backgroundColor: '#ccc',
          zIndex: 1
        }} />
        <Box sx={{
          position: 'absolute',
          top: '50%',
          right: 20,
          transform: 'translateY(-50%)',
          width: 15,
          height: 30,
          borderRadius: '50%',
          backgroundColor: '#ccc',
          zIndex: 1
        }} />
        
        {/* Robot Body */}
        <Box sx={{
          position: 'absolute',
          bottom: 0,
          left: '50%',
          transform: 'translateX(-50%)',
          width: 40,
          height: 20,
          backgroundColor: '#ccc',
          borderBottomLeftRadius: '5px',
          borderBottomRightRadius: '5px',
          zIndex: 1
        }} />
      </Box>
      
      {/* Message */}
      <Typography variant="h6" align="center">
        {message}
      </Typography>
    </Box>
  );
};

export default ChatbotAvatar; 