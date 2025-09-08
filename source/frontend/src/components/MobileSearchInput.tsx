import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Box, InputBase, IconButton, CircularProgress, Typography, Tooltip } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import MicIcon from '@mui/icons-material/Mic';
import MicOffIcon from '@mui/icons-material/MicOff';
import SendIcon from '@mui/icons-material/Send';
import CameraAltIcon from '@mui/icons-material/CameraAlt';
import StopIcon from '@mui/icons-material/Stop';
import { styled, keyframes } from '@mui/material/styles';
import { useAudio, useMessages, useWebSocketConnection, useS2S, useAuth } from '../context';
import { fetchWithAuth } from '../config';

interface MobileSearchInputProps {
  onSend: (message: string) => void;
  placeholder?: string;
  disabled?: boolean;
}

// Define a pulsing animation
const pulse = keyframes`
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.1);
    opacity: 0.9;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
`;

const wavePulse = keyframes`
  0% {
    transform: scaleY(0.2);
  }
  50% {
    transform: scaleY(1);
  }
  100% {
    transform: scaleY(0.2);
  }
`;

const SearchContainer = styled('div')(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  backgroundColor: '#F0F3F7',
  borderRadius: 30,
  padding: '5px 15px',
  width: '100%',
  boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.1)',
}));

const StyledInput = styled(InputBase)(({ theme }) => ({
  marginLeft: theme.spacing(1),
  flex: 1,
  '& input': {
    color: 'black',
  },
  '& input::placeholder': {
    color: '#5F6368',
    opacity: 1,
  },
}));

const IconButtonStyled = styled(IconButton)(({ theme }) => ({
  padding: 10,
  color: '#5F6368',
}));

const SearchIconStyled = styled(IconButton)(({ theme }) => ({
  padding: 8,
  color: '#5F6368',
  '&:hover': {
    backgroundColor: 'transparent',
  },
}));

// Audio visualization bars
const VisualizerContainer = styled('div')({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  height: 24,
  gap: 2,
  marginRight: 8
});

const AudioBar = styled('div')(({ theme }) => ({
  width: 2,
  height: 10,
  backgroundColor: theme.palette.primary.main,
  borderRadius: 1,
  animation: `${wavePulse} 1.2s ease-in-out infinite`,
}));

// Active mic animation
const ActiveMicIcon = styled(MicIcon)({
  color: 'red',
  animation: `${pulse} 1.5s infinite ease-in-out`,
});

// Mic status text
const MicStatusText = styled(Typography)({
  fontSize: '0.75rem',
  color: '#5F6368',
  marginLeft: 8,
});

const MobileSearchInput: React.FC<MobileSearchInputProps> = ({ 
  onSend,
  placeholder = "Search or ask a question", 
  disabled = false 
}) => {
  // Local state
  const [inputValue, setInputValue] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);
  const [isAudioEnabled, setIsAudioEnabled] = useState(true);
  const [showVisualizer, setShowVisualizer] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  
  // Refs
  const inputRef = useRef<HTMLInputElement>(null);
  const micTimeoutRef = useRef<number | null>(null);
  const recordingTimerRef = useRef<number | null>(null);
  const audioVisualizerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Get context hooks
  const { userId,sessionId } = useAuth();
  const { isStreaming, startStreaming, stopStreaming, onAudioData, isMuted, toggleMute } = useAudio();
  const { isLoading, addSystemMessage, addMessage } = useMessages();
  const { isConnected, connect, connectionError } = useWebSocketConnection();
  
  // Start a recording timer to show duration
  const startRecordingTimer = useCallback(() => {
    setRecordingDuration(0);
    
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
    }
    
    recordingTimerRef.current = setInterval(() => {
      setRecordingDuration(prev => prev + 1);
    }, 1000);
  }, []);
  
  // Stop the recording timer
  const stopRecordingTimer = useCallback(() => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
  }, []);
  
  // Format seconds into mm:ss
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Text input handlers
  const handleSend = useCallback(() => {
    if (inputValue.trim() && !disabled) {
      onSend(inputValue.trim());
      setInputValue('');
      setIsExpanded(false);
    }
  }, [inputValue, onSend, disabled]);

  const handleKeyPress = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const handleFocus = useCallback(() => {
    setIsExpanded(true);
  }, []);

  // Audio handling
  const startAudioRecording = useCallback(async () => {
    try {
      console.log("[MobileSearchInput] Starting audio recording");
      
      // Check if microphone is already in use
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter(device => device.kind === 'audioinput');
        console.log("[MobileSearchInput] Available audio inputs:", audioInputs.length);
      } catch (error) {
        console.warn("[MobileSearchInput] Could not enumerate devices:", error);
      }
      
      // Establish WebSocket connection first if not connected
      if (!isConnected) {
        console.log("[MobileSearchInput] WebSocket not connected, establishing connection...");
        try {
          await connect();
          console.log("[MobileSearchInput] WebSocket connection established successfully");
        } catch (error) {
          console.error("[MobileSearchInput] Failed to establish WebSocket connection:", error);
          addSystemMessage("Failed to connect to voice service. Please try again.");
          return;
        }
      }
      
      // Add a small delay to ensure previous cleanup completed
      await new Promise(resolve => setTimeout(resolve, 300));
      
      // Start the streaming process
      await startStreaming();
      
      // Show visual feedback
      setShowVisualizer(true);
      startRecordingTimer();
      
      
    } catch (error) {
      console.error("[MobileSearchInput] Error starting audio recording:", error);
      
      // Check if it's a permission error
      if (error instanceof Error) {
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
          addSystemMessage("Microphone access denied. Please allow microphone access and try again.");
          setIsAudioEnabled(false);
        } else if (error.name === 'NotFoundError') {
          addSystemMessage("No microphone found. Please check your audio devices.");
          setIsAudioEnabled(false);
        } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
          addSystemMessage("Microphone is busy or not available. Please close other applications using the microphone and try again.");
          setIsAudioEnabled(false);
        } else {
          addSystemMessage("Error accessing microphone. Please try again.");
        }
      } else {
        addSystemMessage("Error accessing microphone. Please try again.");
      }
      
      setShowVisualizer(false);
      stopRecordingTimer();
    }
  }, [isConnected, connect, startStreaming, addSystemMessage, startRecordingTimer, stopRecordingTimer]);
  
  const stopAudioRecording = useCallback(async () => {
    try {
      console.log("[MobileSearchInput] Stopping audio recording...");
      
      // First update local UI state
      setShowVisualizer(false);
      stopRecordingTimer();
      
      
      // Then try to stop the streaming - but our local UI should already be updated
      await stopStreaming();
      console.log("[MobileSearchInput] Audio streaming stopped successfully");
    } catch (error) {
      console.error("[MobileSearchInput] Error stopping audio recording:", error);
      addSystemMessage("Error processing audio. Please try again.");
      
      // Even on error, make sure UI is updated and audio is stopped
      setShowVisualizer(false);
      stopRecordingTimer();

    }
  }, [stopStreaming, addSystemMessage, stopRecordingTimer]);
  
  // Handle mic button press
  const handleMicPress = useCallback(async () => {
    if (disabled) return;
    
    // Clear any existing mic timeout
    if (micTimeoutRef.current) {
      clearTimeout(micTimeoutRef.current);
      micTimeoutRef.current = null;
    }
    
    try {
      console.log("[MobileSearchInput] Mic button pressed, current streaming state:", isStreaming);
      
      if (!isStreaming) {
        // Check microphone permissions
        if (!isAudioEnabled) {
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(track => track.stop()); // Clean up test stream
            setIsAudioEnabled(true);
          } catch (error) {
            console.error("[MobileSearchInput] Microphone permission denied:", error);
            addSystemMessage("Microphone access is required for voice input. Please enable microphone access.");
            setIsAudioEnabled(false);
            return;
          }
        }
        
        // Start recording
        await startAudioRecording();
        console.log("[MobileSearchInput] Recording started successfully");
        
        // Auto-stop after 30 seconds
        micTimeoutRef.current = setTimeout(async () => {
          if (isStreaming) {
            console.log("[MobileSearchInput] Auto-stopping recording after 30 seconds");
            await stopAudioRecording();
            addSystemMessage("Voice input stopped after 90 seconds");
          }
        }, 90000);
      } else {
        // Stop recording
        console.log("[MobileSearchInput] Stopping recording via button press");
        await stopAudioRecording();
        setShowVisualizer(false); // Ensure UI is updated
      }
    } catch (error) {
      console.error("[MobileSearchInput] Error handling mic press:", error);
      // Force UI reset on error
      setShowVisualizer(false);
      stopRecordingTimer();
      addSystemMessage("Error with microphone. Please try again.");
    }
  }, [isStreaming, isAudioEnabled, disabled, startAudioRecording, stopAudioRecording, addSystemMessage, stopRecordingTimer]);

  // Search button handler
  const handleSearchClick = useCallback(() => {
    if (inputValue.trim()) {
      handleSend();
    }
  }, [inputValue, handleSend]);

  const handleImageUpload = useCallback(async () => {
    try {
      // Trigger file input click
      fileInputRef.current?.click();
    } catch (error) {
      console.error('Error handling image upload:', error);
      addSystemMessage('Failed to upload image. Please try again.');
    }
  }, [addSystemMessage]);

  const handleFileChange = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {      
      // Create FormData
      const formData = new FormData();
      formData.append('file', file);
      formData.append('userId', userId || '');
      formData.append('sessionId', sessionId || '');

      console.log("[MobileSearchInput] Uploading image...");
      // Log FormData contents properly
      console.log("[MobileSearchInput] formData contents:", {
        file: file.name,
        fileSize: file.size,
        fileType: file.type,
        userId: userId || '',
        sessionId: sessionId || ''
      });
      
      const response = await fetchWithAuth('/uploadimage', {
        method: 'POST',
        data: formData
      });

      const data = response.data;

      // Check if the response has the expected structure
      if (!data || !data.thumbnail) {
        throw new Error('Invalid response format from server');
      }

      // Create data URL for the thumbnail
      const thumbnail = `data:image/jpeg;base64,${data.thumbnail}`;

      // Add system message with image thumbnail
      addMessage(`![Uploaded image](${thumbnail})

*Image uploaded successfully*`, 'user');

      // Clear the file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error) {
      console.error('Error uploading image:', error);
      addSystemMessage('Failed to upload image. Please try again.');
    }
  }, [userId, sessionId, addMessage, addSystemMessage]);

  // Simple mute toggle without complex state verification
  const handleMuteToggle = useCallback(() => {
    // Call toggleMute from context - the context now handles state synchronization properly
    toggleMute();
    
    // Update the button's data attribute for immediate UI feedback
    const muteButton = document.querySelector('[data-muted]');
    if (muteButton) {
      // We can safely use the negation of the current attribute value
      // since this will be updated before the next render
      const currentValue = muteButton.getAttribute('data-muted') === 'true';
      muteButton.setAttribute('data-muted', (!currentValue).toString());
    }
  }, [toggleMute]);

  // Audio data subscription
  useEffect(() => {
    const unsubscribe = onAudioData((audioData) => {
      // If muted, don't process the audio data
      if (isMuted) {
        // Return empty audio data when muted
        return;
      }
      
      // Animate visualizer based on audio levels
      if (showVisualizer && audioVisualizerRef.current) {
        // Update visualizer based on audio levels (you could implement more sophisticated visualization)
        const barElements = audioVisualizerRef.current.querySelectorAll('div');
        barElements.forEach((bar, i) => {
          const height = Math.abs(audioData[i * 50] || 0) / 32767 * 20;
          bar.style.height = `${Math.max(2, height)}px`;
        });
      }
    });
    
    return unsubscribe;
  }, [onAudioData, showVisualizer, isMuted]);

  // Keep UI in sync with streaming state
  useEffect(() => {
    // When streaming stops, ensure visualizer is also stopped
    if (!isStreaming && showVisualizer) {
      console.log("[MobileSearchInput] Synchronizing UI with streaming state (stopped)");
      setShowVisualizer(false);
      stopRecordingTimer();
    }
  }, [isStreaming, showVisualizer, stopRecordingTimer]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (micTimeoutRef.current) {
        clearTimeout(micTimeoutRef.current);
      }
      
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    };
  }, []);

  return (
    <Box sx={{ 
      position: 'fixed',
      bottom: 56, // Height of bottom navigation
      left: 0,
      right: 0,
      p: 2,
      backgroundColor: 'background.paper',
      borderTop: 1,
      borderColor: 'divider',
      zIndex: 1000
    }}>
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        accept="image/*"
        onChange={handleFileChange}
      />
      <SearchContainer>
        {isLoading ? (
          <CircularProgress size={20} color="primary" />
        ) : !showVisualizer && !isStreaming ? (
          <SearchIconStyled disableRipple onClick={handleSearchClick}>
            <SearchIcon />
          </SearchIconStyled>
        ) : null}
        
        {showVisualizer && (
          <VisualizerContainer ref={audioVisualizerRef}>
            <AudioBar style={{ animationDelay: '0ms' }} />
            <AudioBar style={{ animationDelay: '200ms' }} />
            <AudioBar style={{ animationDelay: '400ms' }} />
            <AudioBar style={{ animationDelay: '600ms' }} />
            <AudioBar style={{ animationDelay: '300ms' }} />
          </VisualizerContainer>
        )}
        
        <StyledInput
          placeholder={placeholder}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          onFocus={handleFocus}
          fullWidth
          autoComplete="off"
          disabled={disabled || isLoading || isStreaming || showVisualizer}
          inputRef={inputRef}
        />
        
        {inputValue ? (
          <IconButtonStyled 
            onClick={handleSend}
            disabled={!inputValue.trim() || disabled || isLoading || isStreaming || showVisualizer}
            color="primary"
          >
            <SendIcon />
          </IconButtonStyled>
        ) : (
          <>
            <Tooltip title="Upload image">
              <span>
                <IconButtonStyled onClick={handleImageUpload} disabled={disabled || isLoading}>
                  <CameraAltIcon />
                </IconButtonStyled>
              </span>
            </Tooltip>
            {isStreaming && (
              <Tooltip title={isMuted ? "Unmute Microphone" : "Mute Microphone"}>
                <span>
                  <IconButtonStyled 
                    onClick={handleMuteToggle}
                    data-muted={isMuted ? "true" : "false"}
                  >
                    {isMuted ? <MicOffIcon color="error" /> : <ActiveMicIcon />}
                  </IconButtonStyled>
                </span>
              </Tooltip>
            )}
            <Tooltip title={isStreaming ? "Stop voice session" : "Start voice input"}>
              <span>
                <IconButtonStyled onClick={handleMicPress} disabled={disabled || isLoading}>
                  {isStreaming ? <StopIcon color="primary" /> : (isAudioEnabled ? <MicIcon /> : <MicOffIcon />)}
                </IconButtonStyled>
              </span>
            </Tooltip>
          </>
        )}
      </SearchContainer>
    </Box>
  );
};

export default MobileSearchInput;
