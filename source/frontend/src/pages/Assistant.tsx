import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Box, Typography, IconButton, Button, CircularProgress, Snackbar, Alert } from '@mui/material';
import ThumbUpIcon from '@mui/icons-material/ThumbUp';
import ThumbDownIcon from '@mui/icons-material/ThumbDown';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useMessages, useWebSocketConnection, useAudio, useAuth, useS2S } from '../context';
import { fetchWithAuth } from '../config';
import { DefaultSystemPrompt } from '../consts';

// Import components
import MobileHeader from '../components/MobileHeader';
import MobileSearchInput from '../components/MobileSearchInput';
import ChatbotAvatar from '../components/ChatbotAvatar';
import SuggestedQuestions from '../components/SuggestedQuestions';

// Define the location state type
interface LocationState {
  query?: string;
  isInitialQuery?: boolean;
}

// Message feedback states
type FeedbackState = Record<string, 'up' | 'down' | null>;

const Assistant: React.FC = () => {
  // Navigation and location
  const location = useLocation();
  const navigate = useNavigate();
  const locationState = location.state as LocationState || {};
  const { query, isInitialQuery } = locationState;
  
  // Context hooks
  const { userId, sessionId } = useAuth();
  const { 
    isConnected, 
    connectionId, 
    connect, 
    connectionError,
    promptName
  } = useWebSocketConnection();
  
  const { 
    textStream, 
    addMessage, 
    addSystemMessage, 
    clearMessages, 
    isLoading, 
    setIsLoading,
    sendMessage,
    handleFeedback
  } = useMessages();
  
  const {
    resetContent
  } = useAudio();

  const {
    startS2SSession,
    endS2SSession
  } = useS2S();
  
  // Local state
  const [feedbackState, setFeedbackState] = useState<FeedbackState>(() => {
    // Initialize from localStorage if available
    const saved = localStorage.getItem('messageFeedbackState');
    return saved ? JSON.parse(saved) : {};
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const [taskAdded, setTaskAdded] = useState(false);
  const [processedTaskMessages, setProcessedTaskMessages] = useState<string[]>(() => {
    // Initialize from localStorage if available
    const saved = localStorage.getItem('processedTaskMessages');
    return saved ? JSON.parse(saved) : [];
  });
  const [notification, setNotification] = useState<{message: string, type: 'success' | 'error' | 'info' | 'warning', duration?: number} | null>(null);
  const [sessionStarted, setSessionStarted] = useState(false);
  
  // Refs
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const lastMessageTimeRef = useRef<number>(Date.now());
  const audioContextRef = useRef<AudioContext | null>(null);
  const previouslyDisconnected = useRef(false);
  
  // Suggested questions
  const suggestedQuestions = [
    { text: "What are the inventory levels?" },
    { text: "How do I process a return?" },
    { text: "What's today's staffing schedule?" },
    { text: "Show me sales trends for the week" }
  ];

  // Helper function to extract content from HTML wrapper and render safely
  const extractAndRenderContent = useCallback((htmlString: string) => {
    try {
      // Create a temporary element to parse the HTML
      const tempDiv = document.createElement('div');
      tempDiv.innerHTML = htmlString;
      
      // Extract the content from the div (removing the wrapper)
      const contentElement = tempDiv.firstElementChild as HTMLElement;
      if (contentElement) {
        // Get the inner content (could be markdown, HTML, or plain text)
        const innerContent = contentElement.innerHTML;
        
        // Check if content contains markdown table syntax
        const hasMarkdownTable = /\|.*\|/.test(innerContent);
        
        if (hasMarkdownTable) {
          // If it contains markdown tables, decode HTML entities and return as is
          const decodedContent = innerContent
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>')
            .replace(/&amp;/g, '&')
            .replace(/&quot;/g, '"')
            .replace(/&#39;/g, "'");
          return decodedContent;
        }
        
        // Check if content looks like HTML (contains HTML tags) but not markdown
        const hasHtmlTags = /<[^>]+>/.test(innerContent) && !hasMarkdownTable;
        
        if (hasHtmlTags) {
          // If it contains HTML tags (but not markdown), strip tags
          const textContent = innerContent.replace(/<[^>]*>/g, '');
          return textContent;
        } else {
          // If it's plain text or markdown, return as is
          return innerContent;
        }
      }
      
      // Fallback: return the original string if parsing fails
      return htmlString;
    } catch (error) {
      console.warn('Error parsing message content:', error);
      return htmlString;
    }
  }, []);

  // Handle initial query from another page
  useEffect(() => {
    if (!isInitialQuery || !query || !userId || !sessionId) return;
    
    const sendInitialQuery = async () => {
      try {
        setIsLoading(true);
        
        // Use REST API with increased timeout for initial queries
        const response = await fetchWithAuth(`/chat?query=${encodeURIComponent(query)}&userId=${encodeURIComponent(userId)}&sessionId=${encodeURIComponent(sessionId)}`, {
          method: 'GET',
          timeout: 600000 // 10 minutes timeout
        });
        
        if (response.data && response.data.chat_response) {
          addMessage(response.data.chat_response, 'assistant');
        }
      } catch (error) {
        console.error('Error sending initial query:', error);
        addSystemMessage("Sorry, there was an error processing your request. Please try again.");
      } finally {
        setIsLoading(false);
      }
    };
    
    sendInitialQuery();
    
    // Reset isInitialQuery to prevent re-sending
    if (location.state) {
      (location.state as LocationState).isInitialQuery = false;
    }
  }, [query, isInitialQuery, userId, sessionId, addMessage, addSystemMessage, setIsLoading, location]);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [textStream]);

  // Custom feedback handler that updates local UI state
  const handleFeedbackWithState = useCallback(async (messageId: string, message: string, type: 'up' | 'down') => {
    if (!userId || !sessionId) return;
    
    try {
      // Update local state immediately for UI feedback
      setFeedbackState(prev => {
        const updated = {
          ...prev,
          [messageId]: type
        };
        // Save to localStorage
        localStorage.setItem('messageFeedbackState', JSON.stringify(updated));
        return updated;
      });

      // Call the context feedback handler
      await handleFeedback(messageId, message, type);
      
      // Show success notification
      setNotification({
        message: 'Thank you for your feedback!',
        type: 'success'
      });
    } catch (error) {
      console.error('Error submitting feedback:', error);
      setNotification({
        message: 'Failed to submit feedback. Please try again.',
        type: 'error'
      });
    }
  }, [userId, sessionId, handleFeedback]);

  // Handle clicking a suggested question
  const handleQuestionClick = useCallback((question: string) => {
    sendMessage(question);
  }, [sendMessage]);

  // Clear chat history
  const handleRefresh = useCallback(() => {
    console.log('[Assistant] Refresh button clicked - calling resetContent');
    
    // Cancel any playing audio and clear chat messages
    resetContent();
    
    setTaskAdded(false);
    
    // Clear feedback state and remove from localStorage
    setFeedbackState({});
    localStorage.removeItem('messageFeedbackState');
    
    
  }, [resetContent]);

  // Handle task generation feature
  const handleAddToTasks = useCallback(async (taskContent: string) => {
    if (!userId || !sessionId) {
      addSystemMessage("Cannot add tasks: User not authenticated");
      return;
    }

    setIsGenerating(true);
    
    try {
      // Create tasks API request
      const query = `based on the provided task suggestions, create tasks for each of the users accordingly. Task suggestions: ${taskContent}`;
      
      const response = await fetchWithAuth(`/chat?query=${encodeURIComponent(query)}&userId=${encodeURIComponent(userId)}&sessionId=${encodeURIComponent(sessionId)}`, {
        method: 'GET',
        timeout: 600000 // 10 minutes timeout
      });
      
      // Process the response
      if (response.data && response.data.chat_response) {
        const assistantMessage = response.data.chat_response;
        addMessage(assistantMessage);
      }
      
      setTaskAdded(true);
      
      // Generate a unique identifier for this message to save in localStorage
      const taskMessageId = hashTaskMessage(taskContent);
      setProcessedTaskMessages(prev => {
        const updated = [...prev, taskMessageId];
        localStorage.setItem('processedTaskMessages', JSON.stringify(updated));
        return updated;
      });
      
      addSystemMessage("Tasks have been added to your task list.");
      setNotification({
        message: 'Tasks added successfully!',
        type: 'success'
      });
    } catch (error) {
      console.error("Error adding tasks:", error);
      addSystemMessage("Error adding tasks. Please try again.");
      setNotification({
        message: 'Failed to add tasks. Please try again.',
        type: 'error'
      });
    } finally {
      setIsGenerating(false);
    }
  }, [userId, sessionId, addSystemMessage]);

  // Helper function to check if a message should have feedback buttons
  const shouldShowFeedback = useCallback((text: string) => {
    const isUserMessage = text.includes('data-message-role="user"');
    const isSystemMessage = text.includes('data-message-role="system"');
    const isErrorMessage = text.includes('data-message-type="error"');
    const isSpecialStyle = text.includes('style="color: #666;"');
    
    // Only show feedback for assistant messages that are not errors or specially styled
    return !isUserMessage && !isSystemMessage && !isErrorMessage && !isSpecialStyle;
  }, []);
  
  // Render message content with feedback buttons
  const renderMessage = useCallback((message: string, index: number) => {
    // Extract message type and role from data attributes
    const typeMatch = message.match(/data-message-type="([^"]+)"/);
    const roleMatch = message.match(/data-message-role="([^"]+)"/);
    
    const messageType = typeMatch ? typeMatch[1] : null;
    const messageRole = roleMatch ? roleMatch[1] : null;
    
    // Generate a stable ID for the message
    const messageId = `msg-${index}-${Date.now()}`;
    
    // Use helper function to determine if feedback should be shown
    const showFeedback = shouldShowFeedback(message);
    
    return (
      <Box 
        key={index} 
        className={`message ${messageRole || ''} ${messageType || ''}`}
        sx={{ 
          p: 2, 
          borderRadius: 2, 
          mb: 2,
          backgroundColor: messageRole === 'assistant' ? '#f0f7ff' : 
                           messageRole === 'user' ? '#f5f5f5' : 
                           'transparent',
          maxWidth: '90%',
          alignSelf: messageRole === 'user' ? 'flex-end' : 'flex-start',
          position: 'relative'
        }}
      >
        {messageRole === 'assistant' && !messageType && (
          <ChatbotAvatar />
        )}
        
        <Box
          sx={{
            fontSize: '0.9rem',
            lineHeight: 1.5,
            wordBreak: 'break-word',
            '& code': { fontSize: '0.85rem' },
            '& ul': { paddingInlineStart: '20px' },
            '& h3': { margin: '0.5rem 0' },
            '& table': {
              width: '100%',
              borderCollapse: 'collapse',
              marginBottom: '1rem',
              overflow: 'hidden',
              borderRadius: '8px'
            },
            '& th, & td': {
              border: '1px solid #ddd',
              padding: '8px',
              textAlign: 'left'
            },
            '& th': { backgroundColor: '#f2f2f2' }
          }}
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {extractAndRenderContent(message)}
          </ReactMarkdown>
        </Box>
        
        {showFeedback && (
          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
            <IconButton 
              size="small" 
              onClick={() => handleFeedbackWithState(messageId, message, 'up')}
              color={feedbackState[messageId] === 'up' ? 'primary' : 'default'}
            >
              <ThumbUpIcon fontSize="small" />
            </IconButton>
            <IconButton 
              size="small" 
              onClick={() => handleFeedbackWithState(messageId, message, 'down')}
              color={feedbackState[messageId] === 'down' ? 'error' : 'default'}
            >
              <ThumbDownIcon fontSize="small" />
            </IconButton>
          </Box>
        )}
      </Box>
    );
  }, [feedbackState, handleFeedbackWithState, shouldShowFeedback]);

  // Initialize WebSocket connection on component mount
  useEffect(() => {
    // Don't automatically connect - connection will be established when mic button is clicked
    // connect().catch(error => {
    //   console.error('Error connecting to WebSocket:', error);
    //   setNotification({
    //     type: 'error',
    //     message: 'Could not connect to assistant service.',
    //     duration: 5000
    //   });
    // });

    return () => {
      // Clean up
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Effect to handle WebSocket connection errors
  useEffect(() => {
    if (connectionError) {
      console.log('WebSocket connection error:', connectionError);
      setNotification({
        type: 'error',
        message: `Connection error: ${connectionError}`,
        duration: 5000
      });
    }
  }, [connectionError]);

  // Effect to handle WebSocket status changes
  useEffect(() => {
    if (!isConnected && sessionStarted) {
      // Connection was lost during a session
      setNotification({
        type: 'warning',
        message: 'Connection to assistant service lost. Please wait while we reconnect...',
        duration: 3000
      });
    } else if (isConnected && sessionStarted && previouslyDisconnected.current) {
      previouslyDisconnected.current = false;
      setNotification({
        type: 'success',
        message: 'Connection restored!',
        duration: 3000
      });
    }
  }, [isConnected, sessionStarted]);

  // Track disconnection state
  useEffect(() => {
    if (!isConnected && sessionStarted) {
      previouslyDisconnected.current = true;
    }
  }, [isConnected, sessionStarted]);

  // Helper function to create a hash of the task message for identification
  const hashTaskMessage = useCallback((message: string): string => {
    // Simple hash function for strings
    let hash = 0;
    for (let i = 0; i < message.length; i++) {
      const char = message.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return 'task_' + Math.abs(hash).toString(16);
  }, []);

  return (
<Box 
      component="main"
      sx={{ 
        display: 'flex', 
        flexDirection: 'column',
        height: '100vh',
        width: '100%',
        position: 'relative',
        bgcolor: '#FFFFFF',
        padding: 0,
        margin: 0,
        mt: '56px', // Add top margin to account for fixed header
        overflow: 'hidden',
        '& *': {
          '&::-webkit-scrollbar': {
            display: 'none'
          },
          scrollbarWidth: 'none !important',
          msOverflowStyle: 'none !important'
        }
      }}
    >
      {/* Mobile Header */}
      <MobileHeader 
        title="AnyCompany Assist" 
        onRefresh={handleRefresh}
      />
      
      <Box 
        component="div"
        sx={{ 
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflowY: 'scroll',
          pb: 24, // Increase bottom padding to account for fixed input field (56px + padding + extra space)
          WebkitOverflowScrolling: 'touch'
        }}
      >
        {textStream.length === 0 ? (
          // Show chatbot avatar and suggested questions when no messages
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}>
            <ChatbotAvatar />
            <SuggestedQuestions 
              questions={suggestedQuestions} 
              onQuestionClick={handleQuestionClick} 
            />
          </Box>
        ) : (
          // Show chat messages
          textStream.map((text, index) => {
            // Try a more precise detection for user messages
            const isUserMessage = text.includes('data-message-role="user"');
            
            return (
              <Box 
                key={index}
                sx={{
                  mb: 2,
                  width: '100%',
                  px: 2,
                  display: 'flex',
                  justifyContent: isUserMessage ? 'flex-end' : 'flex-start'
                }}
              >
                <Box
                  sx={{
                    maxWidth: '85%',
                    backgroundColor: isUserMessage ? '#E3F2FD' : '#F5F5F5',
                    borderRadius: isUserMessage ? '12px 12px 0 12px' : '12px',
                    p: 2,
                    position: 'relative'
                  }}
                >
                  {/* Display timestamp and role for messages */}
                  {(text.match(/data-timestamp="([^"]+)"/) || text.includes('data-partial="true"')) && (
                    <Box sx={{ 
                      fontSize: '0.7rem', 
                      color: '#666', 
                      mb: 1,
                      fontFamily: 'monospace',
                      textAlign: isUserMessage ? 'right' : 'left',
                      display: 'flex',
                      flexDirection: isUserMessage ? 'row-reverse' : 'row',
                      alignItems: 'center',
                      gap: '4px'
                    }}>
                      {/* Role badge */}
                      <Box component="span" sx={{ 
                        fontWeight: 'bold',
                        color: isUserMessage 
                          ? '#1976d2' 
                          : text.includes('data-message-role="system"') 
                            ? '#9c27b0' 
                            : '#2e7d32',
                        backgroundColor: isUserMessage 
                          ? '#e3f2fd' 
                          : text.includes('data-message-role="system"') 
                            ? '#f3e5f5' 
                            : '#e8f5e9',
                        padding: '0px 4px',
                        borderRadius: '4px',
                        fontSize: '0.65rem',
                        whiteSpace: 'nowrap' // Prevent role badge from wrapping
                      }}>
                        {isUserMessage ? 'USER' : 
                         text.includes('data-message-role="assistant"') ? 'ASSISTANT' : 
                         text.includes('data-message-role="system"') ? 'SYSTEM' : 'ASSISTANT'}
                      </Box>
                      
                      {/* Timestamp - only show if available */}
                      {text.match(/data-timestamp="([^"]+)"/) && (
                        <Box component="span" sx={{ whiteSpace: 'nowrap' }}>
                          [{text.match(/data-timestamp="([^"]+)"/)![1]}]
                        </Box>
                      )}
                    </Box>
                  )}
                  
                  {/* Message content */}
                  <Box
                    sx={{
                      fontSize: '0.9rem',
                      lineHeight: 1.5,
                      wordBreak: 'break-word',
                      '& code': { fontSize: '0.85rem' },
                      '& ul': { paddingInlineStart: '20px' },
                      '& h3': { margin: '0.5rem 0' },
                      '& table': {
                        width: '100%',
                        borderCollapse: 'collapse',
                        marginBottom: '1rem',
                        overflow: 'hidden',
                        borderRadius: '8px'
                      },
                      '& th, & td': {
                        border: '1px solid #ddd',
                        padding: '8px',
                        textAlign: 'left'
                      },
                      '& th': { backgroundColor: '#f2f2f2' },
                      '& img': {
                        maxWidth: '100%',
                        height: 'auto',
                        borderRadius: '12px',
                        margin: '10px 0',
                        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
                        display: 'block'
                      },
                      '& em': {
                        color: '#666',
                        fontSize: '0.85rem'
                      }
                    }}
                  >
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {extractAndRenderContent(text)}
                    </ReactMarkdown>
                  </Box>
                  
                  {/* Feedback buttons for assistant messages only */}
                  {shouldShowFeedback(text) && (
                    <Box
                      sx={{
                        display: 'flex',
                        gap: 1,
                        justifyContent: 'flex-end',
                        mt: 1
                      }}
                    >
                      <IconButton
                        size="small"
                        onClick={() => {
                          // Generate a message ID if one doesn't exist
                          const messageId = text.match(/data-message-id="([^"]+)"/) ? 
                            text.match(/data-message-id="([^"]+)"/)![1] : 
                            `msg_${index}`; // Use index for stable ID
                          // Extract message text or use the whole message
                          const message = text.match(/data-message-text="([^"]+)"/) ?
                            text.match(/data-message-text="([^"]+)"/)![1] :
                            text.replace(/<[^>]*>/g, '').substring(0, 100);
                          
                          handleFeedbackWithState(messageId, message, 'up');
                        }}
                        color={feedbackState[
                          text.match(/data-message-id="([^"]+)"/) ? 
                            text.match(/data-message-id="([^"]+)"/)![1] : 
                            `msg_${index}`
                        ] === 'up' ? 'primary' : 'default'}
                      >
                        <ThumbUpIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => {
                          // Generate a message ID if one doesn't exist
                          const messageId = text.match(/data-message-id="([^"]+)"/) ? 
                            text.match(/data-message-id="([^"]+)"/)![1] : 
                            `msg_${index}`; // Use index for stable ID
                          // Extract message text or use the whole message
                          const message = text.match(/data-message-text="([^"]+)"/) ?
                            text.match(/data-message-text="([^"]+)"/)![1] :
                            text.replace(/<[^>]*>/g, '').substring(0, 100);
                          
                          handleFeedbackWithState(messageId, message, 'down');
                        }}
                        color={feedbackState[
                          text.match(/data-message-id="([^"]+)"/) ? 
                            text.match(/data-message-id="([^"]+)"/)![1] : 
                            `msg_${index}`
                        ] === 'down' ? 'primary' : 'default'}
                      >
                        <ThumbDownIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  )}
                  
                  {/* Task Approval Button */}
                  {text.includes('data-message-role="assistant"') && 
                    text.toLowerCase().includes('task list') && 
                    !taskAdded && 
                    !processedTaskMessages.includes(hashTaskMessage(text)) && (
                    <Box sx={{ 
                      mt: 2, 
                      display: 'flex', 
                      justifyContent: 'center',
                      width: '100%' // Ensure full width
                    }}>
                      <Button
                        variant="contained"
                        color="primary"
                        onClick={() => handleAddToTasks(text)}
                        disabled={isGenerating}
                        sx={{ width: '100%' }}
                      >
                        {isGenerating ? 'Creating Tasks...' : 'Approve & Create Tasks'}
                      </Button>
                    </Box>
                  )}
                </Box>
              </Box>
            );
          })
        )}
      </Box>
      
     
      
      {/* Input area */}
      <MobileSearchInput 
        onSend={sendMessage}
        disabled={isLoading}
      />
      
      {/* Notification snackbar */}
      <Snackbar
        open={Boolean(notification)}
        autoHideDuration={notification?.duration || 5000}
        onClose={() => setNotification(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        sx={{ display: notification ? 'block' : 'none' }}
      >
        {/* Ensuring we always return a valid ReactElement */}
        <Alert 
          onClose={() => setNotification(null)} 
          severity={notification?.type || 'info'}
          sx={{ 
            width: '100%',
            display: notification ? 'flex' : 'none'
          }}
        >
          {notification?.message || ''}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default Assistant; 