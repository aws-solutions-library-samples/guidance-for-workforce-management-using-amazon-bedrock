import React, { createContext, useContext, useRef, useEffect, useCallback, useState, useMemo, FC, ReactNode } from 'react';
import { WEBSOCKET_URL, fetchWithAuth } from '../config';
import { useAuth } from './AuthContext';
import {
  DefaultInferenceConfiguration,
  DefaultTextConfiguration,
  DefaultAudioOutputConfiguration,
  DefaultAudioInputConfiguration,
  DefaultSystemPrompt
} from '../consts';


// @ts-ignore
import AudioPlayer from "./AudioPlayer.js";

// Audio constants
const SAMPLE_RATE = 16000;
const CHANNELS = 1;

// Security helper function to escape HTML content
const escapeHtml = (text: string): string => {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
};

// -----------------
// Type Definitions
// -----------------

// S2S Protocol Types
interface S2SEvent {
  event: {
    [key: string]: any;
  };
}


// Audio Streaming Options
interface AudioStreamOptions {
  promptName?: string;
  contentName?: string;
}


// Message feedback states
type FeedbackState = Record<string, 'up' | 'down' | null>;

// Main context interface
interface ChatContextType {
  // WebSocket Related
  isConnected: boolean;
  connectionId: string | null;
  connectionError: string | null;
  connect: () => Promise<void>;
  disconnect: () => void;
  promptName: string | null;
  
  // Message Related
  textStream: string[];
  addMessage: (message: string, role?: string) => void;
  addSystemMessage: (message: string) => void; 
  clearMessages: () => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  sendMessage: (message: string) => Promise<void>;
  handleFeedback: (messageId: string, message: string, type: 'up' | 'down') => Promise<void>;
  

  isStreaming: boolean;
  startStreaming: () => Promise<void>;
  stopStreaming: () => void;
  onAudioData: (callback: (audioData: Int16Array) => void) => () => void;
  resetContent: () => Promise<void>;

  
  // S2S Protocol Related
  startS2SSession: (systemPrompt?: string) => Promise<string>;
  endS2SSession: () => Promise<void>;
  
  // Utility Functions
  getPromptUUID: () => string;
  getTextContentUUID: () => string;
  getAudioContentUUID: () => string;
}


// Create the context
const ChatContext = createContext<ChatContextType | undefined>(undefined);

// -----------------
// Context Provider Component
// -----------------

export const ChatProvider: FC<{ children: ReactNode }> = ({ children }): JSX.Element => {
  // WebSocket Integration
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number>();
  const connectionTimeoutRef = useRef<number>();
  const reconnectAttemptsRef = useRef(0);
  const messageHandlersRef = useRef<Set<(event: MessageEvent) => void>>(new Set());
  const isConnectedRef = useRef(false);
  const isMountedRef = useRef(true);
  const [isConnected, setIsConnected] = useState(false);
  const { userId, sessionId, idToken } = useAuth();
  const [connectionId, setConnectionId] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  // const [isStreaming, setIsStreaming] = useState(false);
  const isStreamingRef = useRef(false);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const streamOptionsRef = useRef<AudioStreamOptions>({});

  // Add missing refs for content tracking
  const contentStartRef = useRef<Record<string, {generationStage?: string, role: string, type: string}>>({});
  
  // Add promptName state to fix linter error
  const [promptName, setPromptName] = useState<string | null>(null);
  
  const sessionStartedRef = useRef(false);
  const previouslyDisconnectedRef = useRef(false);
  
  // Add audio context and processor refs for proper cleanup
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  
  // Message state
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  
  // Chat message history
  const [textStream, setTextStream] = useState<string[]>(() => {
    // Initialize from localStorage if available
    const saved = localStorage.getItem('chatMessages');
    return saved ? JSON.parse(saved) : [];
  });
  
  // Text response data storage - matching audio approach
  const textResponseRef = useRef<Record<string, {content: string, role: string, generationStage?: string}>>({});

  // Global array to store all text content for duplicate detection
  const textContentRef = useRef<string[]>([]);

  // Get a prompt UUID and replenish the pool if needed
  const getPromptUUID = useCallback(() => {
    return `prompt-${crypto.randomUUID()}`;
  }, []);
  
  // Get a text content UUID and replenish the pool if needed
  const getTextContentUUID = useCallback(() => {
    return `text-${crypto.randomUUID()}`;
  }, []);
  
  // Get an audio content UUID and replenish the pool if needed
  const getAudioContentUUID = useCallback(() => {
    return `audio-${crypto.randomUUID()}`;
  }, []);

  // Generate client ID for WebSocket connection based on sessionId
  const getClientId = useCallback(() => {
    return sessionId ? sessionId : `client-${crypto.randomUUID()}`;
  }, [sessionId]);

  // -----------------
  // Audio Functionality
  // -----------------
  
  // Audio data subscription
  const audioDataCallbacksRef = useRef<Set<(audioData: Int16Array) => void>>(new Set());
  
  const onAudioData = useCallback((callback: (audioData: Int16Array) => void) => {
    console.log('[Audio] Adding audio data callback');
    audioDataCallbacksRef.current.add(callback);
    
    // Return a function to unsubscribe
    return () => {
      console.log('[Audio] Removing audio data callback');
      audioDataCallbacksRef.current.delete(callback);
    };
  }, []);

  // Process audio for S2S protocol
  const processAudio = useCallback((audioData: Float32Array) => {
    try {
      // Convert Float32Array to Int16Array
      const pcmData = new Int16Array(audioData.length);
      for (let i = 0; i < audioData.length; i++) {
        const int16 = Math.max(-32768, Math.min(32767, Math.round(audioData[i] * 32767)));
        pcmData[i] = int16;
      }

      // Notify all audio data callbacks
      audioDataCallbacksRef.current.forEach(callback => {
        try {
          callback(pcmData);
      } catch (error) {
          console.error('[Audio] Error in audio data callback:', error);
        }
      });

      return pcmData;
    } catch (error) {
      console.error('[Audio] Error processing audio:', error);
      return null;
    }
  }, []);

  // Initialize audio player
  const initAudio = useCallback(async () => {
    // Prevent double initialization in StrictMode
    if (audioPlayerRef.current) {
      console.log('[Audio] AudioPlayer already initialized');
      return;
    }
    
    try {
      console.log('Initializing AudioPlayer...');
      const player = new AudioPlayer();
      await player.start();
      player.onAudioData = processAudio;
      audioPlayerRef.current = player;
      console.log('AudioPlayer initialized successfully');
    } catch (error) {
      console.error('Failed to initialize AudioPlayer:', error);
    }
  }, [processAudio]);
  
  // Convert audio data to base64 string for S2S format
  const convertToBase64 = useCallback((pcmData: Int16Array) => {
    // Create a buffer to hold the binary data
    const buffer = new ArrayBuffer(pcmData.length * 2);
    const view = new DataView(buffer);
    
    // Set the int16 values in the buffer
    for (let i = 0; i < pcmData.length; i++) {
      view.setInt16(i * 2, pcmData[i], true); // true for little-endian
    }
    
    // Convert to binary string
    let binaryString = "";
    for (let i = 0; i < buffer.byteLength; i++) {
      binaryString += String.fromCharCode(new Uint8Array(buffer)[i]);
    }
    
    // Convert to base64
    return btoa(binaryString);
  }, []);

  // Base64 to Float32Array conversion for audio
  const base64ToFloat32Array = useCallback((base64: string): Float32Array => {
    try {
      // First, ensure the base64 string is properly formatted
      const cleanedBase64 = base64.replace(/[\s\r\n]+/g, '');
      
      // Handle potential padding issues
      let paddedBase64 = cleanedBase64;
      while (paddedBase64.length % 4 !== 0) {
        paddedBase64 += '=';
      }
      
      // Convert base64 to binary string
      const binaryString = atob(paddedBase64);
      const len = binaryString.length;
      
      // Create Int16Array from binary data (assuming 16-bit PCM)
      const int16Array = new Int16Array(len / 2);
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // Convert bytes to Int16Array
      const dataView = new DataView(bytes.buffer);
      for (let i = 0; i < int16Array.length; i++) {
        int16Array[i] = dataView.getInt16(i * 2, true); // true for little-endian
      }
      
      // Convert Int16Array to Float32Array (normalize to [-1, 1])
      const float32Array = new Float32Array(int16Array.length);
      for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
      }
      
      console.log(`[Audio] Converted base64 to Float32Array of length: ${float32Array.length}`);
      return float32Array;
    } catch (error) {
      console.error('[Audio] Error converting base64 to Float32Array:', error);
      return new Float32Array(0);
    }
  }, []);
  
  // Initialize AudioPlayer instance
  const audioPlayerRef = useRef<AudioPlayer | null>(null);
  
  // Send audio chunk with S2S protocol
  const sendAudioChunk = useCallback((base64Audio: string) => {
    if (!isStreamingRef.current) {
      console.warn('[WebSocket] Not streaming - ignoring audio chunk');
      return;
    }
    
    if (socketRef.current?.readyState !== WebSocket.OPEN) {
      console.warn('[WebSocket] Not connected - cannot send audio chunk');
      return;
    }
    
    
    if (!streamOptionsRef.current.promptName || !streamOptionsRef.current.contentName) {
      console.error('[WebSocket] Cannot send audio chunk - missing promptName or contentName', {
        promptName: streamOptionsRef.current.promptName,
        contentName: streamOptionsRef.current.contentName,
        isStreaming: isStreamingRef.current,
        streamOptions: streamOptionsRef.current
      });
      
      // If we're trying to stream but don't have content info, force stop streaming state
      if (isStreamingRef.current) {
        console.warn('[WebSocket] Invalid state: streaming without content info. Setting streaming state to false.');
        isStreamingRef.current = false;
        setIsStreaming(false);
      }
      
      return;
    }
    
    try {
      
      // console.log('[WebSocket] Sending audio chunk for prompt:', streamOptionsRef.current.promptName, 'and content:', streamOptionsRef.current.contentName, 'and base64Audio length:', base64Audio.length);
      const audioInputEvent = {
        event: {
          audioInput: {
            promptName: streamOptionsRef.current.promptName,
            contentName: streamOptionsRef.current.contentName,
            content: base64Audio
          }
        }
      };
      
      socketRef.current.send(JSON.stringify(audioInputEvent));
    } catch (error) {
      console.error('[WebSocket] Error sending audio chunk:', error);
      
      // If we encounter an error during streaming, force stop streaming state
      if (isStreamingRef.current) {
        console.warn('[WebSocket] Error during audio chunk send. Setting streaming state to false.');
        isStreamingRef.current = false;
        setIsStreaming(false);
      }
    }
  }, []);
  
  // WebSocket Connection Management
  // ----------------------------
  
  // WebSocket reconnection parameters
  const RECONNECT_INTERVAL = 2000; // 2 seconds
  const MAX_RECONNECT_ATTEMPTS = 5;
  const CONNECTION_TIMEOUT = 10000; // 10 seconds timeout for connections

  // WebSocket message handler
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      console.log('[WebSocket] Received message:', data);

      // Process event-based protocol
      const eventType = Object.keys(data.event)[0];

      // Handle different message types
      switch (eventType) {
        case 'contentStart':
          console.log('[WebSocket][S2S] Content start received:', data.event.contentStart);
          const contentStartEvent = data.event.contentStart;
          const startContentId = contentStartEvent.contentId;
          
          if (startContentId) {
            // Parse additionalModelFields to extract generationStage
            let generationStage: string | undefined;
            if (contentStartEvent.additionalModelFields) {
              try {
                const additionalFields = JSON.parse(contentStartEvent.additionalModelFields);
                generationStage = additionalFields.generationStage;
              } catch (error) {
                console.warn('[WebSocket][S2S] Error parsing additionalModelFields:', error);
              }
            }
            
            // Store content start information
            contentStartRef.current[startContentId] = {
              generationStage,
              role: contentStartEvent.role,
              type: contentStartEvent.type
            };
            
            console.log(`[WebSocket][S2S] Stored contentStart for ${startContentId}:`, {
              generationStage,
              role: contentStartEvent.role,
              type: contentStartEvent.type
            });
          }
          break;

        case 'textOutput':
          console.log('[WebSocket][S2S] Text output received:', {
            contentId: data.event.textOutput.contentId,
            content: data.event.textOutput.content.substring(0, 50) + '...'
          });
          
          const textEvent = data.event.textOutput;
          const textContentId = textEvent.contentId;
          
          if (!textContentId) {
            console.error('[Messages][S2S] Missing contentId in textOutput event');
            break;
          }

          // Get stored content start information
          const contentStartInfo = contentStartRef.current[textContentId];
          if (!contentStartInfo) {
            console.warn(`[Messages][S2S] No contentStart info found for ${textContentId}`);
            break;
          }

          const generationStage = contentStartInfo?.generationStage;
          const role = contentStartInfo?.role?.toLowerCase() || 'assistant';

          // Only process FINAL messages, ignore SPECULATIVE completely
          if (generationStage !== 'FINAL') {
            console.log(`[Messages][S2S] Ignoring ${generationStage} message for ${textContentId}`);
            break;
          }

          console.log(`[Messages][S2S] Processing FINAL ${role} message for ${textContentId}`);
          
          // Check if the text content is already processed to avoid duplicates
          if (textContentRef.current.includes(textEvent.content)) {
            console.log(`[Messages][S2S] Skipping duplicate content for ${textContentId}`);
            break;
          }
          
          // Add to global array to track processed content
          textContentRef.current.push(textEvent.content);

          // Add or append the message to the stream
          const timestamp = new Date().toLocaleTimeString();
          
          setTextStream(stream => {
            // Check if the last message is from the same role
            if (stream.length > 0) {
              const lastMessage = stream[stream.length - 1];
              
              // Extract role from the last message
              const roleMatch = lastMessage.match(/data-message-role="([^"]+)"/);
              const lastRole = roleMatch ? roleMatch[1] : null;
              
              // If same role, append to the existing message
              if (lastRole === role) {
                console.log(`[Messages][S2S] Appending to existing ${role} message`);
                
                // Extract existing text content safely using DOMParser
                const parser = new DOMParser();
                const doc = parser.parseFromString(lastMessage, 'text/html');
                const messageElement = doc.body.firstElementChild as HTMLElement;
                
                // Get existing text content safely
                const existingContent = messageElement ? messageElement.textContent || '' : '';
                
                // Create updated message with appended content
                // Escape the content to prevent XSS attacks
                const updatedMessage = `<div data-message-role="${role}" data-content-id="${textContentId}" data-timestamp="${timestamp}" data-generation-stage="final">${escapeHtml(existingContent + textEvent.content)}</div>`;
                
                // Replace the last message with the updated one
                const newStream = [...stream];
                newStream[newStream.length - 1] = updatedMessage;
                
                console.log(`[Messages][S2S] Appended content to ${role} message`);
                return newStream;
              }
            }
            
            // Different role or no previous messages - create new message
            console.log(`[Messages][S2S] Creating new ${role} message`);
            
            // Escape the content to prevent XSS attacks
            const updatedMessage = `<div data-message-role="${role}" data-content-id="${textContentId}" data-timestamp="${timestamp}" data-generation-stage="final">${escapeHtml(textEvent.content)}</div>`;
            
            return [...stream, updatedMessage];
          });
          
          break;

        case 'audioOutput':
          console.log('[WebSocket][S2S] Audio output received:', data.event.audioOutput);
          if (audioPlayerRef.current) {
            audioPlayerRef.current.playAudio(base64ToFloat32Array(
              data.event.audioOutput.content
            ));
            console.log('[WebSocket][S2S] Audio output played');
          } else {
            console.error('[WebSocket][S2S] Audio player not initialized');
          }
          break;
            
        case 'contentEnd':
          console.log("Content end received:", data.event.contentEnd.type);
          const endContentId = data.event.contentEnd.contentId;
    
          if (data.event.contentEnd.stopReason === "INTERRUPTED") {
            console.log("Content was interrupted by user");
            audioPlayerRef.current?.bargeIn();
            break;
          }

          // Clean up stored data
          delete textResponseRef.current[endContentId];
          delete contentStartRef.current[endContentId];
          
          break;
            
        case 'error':
          console.error('[WebSocket] Server error:', data);
          break;
          
        default:
          console.log('[WebSocket] Unknown message type:', data);
      }
    } catch (error) {
      console.error('[WebSocket] Error handling message:', error);
    }
  }, [base64ToFloat32Array]);

  // Connect to WebSocket server
  const connect = useCallback(async () => {
    return new Promise<void>((resolve, reject) => {
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        console.log('[WebSocket] Already connected');
        resolve();
        return;
      }
      
      // Prevent multiple simultaneous connection attempts
      if (socketRef.current?.readyState === WebSocket.CONNECTING) {
        console.log('[WebSocket] Connection already in progress');
        resolve();
        return;
      }
      
      try {
        const token = idToken;
        if (!token) {
          const error = new Error('No authentication token available');
          setConnectionError('Authentication required');
          reject(error);
          return;
        }
        
        if (!sessionId) {
          const error = new Error('No session ID available');
          setConnectionError('Session ID required');
          reject(error);
          return;
        }
        
        // Clear any previous connection error
        setConnectionError(null);

        // Generate a unique client ID for the session
        const clientId = getClientId();
        
        // Create session-based WebSocket URL
        const wsUrl = `${WEBSOCKET_URL}/${clientId}?token=${encodeURIComponent(token)}&userId=${encodeURIComponent(userId || 'unknown')}`;
        console.log('[WebSocket] Connecting to session-based URL:', wsUrl);
        
        // Set connection timeout
        if (connectionTimeoutRef.current) {
          clearTimeout(connectionTimeoutRef.current);
        }
        
        connectionTimeoutRef.current = setTimeout(() => {
          if (socketRef.current && socketRef.current.readyState !== WebSocket.OPEN) {
            console.error('[WebSocket] Connection timeout');
            if (isMountedRef.current) {
            setConnectionError('Connection timeout');
            }
            socketRef.current.close();
            reject(new Error('Connection timeout'));
          }
        }, CONNECTION_TIMEOUT);
        
        // Create new WebSocket connection
        const socket = new WebSocket(wsUrl);
        socketRef.current = socket;
        
        socket.onopen = () => {
          console.log('[WebSocket] Connection established');
          if (connectionTimeoutRef.current) {
            clearTimeout(connectionTimeoutRef.current);
          }
          isConnectedRef.current = true;
          if (isMountedRef.current) {
          setIsConnected(true);
          }
          reconnectAttemptsRef.current = 0;
          
          
          
          resolve();
        };
        
        socket.onmessage = handleMessage;
        
        socket.onerror = (error) => {
          console.error('[WebSocket] Connection error:', error);
          if (isMountedRef.current) {
          setConnectionError('Connection error occurred');
          }
          if (!isConnectedRef.current) {
            if (connectionTimeoutRef.current) {
              clearTimeout(connectionTimeoutRef.current);
            }
            reject(error);
          }
        };
        
        socket.onclose = (event) => {
          console.log('[WebSocket] Connection closed:', event.code, event.reason);
          isConnectedRef.current = false;
          if (isMountedRef.current) {
          setIsConnected(false);
          }
          
          // Map close codes to user-friendly messages
          if (isMountedRef.current) {
          if (event.code === 1000) {
            setConnectionError(null); // Normal closure
          } else if (event.code === 1001) {
            setConnectionError('Server going away');
          } else if (event.code === 1002 || event.code === 1003) {
            setConnectionError('Protocol error');
          } else if (event.code === 1006) {
            setConnectionError('Connection lost unexpectedly');
          } else if (event.code === 1008) {
            setConnectionError('Policy violation');
          } else if (event.code === 1011) {
            setConnectionError('Server error');
          } else if (event.code === 1012) {
            setConnectionError('Server restarting');
          } else if (event.code === 1013) {
            setConnectionError('Try again later');
          } else {
            setConnectionError(`Connection closed (${event.code})`);
            }
          }
          
          // Attempt reconnection if not intentionally closed and component is still mounted
          if (!event.wasClean && reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS && isMountedRef.current) {
            reconnectAttemptsRef.current++;
            console.log(`[WebSocket] Attempting to reconnect (${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS})...`);
            
            const backoffTime = RECONNECT_INTERVAL * Math.pow(1.5, reconnectAttemptsRef.current - 1);
            console.log(`[WebSocket] Reconnecting in ${backoffTime}ms`);
            
            reconnectTimeoutRef.current = setTimeout(() => {
              if (isMountedRef.current) {
              connect().catch(err => {
                console.error('[WebSocket] Reconnection failed:', err);
                  if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS && isMountedRef.current) {
                  setConnectionError('Max reconnection attempts reached');
                  
                  // Notify message handlers about the permanent disconnect
                  const systemEvent = new MessageEvent('message', {
                    data: JSON.stringify({
                      type: 'system_message',
                      message: 'Connection failed after multiple attempts',
                      severity: 'error'
                    })
                  });
                  messageHandlersRef.current.forEach(handler => handler(systemEvent));
                }
              });
              }
            }, backoffTime);
          } else if (event.wasClean && isMountedRef.current) {
            // Clean disconnection, no need for error
            setConnectionError(null);
          }
        };
        
      } catch (error) {
        console.error('[WebSocket] Error establishing connection:', error);
        if (isMountedRef.current) {
        setConnectionError('Error establishing connection');
        }
        reject(error);
      }
    });
  }, [idToken, userId, sessionId, handleMessage]);

  // Send S2S event
  const sendS2SEvent = useCallback((event: any) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(event));
    } else {
      console.warn('[WebSocket] Not connected - cannot send S2S event');
    }
  }, [socketRef]);

  // Start an S2S session
  const startS2SSession = useCallback(async (systemPrompt?: string) => {
    // Clear accumulated responses at the start of a new session
    textResponseRef.current = {};
    
    // Prevent starting a new session while another operation is in progress
    if (socketRef.current?.readyState !== WebSocket.OPEN) {
      console.log('[S2S] Socket not open, connecting first...');
      await connect();
    }
    
    
    
    // Generate a unique prompt name
    const newPromptName = getPromptUUID();
    
    try {
      console.log('[S2S] Starting new session');
      
      // Start a new S2S session
      const sessionStartEvent = {
        event: {
          sessionStart: {
            inferenceConfiguration: DefaultInferenceConfiguration
          }
        }
      };
      
      // Send session start event
      sendS2SEvent(sessionStartEvent);
      sessionStartedRef.current = true;
      
      // Start a new prompt
      const promptStartEvent = {
        event: {
          promptStart: {
            promptName: newPromptName,
            textOutputConfiguration: DefaultTextConfiguration,
            audioOutputConfiguration: DefaultAudioOutputConfiguration
          }
        }
      };
      
      // Send prompt start event
      sendS2SEvent(promptStartEvent);
      
      // Set the prompt name in state
      setPromptName(newPromptName);
      
      // Send system prompt if provided
      if (systemPrompt) {
        // Use the pre-generated text content UUID for system content
        const systemContentName = getTextContentUUID();
        
        // Start system content
        const contentStartEvent = {
          event: {
            contentStart: {
              promptName: newPromptName,
              contentName: systemContentName,
              type: "TEXT",
              interactive: true,
              role: "SYSTEM",
              textInputConfiguration: {
                mediaType: "text/plain"
              }
            }
          }
        };
        sendS2SEvent(contentStartEvent);
        
        // Send system prompt content
        const textInputEvent = {
          event: {
            textInput: {
              promptName: newPromptName,
              contentName: systemContentName,
              content: systemPrompt
            }
          }
        };
        sendS2SEvent(textInputEvent);
        
        // End system content
        const contentEndEvent = {
          event: {
            contentEnd: {
              promptName: newPromptName,
              contentName: systemContentName
            }
          }
        };
        sendS2SEvent(contentEndEvent);
      }
      
      console.log('[S2S] Session started with promptName:', newPromptName);
      return newPromptName;
    } catch (error: unknown) {
      console.error('[S2S] Error starting session:', error);
      
      sessionStartedRef.current = false;
      throw new Error(`Failed to start S2S session: ${error instanceof Error ? error.message : String(error)}`);
    }
  }, [connect, getPromptUUID, getTextContentUUID, sendS2SEvent]);

  // Comprehensive audio resource cleanup
  const cleanupAudioResources = useCallback(async () => {
    console.log('[Audio] Starting comprehensive audio cleanup...');
    
    try {
      // Disconnect and cleanup ScriptProcessor
      if (processorRef.current) {
        try {
          processorRef.current.disconnect();
          processorRef.current.onaudioprocess = null;
        } catch (error) {
          console.warn('[Audio] Error disconnecting processor:', error);
        }
        processorRef.current = null;
      }

      // Disconnect source node
      if (sourceNodeRef.current) {
        try {
          sourceNodeRef.current.disconnect();
        } catch (error) {
          console.warn('[Audio] Error disconnecting source node:', error);
        }
        sourceNodeRef.current = null;
      }
      
      // Stop all media stream tracks
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          try {
            track.stop();
            console.log('[Audio] Stopped track:', track.kind);
          } catch (error) {
            console.warn('[Audio] Error stopping track:', error);
          }
        });
        streamRef.current = null;
      }
      
      // Close AudioContext
      if (audioContextRef.current) {
        try {
          if (audioContextRef.current.state !== 'closed') {
            await audioContextRef.current.close();
            console.log('[Audio] AudioContext closed');
          }
        } catch (error) {
          console.warn('[Audio] Error closing AudioContext:', error);
        }
        audioContextRef.current = null;
      }
      
      // Reset AudioPlayer if needed
      if (audioPlayerRef.current) {
        try {
          audioPlayerRef.current.bargeIn();
          // Give AudioPlayer time to process barge-in
          await new Promise(resolve => setTimeout(resolve, 100));
        } catch (error) {
          console.warn('[Audio] Error resetting AudioPlayer:', error);
        }
      }
      
      // Clear content tracking
      contentStartRef.current = {};
      textResponseRef.current = {};
      
      // Reset streaming state
      isStreamingRef.current = false;
      setIsStreaming(false);
      streamOptionsRef.current = {};
      
      console.log('[Audio] Audio cleanup completed');
      
      // Give browser time to release resources
      await new Promise(resolve => setTimeout(resolve, 200));
      
    } catch (error) {
      console.error('[Audio] Error during audio cleanup:', error);
    }
  }, []);

  // Modify the startStreaming function to initialize AudioPlayer when needed
  const startStreaming = useCallback(async (options?: AudioStreamOptions) => {
    try {
      // Force comprehensive cleanup of any stale audio resources
      await cleanupAudioResources();
      
      // Initialize AudioPlayer if not already initialized
      if (!audioPlayerRef.current) {
        const player = new AudioPlayer();
        await player.start();
        audioPlayerRef.current = player;
        console.log('[Audio] AudioPlayer initialized successfully');
      }

      // Store options for later use
      streamOptionsRef.current = options || {};

      // Ensure WebSocket is connected
      if (socketRef.current?.readyState !== WebSocket.OPEN) {
        console.log('[Audio] WebSocket not connected, connecting first...');
        await connect();
      }


      console.log('[Audio] Starting new S2S session for audio streaming...');
      
      const newPromptName = await startS2SSession(getFormattedSystemPrompt());

      const currentContentName = getAudioContentUUID();
      // Update streamOptions with the latest values
      streamOptionsRef.current = {
        promptName: newPromptName,
        contentName: currentContentName
      };
      console.log("[Audio] Starting S2S audio content:", { 
        promptName: streamOptionsRef.current.promptName, 
        contentName: streamOptionsRef.current.contentName,
        sessionStarted: sessionStartedRef.current
      });

      // Send content start event
      sendS2SEvent({
        event: {
          contentStart: {
            promptName: streamOptionsRef.current.promptName,
            contentName: streamOptionsRef.current.contentName,
            type: "AUDIO",
            interactive: true,
            role: "USER",
            audioInputConfiguration: {
              mediaType: DefaultAudioInputConfiguration.mediaType,
              sampleRateHertz: DefaultAudioInputConfiguration.sampleRateHertz,
              sampleSizeBits: DefaultAudioInputConfiguration.sampleSizeBits,
              channelCount: DefaultAudioInputConfiguration.channelCount,
              audioType: DefaultAudioInputConfiguration.audioType,
              encoding: DefaultAudioInputConfiguration.encoding
            }
          }
        }
      });

      // Get microphone access with specific audio constraints
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: CHANNELS,
          sampleRate: SAMPLE_RATE,
          sampleSize: 16,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // Create AudioContext for processing
      const audioContext = new AudioContext({
        sampleRate: SAMPLE_RATE,
        latencyHint: "interactive",
      });
      
      // Store references for cleanup
      audioContextRef.current = audioContext;
      streamRef.current = stream;

      // Create MediaStreamSource
      const source = audioContext.createMediaStreamSource(stream);
      sourceNodeRef.current = source;

      // Create ScriptProcessor for raw PCM data
      const processor = audioContext.createScriptProcessor(1024, 1, 1);
      processorRef.current = processor;

      // Connect the audio nodes
      source.connect(processor);
      processor.connect(audioContext.destination);

      // Variables for user speech detection
      let userIsSpeaking = false;
      let silenceTimer: number | null = null;
      let speakingStarted = false;
      const SILENCE_THRESHOLD = 0.01;
      const SPEECH_THRESHOLD = 0.015;
      const SILENCE_DURATION = 1000;
      const MIN_SPEECH_SAMPLES = 5;
      let speechSampleCount = 0;

      processor.onaudioprocess = (e) => {
        if (!isStreamingRef.current) return;

        const inputData = e.inputBuffer.getChannelData(0);

        // Calculate audio level for this chunk (for speech detection)
        const audioLevel = Math.max(...Array.from(inputData).map(Math.abs));

        // Speech detection logic
        if (audioLevel > SPEECH_THRESHOLD) {
          speechSampleCount++;

          if (speechSampleCount >= MIN_SPEECH_SAMPLES && !userIsSpeaking) {
            userIsSpeaking = true;
            if (!speakingStarted) {
              speakingStarted = true;
              // Notify that user started speaking
              console.log('[Audio] User started speaking');
            }
          }

          if (silenceTimer) {
            clearTimeout(silenceTimer);
            silenceTimer = null;
          }
        } else if (audioLevel < SILENCE_THRESHOLD && userIsSpeaking) {
          speechSampleCount = 0;

          if (!silenceTimer) {
            silenceTimer = setTimeout(() => {
              userIsSpeaking = false;
              speakingStarted = false;
              console.log('[Audio] User stopped speaking');
              silenceTimer = null;
            }, SILENCE_DURATION);
          }
        } else {
          speechSampleCount = 0;
        }

        // Direct audio processing without redundant conversions
        try {
          // Convert Float32Array directly to Int16Array (single conversion)
          const pcmData = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]));
            pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }

          // Notify audio data callbacks
          audioDataCallbacksRef.current.forEach(callback => {
            try {
              callback(pcmData);
            } catch (error) {
              console.error('[Audio] Error in audio data callback:', error);
            }
          });

          // Convert directly to base64 using the more reliable method
          if (pcmData.length > 0) {
            const base64Audio = convertToBase64(pcmData);
            sendAudioChunk(base64Audio);
          }
        } catch (error) {
          console.error("Error processing audio data:", error);
        }
      };

      // Update streaming state
      setIsStreaming(true);
      isStreamingRef.current = true;
      
      console.log("Audio streaming started successfully");

    } catch (error) {
      console.error('Error starting streaming:', error);
      await stopStreaming();
      throw error;
    }
  }, [processAudio, convertToBase64, sendAudioChunk, isStreaming, startS2SSession, cleanupAudioResources]);

  // -----------------
  // WebSocket Connection
  // -----------------

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    // Clear timeouts if active
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = undefined;
    }
    
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
      connectionTimeoutRef.current = undefined;
    }
    
    // Close the socket if open
    if (socketRef.current) {
      if (socketRef.current.readyState === WebSocket.OPEN) {
        // Try to end S2S session if active
        if (sessionStartedRef.current) {
          try {
            const sessionEndEvent = {
              event: {
                sessionEnd: {}
              }
            };
            socketRef.current.send(JSON.stringify(sessionEndEvent));
          } catch (error) {
            console.warn('[WebSocket] Error sending session end event:', error);
          }
        }
        
        socketRef.current.close(1000, 'Client initiated disconnect');
      }
      socketRef.current = null;
    }
    
    // Update state
    isConnectedRef.current = false;
    setIsConnected(false);
    setConnectionId(null);
    sessionStartedRef.current = false;
    setConnectionError(null);
    
    console.log('[WebSocket] Disconnected');
  }, []);

  // -----------------
  // S2S Session Management  
  // -----------------

  // End an S2S session
  const endS2SSession = useCallback(async () => {
    if (!sessionStartedRef.current) {
      console.log('[S2S] No active session to end');
      return;
    }
    
    try {
      console.log('[S2S] Ending session with prompt:', streamOptionsRef.current.promptName);
      
      // End current prompt if active
      if (streamOptionsRef.current.promptName) {
        // First end any active content if needed
        if (streamOptionsRef.current.contentName) {
          const audioContentEndEvent = {
            event: {
              contentEnd: {
                promptName: streamOptionsRef.current.promptName,
                contentName: streamOptionsRef.current.contentName
              }
            }
          };
          sendS2SEvent(audioContentEndEvent);
          
          // Add a small delay between events
          await new Promise(resolve => setTimeout(resolve, 50));
        }
        
        // Now end the prompt
        const promptEndEvent = {
          event: {
            promptEnd: {
              promptName: streamOptionsRef.current.promptName
            }
          }
        };
        sendS2SEvent(promptEndEvent);
        
        // Wait for prompt end to be processed before ending session
        await new Promise(resolve => setTimeout(resolve, 200));
      }
      
      // End the session only if we're still connected
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        const sessionEndEvent = {
          event: {
            sessionEnd: {}
          }
        };
        sendS2SEvent(sessionEndEvent);
      }
      
      // Reset session state regardless of whether we could send the event
      sessionStartedRef.current = false;
      
      // Clean up accumulated responses
      textResponseRef.current = {};
      contentStartRef.current = {};
      
      console.log('[S2S] Session ended');
    } catch (error) {
      console.error('[S2S] Error ending session:', error);
      
      // Reset state even if there was an error
      sessionStartedRef.current = false;
      
      // Clean up accumulated responses even on error
      textResponseRef.current = {};
      contentStartRef.current = {};
      
      throw error;
    }
  }, [sendS2SEvent]);

  // Update stopStreaming to include proper cleanup  
  const stopStreaming = useCallback(async () => {
    console.log("Stopping streaming...");

    // Use comprehensive cleanup
    await cleanupAudioResources();

    console.log("Audio streaming stopped");

    // End the S2S session
    await endS2SSession();
    console.log("S2S session ended");
    
    // disconnect web socket connection
    disconnect();
  }, [cleanupAudioResources, endS2SSession, disconnect]);

  // -----------------
  // Message Handling Functions
  // -----------------

  // Add a regular message to the chat
  const addMessage = useCallback((message: string, role: string = 'user') => {
    // Format as HTML with data attributes for easy parsing
    const formattedContent = message;
    const timestamp = new Date().toLocaleTimeString();
    
    // Generate a unique content ID for this message
    const contentId = `manual-${role}-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    
    // Make sure user messages are clearly marked
    // Use a more explicit approach with attributes that match assistant messages
    let formattedMessage;
    if (role === 'user') {
      // Include both data-role and data-message-role to handle both formats
      formattedMessage = `<div data-role="user" data-message-role="user" data-content-id="${contentId}" data-timestamp="${timestamp}">${formattedContent}</div>`;
    } else {
      formattedMessage = `<div data-message-role="${role}" data-content-id="${contentId}" data-timestamp="${timestamp}">${formattedContent}</div>`;
    }
    
    setTextStream(prevStream => [...prevStream, formattedMessage]);
  }, []);

  // Add a system message (notifications, errors, etc.)
  const addSystemMessage = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    // Generate a unique content ID for this system message
    const contentId = `system-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    const formattedMessage = `<div data-message-role="system" data-message-type="notification" data-content-id="${contentId}" data-timestamp="${timestamp}">${message}</div>`;
    setTextStream(prevStream => [...prevStream, formattedMessage]);
  }, []);

  // Clear all messages
  const clearMessages = useCallback(() => {
    setTextStream([]);
    localStorage.removeItem('chatMessages');
  }, []);

  // Format system prompt with user information
  const getFormattedSystemPrompt = useCallback(() => {
    if (!userId) {
      return DefaultSystemPrompt; // Return default if no userId
    }
    // Replace {current_user} placeholder with actual userId
    const formattedSystemPrompt = DefaultSystemPrompt.replace('{userId}', userId).replace('{date}', new Date().toLocaleDateString()).replace('{sessionId}', sessionId);
    console.log("Formatted system prompt:", formattedSystemPrompt);
    return formattedSystemPrompt;
  }, [userId, sessionId]);

  // Modify the sendMessage function to use the formatted system prompt
  const sendMessage = useCallback(async (message: string) => {
    if (!userId || !sessionId) return;
    
    setIsLoading(true);
    
    try {
      // Add user message to the UI
      addMessage(message, 'user');
      
      // Send query through REST API
      const response = await fetchWithAuth(`/chat?query=${encodeURIComponent(message)}&userId=${encodeURIComponent(userId)}&sessionId=${encodeURIComponent(sessionId)}`, {
        method: 'GET',
        timeout: 600000 // 10 minutes timeout
      });
      
      // Process the response
      if (response.data && response.data.chat_response) {
        const assistantMessage = response.data.chat_response;
        addMessage(assistantMessage, 'assistant');
      } else {
        // Handle empty or invalid response
        addSystemMessage("Received an invalid response. Please try again.");
      }
    } catch (error) {
      console.error('Error sending message:', error);
      addSystemMessage("Sorry, there was an error processing your request. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }, [userId, sessionId, addMessage, addSystemMessage]);

  // Handle sending feedback for a message
  const handleFeedback = useCallback(async (messageId: string, message: string, type: 'up' | 'down') => {
    if (!userId || !sessionId) return;
    
    try {
      // Send feedback to backend
      await fetchWithAuth('/feedback', {
        method: 'POST',
        body: JSON.stringify({ 
          messageId, 
          message,
          feedback: type, 
          userId,
          sessionId,
          timestamp: new Date().toISOString()
        })
      });
      
      // Show success notification
      addSystemMessage('Thank you for your feedback!');
    } catch (error) {
      console.error('Error submitting feedback:', error);
      addSystemMessage('Failed to submit feedback. Please try again.');
    }
  }, [userId, sessionId, addSystemMessage]);

  
  // -----------------
  // Effects
  // -----------------

  // Save messages to localStorage when they change
  useEffect(() => {
    localStorage.setItem('chatMessages', JSON.stringify(textStream));
  }, [textStream]);

  // Handle WebSocket connection status changes
  useEffect(() => {
    if (!isConnected && sessionStartedRef.current) {
      // Connection was lost during a session
      previouslyDisconnectedRef.current = true;
      addSystemMessage('Connection to assistant service lost. Please wait while we reconnect...');
    } else if (isConnected && sessionStartedRef.current && previouslyDisconnectedRef.current) {
      previouslyDisconnectedRef.current = false;
      addSystemMessage('Connection restored!');
    }
  }, [isConnected, addSystemMessage]);

  // Clean up on unmount
  useEffect(() => {
    // Mark component as mounted
    isMountedRef.current = true;
    
    return () => {
      // Mark component as unmounted
      isMountedRef.current = false;
      
      // Cleanup all resources
      disconnect();
      
      // Clean up audio resources
      cleanupAudioResources();
    };
  }, [disconnect, cleanupAudioResources]);

  // Add the resetContent function implementation
  const resetContent = useCallback(async () => {
    console.log('[ChatContext] Canceling audio playbook and clearing state...');
    
    try {
      // Use the comprehensive audio cleanup instead of manual resets
      await cleanupAudioResources();

      // Clear chat history from localStorage
      localStorage.removeItem('chatMessages');
      
      // Clear messages from state
      setTextStream([]);

      // set the chat history to inactive
      if (userId && sessionId) {
        console.log('[ChatContext] Calling reset_chat API with userId:', userId, 'sessionId:', sessionId);
        await fetchWithAuth(`/reset_chat?userId=${encodeURIComponent(userId)}&sessionId=${encodeURIComponent(sessionId)}`, {
          method: 'GET'
        });
        console.log('[ChatContext] Chat history reset');
      }

      // Clear global text content tracking
      textContentRef.current = [];
      
      console.log('[ChatContext] Audio playbook canceled and state cleared');

      // // hard reset as workaround
      // window.location.reload();
    } catch (error) {
      console.error('[ChatContext] Error during resetContent:', error);
      throw error;
    }
  }, [cleanupAudioResources, userId, sessionId]);

  // Define the context value
  const contextValue = useMemo(() => ({
    // WebSocket Related
    isConnected,
    connectionId,
    connectionError,
    connect,
    disconnect,
    promptName,
    
    // Message Related
    textStream,
    addMessage,
    addSystemMessage,
    clearMessages,
    isLoading,
    setIsLoading,
    sendMessage,
    handleFeedback,
    isStreaming,
    startStreaming,
    stopStreaming,
    onAudioData,
    resetContent,
    
    // S2S Protocol Related
    startS2SSession,
    endS2SSession,
    
    // Utility Functions
    getPromptUUID,
    getTextContentUUID,
    getAudioContentUUID
  }), [
    isConnected, connectionId, connectionError, connect, disconnect, promptName,
    textStream, addMessage, addSystemMessage, clearMessages, isLoading, isStreaming, setIsLoading, sendMessage, handleFeedback, 
    startStreaming, stopStreaming, onAudioData, resetContent,
    startS2SSession, endS2SSession, getPromptUUID, getTextContentUUID, getAudioContentUUID
  ]);

  return (
    <ChatContext.Provider value={contextValue}>
      {children}
    </ChatContext.Provider>
  );
};

// Custom hook to use chat context
export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};

// Helper hooks for specific functionality
export const useWebSocketConnection = () => {
  const { 
    isConnected, connectionId, connectionError, 
    connect, disconnect, promptName 
  } = useChat();
  return { isConnected, connectionId, connectionError, connect, disconnect, promptName };
};

export const useMessages = () => {
  const { 
    textStream, addMessage, addSystemMessage, clearMessages,
    isLoading, setIsLoading, sendMessage, handleFeedback
  } = useChat();
  return { textStream, addMessage, addSystemMessage, clearMessages, isLoading, setIsLoading, sendMessage, handleFeedback };
};

export const useAudio = () => {
  const { 
    isStreaming, startStreaming, stopStreaming, onAudioData, resetContent
  } = useChat();
  return { isStreaming, startStreaming, stopStreaming, onAudioData, resetContent };
};

export const useS2S = () => {
  const { 
    startS2SSession, endS2SSession, 
    getPromptUUID, getTextContentUUID, getAudioContentUUID 
  } = useChat();
  return { startS2SSession, endS2SSession, getPromptUUID, getTextContentUUID, getAudioContentUUID };
}; 
