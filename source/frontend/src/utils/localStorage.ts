/**
 * Utility functions for managing scoped localStorage in the chat application
 */

// Storage key generators
export const getChatStorageKey = (userId: string, sessionId: string): string => 
  `chatMessages_${userId}_${sessionId}`;

export const getFeedbackStorageKey = (userId: string, sessionId: string): string => 
  `messageFeedbackState_${userId}_${sessionId}`;

export const getTaskStorageKey = (userId: string, sessionId: string): string => 
  `processedTaskMessages_${userId}_${sessionId}`;

// Cleanup functions
export const cleanupUserSessions = (userId: string, keepSessionId?: string): void => {
  const keysToRemove: string[] = [];
  
  // Iterate through all localStorage keys
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key) {
      // Check if key belongs to this user
      const userPrefix = `_${userId}_`;
      if (key.includes(userPrefix)) {
        // If keepSessionId is provided, don't remove that session's data
        if (keepSessionId && key.includes(`_${userId}_${keepSessionId}`)) {
          continue;
        }
        keysToRemove.push(key);
      }
    }
  }
  
  // Remove the identified keys
  keysToRemove.forEach(key => {
    console.log(`Cleaning up localStorage key: ${key}`);
    localStorage.removeItem(key);
  });
};

export const cleanupOldUnscopedKeys = (): void => {
  const oldKeys = ['chatMessages', 'messageFeedbackState', 'processedTaskMessages'];
  oldKeys.forEach(key => {
    if (localStorage.getItem(key)) {
      console.log(`Removing old unscoped localStorage key: ${key}`);
      localStorage.removeItem(key);
    }
  });
};

export const getAllUserSessions = (userId: string): string[] => {
  const sessions: string[] = [];
  const userPrefix = `_${userId}_`;
  
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && key.includes(userPrefix)) {
      // Extract session ID from key
      const parts = key.split('_');
      const userIndex = parts.indexOf(userId);
      if (userIndex !== -1 && userIndex + 1 < parts.length) {
        const sessionId = parts[userIndex + 1];
        if (!sessions.includes(sessionId)) {
          sessions.push(sessionId);
        }
      }
    }
  }
  
  return sessions;
};

// Storage size management
export const getStorageSize = (): { used: number; total: number } => {
  let used = 0;
  for (let key in localStorage) {
    if (localStorage.hasOwnProperty(key)) {
      used += localStorage[key].length + key.length;
    }
  }
  
  // Most browsers have a 5-10MB limit for localStorage
  const total = 5 * 1024 * 1024; // 5MB estimate
  
  return { used, total };
};

export const cleanupOldestSessions = (userId: string, maxSessions: number = 10): void => {
  const sessions = getAllUserSessions(userId);
  
  if (sessions.length <= maxSessions) {
    return;
  }
  
  // Sort sessions by last modified time (approximate based on localStorage access)
  const sessionData = sessions.map(sessionId => {
    const chatKey = getChatStorageKey(userId, sessionId);
    const data = localStorage.getItem(chatKey);
    let lastModified = 0;
    
    if (data) {
      try {
        const messages = JSON.parse(data);
        if (Array.isArray(messages) && messages.length > 0) {
          // Try to extract timestamp from last message
          const lastMessage = messages[messages.length - 1];
          const timestampMatch = lastMessage.match(/data-timestamp="([^"]+)"/);
          if (timestampMatch) {
            lastModified = new Date(timestampMatch[1]).getTime();
          }
        }
      } catch (error) {
        console.warn('Error parsing session data for cleanup:', error);
      }
    }
    
    return { sessionId, lastModified };
  });
  
  // Sort by last modified (oldest first)
  sessionData.sort((a, b) => a.lastModified - b.lastModified);
  
  // Remove oldest sessions
  const sessionsToRemove = sessionData.slice(0, sessions.length - maxSessions);
  sessionsToRemove.forEach(({ sessionId }) => {
    console.log(`Cleaning up old session: ${sessionId}`);
    localStorage.removeItem(getChatStorageKey(userId, sessionId));
    localStorage.removeItem(getFeedbackStorageKey(userId, sessionId));
    localStorage.removeItem(getTaskStorageKey(userId, sessionId));
  });
};