import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../context/AuthContext';

/**
 * Custom hook for scoped localStorage that automatically includes userId and sessionId
 */
export function useScopedLocalStorage<T>(
  keyPrefix: string,
  defaultValue: T,
  serializer?: {
    serialize: (value: T) => string;
    deserialize: (value: string) => T;
  }
) {
  const { userId, sessionId } = useAuth();
  
  const defaultSerializer = {
    serialize: JSON.stringify,
    deserialize: JSON.parse
  };
  
  const { serialize, deserialize } = serializer || defaultSerializer;
  
  // Generate scoped key
  const getScopedKey = useCallback(() => {
    if (!userId || !sessionId) return null;
    return `${keyPrefix}_${userId}_${sessionId}`;
  }, [keyPrefix, userId, sessionId]);
  
  // State to hold the current value
  const [storedValue, setStoredValue] = useState<T>(defaultValue);
  
  // Load value from localStorage when key becomes available
  useEffect(() => {
    const scopedKey = getScopedKey();
    if (scopedKey) {
      try {
        const item = localStorage.getItem(scopedKey);
        if (item !== null) {
          setStoredValue(deserialize(item));
        }
      } catch (error) {
        console.warn(`Error loading from localStorage key "${scopedKey}":`, error);
        setStoredValue(defaultValue);
      }
    }
  }, [getScopedKey, defaultValue, deserialize]);
  
  // Function to update the stored value
  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      // Allow value to be a function so we have the same API as useState
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      
      // Save state
      setStoredValue(valueToStore);
      
      // Save to localStorage if key is available
      const scopedKey = getScopedKey();
      if (scopedKey) {
        localStorage.setItem(scopedKey, serialize(valueToStore));
      }
    } catch (error) {
      console.warn('Error saving to localStorage:', error);
    }
  }, [getScopedKey, serialize, storedValue]);
  
  // Function to remove the item from localStorage
  const removeValue = useCallback(() => {
    const scopedKey = getScopedKey();
    if (scopedKey) {
      localStorage.removeItem(scopedKey);
    }
    setStoredValue(defaultValue);
  }, [getScopedKey, defaultValue]);
  
  return [storedValue, setValue, removeValue] as const;
}

/**
 * Specialized hooks for common use cases
 */

export function useScopedChatMessages() {
  return useScopedLocalStorage<string[]>('chatMessages', []);
}

export function useScopedFeedbackState() {
  return useScopedLocalStorage<Record<string, 'up' | 'down' | null>>('messageFeedbackState', {});
}

export function useScopedTaskMessages() {
  return useScopedLocalStorage<string[]>('processedTaskMessages', []);
}