/**
 * Tests for localStorage utility functions
 */

import {
  getChatStorageKey,
  getFeedbackStorageKey,
  getTaskStorageKey,
  cleanupUserSessions,
  cleanupOldUnscopedKeys,
  getAllUserSessions
} from '../localStorage';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
    get length() {
      return Object.keys(store).length;
    },
    key: (index: number) => {
      const keys = Object.keys(store);
      return keys[index] || null;
    },
    hasOwnProperty: (key: string) => key in store
  };
})();

// Replace global localStorage with mock
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
});

describe('localStorage utilities', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  describe('Storage key generators', () => {
    test('getChatStorageKey generates correct key', () => {
      const key = getChatStorageKey('user123', 'session456');
      expect(key).toBe('chatMessages_user123_session456');
    });

    test('getFeedbackStorageKey generates correct key', () => {
      const key = getFeedbackStorageKey('user123', 'session456');
      expect(key).toBe('messageFeedbackState_user123_session456');
    });

    test('getTaskStorageKey generates correct key', () => {
      const key = getTaskStorageKey('user123', 'session456');
      expect(key).toBe('processedTaskMessages_user123_session456');
    });
  });

  describe('Session management', () => {
    test('getAllUserSessions returns correct sessions', () => {
      // Set up test data
      localStorage.setItem('chatMessages_user123_session1', '[]');
      localStorage.setItem('messageFeedbackState_user123_session1', '{}');
      localStorage.setItem('chatMessages_user123_session2', '[]');
      localStorage.setItem('chatMessages_user456_session3', '[]');
      localStorage.setItem('unrelated_key', 'value');

      const sessions = getAllUserSessions('user123');
      expect(sessions).toContain('session1');
      expect(sessions).toContain('session2');
      expect(sessions).not.toContain('session3');
      expect(sessions.length).toBe(2);
    });

    test('cleanupUserSessions removes correct keys', () => {
      // Set up test data
      localStorage.setItem('chatMessages_user123_session1', '[]');
      localStorage.setItem('messageFeedbackState_user123_session1', '{}');
      localStorage.setItem('chatMessages_user123_session2', '[]');
      localStorage.setItem('chatMessages_user456_session3', '[]');
      localStorage.setItem('unrelated_key', 'value');

      cleanupUserSessions('user123', 'session1');

      // Should keep session1 and unrelated keys
      expect(localStorage.getItem('chatMessages_user123_session1')).toBe('[]');
      expect(localStorage.getItem('messageFeedbackState_user123_session1')).toBe('{}');
      expect(localStorage.getItem('chatMessages_user456_session3')).toBe('[]');
      expect(localStorage.getItem('unrelated_key')).toBe('value');

      // Should remove session2
      expect(localStorage.getItem('chatMessages_user123_session2')).toBeNull();
    });
  });

  describe('Cleanup functions', () => {
    test('cleanupOldUnscopedKeys removes old keys', () => {
      // Set up old unscoped keys
      localStorage.setItem('chatMessages', '[]');
      localStorage.setItem('messageFeedbackState', '{}');
      localStorage.setItem('processedTaskMessages', '[]');
      localStorage.setItem('someOtherKey', 'value');

      cleanupOldUnscopedKeys();

      // Old keys should be removed
      expect(localStorage.getItem('chatMessages')).toBeNull();
      expect(localStorage.getItem('messageFeedbackState')).toBeNull();
      expect(localStorage.getItem('processedTaskMessages')).toBeNull();

      // Other keys should remain
      expect(localStorage.getItem('someOtherKey')).toBe('value');
    });
  });
});