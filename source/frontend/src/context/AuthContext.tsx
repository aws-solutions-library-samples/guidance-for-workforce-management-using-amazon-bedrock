import React, { createContext, useContext, useEffect, useState } from 'react';
import { AuthUser } from '@aws-amplify/auth';
import { getCurrentUser, fetchAuthSession } from 'aws-amplify/auth';

interface AuthContextType {
  user: AuthUser | undefined;
  signOut: () => Promise<void>;
  userId: string | null;
  isAuthenticated: boolean;
  sessionId: string;
  idToken: string | null;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{
  children: React.ReactNode;
  user: AuthUser;
  signOut: () => Promise<void>;
}> = ({ children, user, signOut }) => {
  const [userId, setUserId] = useState<string | null>(null);
  const [idToken, setIdToken] = useState<string | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [sessionId] = useState(() => 
    localStorage.getItem('sessionId') || crypto.randomUUID()
  );

  useEffect(() => {
    if (user) {
      const getAuthDetails = async () => {
        try {
          const session = await fetchAuthSession();
          if (!session) {
            throw new Error('No valid session found');
          }
          
          const token = session.tokens?.idToken?.toString();
          if (token) {
            setIdToken(token);
            
            // Parse the JWT token to extract the username/email
            try {
              // Decode JWT payload (second part of the token)
              const payload = JSON.parse(atob(token.split('.')[1]));
              console.log('[AuthContext] JWT payload:', payload);
              
              // Try to get the username from various sources in order of preference
              let extractedUsername = 
                payload.email ||                    // Email address
                payload.preferred_username ||       // Preferred username
                payload['cognito:username'] ||      // Cognito username
                payload.username ||                 // Username claim
                user.username ||                    // User object username
                user.userId;                        // Fallback to userId
              
              console.log('[AuthContext] Extracted username:', extractedUsername);
              setUserId(extractedUsername);
              localStorage.setItem('currentUserId', extractedUsername);
            } catch (jwtError) {
              console.error('[AuthContext] Error parsing JWT token:', jwtError);
              // Fallback to user object properties
              const fallbackUsername = user.username || user.userId;
              setUserId(fallbackUsername);
              localStorage.setItem('currentUserId', fallbackUsername);
            }
          }

          localStorage.setItem('sessionId', sessionId);
        } catch (error) {
          console.error('Error getting session:', error);
          setIdToken(null);
          
          // Fallback to user object properties if session fetch fails
          const fallbackUsername = user.username || user.userId;
          setUserId(fallbackUsername);
          localStorage.setItem('currentUserId', fallbackUsername);
          localStorage.setItem('sessionId', sessionId);
        } finally {
          setIsReady(true);
        }
      };
      getAuthDetails();
    } else {
      setIsReady(true);
    }
  }, [user, sessionId]);

  const handleSignOut = async () => {
    try {
      await signOut();
      localStorage.removeItem('currentUserId');
      localStorage.removeItem('sessionId');
      setUserId(null);
      setIdToken(null);
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  return isReady ? (
    <AuthContext.Provider 
      value={{ 
        user, 
        signOut: handleSignOut, 
        userId, 
        isAuthenticated: !!user,
        sessionId,
        idToken,
      }}
    >
      {children}
    </AuthContext.Provider>
  ) : null;
}; 