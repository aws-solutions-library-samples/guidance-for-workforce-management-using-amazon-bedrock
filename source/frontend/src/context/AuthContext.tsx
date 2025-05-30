import React, { createContext, useContext, useEffect, useState } from 'react';
import { AuthUser } from '@aws-amplify/auth';
import { Auth } from 'aws-amplify';

interface AuthContextType {
  user: AuthUser | undefined;
  signOut: () => Promise<void>;
  userId: string | null;
  isAuthenticated: boolean;
  sessionId: string;
  idToken: string | null;
  credentials: {
    accessKeyId: string;
    secretAccessKey: string;
    sessionToken: string;
  } | null;
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
  const [credentials, setCredentials] = useState<AuthContextType['credentials']>(null);
  const [sessionId] = useState(() => 
    localStorage.getItem('sessionId') || crypto.randomUUID()
  );

  useEffect(() => {
    if (user) {
      const username = user.attributes?.email || user.username;
      setUserId(username);
      localStorage.setItem('currentUserId', username);
      localStorage.setItem('sessionId', sessionId);
      
      const getAuthDetails = async () => {
        try {
          const session = await Auth.currentSession();
          if (!session) {
            throw new Error('No valid session found');
          }
          
          const token = session.getIdToken().getJwtToken();
          setIdToken(token);

          // Only fetch credentials if we have a valid authenticated session
          if (session.isValid()) {
            try {
              // Ensure we're using the authenticated credentials
              const currentCredentials = await Auth.currentUserCredentials();
              // console.log('currentCredentials', currentCredentials);
              if (currentCredentials && !currentCredentials.isGuest) {
                setCredentials({
                  accessKeyId: currentCredentials.accessKeyId,
                  secretAccessKey: currentCredentials.secretAccessKey,
                  sessionToken: currentCredentials.sessionToken,
                });
              }
            } catch (credError) {
              console.warn('Unable to get AWS credentials:', credError);
              setCredentials(null);
            }
          }
        } catch (error) {
          console.error('Error getting session:', error);
          setIdToken(null);
          setCredentials(null);
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
      setCredentials(null);
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
        credentials
      }}
    >
      {children}
    </AuthContext.Provider>
  ) : null;
}; 