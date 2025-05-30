import React from 'react';
import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import Loading from './Loading';

export const RequiredAuth: React.FC = () => {
  const { isAuthenticated, user } = useAuth();
  const location = useLocation();

  if (user === undefined) {
    return <Loading />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/" state={{ from: location }} replace />;
  }

  return <Outlet />;
};
