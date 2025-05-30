import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { RequiredAuth } from './SecureRoute';

import Home from '../pages/Home';
import Analytics from '../pages/Analytics';
import TaskList from '../pages/TaskList';
import Assistant from '../pages/Assistant';

const AppRoutes = () => {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route element={<RequiredAuth />}>
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/chat" element={<Navigate to="/assistant" replace />} />
        <Route path="/todolist" element={<TaskList />} />
        <Route path="/assistant" element={<Assistant />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
};

export default AppRoutes;
