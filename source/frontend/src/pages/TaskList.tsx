import React, { useState, useEffect } from 'react';
import {
  Container,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Paper,
  Box,
  CircularProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Typography,
  SelectChangeEvent,
  Collapse,
  Pagination
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import useMediaQuery from '@mui/material/useMediaQuery';
import MobileHeader from '../components/MobileHeader';

import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import SaveIcon from '@mui/icons-material/Save';

import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import SearchIcon from '@mui/icons-material/Search';

interface Todo {
  userId: string;
  taskId: string;
  taskOwner: string;
  text: string;
  description?: string;
  status: string;
  createdAt: string;
}
import { fetchWithAuth } from '../config';

const getStatusColor = (status: string) => {
  switch (status) {
    case 'completed':
      return 'success.main';  // Green
    case 'in_progress':
      return 'warning.main';  // Orange/Yellow
    case 'open':
      return 'info.main';     // Blue
    default:
      return 'info.main';     // Default to blue
  }
};

const TaskList: React.FC = () => {
  const { userId, isAuthenticated } = useAuth();
  const [todos, setTodos] = useState<Todo[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [users, setUsers] = useState<{ userId: string; userRole: string }[]>([]);
  const [selectedOwner, setSelectedOwner] = useState('');
  
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const [editMode, setEditMode] = useState<{ [key: string]: { status?: boolean; description?: boolean } }>({});
  const [tempEdits, setTempEdits] = useState<{ [key: string]: { status?: string; description?: string } }>({});

  const [page, setPage] = useState(1);
  const itemsPerPage = 5;
  const navigate = useNavigate();
  const [searchingSOPs, setSearchingSOPs] = useState<{ [key: string]: boolean }>({});
  const matches = useMediaQuery('(min-width:600px)');
  const isMobile = useMediaQuery('(max-width:600px)');

  const [newTaskStatus, setNewTaskStatus] = useState('open');

  const handlePageChange = (event: React.ChangeEvent<unknown>, value: number) => {
    setPage(value);
  };

  // Calculate pagination
  const totalPages = Math.ceil(todos.length / itemsPerPage);
  const startIndex = (page - 1) * itemsPerPage;
  const displayedTodos = isMobile ? todos : todos.slice(startIndex, startIndex + itemsPerPage);


  const handleExpand = (taskId: string) => {
    setExpandedId(expandedId === taskId ? null : taskId);
  };
  

  useEffect(() => {
    if (isAuthenticated && userId) {
      fetchTodos();
      fetchUsers().then(() => {
        setSelectedOwner(userId);
      });
    }
  }, [isAuthenticated, userId]);

  const fetchUsers = async () => {
    try {
      const response = await fetchWithAuth('/users');
      const data = response.data.users;
      console.log(data);
      setUsers(data);
    } catch (error) {
      console.error('Error fetching users:', error);
    }
  };

  const fetchTodos = async () => {
    if (!isAuthenticated || !userId) return;
    try {
      setLoading(true);
      const response = await fetchWithAuth(`/todos?userId=${userId}`);
      const data = response.data.todos;
      console.log(data);
      setTodos(data);
    } catch (error) {
      console.error('Error fetching todos:', error);
    } finally {
      setLoading(false);
    }
  };

  const addTodo = async () => {
    if (!isAuthenticated || !userId) return;
    
    if (input.trim() !== '' && selectedOwner) {
      try {
        const response = await fetchWithAuth('/todos', {
          method: 'POST',
          body: JSON.stringify({ 
            userId,
            taskOwner: selectedOwner,
            text: input.trim(),
            description: '',
            status: newTaskStatus
          }),
        });
        const newTodo = response.data;
        setTodos([...todos, newTodo]);
        setInput('');
        setNewTaskStatus('open');
      } catch (error) {
        console.error('Error adding todo:', error);
      }
    } else {
      alert('Please enter a task and select a task owner');
    }
  };

  const deleteTodo = async (taskId: string) => {
    if (!isAuthenticated || !userId) return;
    
    try {
      await fetchWithAuth(`/todos/${userId}/${taskId}`, {
        method: 'DELETE',
      });
      setTodos(prevTodos => prevTodos.filter(todo => todo.taskId !== taskId));
    } catch (error) {
      console.error('Error deleting todo:', error);
    }
  };


  const handleDescriptionChange = (taskId: string, newDescription: string) => {
    setTempEdits(prev => ({
      ...prev,
      [taskId]: { ...prev[taskId], description: newDescription }
    }));
  };

  const handleStatusChange = (taskId: string, newStatus: string) => {
    setTempEdits(prev => ({
      ...prev,
      [taskId]: { ...prev[taskId], status: newStatus }
    }));
  };

  // Add new functions to handle edit mode
  const toggleEditMode = (taskId: string, field: 'status' | 'description') => {
    setEditMode(prev => ({
      ...prev,
      [taskId]: { ...prev[taskId], [field]: !prev[taskId]?.[field] }
    }));
    
    // Find the correct todo item
    const todoItem = todos.find(t => t.taskId === taskId);
    
    // Initialize temp edits with current values
    setTempEdits(prev => ({
      ...prev,
      [taskId]: {
        ...prev[taskId],
        [field]: field === 'status' ? 
          todoItem?.status ?? 'open' :
          todoItem?.description ?? ''
      }
    }));
  };

  const handleSave = async (taskId: string, field: 'status' | 'description') => {
    const updates = {
      [field]: tempEdits[taskId]?.[field]
    };
    
    try {
      await updateTodo(taskId, updates);
      
      // Update todos directly instead of using separate state
      setTodos(prevTodos =>
        prevTodos.map(todo =>
          todo.taskId === taskId
            ? { ...todo, ...updates }
            : todo
        )
      );
      
      // Clear edit mode
      setEditMode(prev => ({
        ...prev,
        [taskId]: { ...prev[taskId], [field]: false }
      }));
    } catch (error) {
      console.error(`Error saving ${field}:`, error);
    }
  };

  // Combined update function for both description and status
  const updateTodo = async (taskId: string, updates: { description?: string; status?: string }) => {
    if (!isAuthenticated || !userId) return;
    try {
      const response = await fetchWithAuth(`/todos/${userId}/${taskId}`, {
        method: 'PATCH',
        body: JSON.stringify(updates),
      });
      
      setTodos(prevTodos =>
        prevTodos.map(todo =>
          todo.taskId === taskId
            ? { ...todo, ...updates }
            : todo
        )
      );
    } catch (error) {
      console.error('Error updating todo:', error);
    }
  };

  const handleSOPSearch = async (taskId: string, taskText: string) => {

    navigate('/assistant', { 
      replace: false, 
      state: { 
        query: "SOP for " + taskText + ". Do not include any background color in the response.",
        isInitialQuery: true
      } 
    });
  };

  return (
    <Box sx={{ 
      width: '100%', 
      height: '100%', 
      display: 'flex', 
      flexDirection: 'column',
      position: 'relative',
      flex: 1,
      overflow: 'hidden',
      mt: '56px', // Simplified top margin
      mb: 2,
      px: {
        xs: 1,
        sm: 2,
        md: 3
      },
      pb: {
        xs: 1,
        sm: 2
      }
    }}>
      {/* Mobile Header */}
      <MobileHeader 
        title="Task List" 
        showRefreshButton={false}
      />
      
      {isAuthenticated ? (
        <Paper 
          elevation={3} 
          sx={{ 
            p: { xs: 1.5, sm: 2 },
            flexGrow: 1,
            display: 'flex',
            flexDirection: 'column',
            borderRadius: {
              xs: 0,
              sm: '4px'
            },
            overflow: 'hidden'
          }}
        >
          <Box sx={{ display: 'flex', mb: 2 }}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Add a new task"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              sx={{ mr: 1 }}
            />
            <FormControl variant="outlined" sx={{ minWidth: 120 }} required>
              <InputLabel id="status-label">Status</InputLabel>
              <Select
                labelId="status-label"
                id="status-select"
                value={newTaskStatus}
                onChange={(e) => setNewTaskStatus(e.target.value)}
                label="Status"
              >
                <MenuItem value="open">Open</MenuItem>
                <MenuItem value="in_progress">In Progress</MenuItem>
                <MenuItem value="completed">Completed</MenuItem>
              </Select>
            </FormControl>
            <FormControl variant="outlined" sx={{ minWidth: 120 }} required>
              <InputLabel id="task-owner-label">Task Owner</InputLabel>
              <Select
                labelId="task-owner-label"
                value={selectedOwner || ''}
                onChange={(e) => setSelectedOwner(e.target.value as string)}
                label="Task Owner"
              >
                {users.length > 0 ? (
                  users.map((user) => (
                    <MenuItem key={user.userId} value={user.userId}>
                      {user.userId} ({user.userRole})
                    </MenuItem>
                  ))
                ) : (
                  <MenuItem value="">No users available</MenuItem>
                )}
              </Select>
            </FormControl>
          </Box>
          <Button
            variant="contained"
            color="primary"
            fullWidth
            startIcon={<AddIcon />}
            onClick={addTodo}
            disabled={!input.trim() || !selectedOwner}
          >
            Add Task
          </Button>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
              <CircularProgress />
            </Box>
          ) : (
            <Box sx={{ 
              flexGrow: 1,
              overflowY: 'auto',
              WebkitOverflowScrolling: 'touch',
              display: 'flex',
              flexDirection: 'column'
            }}>
              <List sx={{ 
                mt: 2, 
                flexGrow: 1, 
                overflow: isMobile ? 'auto' : 'visible',
                WebkitOverflowScrolling: isMobile ? 'touch' : 'auto',
                maxHeight: isMobile ? '100%' : 'auto'
              }}>
                {displayedTodos.map((todo) => (
                  <React.Fragment key={todo.taskId}>
                    <ListItem
                      onClick={() => handleExpand(todo.taskId)}
                      sx={{ 
                        cursor: 'pointer',
                        '&:hover': { 
                          backgroundColor: 'rgba(0, 0, 0, 0.04)' 
                        }
                      }}
                      secondaryAction={
                        <Box onClick={(e) => e.stopPropagation()}>
                          <IconButton 
                            edge="end" 
                            onClick={() => handleExpand(todo.taskId)}
                          >
                            {expandedId === todo.taskId ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                          </IconButton>
                        </Box>
                      }
                    >
                      <ListItemText 
                        primary={todo.text} 
                        secondary={
                          <Typography
                            component="span"
                            variant="body2"
                            sx={{ display: 'flex', alignItems: 'center', gap: 1 }}
                          >
                            <Box
                              component="span"
                              sx={{
                                width: 8,
                                height: 8,
                                borderRadius: '50%',
                                bgcolor: getStatusColor(todo.status ?? 'open'),
                                display: 'inline-block'
                              }}
                            />
                            {`Owner: ${todo.taskOwner} • Status: ${
                              (todo.status ?? 'open')
                                .replace('_', ' ')
                                .replace(/\b\w/g, l => l.toUpperCase())
                            }`}
                          </Typography>
                        }
                      />
                    </ListItem>
                    <Collapse in={expandedId === todo.taskId} timeout="auto" unmountOnExit>
                      <Box sx={{ pl: 4, pr: 4, pb: 2 }}>
                        <Box sx={{ display: 'flex', gap: 2, mb: 2, alignItems: 'center' }}>
                          <FormControl fullWidth variant="outlined">
                            <InputLabel id={`status-label-${todo.taskId}`}>Status</InputLabel>
                            <Select
                              labelId={`status-label-${todo.taskId}`}
                              value={editMode[todo.taskId]?.status ?
                                (tempEdits[todo.taskId]?.status || todo.status || 'open') :
                                (todo.status ?? 'open')
                              }
                              onChange={(e) => handleStatusChange(todo.taskId, e.target.value)}
                              label="Status"
                              disabled={!editMode[todo.taskId]?.status}
                            >
                              <MenuItem value="open">Open</MenuItem>
                              <MenuItem value="in_progress">In Progress</MenuItem>
                              <MenuItem value="completed">Completed</MenuItem>
                            </Select>
                          </FormControl>
                          {editMode[todo.taskId]?.status ? (
                            <IconButton onClick={() => handleSave(todo.taskId, 'status')} color="primary">
                              <SaveIcon />
                            </IconButton>
                          ) : (
                            <IconButton onClick={() => toggleEditMode(todo.taskId, 'status')}>
                              <EditIcon />
                            </IconButton>
                          )}
                        </Box>
                        <Box sx={{ display: 'flex', gap: 2, mb: 2, alignItems: 'flex-start' }}>
                          <TextField
                            fullWidth
                            multiline
                            label="Description"
                            rows={3}
                            variant="outlined"
                            placeholder="Add a description (optional)"
                            value={editMode[todo.taskId]?.description ?
                              (tempEdits[todo.taskId]?.description || todo.description || '') :
                              (todo.description ?? '')
                            }
                            onChange={(e) => handleDescriptionChange(todo.taskId, e.target.value)}
                            disabled={!editMode[todo.taskId]?.description}
                          />
                          {editMode[todo.taskId]?.description ? (
                            <IconButton onClick={() => handleSave(todo.taskId, 'description')} color="primary">
                              <SaveIcon />
                            </IconButton>
                          ) : (
                            <IconButton onClick={() => toggleEditMode(todo.taskId, 'description')}>
                              <EditIcon />
                            </IconButton>
                          )}
                        </Box>

                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                          <Button
                            variant="outlined"
                            startIcon={searchingSOPs[todo.taskId] ? (
                              <CircularProgress size={20} color="inherit" />
                            ) : (
                              <SearchIcon />
                            )}
                            onClick={() => handleSOPSearch(todo.taskId, todo.text)}
                            disabled={searchingSOPs[todo.taskId]}
                            fullWidth
                          >
                            {searchingSOPs[todo.taskId] ? 'Searching...' : 'SOP for this task'}
                          </Button>
                          <Button
                            variant="outlined"
                            color="error"
                            startIcon={<DeleteIcon />}
                            disabled={searchingSOPs[todo.taskId]}
                            onClick={() => deleteTodo(todo.taskId)}
                            fullWidth
                          >
                            Delete Task
                          </Button>
                        </Box>
                      </Box>
                    </Collapse>
                  </React.Fragment>
                ))}
              </List>
              {!isMobile && totalPages > 1 && (
                <Box sx={{ 
                  position: 'sticky',
                  bottom: 0,
                  py: 2,
                  borderTop: 1,
                  borderColor: 'divider',
                  display: 'flex',
                  justifyContent: 'center'
                }}>
                  <Pagination 
                    count={totalPages}
                    page={page}
                    onChange={handlePageChange}
                    color="primary"
                    showFirstButton 
                    showLastButton
                    size="medium"
                  />
                </Box>
              )}
            </Box>
          )}
        </Paper>
      ) : (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
          <Typography variant="h6">Please log in to view and manage tasks.</Typography>
        </Box>
      )}
    </Box>
  );
};

export default TaskList;
