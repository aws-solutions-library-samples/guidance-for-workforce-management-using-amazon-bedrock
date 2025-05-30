import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { Paper, Typography, Box, Button, IconButton } from '@mui/material';
import { fetchWithAuth } from '../config';
import { useAuth } from '../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import useMediaQuery from '@mui/material/useMediaQuery';
import CircularProgress from '@mui/material/CircularProgress';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import MobileHeader from '../components/MobileHeader';

import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Pie } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

// Add interface for feedback data
interface Feedback {
  userId: string;
  session_id: string;
  message: string;
  feedback: 'up' | 'down';
  timestamp: string;
}

interface KPIData {
  value: number;
  target: number;
  unit: string;
  achievement_percentage: number;
}

interface DepartmentData {
  aisle: string | number;
  bestsellers: string[];
  revenue: string;
  market_share: string;
}

interface InventoryAlert {
  item: string;
  units_left: number;
  status: string;
  reorder_point: number;
}

interface Recommendation {
  department: string;
  aisle: string | number;
  action: string;
  expected_impact: string;
}

interface StoreDataFormat {
  kpi_data: {
    [key: string]: {
      value: number;
      target: number;
      unit: string;
      achievement_percentage: number;
    };
  };
  department_data: {
    [key: string]: {
      aisle: string | number;
      bestsellers: string[];
      revenue: string;
      market_share: string;
    };
  };
  inventory_alerts: {
    [key: string]: {
      item: string;
      units_left: number;
      status: string;
      reorder_point: number;
    };
  };
  recommendations: Array<{
    department: string;
    aisle: string | number;
    action: string;
    expected_impact: string;
  }>;
  // Make these optional since we don't use them in the UI
  upcoming_events?: {
    holiday_sales: {
      event: string;
      dates: string;
      recommended_discounts: {
        [key: string]: string;
      };
    };
  };
  staff_training_recommendations?: Array<{
    department: string;
    focus: string;
    duration: string;
  }>;
}


const Analytics: React.FC = () => {
  const [isGenerating, setIsGenerating] = useState(false);
  const navigate = useNavigate();
  const { userId } = useAuth();
  const [feedbacks, setFeedbacks] = useState<Feedback[]>([]);
  const [feedbackLoading, setFeedbackLoading] = useState(true);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  const [storeData, setStoreData] = useState<StoreDataFormat | null>(null);
  const [storeLoading, setStoreLoading] = useState(true);
  const [storeError, setStoreError] = useState<string | null>(null);
  const [generingError, setGeneratingError] = useState<string | null>(null);

  const matches = useMediaQuery('(min-width:600px)');

  // Generate a session ID if needed
  const sessionId = useRef(localStorage.getItem('sessionId') || crypto.randomUUID()).current;
  
  useEffect(() => {
    localStorage.setItem('sessionId', sessionId);
  }, [sessionId]);

  useEffect(() => {
    const fetchFeedbacks = async () => {
      try {
        const response = await fetchWithAuth('/feedbacks');
        
        // Ensure we're getting the correct data structure
        const feedbacksArray = Array.isArray(response.data) 
        ? response.data 
        : Array.isArray(response.data?.feedbacks)
          ? response.data.feedbacks
          : [];
      
        setFeedbacks(feedbacksArray);
      } catch (err) {
        console.error('Error fetching feedbacks:', err);
        setFeedbackError(err instanceof Error ? err.message : 'An error occurred');
        setFeedbacks([]); // Set empty array on error
      } finally {
        setFeedbackLoading(false);
      }
    };
  
    fetchFeedbacks();
  }, []);

  useEffect(() => {
    const fetchStoreData = async () => {
      try {
        const response = await fetchWithAuth('/store');
        
        // Get the raw response
        const parsedData = response.data;
        console.log('Raw response:', parsedData);
        
        // Validate the data
        if (!parsedData || typeof parsedData !== 'object') {
          throw new Error('Invalid data format');
        }

        const requiredFields = ['kpi_data', 'department_data', 'inventory_alerts', 'recommendations'];
        const missingFields = requiredFields.filter(field => !parsedData[field]);
        
        if (missingFields.length > 0) {
          throw new Error(`Missing required fields: ${missingFields.join(', ')}`);
        }

        setStoreData(parsedData as StoreDataFormat);
      } catch (err) {
        console.error('Store data fetch error:', err);
        setStoreError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setStoreLoading(false);
      }
    };

    fetchStoreData();
  }, []);

  // Prepare data for pie chart
  const feedbackChartData = {
    labels: ['Positive Feedback', 'Negative Feedback'],
    datasets: [{
      data: Array.isArray(feedbacks) && feedbacks.length > 0
      ? [
          feedbacks.filter(f => f.feedback === 'up').length,
          feedbacks.filter(f => f.feedback === 'down').length
        ]
      : [0, 0], // Default values for empty or invalid array
      backgroundColor: ['#4CAF50', '#f44336'],
      borderColor: ['#388E3C', '#D32F2F'],
      borderWidth: 1,
    }],
  };


  const handleGenerateTasks = async () => {
    if (!userId) {
      console.error('No user ID available');
      return;
    }

    // Add null check for storeData
    if (!storeData?.recommendations) {
      throw new Error('Store data not available');
    }
    
    const taskText = `Draft tasks for available staff based on the below recommendations. Skip any preamble, only include actual response

    Recommendations: ${storeData.recommendations.map(rec => `${rec.department} (Aisle ${rec.aisle}): ${rec.action}`).join('\n')}

    But do not create tasks. Instead return the task list for review to the user in an markdown list, grouped by task_owner and focus on high-level tasks, following this structure:

    ### Task List:

    - Assigned to [task_owner]:
        - Task 1: [task_name]
        - Task 2: [task_name]
        - Task 3: [task_name]
    - Assigned to [task_owner]:
        - Task 1: [task_name]
        - Task 2: [task_name]
        - Task 3: [task_name]

    Ensure that the task list is assigned to actual users.`;

      navigate('/assistant', { 
        replace: false, 
        state: { 
          query: taskText,
          isInitialQuery: true
        } 
      });
  }

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
      <MobileHeader 
        title="Analytics" 
        showBackButton={true}
        showRefreshButton={true}
      />
      
      <Box 
        component="div"
        sx={{ 
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflowY: 'scroll',
          px: { xs: 1.5, sm: 2 },
          py: 2,
          WebkitOverflowScrolling: 'touch'
        }}
      >
        <Box sx={{ 
          display: 'flex',
          flexDirection: 'column',
          gap: 4,
          width: '100%'
        }}>
          {storeLoading && (
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              width: '100%' 
            }}>
              <CircularProgress />
            </Box>
          )}
          {storeError && <Typography color="error">{storeError}</Typography>}
          {!storeLoading && !storeError && storeData && (
            <Box sx={{ width: '100%' }}>
              {/* KPIs Section */}
              <Typography variant="h6" sx={{ mb: 2 }}>Key Performance Indicators</Typography>
              <Box sx={{ 
                display: 'grid', 
                gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: '1fr 1fr 1fr' },
                gap: 2,
                mb: 4
              }}>
                {Object.entries(storeData.kpi_data).map(([key, data]) => (
                  <Paper
                    key={key}
                    elevation={2}
                    sx={{
                      p: 2,
                      borderRadius: 1,
                      backgroundColor: data.achievement_percentage >= 95 
                        ? '#81c784' // Green
                        : data.achievement_percentage >= 90
                          ? '#ffb74d' // Orange
                          : '#e57373', // Red
                      transition: 'background-color 0.3s ease',
                      '&:hover': {
                        backgroundColor: data.achievement_percentage >= 95
                          ? '#66bb6a' 
                          : data.achievement_percentage >= 90
                            ? '#ffa726'
                            : '#ef5350',
                      },
                      color: '#ffffff', // White text for better contrast
                      backgroundImage: 'none',  // Remove any default gradient
                      backdropFilter: 'none',   // Remove any backdrop filter
                    }}
                  >
                    <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.9)' }}>
                      {key}
                    </Typography>
                    <Typography variant="h6" sx={{ mt: 1, color: '#ffffff', fontWeight: 'bold' }}>
                      {data.unit}{data.value.toLocaleString()}
                    </Typography>
                    <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.85)' }}>
                      Target: {data.unit}{data.target.toLocaleString()} ({data.achievement_percentage}%)
                    </Typography>
                  </Paper>
                ))}
              </Box>

              {/* Department Performance */}
              <Typography variant="h6" sx={{ mb: 2, mt: 4 }}>Department Performance</Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr 1fr' }, gap: 2, mb: 4 }}>
                {Object.entries(storeData.department_data).map(([dept, data]) => (
                  <Paper key={dept} elevation={2} sx={{ 
                    p: 2,
                    height: '100%',  // Ensure consistent height
                    display: 'flex',
                    flexDirection: 'column'
                  }}>
                    <Typography variant="subtitle1" color="primary" noWrap>{dept}</Typography>
                    <Typography variant="body2">Aisle: {data.aisle}</Typography>
                    <Typography variant="body2">Revenue: {data.revenue}</Typography>
                    <Typography variant="body2">Market Share: {data.market_share}</Typography>
                    <Typography variant="body2" sx={{ mt: 1 }}>Top Sellers:</Typography>
                    <Box sx={{ overflow: 'auto', maxHeight: '100px' }}> {/* Scrollable area for long lists */}
                      <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                        {data.bestsellers.map((item, i) => (
                          <li key={i}>
                            <Typography variant="body2" noWrap>{item}</Typography>
                          </li>
                        ))}
                      </ul>
                    </Box>
                  </Paper>
                ))}
              </Box>

              {/* Inventory Alerts */}
              <Typography variant="h6" sx={{ mb: 2, mt: 4 }}>Inventory Alerts</Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr 1fr' }, gap: 2, mb: 4 }}>
                {Object.entries(storeData.inventory_alerts).map(([dept, alert]) => (
                  <Paper 
                    key={dept} 
                    elevation={2} 
                    sx={{ 
                      p: 2,
                      backgroundColor: alert.status === 'Urgent reorder' 
                        ? '#ef5350' // Much more saturated red
                        : '#ffa726', // Much more saturated orange
                      transition: 'background-color 0.3s ease',
                      '&:hover': {
                        backgroundColor: alert.status === 'Urgent reorder'
                          ? '#e53935' // Even darker red on hover
                          : '#fb8c00', // Even darker orange on hover
                      },
                      backgroundImage: 'none',
                      backdropFilter: 'none',
                      // Add these to ensure color vibrancy
                      border: 'none',
                      boxShadow: (theme) => `0 2px 4px ${
                        alert.status === 'Urgent reorder' 
                          ? 'rgba(239, 83, 80, 0.25)'
                          : 'rgba(255, 167, 38, 0.25)'
                      }`,
                      // Adjust text colors for better contrast
                      '& .MuiTypography-root': {
                        color: '#fff'  // White text for better contrast
                      },
                      '& .MuiTypography-primary': {
                        color: '#fff',  // Keep header text white
                        fontWeight: 'bold'
                      }
                    }}
                  >
                    <Typography variant="subtitle1">{dept}</Typography>
                    <Typography variant="body1">{alert.item}</Typography>
                    <Typography variant="body2" sx={{ 
                      color: '#fff !important',  // Force white color
                      opacity: 0.9  // Slightly dimmed for hierarchy
                    }}>
                      Units Left: {alert.units_left} (Reorder at: {alert.reorder_point})
                    </Typography>
                    <Typography variant="body2" sx={{ 
                      fontWeight: 'bold', 
                      mt: 1,
                      color: '#fff !important'  // Force white color
                    }}>
                      Status: {alert.status}
                    </Typography>
                  </Paper>
                ))}
              </Box>

              {/* Store Recommendations */}
              <Box>
                <Typography variant="h6" sx={{ mb: 2 }}>Store Recommendations</Typography>
                <Box sx={{ 
                  display: 'grid', 
                  gridTemplateColumns: { 
                    xs: '1fr',     // 1 column on mobile
                    md: '1fr 1fr'  // 2 columns on desktop
                  },
                  gap: 2
                }}>
                  {storeData.recommendations.map((rec, index) => (
                    <Paper key={index} elevation={1} sx={{ 
                      p: 2,
                      height: '100%',
                      display: 'flex',
                      flexDirection: 'column'
                    }}>
                      <Typography variant="subtitle1" color="primary" sx={{ mb: 1 }}>
                        {rec.department} (Aisle {rec.aisle})
                      </Typography>
                      <Box sx={{ flex: 1 }}> {/* Take remaining space */}
                        <Typography variant="body1">
                          {rec.action}
                        </Typography>
                        <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                          Expected Impact: {rec.expected_impact}
                        </Typography>
                      </Box>
                    </Paper>
                  ))}
                </Box>
              </Box>
              <Box sx={{ 
                mt: 4, 
                display: 'flex', 
                justifyContent: 'center'
              }}>
                <Button 
                  variant="contained"
                  color="primary"
                  onClick={handleGenerateTasks}
                  disabled={isGenerating || !userId}
                  startIcon={isGenerating ? <CircularProgress size={20} /> : <AutoAwesomeIcon />}
                  sx={{ width: { xs: '100%', sm: 'auto' } }}  // Full width on mobile
                >
                  Generate tasks based on recommendations
                </Button> 
              </Box>
            </Box>
          )}

          <Box sx={{ 
            display: 'flex',
            flexDirection: 'column', 
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto', 
            width: '100%',
            mt: 4
          }}>
            <Typography variant="h6" sx={{ mb: 4 }}>User Feedback Analytics</Typography>
            {feedbackLoading && <CircularProgress />}
            {feedbackError && <Typography color="error">{feedbackError}</Typography>}
            {!feedbackLoading && !feedbackError && (
              <Box sx={{ 
                width: '100%', 
                maxWidth: 400,
                margin: '0 auto'  // Center the chart
              }}>
                {feedbacks.length === 0 ? (
                  <Typography variant="body1" align="center" color="textSecondary">
                    No feedback data available yet
                  </Typography>
                ) : (
                  <Pie data={feedbackChartData} />
                )}
              </Box>
            )}
          </Box>
        </Box>
      </Box>
    </Box>
  );
};

export default Analytics;