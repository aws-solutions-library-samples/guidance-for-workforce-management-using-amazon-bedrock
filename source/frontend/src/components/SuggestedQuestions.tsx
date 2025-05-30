import React from 'react';
import { Box, Paper, Typography } from '@mui/material';

interface SuggestedQuestion {
  text: string;
}

interface SuggestedQuestionsProps {
  questions: SuggestedQuestion[];
  onQuestionClick: (question: string) => void;
}

const SuggestedQuestions: React.FC<SuggestedQuestionsProps> = ({ 
  questions, 
  onQuestionClick 
}) => {
  return (
    <Box sx={{ 
      display: 'flex', 
      flexWrap: 'wrap', 
      gap: 1.5,
      mt: 2,
      mb: 2,
      px: 1
    }}>
      {questions.map((question, index) => (
        <Paper
          key={index}
          elevation={1}
          onClick={() => onQuestionClick(question.text)}
          sx={{
            p: 1.5,
            borderRadius: 2,
            cursor: 'pointer',
            flex: '1 0 calc(50% - 12px)',
            minWidth: '150px',
            maxWidth: 'calc(50% - 12px)',
            border: '1px solid rgba(0, 0, 0, 0.12)',
            '&:hover': {
              backgroundColor: 'rgba(0, 0, 0, 0.04)'
            }
          }}
        >
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            {question.text}
          </Typography>
        </Paper>
      ))}
    </Box>
  );
};

export default SuggestedQuestions; 