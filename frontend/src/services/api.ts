import axios from 'axios';

// Set base URL for API requests
const API_BASE_URL = 'http://localhost:8000'; // Update with your backend URL

// Create axios instance with base configuration
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API functions
export const analyzeStartup = async (startupName: string, queryMode = 'hybrid', enrichContent = false) => {
  try {
    const response = await api.post('/analyze-startup', {
      startup_name: startupName,
      query_mode: queryMode,
      enrich_content: enrichContent
    });
    return response.data;
  } catch (error) {
    console.error('Error analyzing startup:', error);
    throw error;
  }
};

export const getStartupProfile = async (startupName: string) => {
  try {
    const response = await api.get(`/startup/${encodeURIComponent(startupName)}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching startup profile:', error);
    throw error;
  }
};

export const saveStartupNotes = async (startupName: string, notes: string) => {
  try {
    const response = await api.post(`/startup/${encodeURIComponent(startupName)}/notes`, {
      notes
    });
    return response.data;
  } catch (error) {
    console.error('Error saving startup notes:', error);
    throw error;
  }
};

export default api;