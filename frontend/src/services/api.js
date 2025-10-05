import axios from 'axios';

// Dynamic backend URL: use relative path when accessed from preview, localhost otherwise
const getBackendURL = () => {
  // If accessed from preview domain, use relative URLs (proxy will route to backend)
  if (window.location.hostname.includes('preview.emergentagent.com')) {
    return window.location.origin; // Same origin, proxy routes /api/* to backend
  }
  // Otherwise use localhost
  return import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001';
};

const BACKEND_URL = getBackendURL();

const api = axios.create({
  baseURL: BACKEND_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const problemAPI = {
  create: (data) => api.post('/api/problem/create', data),
  get: (id) => api.get(`/api/problem/${id}`),
  list: () => api.get('/api/problem/'),
};

export const analysisAPI = {
  analyze: (problemId, problemInput) => 
    api.post(`/api/analysis/analyze/${problemId}`, problemInput),
  get: (problemId) => api.get(`/api/analysis/${problemId}`),
  generateCode: (problemId, codeType) => 
    api.post(`/api/analysis/generate-code/${problemId}?code_type=${codeType}`),
};

export const evolutionAPI = {
  configure: (problemId, userConfig) =>
    api.post(`/api/evolution/configure/${problemId}`, userConfig),
  start: (sessionId) =>
    api.post(`/api/evolution/start/${sessionId}`),
  getStatus: (sessionId) =>
    api.get(`/api/evolution/status/${sessionId}`),
};

export const healthAPI = {
  check: () => api.get('/api/health'),
};

export default api;