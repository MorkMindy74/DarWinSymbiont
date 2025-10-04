import axios from 'axios';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001';

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

export const healthAPI = {
  check: () => api.get('/api/health'),
};

export default api;