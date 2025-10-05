import { useState, useEffect, useRef } from 'react';

/**
 * Custom hook for evolution WebSocket connection
 */
export function useEvolutionWebSocket(sessionId) {
  const [isConnected, setIsConnected] = useState(false);
  const [evolutionState, setEvolutionState] = useState({
    status: 'connecting',
    generation: 0,
    bestFitness: 0,
    avgFitness: 0,
    diversity: 0,
    history: [],
    islands: {},
    programs: [],
    error: null,
  });

  const ws = useRef(null);

  useEffect(() => {
    if (!sessionId) return;

    // Construct WebSocket URL with dynamic detection (same logic as api.js)
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    let host;
    
    // If accessed from preview domain, use same host (proxy will route)
    if (window.location.hostname.includes('preview.emergentagent.com')) {
      host = window.location.host;
    } else {
      // Otherwise use localhost
      host = 'localhost:8001';
    }
    
    const wsUrl = `${protocol}//${host}/api/evolution/ws/${sessionId}`;

    console.log('🔌 Connecting to WebSocket:', wsUrl);

    // Create WebSocket connection
    ws.current = new WebSocket(wsUrl);

    ws.current.onopen = () => {
      console.log('✅ WebSocket connected');
      setIsConnected(true);
      setEvolutionState(prev => ({ ...prev, status: 'connected' }));
    };

    ws.current.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        console.log('📨 WebSocket message:', message.type, message);

        handleMessage(message);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    ws.current.onerror = (error) => {
      console.error('❌ WebSocket error:', error);
      setEvolutionState(prev => ({
        ...prev,
        status: 'error',
        error: 'WebSocket connection error'
      }));
    };

    ws.current.onclose = () => {
      console.log('🔌 WebSocket disconnected');
      setIsConnected(false);
      setEvolutionState(prev => ({ ...prev, status: 'disconnected' }));
    };

    // Cleanup on unmount
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [sessionId]);

  const handleMessage = (message) => {
    switch (message.type) {
      case 'connected':
        setEvolutionState(prev => ({ ...prev, status: 'connected' }));
        break;

      case 'evolution_start':
        setEvolutionState(prev => ({ ...prev, status: 'running' }));
        break;

      case 'generation_complete':
        setEvolutionState(prev => ({
          ...prev,
          status: 'running',
          generation: message.generation,
          bestFitness: message.best_fitness,
          avgFitness: message.avg_fitness,
          diversity: message.diversity,
          programs: message.programs || [],
          history: [
            ...prev.history,
            {
              generation: message.generation,
              bestFitness: message.best_fitness,
              avgFitness: message.avg_fitness,
              diversity: message.diversity,
              timestamp: message.timestamp,
            }
          ]
        }));
        break;

      case 'islands_update':
        setEvolutionState(prev => ({
          ...prev,
          islands: message.islands || {}
        }));
        break;

      case 'evolution_complete':
        setEvolutionState(prev => ({
          ...prev,
          status: 'completed',
          bestSolution: message.best_solution
        }));
        break;

      case 'evolution_error':
        setEvolutionState(prev => ({
          ...prev,
          status: 'error',
          error: message.error || message.message
        }));
        break;

      case 'pong':
        // Heartbeat response
        break;

      case 'heartbeat':
        // Server heartbeat - connection is alive, no action needed
        console.log('💓 Heartbeat received');
        break;

      default:
        console.warn('Unknown message type:', message.type);
    }
  };

  const sendMessage = (message) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    }
  };

  return {
    isConnected,
    evolutionState,
    sendMessage,
  };
}
