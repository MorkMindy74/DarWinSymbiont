import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Loader, CheckCircle, AlertCircle, Play, Pause } from 'lucide-react';
import { useEvolutionWebSocket } from '../hooks/useEvolutionWebSocket';
import { evolutionAPI, problemAPI } from '../services/api';
import toast, { Toaster } from 'react-hot-toast';

import MetricsBar from '../components/Evolution/MetricsBar';
import FitnessChart from '../components/Evolution/FitnessChart';
import IslandMap from '../components/Evolution/IslandMap';
import LiveFeed from '../components/Evolution/LiveFeed';

function EvolutionDashboard() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [problem, setProblem] = useState(null);
  const [sessionInfo, setSessionInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  const { isConnected, evolutionState, sendMessage } = useEvolutionWebSocket(sessionId);

  useEffect(() => {
    loadSessionInfo();
  }, [sessionId]);

  const loadSessionInfo = async () => {
    try {
      const response = await evolutionAPI.getStatus(sessionId);
      setSessionInfo(response.data);

      // Load problem info
      const problemResponse = await problemAPI.get(response.data.problem_id);
      setProblem(problemResponse.data.problem);
    } catch (error) {
      console.error('Failed to load session info:', error);
      toast.error('Failed to load session information');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = () => {
    switch (evolutionState.status) {
      case 'running':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'completed':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'error':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getStatusIcon = () => {
    switch (evolutionState.status) {
      case 'running':
        return <Loader className="w-5 h-5 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5" />;
      case 'error':
        return <AlertCircle className="w-5 h-5" />;
      default:
        return <Pause className="w-5 h-5" />;
    }
  };

  const getStatusText = () => {
    switch (evolutionState.status) {
      case 'connecting':
        return 'Connecting...';
      case 'connected':
        return 'Connected - Waiting';
      case 'running':
        return 'Evolution Running';
      case 'completed':
        return 'Evolution Complete';
      case 'error':
        return 'Error';
      default:
        return 'Unknown';
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <Loader className="w-12 h-12 text-primary-600 animate-spin" />
        <p className="text-lg text-gray-600">Loading evolution session...</p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <Toaster position="top-right" />

      {/* Header */}
      <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center space-x-3 mb-2">
              <div className={`px-3 py-1 rounded-full border font-medium flex items-center space-x-2 ${getStatusColor()}`}>
                {getStatusIcon()}
                <span>{getStatusText()}</span>
              </div>
              {isConnected ? (
                <div className="flex items-center space-x-1 text-green-600">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-sm font-medium">WebSocket Connected</span>
                </div>
              ) : (
                <div className="flex items-center space-x-1 text-gray-500">
                  <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                  <span className="text-sm font-medium">Disconnected</span>
                </div>
              )}
            </div>
            
            {problem && (
              <>
                <h1 className="text-2xl font-bold mb-1">{problem.problem_input.title}</h1>
                <p className="text-gray-600">
                  Type: <span className="font-medium">{problem.problem_input.problem_type.toUpperCase()}</span>
                  {' • '}
                  Session: <span className="font-mono text-sm">{sessionId.slice(0, 8)}...</span>
                </p>
              </>
            )}
          </div>

          <button
            onClick={() => navigate('/')}
            className="btn-secondary"
          >
            Back to Home
          </button>
        </div>

        {evolutionState.error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-800 font-medium">Error: {evolutionState.error}</p>
          </div>
        )}
      </div>

      {/* Metrics Bar */}
      <MetricsBar
        generation={evolutionState.generation}
        bestFitness={evolutionState.bestFitness}
        avgFitness={evolutionState.avgFitness}
        diversity={evolutionState.diversity}
      />

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Charts */}
        <div className="lg:col-span-2 space-y-6">
          <FitnessChart history={evolutionState.history} />
          <IslandMap islands={evolutionState.islands} />
        </div>

        {/* Right Column - Live Feed */}
        <div className="lg:col-span-1">
          <LiveFeed history={evolutionState.history} />
        </div>
      </div>

      {/* Best Solution Preview */}
      {evolutionState.bestSolution && (
        <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
          <h3 className="text-lg font-semibold mb-4">🏆 Best Solution Found</h3>
          <div className="grid md:grid-cols-2 gap-4 mb-4">
            <div className="border border-gray-200 rounded-lg p-4">
              <span className="text-sm text-gray-600">Fitness</span>
              <p className="text-2xl font-bold text-green-600">
                {evolutionState.bestSolution.fitness?.toFixed(2)}
              </p>
            </div>
            <div className="border border-gray-200 rounded-lg p-4">
              <span className="text-sm text-gray-600">Generation</span>
              <p className="text-2xl font-bold text-blue-600">
                {evolutionState.bestSolution.generation}
              </p>
            </div>
          </div>
          
          <button
            onClick={() => navigate(`/results/${sessionId}`)}
            className="btn-primary w-full"
          >
            View Complete Results →
          </button>
        </div>
      )}

      {/* Completion Message */}
      {evolutionState.status === 'completed' && (
        <div className="bg-gradient-to-r from-green-500 to-emerald-600 rounded-2xl p-8 text-white text-center shadow-xl">
          <CheckCircle className="w-16 h-16 mx-auto mb-4" />
          <h2 className="text-3xl font-bold mb-2">Evolution Complete! 🎉</h2>
          <p className="text-lg mb-6 opacity-90">
            Successfully completed {evolutionState.generation} generations
          </p>
          <button
            onClick={() => navigate(`/results/${sessionId}`)}
            className="bg-white text-green-600 hover:bg-gray-100 font-semibold py-3 px-8 rounded-lg transition transform hover:scale-105 shadow-lg"
          >
            View Final Results
          </button>
        </div>
      )}
    </div>
  );
}

export default EvolutionDashboard;
