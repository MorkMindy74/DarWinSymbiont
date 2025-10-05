import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Loader, Download, CheckCircle, TrendingDown, Code, ChevronDown, ChevronUp } from 'lucide-react';
import { evolutionAPI, problemAPI } from '../services/api';
import toast, { Toaster } from 'react-hot-toast';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

function Results() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [sessionInfo, setSessionInfo] = useState(null);
  const [problem, setProblem] = useState(null);
  const [showBestCode, setShowBestCode] = useState(false);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    loadResults();
  }, [sessionId]);

  const loadResults = async () => {
    try {
      // Load session info
      const response = await evolutionAPI.getStatus(sessionId);
      setSessionInfo(response.data);

      // Load problem info
      if (response.data.problem_id) {
        const problemResponse = await problemAPI.get(response.data.problem_id);
        setProblem(problemResponse.data.problem);
      }
    } catch (error) {
      console.error('Failed to load results:', error);
      toast.error('Failed to load evolution results');
    } finally {
      setLoading(false);
    }
  };

  const exportResults = async (format) => {
    setExporting(true);
    try {
      const data = {
        session_id: sessionId,
        problem: problem,
        runtime: sessionInfo.runtime,
        best_solution: sessionInfo.runtime?.best_solution,
        configuration: sessionInfo.user_config,
        timestamp: new Date().toISOString()
      };

      if (format === 'json') {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `evolution-results-${sessionId.slice(0, 8)}.json`;
        a.click();
        URL.revokeObjectURL(url);
        toast.success('Results exported as JSON');
      } else if (format === 'csv') {
        // Export simplified CSV
        const csvContent = [
          ['Metric', 'Value'],
          ['Session ID', sessionId],
          ['Problem Type', problem?.problem_input?.problem_type || 'N/A'],
          ['Problem Title', problem?.problem_input?.title || 'N/A'],
          ['Total Generations', sessionInfo.user_config?.num_generations || 'N/A'],
          ['Best Fitness', sessionInfo.runtime?.best_solution?.fitness || 'N/A'],
          ['Final Generation', sessionInfo.runtime?.best_solution?.generation || 'N/A']
        ].map(row => row.join(',')).join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `evolution-results-${sessionId.slice(0, 8)}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        toast.success('Results exported as CSV');
      }
    } catch (error) {
      console.error('Export failed:', error);
      toast.error('Failed to export results');
    } finally {
      setExporting(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <Loader className="w-12 h-12 text-primary-600 animate-spin" />
        <p className="text-lg text-gray-600">Loading results...</p>
      </div>
    );
  }

  const bestSolution = sessionInfo?.runtime?.best_solution;
  const chartData = sessionInfo?.runtime?.history || [];

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <Toaster position="top-right" />

      {/* Header */}
      <div className="bg-gradient-to-r from-green-500 to-emerald-600 rounded-2xl p-8 text-white shadow-xl">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center space-x-3 mb-4">
              <CheckCircle className="w-10 h-10" />
              <h1 className="text-3xl font-bold">Evolution Complete!</h1>
            </div>
            {problem && (
              <>
                <h2 className="text-xl mb-2">{problem.problem_input.title}</h2>
                <p className="text-green-100">
                  Type: <span className="font-medium">{problem.problem_input.problem_type.toUpperCase()}</span>
                  {' • '}
                  Session: <span className="font-mono text-sm">{sessionId.slice(0, 8)}...</span>
                </p>
              </>
            )}
          </div>

          <button
            onClick={() => navigate('/')}
            className="bg-white text-green-600 hover:bg-gray-100 font-semibold py-2 px-6 rounded-lg transition"
          >
            Back to Home
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-600">Total Generations</span>
            <TrendingDown className="w-5 h-5 text-blue-500" />
          </div>
          <p className="text-3xl font-bold text-gray-900">
            {sessionInfo?.user_config?.num_generations || 'N/A'}
          </p>
        </div>

        <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-600">Best Fitness</span>
            <CheckCircle className="w-5 h-5 text-green-500" />
          </div>
          <p className="text-3xl font-bold text-gray-900">
            {bestSolution?.fitness?.toFixed(4) || '0.0000'}
          </p>
        </div>

        <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <span className="text-gray-600">Found at Generation</span>
            <Code className="w-5 h-5 text-purple-500" />
          </div>
          <p className="text-3xl font-bold text-gray-900">
            {bestSolution?.generation || '0'}
          </p>
        </div>
      </div>

      {/* Export Section */}
      <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Download className="w-5 h-5 mr-2" />
          Export Results
        </h3>
        <div className="flex space-x-4">
          <button
            onClick={() => exportResults('json')}
            disabled={exporting}
            className="btn-primary flex items-center space-x-2"
          >
            <Download className="w-4 h-4" />
            <span>Export as JSON</span>
          </button>
          <button
            onClick={() => exportResults('csv')}
            disabled={exporting}
            className="btn-secondary flex items-center space-x-2"
          >
            <Download className="w-4 h-4" />
            <span>Export as CSV</span>
          </button>
        </div>
      </div>

      {/* Evolution Progress Chart */}
      {chartData.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
          <h3 className="text-lg font-semibold mb-4">Evolution Progress</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="generation"
                label={{ value: 'Generation', position: 'insideBottom', offset: -5 }}
              />
              <YAxis label={{ value: 'Fitness', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="bestFitness"
                stroke="#10b981"
                name="Best Fitness"
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="avgFitness"
                stroke="#6366f1"
                name="Average Fitness"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Best Solution Code */}
      {bestSolution && (
        <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
          <button
            onClick={() => setShowBestCode(!showBestCode)}
            className="w-full flex items-center justify-between text-left mb-4"
          >
            <h3 className="text-lg font-semibold flex items-center">
              <Code className="w-5 h-5 mr-2" />
              Best Solution Code
            </h3>
            {showBestCode ? (
              <ChevronUp className="w-5 h-5 text-gray-500" />
            ) : (
              <ChevronDown className="w-5 h-5 text-gray-500" />
            )}
          </button>
          
          {showBestCode && bestSolution.code && (
            <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-sm text-green-400 font-mono">
                <code>{bestSolution.code}</code>
              </pre>
            </div>
          )}
          
          {showBestCode && !bestSolution.code && (
            <p className="text-gray-500 italic">No code available for best solution</p>
          )}
        </div>
      )}

      {/* Configuration Details */}
      <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">Evolution Configuration</h3>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Number of Generations:</span>
            <span className="ml-2 font-medium">{sessionInfo?.user_config?.num_generations || 'N/A'}</span>
          </div>
          <div>
            <span className="text-gray-600">Number of Islands:</span>
            <span className="ml-2 font-medium">{sessionInfo?.user_config?.num_islands || 'N/A'}</span>
          </div>
          <div>
            <span className="text-gray-600">Parallel Jobs:</span>
            <span className="ml-2 font-medium">{sessionInfo?.user_config?.max_parallel_jobs || 'N/A'}</span>
          </div>
          <div>
            <span className="text-gray-600">Archive Size:</span>
            <span className="ml-2 font-medium">{sessionInfo?.user_config?.archive_size || 'N/A'}</span>
          </div>
          <div>
            <span className="text-gray-600">Migration Interval:</span>
            <span className="ml-2 font-medium">{sessionInfo?.user_config?.migration_interval || 'N/A'}</span>
          </div>
          <div>
            <span className="text-gray-600">LLM Models:</span>
            <span className="ml-2 font-medium">
              {sessionInfo?.user_config?.llm_models?.join(', ') || 'N/A'}
            </span>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex justify-center space-x-4">
        <button
          onClick={() => navigate('/')}
          className="btn-secondary"
        >
          Start New Problem
        </button>
        <button
          onClick={() => navigate(`/dashboard/${sessionId}`)}
          className="btn-primary"
        >
          View Dashboard
        </button>
      </div>
    </div>
  );
}

export default Results;
