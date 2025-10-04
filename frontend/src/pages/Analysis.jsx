import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Loader, 
  CheckCircle, 
  AlertCircle, 
  TrendingUp, 
  Settings,
  Code,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { problemAPI, analysisAPI } from '../services/api';
import toast, { Toaster } from 'react-hot-toast';

function Analysis() {
  const { problemId } = useParams();
  const navigate = useNavigate();
  const [problem, setProblem] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [expandedSections, setExpandedSections] = useState({
    challenges: true,
    parameters: true,
    constraints: true,
    strategy: true,
  });

  useEffect(() => {
    loadProblem();
  }, [problemId]);

  const loadProblem = async () => {
    try {
      const response = await problemAPI.get(problemId);
      setProblem(response.data.problem);
      setAnalysis(response.data.analysis);
    } catch (error) {
      console.error('Error loading problem:', error);
      toast.error('Failed to load problem');
    } finally {
      setLoading(false);
    }
  };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const getImportanceBadge = (importance) => {
    const colors = {
      critical: 'bg-red-100 text-red-800 border-red-200',
      high: 'bg-orange-100 text-orange-800 border-orange-200',
      medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      low: 'bg-blue-100 text-blue-800 border-blue-200',
    };
    return colors[importance] || colors.medium;
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <Loader className="w-12 h-12 text-primary-600 animate-spin" />
        <p className="text-lg text-gray-600">Loading analysis...</p>
      </div>
    );
  }

  if (!problem) {
    return (
      <div className="card text-center py-12">
        <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold mb-2">Problem Not Found</h2>
        <p className="text-gray-600 mb-6">The requested problem could not be found.</p>
        <button onClick={() => navigate('/')} className="btn-primary">
          Go Home
        </button>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="card text-center py-12">
        <AlertCircle className="w-16 h-16 text-yellow-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold mb-2">Analysis Not Available</h2>
        <p className="text-gray-600 mb-6">
          This problem hasn't been analyzed yet. Please run the analysis first.
        </p>
        <button onClick={() => navigate('/problem/new')} className="btn-primary">
          Create New Problem
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <Toaster position="top-right" />

      {/* Header */}
      <div className="card">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center space-x-2 mb-2">
              <CheckCircle className="w-6 h-6 text-green-500" />
              <span className="text-sm font-medium text-green-600">Analysis Complete</span>
            </div>
            <h1 className="text-3xl font-bold mb-2">{problem.problem_input.title}</h1>
            <p className="text-gray-600">
              Type: <span className="font-medium">{problem.problem_input.problem_type.toUpperCase()}</span>
            </p>
          </div>
          <button
            onClick={() => navigate('/')}
            className="btn-secondary"
          >
            Back to Home
          </button>
        </div>
      </div>

      {/* Problem Characterization */}
      <div className="card">
        <h2 className="text-2xl font-bold mb-4 flex items-center space-x-2">
          <TrendingUp className="w-6 h-6 text-primary-600" />
          <span>Problem Characterization</span>
        </h2>
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
          <p className="text-gray-800">{analysis.problem_characterization}</p>
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="border border-gray-200 rounded-lg p-4">
            <h3 className="font-semibold text-sm text-gray-500 mb-2">Complexity Assessment</h3>
            <p className="text-gray-800">{analysis.complexity_assessment}</p>
          </div>
          <div className="border border-gray-200 rounded-lg p-4">
            <h3 className="font-semibold text-sm text-gray-500 mb-2">Search Space</h3>
            <p className="text-gray-800 font-mono text-lg">{analysis.estimated_search_space}</p>
          </div>
        </div>
      </div>

      {/* Key Challenges */}
      <div className="card">
        <button
          onClick={() => toggleSection('challenges')}
          className="w-full flex items-center justify-between mb-4"
        >
          <h2 className="text-2xl font-bold flex items-center space-x-2">
            <AlertCircle className="w-6 h-6 text-orange-500" />
            <span>Key Challenges</span>
          </h2>
          {expandedSections.challenges ? <ChevronUp /> : <ChevronDown />}
        </button>
        {expandedSections.challenges && (
          <ul className="space-y-2">
            {analysis.key_challenges.map((challenge, index) => (
              <li key={index} className="flex items-start space-x-2">
                <span className="text-primary-600 font-bold">•</span>
                <span className="text-gray-700">{challenge}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Parameter Suggestions */}
      <div className="card">
        <button
          onClick={() => toggleSection('parameters')}
          className="w-full flex items-center justify-between mb-4"
        >
          <h2 className="text-2xl font-bold flex items-center space-x-2">
            <Settings className="w-6 h-6 text-primary-600" />
            <span>Recommended Parameters</span>
          </h2>
          {expandedSections.parameters ? <ChevronUp /> : <ChevronDown />}
        </button>
        {expandedSections.parameters && (
          <div className="space-y-3">
            {analysis.parameter_suggestions.map((param, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-lg">{param.name}</h3>
                  <span className="text-primary-600 font-mono text-lg">{JSON.stringify(param.value)}</span>
                </div>
                <p className="text-sm text-gray-600 mb-1">{param.description}</p>
                <p className="text-xs text-gray-500 italic">Rationale: {param.rationale}</p>
                {param.adjustable && (
                  <span className="inline-block mt-2 text-xs bg-green-100 text-green-800 px-2 py-1 rounded">
                    Adjustable
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Constraints Analysis */}
      <div className="card">
        <button
          onClick={() => toggleSection('constraints')}
          className="w-full flex items-center justify-between mb-4"
        >
          <h2 className="text-2xl font-bold flex items-center space-x-2">
            <AlertCircle className="w-6 h-6 text-red-500" />
            <span>Constraints Analysis</span>
          </h2>
          {expandedSections.constraints ? <ChevronUp /> : <ChevronDown />}
        </button>
        {expandedSections.constraints && (
          <div className="space-y-3">
            {analysis.constraints_analysis.map((constraint, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-lg">{constraint.constraint_type}</h3>
                  <span className={`text-xs font-medium px-3 py-1 rounded-full border ${getImportanceBadge(constraint.importance)}`}>
                    {constraint.importance}
                  </span>
                </div>
                <p className="text-sm text-gray-600 mb-1">{constraint.description}</p>
                <p className="text-xs text-gray-500 italic">Impact: {constraint.impact_on_solution}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Solution Strategy */}
      <div className="card">
        <button
          onClick={() => toggleSection('strategy')}
          className="w-full flex items-center justify-between mb-4"
        >
          <h2 className="text-2xl font-bold flex items-center space-x-2">
            <Code className="w-6 h-6 text-purple-600" />
            <span>Solution Strategy</span>
          </h2>
          {expandedSections.strategy ? <ChevronUp /> : <ChevronDown />}
        </button>
        {expandedSections.strategy && (
          <div className="space-y-4">
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
              <p className="text-gray-800 whitespace-pre-line">{analysis.solution_strategy}</p>
            </div>
            
            {/* Recommended Evolution Config */}
            <div className="border-t pt-4">
              <h3 className="font-semibold mb-3">Recommended Evolution Configuration</h3>
              <div className="bg-gray-50 rounded-lg p-4 font-mono text-sm">
                <pre className="whitespace-pre-wrap">
                  {JSON.stringify(analysis.recommended_evolution_config, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Next Steps */}
      <div className="bg-gradient-to-r from-primary-600 to-purple-600 rounded-2xl p-8 text-white">
        <h2 className="text-2xl font-bold mb-4">Next Steps</h2>
        <p className="text-lg mb-6 opacity-90">
          Ready to run the evolution? Configure parameters and start the optimization process.
        </p>
        <button
          className="bg-white text-primary-600 hover:bg-gray-100 font-semibold py-3 px-8 rounded-lg transition transform hover:scale-105 shadow-lg"
          onClick={() => toast.info('Evolution simulation coming in Phase 3-4!')}
        >
          Start Evolution (Coming Soon)
        </button>
      </div>
    </div>
  );
}

export default Analysis;
