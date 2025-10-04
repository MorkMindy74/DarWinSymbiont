import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Zap, Brain, TrendingUp, ArrowRight } from 'lucide-react';
import { problemAPI } from '../services/api';
import toast, { Toaster } from 'react-hot-toast';

function Home() {
  const navigate = useNavigate();
  const [recentProblems, setRecentProblems] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadRecentProblems();
  }, []);

  const loadRecentProblems = async () => {
    try {
      const response = await problemAPI.list();
      setRecentProblems(response.data.slice(0, 5));
    } catch (error) {
      console.error('Failed to load problems:', error);
    } finally {
      setLoading(false);
    }
  };

  const features = [
    {
      icon: <Brain className="w-6 h-6" />,
      title: "AI-Powered Analysis",
      description: "LLM analyzes your problem and suggests optimal parameters"
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: "Adaptive Evolution",
      description: "Context-aware Thompson Sampling for intelligent model selection"
    },
    {
      icon: <TrendingUp className="w-6 h-6" />,
      title: "Real-time Insights",
      description: "Watch your solutions evolve with live metrics and visualizations"
    }
  ];

  const problemTypes = [
    {
      name: "TSP",
      description: "Traveling Salesman Problem",
      color: "bg-blue-500"
    },
    {
      name: "TSP-TW",
      description: "TSP with Time Windows",
      color: "bg-purple-500"
    },
    {
      name: "Scheduling",
      description: "Resource Scheduling",
      color: "bg-green-500"
    }
  ];

  return (
    <div className="space-y-12">
      <Toaster position="top-right" />
      
      {/* Hero Section */}
      <section className="text-center space-y-6 py-12">
        <h1 className="text-5xl font-bold bg-gradient-to-r from-primary-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
          AI-Powered Optimization Platform
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Transform your optimization challenges into evolved solutions with intelligent 
          problem analysis and adaptive evolutionary algorithms
        </p>
        <div className="flex justify-center gap-4">
          <button
            onClick={() => navigate('/problem/new')}
            className="btn-primary text-lg px-8 py-3 flex items-center space-x-2 shadow-lg hover:shadow-xl transform hover:scale-105 transition"
          >
            <Plus className="w-5 h-5" />
            <span>Start New Problem</span>
          </button>
        </div>
      </section>

      {/* Features */}
      <section className="grid md:grid-cols-3 gap-6">
        {features.map((feature, index) => (
          <div key={index} className="card hover:shadow-lg transition group">
            <div className="flex items-start space-x-4">
              <div className="p-3 bg-primary-100 text-primary-600 rounded-lg group-hover:scale-110 transition">
                {feature.icon}
              </div>
              <div>
                <h3 className="font-semibold text-lg mb-2">{feature.title}</h3>
                <p className="text-gray-600 text-sm">{feature.description}</p>
              </div>
            </div>
          </div>
        ))}
      </section>

      {/* Problem Types */}
      <section className="card">
        <h2 className="text-2xl font-bold mb-6">Supported Problem Types</h2>
        <div className="grid md:grid-cols-3 gap-4">
          {problemTypes.map((type, index) => (
            <div 
              key={index}
              className="border border-gray-200 rounded-lg p-4 hover:border-primary-500 transition cursor-pointer group"
              onClick={() => navigate('/problem/new')}
            >
              <div className={`w-12 h-12 ${type.color} rounded-lg mb-3 group-hover:scale-110 transition`}></div>
              <h3 className="font-semibold text-lg">{type.name}</h3>
              <p className="text-sm text-gray-600">{type.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Recent Problems */}
      {recentProblems.length > 0 && (
        <section className="card">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold">Recent Problems</h2>
          </div>
          <div className="space-y-3">
            {recentProblems.map((problem) => (
              <div
                key={problem.problem_id}
                className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer transition group"
                onClick={() => navigate(`/analysis/${problem.problem_id}`)}
              >
                <div className="flex-1">
                  <h3 className="font-medium group-hover:text-primary-600 transition">
                    {problem.problem_input.title}
                  </h3>
                  <p className="text-sm text-gray-500 mt-1">
                    Type: {problem.problem_input.problem_type.toUpperCase()} • 
                    Created: {new Date(problem.created_at).toLocaleDateString()}
                  </p>
                </div>
                <ArrowRight className="w-5 h-5 text-gray-400 group-hover:text-primary-600 group-hover:translate-x-1 transition" />
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Getting Started */}
      <section className="bg-gradient-to-r from-primary-600 to-purple-600 rounded-2xl p-8 text-white text-center shadow-xl">
        <h2 className="text-3xl font-bold mb-4">Ready to Optimize?</h2>
        <p className="text-lg mb-6 opacity-90">
          Describe your problem, let AI analyze it, and watch evolution find the optimal solution
        </p>
        <button
          onClick={() => navigate('/problem/new')}
          className="bg-white text-primary-600 hover:bg-gray-100 font-semibold py-3 px-8 rounded-lg transition transform hover:scale-105 shadow-lg"
        >
          Get Started Now
        </button>
      </section>
    </div>
  );
}

export default Home;
