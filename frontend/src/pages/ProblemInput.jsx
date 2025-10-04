import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Loader } from 'lucide-react';
import { problemAPI, analysisAPI } from '../services/api';
import toast, { Toaster } from 'react-hot-toast';

function ProblemInput() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    problem_type: 'tsp',
    title: '',
    description: '',
    constraints: {
      num_locations: null,
      max_distance: null,
      max_time: null,
      time_windows: null,
      vehicles: null,
      capacity: null,
    }
  });
  const [loading, setLoading] = useState(false);

  const problemTypes = [
    { value: 'tsp', label: 'TSP - Traveling Salesman Problem', priority: true },
    { value: 'tsp_tw', label: 'TSP with Time Windows', priority: true },
    { value: 'scheduling', label: 'Scheduling Problem', priority: true },
    { value: 'circle_packing', label: 'Circle Packing', priority: false },
    { value: 'nas', label: 'Neural Architecture Search', priority: false },
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.title || !formData.description) {
      toast.error('Please fill in all required fields');
      return;
    }

    setLoading(true);
    
    try {
      // Step 1: Create problem
      const problemResponse = await problemAPI.create({
        problem_type: formData.problem_type,
        title: formData.title,
        description: formData.description,
        constraints: formData.constraints,
      });

      const problemId = problemResponse.data.problem_id;
      toast.success('Problem created successfully!');

      // Step 2: Start analysis
      toast.loading('Starting AI analysis...', { id: 'analysis' });
      
      const analysisResponse = await analysisAPI.analyze(
        problemId,
        {
          problem_type: formData.problem_type,
          title: formData.title,
          description: formData.description,
          constraints: formData.constraints,
        }
      );

      toast.success('Analysis complete!', { id: 'analysis' });
      
      // Navigate to analysis page
      navigate(`/analysis/${problemId}`);
      
    } catch (error) {
      console.error('Error:', error);
      toast.error('Failed to create problem: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleConstraintChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      constraints: {
        ...prev.constraints,
        [field]: value === '' ? null : (isNaN(value) ? value : Number(value))
      }
    }));
  };

  const renderConstraints = () => {
    const { problem_type } = formData;

    if (problem_type === 'tsp') {
      return (
        <>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Number of Locations *
            </label>
            <input
              type="number"
              className="input-field"
              value={formData.constraints.num_locations || ''}
              onChange={(e) => handleConstraintChange('num_locations', e.target.value)}
              placeholder="e.g., 10"
              min="2"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Maximum Distance (optional)
            </label>
            <input
              type="number"
              className="input-field"
              value={formData.constraints.max_distance || ''}
              onChange={(e) => handleConstraintChange('max_distance', e.target.value)}
              placeholder="e.g., 1000"
            />
          </div>
        </>
      );
    }

    if (problem_type === 'tsp_tw') {
      return (
        <>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Number of Locations *
            </label>
            <input
              type="number"
              className="input-field"
              value={formData.constraints.num_locations || ''}
              onChange={(e) => handleConstraintChange('num_locations', e.target.value)}
              placeholder="e.g., 10"
              min="2"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Maximum Time (optional)
            </label>
            <input
              type="number"
              className="input-field"
              value={formData.constraints.max_time || ''}
              onChange={(e) => handleConstraintChange('max_time', e.target.value)}
              placeholder="e.g., 480 (minutes)"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Number of Vehicles (optional)
            </label>
            <input
              type="number"
              className="input-field"
              value={formData.constraints.vehicles || ''}
              onChange={(e) => handleConstraintChange('vehicles', e.target.value)}
              placeholder="e.g., 3"
              min="1"
            />
          </div>
        </>
      );
    }

    if (problem_type === 'scheduling') {
      return (
        <>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Maximum Time (optional)
            </label>
            <input
              type="number"
              className="input-field"
              value={formData.constraints.max_time || ''}
              onChange={(e) => handleConstraintChange('max_time', e.target.value)}
              placeholder="e.g., 480 (minutes)"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Resource Capacity (optional)
            </label>
            <input
              type="number"
              className="input-field"
              value={formData.constraints.capacity || ''}
              onChange={(e) => handleConstraintChange('capacity', e.target.value)}
              placeholder="e.g., 100"
            />
          </div>
        </>
      );
    }

    return null;
  };

  return (
    <div className="max-w-4xl mx-auto">
      <Toaster position="top-right" />
      
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Create New Problem</h1>
        <p className="text-gray-600">
          Describe your optimization problem and let AI analyze it for you
        </p>
      </div>

      {/* Form */}
      <form onSubmit={handleSubmit} className="card space-y-6">
        {/* Problem Type */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Problem Type *
          </label>
          <select
            className="input-field"
            value={formData.problem_type}
            onChange={(e) => handleChange('problem_type', e.target.value)}
            required
          >
            {problemTypes.map(type => (
              <option key={type.value} value={type.value}>
                {type.label} {type.priority ? '⭐' : ''}
              </option>
            ))}
          </select>
          <p className="text-xs text-gray-500 mt-1">
            ⭐ = Priority problems with enhanced support
          </p>
        </div>

        {/* Title */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Problem Title *
          </label>
          <input
            type="text"
            className="input-field"
            value={formData.title}
            onChange={(e) => handleChange('title', e.target.value)}
            placeholder="e.g., Optimize delivery routes for 50 cities"
            required
            minLength={3}
            maxLength={200}
          />
        </div>

        {/* Description */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Problem Description *
          </label>
          <textarea
            className="input-field"
            rows={6}
            value={formData.description}
            onChange={(e) => handleChange('description', e.target.value)}
            placeholder="Describe your optimization problem in detail. Include objectives, constraints, and any specific requirements..."
            required
            minLength={10}
          />
          <p className="text-xs text-gray-500 mt-1">
            Be as detailed as possible. Better descriptions lead to better AI analysis.
          </p>
        </div>

        {/* Dynamic Constraints */}
        <div className="border-t pt-6">
          <h3 className="text-lg font-semibold mb-4">Constraints</h3>
          <div className="grid md:grid-cols-2 gap-4">
            {renderConstraints()}
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-end space-x-3 pt-6 border-t">
          <button
            type="button"
            onClick={() => navigate('/')}
            className="btn-secondary"
            disabled={loading}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="btn-primary flex items-center space-x-2"
            disabled={loading}
          >
            {loading ? (
              <>
                <Loader className="w-4 h-4 animate-spin" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <span>Analyze Problem</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}

export default ProblemInput;
