import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function FitnessChart({ history }) {
  if (!history || history.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6 border border-gray-200 flex items-center justify-center h-96">
        <p className="text-gray-500">Waiting for evolution data...</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
      <h3 className="text-lg font-semibold mb-4">Fitness Evolution</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={history}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis 
            dataKey="generation" 
            label={{ value: 'Generation', position: 'insideBottom', offset: -5 }}
            stroke="#6b7280"
          />
          <YAxis 
            label={{ value: 'Fitness', angle: -90, position: 'insideLeft' }}
            stroke="#6b7280"
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#fff', 
              border: '1px solid #e5e7eb',
              borderRadius: '0.5rem'
            }}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="bestFitness" 
            stroke="#10b981" 
            strokeWidth={2}
            name="Best Fitness"
            dot={{ fill: '#10b981', r: 3 }}
            activeDot={{ r: 5 }}
          />
          <Line 
            type="monotone" 
            dataKey="avgFitness" 
            stroke="#6366f1" 
            strokeWidth={2}
            name="Average Fitness"
            dot={{ fill: '#6366f1', r: 3 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default FitnessChart;