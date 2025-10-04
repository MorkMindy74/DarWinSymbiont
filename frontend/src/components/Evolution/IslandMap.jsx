import React from 'react';
import { TrendingUp, Users } from 'lucide-react';

function IslandMap({ islands }) {
  if (!islands || Object.keys(islands).length === 0) {
    return (
      <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
        <h3 className="text-lg font-semibold mb-4">🏝️ Evolutionary Islands</h3>
        <p className="text-gray-500 text-center py-8">No island data available yet...</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
      <h3 className="text-lg font-semibold mb-4">🏝️ Evolutionary Islands</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Object.entries(islands).map(([islandId, island]) => (
          <div 
            key={islandId}
            className="border border-gray-200 rounded-lg p-4 hover:border-primary-500 transition"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-gray-700">Island {islandId}</span>
              <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse"></div>
            </div>
            
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Best:</span>
                <span className="font-medium text-green-600">
                  {island.best_fitness?.toFixed(2) || '0.00'}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Avg:</span>
                <span className="font-medium text-blue-600">
                  {island.avg_fitness?.toFixed(2) || '0.00'}
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Gen:</span>
                <span className="font-medium text-gray-700">
                  {island.latest_generation || 0}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default IslandMap;