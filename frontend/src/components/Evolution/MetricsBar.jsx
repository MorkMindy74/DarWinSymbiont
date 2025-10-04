import React from 'react';
import { TrendingUp, Layers, Zap, Target } from 'lucide-react';

function MetricsBar({ generation, bestFitness, avgFitness, diversity }) {
  const calculateImprovement = () => {
    if (!avgFitness || avgFitness === 0) return 0;
    const improvement = ((bestFitness - avgFitness) / Math.abs(avgFitness)) * 100;
    return improvement.toFixed(1);
  };

  const metrics = [
    {
      icon: Target,
      label: 'Generation',
      value: generation,
      color: 'text-blue-600',
      bgColor: 'bg-blue-100'
    },
    {
      icon: TrendingUp,
      label: 'Best Fitness',
      value: bestFitness?.toFixed(2) || '0.00',
      color: 'text-green-600',
      bgColor: 'bg-green-100'
    },
    {
      icon: Zap,
      label: 'Improvement',
      value: `${calculateImprovement()}%`,
      color: 'text-purple-600',
      bgColor: 'bg-purple-100'
    },
    {
      icon: Layers,
      label: 'Diversity',
      value: diversity || 0,
      color: 'text-orange-600',
      bgColor: 'bg-orange-100'
    }
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      {metrics.map((metric, index) => {
        const Icon = metric.icon;
        return (
          <div key={index} className="bg-white rounded-lg shadow p-4 border border-gray-200">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-600">{metric.label}</span>
              <div className={`p-2 rounded-lg ${metric.bgColor}`}>
                <Icon className={`w-4 h-4 ${metric.color}`} />
              </div>
            </div>
            <div className={`text-2xl font-bold ${metric.color}`}>
              {metric.value}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default MetricsBar;