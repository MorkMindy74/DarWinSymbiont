import React, { useEffect, useRef } from 'react';
import { Sparkles } from 'lucide-react';

function LiveFeed({ history }) {
  const feedRef = useRef(null);

  useEffect(() => {
    // Auto-scroll to bottom when new messages arrive
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [history]);

  const generateCommentary = (entry) => {
    if (!entry) return '';
    
    const improvement = entry.bestFitness - (history[history.length - 2]?.bestFitness || 0);
    
    if (improvement > 0) {
      return `🎯 Generation ${entry.generation}: Found improvement! Best fitness increased to ${entry.bestFitness.toFixed(2)} (+${improvement.toFixed(2)})`;
    } else if (entry.generation % 10 === 0) {
      return `🔍 Generation ${entry.generation}: Exploring solution space. Best: ${entry.bestFitness.toFixed(2)}, Diversity: ${entry.diversity}`;
    } else {
      return `🧬 Generation ${entry.generation}: Evolution in progress. Best: ${entry.bestFitness.toFixed(2)}`;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
      <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
        <Sparkles className="w-5 h-5 text-primary-600" />
        <span>Live Evolution Feed</span>
      </h3>
      
      <div 
        ref={feedRef}
        className="space-y-2 overflow-y-auto max-h-96 pr-2"
        style={{ scrollbarWidth: 'thin' }}
      >
        {history.length === 0 ? (
          <p className="text-gray-500 text-center py-8">Waiting for evolution to start...</p>
        ) : (
          history.slice(-15).map((entry, index) => (
            <div 
              key={index}
              className="p-3 bg-gray-50 rounded-lg border border-gray-200 text-sm animate-fadeIn"
            >
              <p className="text-gray-700">{generateCommentary(entry)}</p>
              <span className="text-xs text-gray-500">
                {new Date(entry.timestamp).toLocaleTimeString()}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default LiveFeed;