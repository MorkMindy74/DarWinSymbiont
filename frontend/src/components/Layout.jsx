import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Sparkles, Home, Plus } from 'lucide-react';

function Layout({ children }) {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link to="/" className="flex items-center space-x-2 hover:opacity-80 transition">
              <Sparkles className="w-8 h-8 text-primary-600" />
              <span className="text-2xl font-bold bg-gradient-to-r from-primary-600 to-purple-600 bg-clip-text text-transparent">
                EMERGENT
              </span>
            </Link>
            
            <nav className="flex items-center space-x-4">
              <Link
                to="/"
                className="flex items-center space-x-1 px-3 py-2 rounded-lg hover:bg-gray-100 transition"
              >
                <Home className="w-4 h-4" />
                <span className="text-sm font-medium">Home</span>
              </Link>
              <button
                onClick={() => navigate('/problem/new')}
                className="btn-primary flex items-center space-x-1"
              >
                <Plus className="w-4 h-4" />
                <span>New Problem</span>
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">
            EMERGENT - AI-Powered Optimization Platform powered by ShinkaEvolve
          </p>
        </div>
      </footer>
    </div>
  );
}

export default Layout;