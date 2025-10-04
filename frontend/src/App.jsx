import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import ProblemInput from './pages/ProblemInput';
import Analysis from './pages/Analysis';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/problem/new" element={<ProblemInput />} />
          <Route path="/analysis/:problemId" element={<Analysis />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;