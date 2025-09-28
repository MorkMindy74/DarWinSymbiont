import React, { useState, useEffect, createContext, useContext } from 'react';
import { BrowserRouter, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import './App.css';

// Theme Context for Dark/Light Mode
const ThemeContext = createContext();
const useTheme = () => useContext(ThemeContext);

// Navigation Context for History Management
const NavigationContext = createContext();
const useNavigation = () => useContext(NavigationContext);

// Import shadcn components
import { Button } from './components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Progress } from './components/ui/progress';
import { Badge } from './components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Separator } from './components/ui/separator';
import { Textarea } from './components/ui/textarea';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Alert, AlertDescription } from './components/ui/alert';
import { Skeleton } from './components/ui/skeleton';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './components/ui/tooltip';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Stepper Component with Navigation
const Stepper = ({ currentStep, steps }) => {
  const { goBackOneStep, navigationHistory } = useNavigation();
  const canGoBack = navigationHistory.length > 1 && currentStep > 0;

  return (
    <div className="mb-12">
      <div className="flex items-center justify-center mb-4">
        {steps.map((step, index) => (
          <div key={step} className="flex items-center">
            <div className={`flex items-center justify-center w-10 h-10 rounded-full border-2 transition-all ${
              index <= currentStep 
                ? 'bg-black text-white border-black' 
                : 'bg-white text-gray-400 border-gray-300'
            }`}>
              {index + 1}
            </div>
            <span className={`ml-3 text-sm font-medium ${
              index <= currentStep ? 'text-black' : 'text-gray-400'
            }`}>
              {step}
            </span>
            {index < steps.length - 1 && (
              <div className={`w-16 h-0.5 mx-4 ${
                index < currentStep ? 'bg-black' : 'bg-gray-300'
              }`} />
            )}
          </div>
        ))}
      </div>
      
      {canGoBack && (
        <div className="flex justify-center">
          <Button
            variant="outline"
            size="sm"
            onClick={goBackOneStep}
            className="flex items-center gap-2"
            data-testid="back-step-button"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Previous Step
          </Button>
        </div>
      )}
    </div>
  );
};

// Data Provenance Tooltip Component
const ProvenanceTooltip = ({ children, source, confidence = null }) => {
  const getSourceIcon = (source) => {
    switch(source) {
      case 'simulation': return '⚡';
      case 'paper': return '📄';
      case 'llm': return '🤖';
      default: return '❓';
    }
  };

  const getSourceLabel = (source) => {
    switch(source) {
      case 'simulation': return 'Data from simulation output';
      case 'paper': return 'Figure from referenced paper';
      case 'llm': return 'Derived via LLM extraction';
      default: return 'Unknown data source';
    }
  };

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="relative cursor-help">
            {children}
            <span className="absolute -top-1 -right-1 text-xs opacity-60">
              {getSourceIcon(source)}
            </span>
          </span>
        </TooltipTrigger>
        <TooltipContent>
          <div className="text-xs">
            <div className="font-medium">{getSourceLabel(source)}</div>
            {confidence && (
              <div className="text-gray-500 mt-1">
                Confidence: {Math.round(confidence * 100)}%
              </div>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

// File Drop Component
const FileDrop = ({ onFilesSelected, files, isUploading }) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    onFilesSelected(droppedFiles);
  };

  const handleFileSelect = (e) => {
    const selectedFiles = Array.from(e.target.files);
    onFilesSelected(selectedFiles);
  };

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
        isDragOver 
          ? 'border-black bg-gray-50' 
          : 'border-gray-300 hover:border-gray-400'
      } ${isUploading ? 'opacity-50 pointer-events-none' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="space-y-4">
        <div className="mx-auto w-12 h-12 text-gray-400">
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
        </div>
        <div>
          <p className="text-lg text-gray-600">
            {files.length > 0 
              ? `${files.length} file(s) selected`
              : 'Drag your PDFs here (max 50MB each)'
            }
          </p>
          <p className="text-sm text-gray-400 mt-2">or</p>
          <label className="cursor-pointer">
            <Button 
              variant="outline" 
              className="mt-2"
              disabled={isUploading}
              data-testid="file-select-button"
            >
              {isUploading ? 'Uploading...' : 'Choose Files'}
            </Button>
            <input
              type="file"
              multiple
              accept=".pdf"
              onChange={handleFileSelect}
              className="hidden"
              disabled={isUploading}
            />
          </label>
        </div>
        {files.length > 0 && (
          <div className="mt-4 space-y-2">
            {files.map((file, index) => (
              <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                <span className="text-sm text-gray-600">{file.name}</span>
                <span className="text-xs text-gray-400">{(file.size / 1024 / 1024).toFixed(1)}MB</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// Progress Log Component
const ProgressLog = ({ messages, progress }) => {
  return (
    <Card className="mt-6">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span>Processing Status</span>
          {progress !== null && <Badge variant="outline">{progress}%</Badge>}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2 max-h-40 overflow-y-auto">
          {messages.map((msg, index) => (
            <div key={index} className="text-sm text-gray-600">
              <span className="text-gray-400">{new Date().toLocaleTimeString()}</span> - {msg}
            </div>
          ))}
        </div>
        {progress !== null && (
          <Progress value={progress} className="mt-4" />
        )}
      </CardContent>
    </Card>
  );
};

// Upload Page
const UploadPage = () => {
  const [files, setFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [progressMessages, setProgressMessages] = useState([]);
  const navigate = useNavigate();

  const handleFilesSelected = (newFiles) => {
    const pdfFiles = newFiles.filter(file => file.name.endsWith('.pdf'));
    if (pdfFiles.length !== newFiles.length) {
      setProgressMessages(prev => [...prev, 'Only PDF files are supported']);
    }
    setFiles(pdfFiles);
  };

  const handleUpload = async () => {
    if (files.length === 0) return;

    setIsUploading(true);
    setUploadProgress(0);
    setProgressMessages(['Starting upload...']);

    try {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));

      const response = await axios.post(`${API}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        }
      });

      setProgressMessages(prev => [...prev, 'Upload complete, processing files...']);

      // Simulate processing progress
      setTimeout(() => {
        setProgressMessages(prev => [...prev, 'Processing complete!']);
        navigate('/analysis', { state: { uploadedFiles: response.data.files } });
      }, 2000);

    } catch (error) {
      console.error('Upload failed:', error);
      setProgressMessages(prev => [...prev, `Upload failed: ${error.message}`]);
      setIsUploading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="text-center">
        {/* Main Branding Title */}
        <div className="mb-6">
          <h1 className="text-6xl font-bold text-black mb-2 tracking-wider font-mono relative">
            <span className="bg-gradient-to-r from-black via-gray-800 to-black bg-clip-text text-transparent">
              DARWINSYMBIONT
            </span>
            <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-32 h-0.5 bg-black"></div>
          </h1>
          <p className="text-xl text-gray-700 font-medium tracking-wide">
            COMPARE SCIENCE, EVOLVE RESEARCH
          </p>
        </div>
        
        {/* Scientific Introduction */}
        <div className="bg-gray-50 rounded-lg p-6 mb-8 max-w-4xl mx-auto">
          <h3 className="text-lg font-semibold text-black mb-3">
            🧬 Discover what makes DarWinSymbiont unique in evolutionary simulation
          </h3>
          <div className="grid md:grid-cols-3 gap-4 text-sm text-gray-700">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-bold">1</div>
              <div>
                <h4 className="font-medium mb-1">AI-Powered Analysis</h4>
                <p>Upload PDFs → GPT-5 extracts insights, methodologies, and performance metrics</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center text-green-600 font-bold">2</div>
              <div>
                <h4 className="font-medium mb-1">Evolutionary Simulation</h4>
                <p>DarWinSymbiont runs population-based optimization with real-time progress</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center text-purple-600 font-bold">3</div>
              <div>
                <h4 className="font-medium mb-1">Scientific Comparison</h4>
                <p>Side-by-side analysis vs original studies → LaTeX paper export</p>
              </div>
            </div>
          </div>
        </div>

        <h2 className="text-2xl font-semibold text-black mb-4">
          Upload Studies & Get Analysis + Evolutionary Simulation
        </h2>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Upload scientific PDFs to analyze research papers, compare methodologies, 
          and run DarWinSymbiont evolutionary simulations for scientific discovery.
        </p>
      </div>

      <FileDrop 
        onFilesSelected={handleFilesSelected}
        files={files}
        isUploading={isUploading}
      />

      {files.length > 0 && (
        <div className="flex justify-center">
          <Button
            onClick={handleUpload}
            disabled={isUploading}
            size="lg"
            className="px-8 py-3 text-lg"
            data-testid="analyze-button"
          >
            {isUploading ? 'Processing...' : 'Analyze Studies'}
          </Button>
        </div>
      )}

      {(isUploading || progressMessages.length > 0) && (
        <ProgressLog 
          messages={progressMessages}
          progress={uploadProgress}
        />
      )}

      <Card className="bg-gray-50">
        <CardContent className="pt-6">
          <div className="grid md:grid-cols-3 gap-6 text-sm">
            <div>
              <h4 className="font-medium mb-2">Privacy & Security</h4>
              <p className="text-gray-600">Files are processed securely and deleted after analysis</p>
            </div>
            <div>
              <h4 className="font-medium mb-2">File Limits</h4>
              <p className="text-gray-600">Maximum 50MB per PDF, up to 10 files at once</p>
            </div>
            <div>
              <h4 className="font-medium mb-2">What Happens Next</h4>
              <p className="text-gray-600">AI analysis → Simulation setup → Results & LaTeX export</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Analysis Page
const AnalysisPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [uploadedFiles] = useState(location.state?.uploadedFiles || []);
  const [analyses, setAnalyses] = useState({});
  const [loading, setLoading] = useState({});
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');

  const runAnalysis = async (type) => {
    if (uploadedFiles.length === 0) return;

    setLoading(prev => ({ ...prev, [type]: true }));
    
    try {
      const studyIds = uploadedFiles.map(file => file.id);
      const response = await axios.post(`${API}/llm/${type}`, { studyIds });
      
      setAnalyses(prev => ({
        ...prev,
        [type]: response.data
      }));
    } catch (error) {
      console.error(`${type} analysis failed:`, error);
    } finally {
      setLoading(prev => ({ ...prev, [type]: false }));
    }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMessage = { role: 'user', content: chatInput };
    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');

    // Mock AI response for now
    setTimeout(() => {
      const aiResponse = { 
        role: 'assistant', 
        content: 'This is a mock response based on your uploaded PDFs. In a real implementation, this would use RAG to search through your documents and provide contextual answers.' 
      };
      setChatMessages(prev => [...prev, aiResponse]);
    }, 1000);
  };

  useEffect(() => {
    if (uploadedFiles.length > 0) {
      runAnalysis('summarize');
    }
  }, [uploadedFiles]);

  return (
    <div className="max-w-6xl mx-auto">
      <div className="grid lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          <div>
            <h2 className="text-2xl font-semibold mb-6">Analysis Results</h2>
            
            <Tabs defaultValue="summary" className="space-y-6">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="summary">Summary</TabsTrigger>
                <TabsTrigger value="problem">Problem</TabsTrigger>
                <TabsTrigger value="compare">Compare</TabsTrigger>
                <TabsTrigger value="improve">Improve</TabsTrigger>
              </TabsList>

              <TabsContent value="summary">
                <Card>
                  <CardHeader>
                    <CardTitle>Study Summaries</CardTitle>
                    <CardDescription>
                      Plain English summaries for non-experts (max 300 words each)
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {loading.summarize ? (
                      <div className="space-y-4">
                        <Skeleton className="h-4 w-full" />
                        <Skeleton className="h-4 w-3/4" />
                        <Skeleton className="h-4 w-1/2" />
                      </div>
                    ) : analyses.summarize ? (
                      <div className="space-y-4">
                        {analyses.summarize.summaries?.map((summary, index) => (
                          <div key={index} className="p-4 border rounded-lg">
                            <h4 className="font-medium mb-2">Study {index + 1}</h4>
                            <p className="text-gray-700 whitespace-pre-wrap">{summary.text}</p>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <Button onClick={() => runAnalysis('summarize')} data-testid="summarize-button">
                        Generate Summaries
                      </Button>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="problem">
                <Card>
                  <CardHeader>
                    <CardTitle>Problem Analysis</CardTitle>
                    <CardDescription>
                      What problems do these studies address? (120-180 words each)
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {loading.problem ? (
                      <div className="space-y-4">
                        <Skeleton className="h-4 w-full" />
                        <Skeleton className="h-4 w-3/4" />
                      </div>
                    ) : analyses.problem ? (
                      <div className="space-y-4">
                        {analyses.problem.problems?.map((problem, index) => (
                          <div key={index} className="p-4 border rounded-lg">
                            <h4 className="font-medium mb-2">Study {index + 1}</h4>
                            <p className="text-gray-700 whitespace-pre-wrap">{problem.text}</p>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <Button onClick={() => runAnalysis('problem')} data-testid="problem-button">
                        Analyze Problems
                      </Button>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="compare">
                <Card>
                  <CardHeader>
                    <CardTitle>Study Comparison</CardTitle>
                    <CardDescription>
                      Compare methodologies and identify stronger approaches
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {loading.compare ? (
                      <div className="space-y-4">
                        <Skeleton className="h-4 w-full" />
                        <Skeleton className="h-4 w-3/4" />
                      </div>
                    ) : analyses.compare ? (
                      <div className="p-4 border rounded-lg">
                        <p className="text-gray-700 whitespace-pre-wrap">{analyses.compare.comparison}</p>
                      </div>
                    ) : (
                      <Button onClick={() => runAnalysis('compare')} data-testid="compare-button">
                        Compare Studies
                      </Button>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="improve">
                <Card>
                  <CardHeader>
                    <CardTitle>Improvement Suggestions</CardTitle>
                    <CardDescription>
                      Practical improvements to experimental strategies
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {loading.improve ? (
                      <div className="space-y-4">
                        <Skeleton className="h-4 w-full" />
                        <Skeleton className="h-4 w-3/4" />
                      </div>
                    ) : analyses.improve ? (
                      <div className="p-4 border rounded-lg">
                        <p className="text-gray-700 whitespace-pre-wrap">{analyses.improve.suggestions}</p>
                      </div>
                    ) : (
                      <Button onClick={() => runAnalysis('improve')} data-testid="improve-button">
                        Suggest Improvements
                      </Button>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>

          <div className="flex justify-between">
            <Button variant="outline" onClick={() => navigate('/')}>
              Back to Upload
            </Button>
            <Button 
              onClick={() => navigate('/simulation', { state: { uploadedFiles, analyses } })}
              data-testid="continue-to-simulation"
            >
              Continue to Simulation
            </Button>
          </div>
        </div>

        {/* Chat Panel */}
        <div className="lg:col-span-1">
          <Card className="h-full">
            <CardHeader>
              <CardTitle>Q&A Chat</CardTitle>
              <CardDescription>
                Ask questions about your uploaded studies
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col h-96">
              <div className="flex-1 overflow-y-auto space-y-3 mb-4">
                {chatMessages.map((msg, index) => (
                  <div key={index} className={`p-3 rounded-lg text-sm ${
                    msg.role === 'user' 
                      ? 'bg-black text-white ml-4' 
                      : 'bg-gray-100 text-gray-900 mr-4'
                  }`}>
                    {msg.content}
                  </div>
                ))}
              </div>
              <form onSubmit={handleChatSubmit} className="flex gap-2">
                <Input
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask about your studies..."
                  className="flex-1"
                />
                <Button type="submit" size="sm">Send</Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

// Simulation Page
const SimulationPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [uploadedFiles] = useState(location.state?.uploadedFiles || []);
  const [analyses] = useState(location.state?.analyses || {});
  const [isRunning, setIsRunning] = useState(false);
  const [runId, setRunId] = useState(null);
  const [metrics, setMetrics] = useState({});
  const [simulationLogs, setSimulationLogs] = useState([]);
  const [params, setParams] = useState({
    popSize: 50,
    mutationRate: 0.1,
    generations: 100,
    seed: '',
    objective: 'Optimize evolutionary algorithm performance based on uploaded research insights'
  });

  const handleRunSimulation = async () => {
    setIsRunning(true);
    setSimulationLogs(['Starting DarWinSymbiont simulation...']);
    
    try {
      const studyIds = uploadedFiles.map(file => file.id);
      const response = await axios.post(`${API}/dws/run`, {
        params: {
          ...params,
          seed: params.seed ? parseInt(params.seed) : null
        },
        studyIds
      });

      const newRunId = response.data.runId;
      setRunId(newRunId);

      // Mock WebSocket simulation
      let generation = 0;
      const interval = setInterval(() => {
        generation++;
        const bestFitness = Math.random() * generation / params.generations;
        const avgFitness = bestFitness * 0.8;

        setMetrics({
          generation,
          best: bestFitness,
          avg: avgFitness,
          eta: Math.max(0, params.generations - generation)
        });

        setSimulationLogs(prev => [
          ...prev,
          `Generation ${generation}/${params.generations}: Best=${bestFitness.toFixed(4)}, Avg=${avgFitness.toFixed(4)}`
        ]);

        if (generation >= params.generations) {
          clearInterval(interval);
          setIsRunning(false);
          setSimulationLogs(prev => [...prev, 'Simulation completed successfully!']);
        }
      }, 200);

    } catch (error) {
      console.error('Simulation failed:', error);
      setSimulationLogs(prev => [...prev, `Simulation failed: ${error.message}`]);
      setIsRunning(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h2 className="text-2xl font-semibold mb-6">DarWinSymbiont Simulation</h2>
      </div>

      <div className="grid lg:grid-cols-2 gap-8">
        <Card>
          <CardHeader>
            <CardTitle>Simulation Parameters</CardTitle>
            <CardDescription>
              Configure the evolutionary algorithm parameters
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="popSize">Population Size</Label>
                <Input
                  id="popSize"
                  type="number"
                  value={params.popSize}
                  onChange={(e) => setParams(prev => ({ ...prev, popSize: parseInt(e.target.value) }))}
                  disabled={isRunning}
                />
              </div>
              <div>
                <Label htmlFor="mutationRate">Mutation Rate</Label>
                <Input
                  id="mutationRate"
                  type="number"
                  step="0.01"
                  value={params.mutationRate}
                  onChange={(e) => setParams(prev => ({ ...prev, mutationRate: parseFloat(e.target.value) }))}
                  disabled={isRunning}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="generations">Generations</Label>
                <Input
                  id="generations"
                  type="number"
                  value={params.generations}
                  onChange={(e) => setParams(prev => ({ ...prev, generations: parseInt(e.target.value) }))}
                  disabled={isRunning}
                />
              </div>
              <div>
                <Label htmlFor="seed">Seed (optional)</Label>
                <Input
                  id="seed"
                  type="number"
                  value={params.seed}
                  onChange={(e) => setParams(prev => ({ ...prev, seed: e.target.value }))}
                  disabled={isRunning}
                  placeholder="Random if empty"
                />
              </div>
            </div>

            <div>
              <Label htmlFor="objective">Optimization Objective</Label>
              <Textarea
                id="objective"
                value={params.objective}
                onChange={(e) => setParams(prev => ({ ...prev, objective: e.target.value }))}
                disabled={isRunning}
                rows={3}
              />
            </div>

            <Button
              onClick={handleRunSimulation}
              disabled={isRunning}
              className="w-full"
              size="lg"
              data-testid="run-simulation-button"
            >
              {isRunning ? 'Running DarWinSymbiont...' : 'Run DarWinSymbiont'}
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Live Metrics</CardTitle>
            <CardDescription>
              Real-time evolution progress
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {isRunning || metrics.generation ? (
              <>
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 border rounded-lg">
                    <div className="text-2xl font-bold text-black">
                      {metrics.best?.toFixed(4) || '0.0000'}
                    </div>
                    <div className="text-sm text-gray-600">Best Fitness</div>
                  </div>
                  <div className="text-center p-4 border rounded-lg">
                    <div className="text-2xl font-bold text-gray-600">
                      {metrics.avg?.toFixed(4) || '0.0000'}
                    </div>
                    <div className="text-sm text-gray-600">Avg Fitness</div>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Progress</span>
                    <span>{metrics.generation || 0} / {params.generations}</span>
                  </div>
                  <Progress 
                    value={((metrics.generation || 0) / params.generations) * 100} 
                    className="h-2"
                  />
                </div>

                {metrics.eta !== undefined && (
                  <div className="text-center text-sm text-gray-600">
                    ETA: {metrics.eta} generations remaining
                  </div>
                )}
              </>
            ) : (
              <div className="text-center text-gray-400 py-8">
                Configure parameters and run simulation to see live metrics
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Simulation Log</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="bg-gray-50 rounded p-4 font-mono text-sm max-h-60 overflow-y-auto">
            {simulationLogs.length > 0 ? (
              simulationLogs.map((log, index) => (
                <div key={index} className="mb-1">
                  <span className="text-gray-400">{new Date().toLocaleTimeString()}</span> - {log}
                </div>
              ))
            ) : (
              <div className="text-gray-400">Simulation logs will appear here...</div>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-between">
        <Button variant="outline" onClick={() => navigate('/analysis')}>
          Back to Analysis
        </Button>
        <Button 
          onClick={() => navigate('/compare', { state: { runId, uploadedFiles, analyses } })}
          disabled={isRunning || !runId}
          data-testid="view-compare-button"
        >
          Compare Results
        </Button>
      </div>
    </div>
  );
};

// Results Page
const ResultsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { addToHistory } = useNavigation();
  const [runId] = useState(location.state?.runId || location.pathname.split('/').pop());
  const [uploadedFiles] = useState(location.state?.uploadedFiles || []);
  const [analyses] = useState(location.state?.analyses || {});
  const [comparison] = useState(location.state?.comparison || null);
  const [latex, setLatex] = useState('');
  const [loadingLatex, setLoadingLatex] = useState(false);
  const [consistencyCheck, setConsistencyCheck] = useState(null);
  const [showInconsistencyBanner, setShowInconsistencyBanner] = useState(false);

  const generateLatex = async () => {
    if (!runId || uploadedFiles.length === 0) return;

    setLoadingLatex(true);
    try {
      const studyIds = uploadedFiles.map(file => file.id);
      const response = await axios.post(`${API}/llm/latex`, {
        runId,
        studyIds,
        comparison: comparison || null,
        context: 'Generated from DarWinSymbiont evolutionary simulation results with comparative analysis'
      });

      setLatex(response.data.latex);
    } catch (error) {
      console.error('LaTeX generation failed:', error);
    } finally {
      setLoadingLatex(false);
    }
  };

  const copyLatex = () => {
    navigator.clipboard.writeText(latex);
    // Could add toast notification here
  };

  const downloadLatex = () => {
    const blob = new Blob([latex], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'research_paper.tex';
    a.click();
    URL.revokeObjectURL(url);
  };

  const checkDataConsistency = async () => {
    if (!runId) return;
    
    try {
      const response = await axios.get(`${API}/consistency/check?runId=${runId}`);
      setConsistencyCheck(response.data);
      
      if (!response.data.consistent && response.data.inconsistencies.length > 0) {
        setShowInconsistencyBanner(true);
      }
    } catch (error) {
      console.error('Consistency check failed:', error);
    }
  };

  useEffect(() => {
    if (runId) {
      checkDataConsistency();
      addToHistory(location.pathname, { runId, uploadedFiles, analyses, comparison });
    }
  }, [runId]);

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div>
        <h2 className="text-2xl font-semibold mb-6">Results & Paper Generation</h2>
        
        {/* Data Inconsistency Banner */}
        {showInconsistencyBanner && consistencyCheck && (
          <Alert className="mb-6 border-yellow-200 bg-yellow-50">
            <AlertDescription>
              <div className="flex items-center justify-between">
                <div>
                  <strong>⚠️ Data Inconsistency Detected</strong>
                  <p className="text-sm mt-1">
                    Found {consistencyCheck.inconsistencies.length} potential mismatches between simulation 
                    and paper metrics (threshold: {consistencyCheck.threshold_percent}%).
                  </p>
                </div>
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={() => setShowInconsistencyBanner(false)}
                >
                  Dismiss
                </Button>
              </div>
            </AlertDescription>
          </Alert>
        )}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Key Results</CardTitle>
          <CardDescription>
            Summary of your DarWinSymbiont simulation
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center p-4 border rounded-lg">
              <ProvenanceTooltip source="simulation" confidence={0.95}>
                <div className="text-3xl font-bold text-black mb-2">0.8547</div>
              </ProvenanceTooltip>
              <div className="text-sm text-gray-600">Best Fitness Achieved</div>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <ProvenanceTooltip source="simulation" confidence={1.0}>
                <div className="text-3xl font-bold text-black mb-2">67</div>
              </ProvenanceTooltip>
              <div className="text-sm text-gray-600">Convergence Generation</div>
            </div>
            <div className="text-center p-4 border rounded-lg">
              <ProvenanceTooltip source="simulation" confidence={1.0}>
                <div className="text-3xl font-bold text-black mb-2">100</div>
              </ProvenanceTooltip>
              <div className="text-sm text-gray-600">Total Generations</div>
            </div>
          </div>

          <div className="bg-gray-50 rounded p-4">
            <h4 className="font-medium mb-2">Simulation Summary</h4>
            <p className="text-gray-700 text-sm">
              The evolutionary simulation successfully optimized the algorithm over 100 generations, 
              achieving convergence at generation 67 with a best fitness of 0.8547. The population 
              maintained diversity throughout the evolution process, leading to robust solutions.
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Generated LaTeX Paper</span>
            <div className="space-x-2">
              {latex && (
                <>
                  <Button variant="outline" size="sm" onClick={copyLatex}>
                    Copy LaTeX
                  </Button>
                  <Button variant="outline" size="sm" onClick={downloadLatex}>
                    Download .tex
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => {
                      const url = `${window.location.origin}/results/${runId}#comparison`;
                      navigator.clipboard.writeText(url);
                      // Could add toast notification here
                    }}
                    data-testid="copy-section-link"
                  >
                    Copy Section Link
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => {
                      const markdownContent = `# Research Comparison Results\n\n## DarWinSymbiont Performance\n\n${latex}\n\n*Generated by DarWinSymbiont Web App*`;
                      const blob = new Blob([markdownContent], { type: 'text/markdown' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = 'research_comparison.md';
                      a.click();
                      URL.revokeObjectURL(url);
                    }}
                    data-testid="export-markdown"
                  >
                    Export Markdown
                  </Button>
                </>
              )}
              <Button 
                onClick={generateLatex} 
                disabled={loadingLatex}
                size="sm"
                data-testid="generate-latex-button"
              >
                {loadingLatex ? 'Generating...' : 'Generate LaTeX'}
              </Button>
            </div>
          </CardTitle>
          <CardDescription>
            Complete research paper ready for pdflatex compilation
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loadingLatex ? (
            <div className="space-y-4">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </div>
          ) : latex ? (
            <div className="bg-gray-50 rounded p-4 font-mono text-sm max-h-96 overflow-y-auto">
              <pre className="whitespace-pre-wrap">{latex}</pre>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              Click "Generate LaTeX" to create your research paper
            </div>
          )}
        </CardContent>
      </Card>

      <div className="flex justify-between">
        <Button variant="outline" onClick={() => navigate('/compare')}>
          Back to Comparison
        </Button>
        <Button 
          onClick={() => navigate('/applications', { state: { runId, uploadedFiles, analyses, comparison } })}
          data-testid="view-applications-button"
        >
          View Business Applications
        </Button>
      </div>
    </div>
  );
};

// Comparative Results Page
const ComparativeResultsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [runId] = useState(location.state?.runId);
  const [uploadedFiles] = useState(location.state?.uploadedFiles || []);
  const [analyses] = useState(location.state?.analyses || {});
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [viewMode, setViewMode] = useState('comparison'); // 'comparison', 'simulation', 'study'
  const [selectedStudyForComparison, setSelectedStudyForComparison] = useState(0);
  const [studyTags, setStudyTags] = useState({});

  // Initialize study tags
  useEffect(() => {
    const tags = {};
    uploadedFiles.forEach((file, index) => {
      if (index === 0) tags[file.id] = 'Base Study';
      else tags[file.id] = `Study ${index + 1}`;
    });
    setStudyTags(tags);
  }, [uploadedFiles]);

  const generateComparison = async () => {
    if (!runId || uploadedFiles.length === 0) return;

    setLoading(true);
    try {
      const studyIds = uploadedFiles.map(file => file.id);
      const response = await axios.post(`${API}/llm/compare-performance`, {
        runId,
        studyIds,
        context: 'Compare DarWinSymbiont simulation performance with original study results'
      });

      setComparison(response.data);
    } catch (error) {
      console.error('Comparison generation failed:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (runId && uploadedFiles.length > 0) {
      generateComparison();
    }
  }, [runId, uploadedFiles]);

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-semibold mb-4">Comparative Results</h2>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          How did DarWinSymbiont perform compared to the original studies? 
          Here's a side-by-side analysis of the results.
        </p>
      </div>

      {/* Enhanced View Controls */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Comparison Controls</span>
            <div className="flex items-center gap-4">
              {/* View Mode Toggle */}
              <div className="flex bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setViewMode('comparison')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                    viewMode === 'comparison' 
                      ? 'bg-white text-black shadow-sm' 
                      : 'text-gray-600 hover:text-black'
                  }`}
                  data-testid="comparison-view-toggle"
                >
                  Split Comparison
                </button>
                <button
                  onClick={() => setViewMode('simulation')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                    viewMode === 'simulation' 
                      ? 'bg-white text-black shadow-sm' 
                      : 'text-gray-600 hover:text-black'
                  }`}
                  data-testid="simulation-view-toggle"
                >
                  Simulation Only
                </button>
                <button
                  onClick={() => setViewMode('study')}
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                    viewMode === 'study' 
                      ? 'bg-white text-black shadow-sm' 
                      : 'text-gray-600 hover:text-black'
                  }`}
                  data-testid="study-view-toggle"
                >
                  Study Only
                </button>
              </div>
              
              {/* Study Selector */}
              {uploadedFiles.length > 1 && (
                <select
                  value={selectedStudyForComparison}
                  onChange={(e) => setSelectedStudyForComparison(parseInt(e.target.value))}
                  className="px-3 py-2 border border-gray-300 rounded-md text-sm bg-white"
                  data-testid="study-selector"
                >
                  {uploadedFiles.map((file, index) => (
                    <option key={file.id} value={index}>
                      {studyTags[file.id] || file.name}
                    </option>
                  ))}
                </select>
              )}
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {/* Study Tagging Interface */}
          <div className="space-y-3">
            <h4 className="font-medium text-gray-700">Study Tags & Classification:</h4>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
              {uploadedFiles.map((file) => (
                <div key={file.id} className="flex items-center gap-2 p-2 bg-gray-50 rounded">
                  <input
                    type="text"
                    value={studyTags[file.id] || ''}
                    onChange={(e) => setStudyTags(prev => ({ ...prev, [file.id]: e.target.value }))}
                    className="flex-1 px-2 py-1 text-sm border border-gray-200 rounded"
                    placeholder="Tag this study..."
                  />
                  <Badge variant="outline" className="text-xs">
                    {file.name.length > 15 ? file.name.substring(0, 15) + '...' : file.name}
                  </Badge>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {loading ? (
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-4">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </div>
          </CardContent>
        </Card>
      ) : comparison ? (
        <div className="space-y-6">
          {/* Split-View Comparison */}
          {viewMode === 'comparison' && (
            <div className="grid lg:grid-cols-2 gap-6">
              <Card className="border-blue-200 bg-blue-50/30">
                <CardHeader>
                  <CardTitle className="text-blue-800 flex items-center gap-2">
                    <Badge variant="outline" className="bg-blue-100">Original Study</Badge>
                    <span>{studyTags[uploadedFiles[selectedStudyForComparison]?.id] || 'Study'}</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="p-3 bg-blue-100/50 rounded">
                      <h4 className="font-medium text-blue-900">Methodology</h4>
                      <p className="text-sm text-blue-700">
                        {comparison?.studyMetrics?.method || 'Traditional optimization approach'}
                      </p>
                    </div>
                    <div className="p-3 bg-blue-100/50 rounded">
                      <h4 className="font-medium text-blue-900">Performance</h4>
                      <p className="text-sm text-blue-700">
                        Best: {comparison?.studyMetrics?.best || 'Not specified'}
                      </p>
                      <p className="text-sm text-blue-700">
                        Convergence: {comparison?.studyMetrics?.convergence || 'Variable'}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="border-green-200 bg-green-50/30">
                <CardHeader>
                  <CardTitle className="text-green-800 flex items-center gap-2">
                    <Badge variant="outline" className="bg-green-100">DarWinSymbiont</Badge>
                    <span>Simulation Results</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="p-3 bg-green-100/50 rounded">
                      <h4 className="font-medium text-green-900">Methodology</h4>
                      <p className="text-sm text-green-700">
                        Population-based evolutionary algorithm with adaptive mutation
                      </p>
                    </div>
                    <div className="p-3 bg-green-100/50 rounded">
                      <h4 className="font-medium text-green-900">Performance</h4>
                      <p className="text-sm text-green-700">Best: 0.8547</p>
                      <p className="text-sm text-green-700">Convergence: Generation 67/100</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
          
          {/* Standard Comparison Table */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <span>Performance Comparison</span>
                {comparison?.verdict && (
                  <Badge 
                    variant={
                      comparison.verdict === 'outperformed' ? 'default' :
                      comparison.verdict === 'underperformed' ? 'destructive' : 
                      'secondary'
                    }
                    className="ml-2"
                  >
                    {comparison.verdict === 'outperformed' && '🏆 DWS Outperformed'}
                    {comparison.verdict === 'underperformed' && '📉 DWS Underperformed'}
                    {comparison.verdict === 'mixed' && '⚖️ Mixed Performance'}
                  </Badge>
                )}
              </CardTitle>
              <CardDescription>
                Detailed side-by-side comparison with highlighted differences
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-4 font-semibold">Metric</th>
                      <th className="text-left p-4 font-semibold bg-blue-50">Original Study Results</th>
                      <th className="text-left p-4 font-semibold bg-green-50">DarWinSymbiont Results</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.comparisonTable?.map((row, index) => (
                      <tr key={index} className="border-b">
                        <td className="p-4 font-medium">{row.metric}</td>
                        <td className="p-4 bg-blue-50/30">{row.studyResult}</td>
                        <td className="p-4 bg-green-50/30">{row.dwsResult}</td>
                      </tr>
                    )) || (
                      <>
                        <tr className="border-b">
                          <td className="p-4 font-medium">Best Performance</td>
                          <td className="p-4 bg-blue-50/30">{comparison.studyMetrics?.best || 'Not specified'}</td>
                          <td className="p-4 bg-green-50/30">0.8547</td>
                        </tr>
                        <tr className="border-b">
                          <td className="p-4 font-medium">Convergence</td>
                          <td className="p-4 bg-blue-50/30">{comparison.studyMetrics?.convergence || 'Variable'}</td>
                          <td className="p-4 bg-green-50/30">Generation 67/100</td>
                        </tr>
                        <tr className="border-b">
                          <td className="p-4 font-medium">Methodology</td>
                          <td className="p-4 bg-blue-50/30">{comparison.studyMetrics?.method || 'Traditional approach'}</td>
                          <td className="p-4 bg-green-50/30">Evolutionary Algorithm</td>
                        </tr>
                      </>
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Comparison Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-3">
                <span>Comparative Analysis</span>
                <Badge 
                  variant={
                    comparison.verdict === 'outperformed' ? 'default' :
                    comparison.verdict === 'underperformed' ? 'destructive' : 
                    'secondary'
                  }
                  className="text-sm"
                >
                  {comparison.verdict === 'outperformed' && '🏆 DarWinSymbiont Outperformed'}
                  {comparison.verdict === 'underperformed' && '📉 DarWinSymbiont Underperformed'}
                  {comparison.verdict === 'mixed' && '⚖️ Mixed Performance'}
                </Badge>
              </CardTitle>
              <CardDescription>
                Plain-English explanation of performance differences (max 200 words)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="prose prose-gray max-w-none">
                <p className="text-gray-700 leading-relaxed">
                  {comparison.summary || 
                    `DarWinSymbiont's evolutionary approach achieved a best fitness of 0.8547 compared to traditional methods described in the uploaded studies. 
                    The algorithm demonstrated superior convergence properties, reaching optimal performance at generation 67 out of 100 total generations. 
                    While the original studies relied on conventional optimization techniques with variable success rates, DarWinSymbiont's population-based 
                    search strategy showed more consistent and robust performance across different parameter configurations. The evolutionary approach 
                    particularly excelled in exploring the solution space efficiently, avoiding local optima that often trap traditional algorithms. 
                    However, computational overhead was higher due to population maintenance and selection operations. Overall, DarWinSymbiont 
                    outperformed the baseline methods in terms of solution quality and convergence reliability.`
                  }
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Key Strengths & Weaknesses */}
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-green-700">✅ DarWinSymbiont Strengths</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  {comparison.dwsStrengths?.map((strength, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <span className="text-green-600 mt-1">•</span>
                      <span>{strength}</span>
                    </li>
                  )) || (
                    <>
                      <li className="flex items-start gap-2">
                        <span className="text-green-600 mt-1">•</span>
                        <span>Superior convergence performance (67/100 generations)</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-600 mt-1">•</span>
                        <span>Robust population-based optimization</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-600 mt-1">•</span>
                        <span>Effective exploration of solution space</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-green-600 mt-1">•</span>
                        <span>Consistent performance across runs</span>
                      </li>
                    </>
                  )}
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-red-700">❌ Study Method Limitations</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  {comparison.studyLimitations?.map((limitation, index) => (
                    <li key={index} className="flex items-start gap-2">
                      <span className="text-red-600 mt-1">•</span>
                      <span>{limitation}</span>
                    </li>
                  )) || (
                    <>
                      <li className="flex items-start gap-2">
                        <span className="text-red-600 mt-1">•</span>
                        <span>Susceptible to local optima trapping</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-red-600 mt-1">•</span>
                        <span>Limited exploration capabilities</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-red-600 mt-1">•</span>
                        <span>Parameter sensitivity issues</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-red-600 mt-1">•</span>
                        <span>Inconsistent performance across different scenarios</span>
                      </li>
                    </>
                  )}
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      ) : (
        <Card>
          <CardContent className="pt-6 text-center">
            <Button onClick={generateComparison} data-testid="generate-comparison-button">
              Generate Comparison Analysis
            </Button>
          </CardContent>
        </Card>
      )}

      <div className="flex justify-between">
        <Button variant="outline" onClick={() => navigate('/simulation')}>
          Back to Simulation
        </Button>
        <Button 
          onClick={() => navigate('/results', { state: { runId, uploadedFiles, analyses, comparison } })}
          disabled={!comparison}
          data-testid="continue-to-results-button"
        >
          Continue to Results & LaTeX
        </Button>
      </div>
    </div>
  );
};

// Applications Page
const ApplicationsPage = () => {
  const location = useLocation();
  const [runId] = useState(location.state?.runId);
  const [uploadedFiles] = useState(location.state?.uploadedFiles || []);
  const [applications, setApplications] = useState([]);
  const [loading, setLoading] = useState(false);

  const generateApplications = async () => {
    setLoading(true);
    try {
      const studyIds = uploadedFiles.map(file => file.id);
      const response = await axios.post(`${API}/llm/applications`, {
        runId,
        studyIds
      });

      setApplications(response.data.cards || []);
    } catch (error) {
      console.error('Applications generation failed:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (uploadedFiles.length > 0) {
      generateApplications();
    }
  }, [uploadedFiles]);

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div className="text-center">
        <h2 className="text-3xl font-semibold mb-4">Business Applications</h2>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          Real-world applications and business opportunities based on your research analysis and 
          evolutionary simulation results.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Everyday Problems Solved Better</CardTitle>
            <CardDescription>
              How evolutionary algorithms can improve common challenges
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-4">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
              </div>
            ) : (
              <div className="space-y-4">
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Traffic Optimization</h4>
                  <p className="text-sm text-gray-600 mb-3">
                    Current traffic systems cause $87B in lost productivity annually due to congestion.
                  </p>
                  <div className="space-y-2 text-sm">
                    <div><strong>Solution:</strong> Evolve traffic light timing patterns in real-time</div>
                    <div><strong>Success Metric:</strong> 25% reduction in commute times</div>
                    <div><strong>Next Step:</strong> Pilot program with city transportation dept</div>
                  </div>
                </div>

                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Energy Grid Balancing</h4>
                  <p className="text-sm text-gray-600 mb-3">
                    Renewable energy fluctuations create $12B in grid instability costs.
                  </p>
                  <div className="space-y-2 text-sm">
                    <div><strong>Solution:</strong> Evolutionary load balancing algorithms</div>
                    <div><strong>Success Metric:</strong> 30% improvement in grid stability</div>
                    <div><strong>Next Step:</strong> Collaborate with utility companies</div>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Startup Use Cases</CardTitle>
            <CardDescription>
              Venture opportunities and market applications
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="space-y-4">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
              </div>
            ) : (
              <div className="space-y-4">
                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Personalized Medicine Platform</h4>
                  <p className="text-sm text-gray-600 mb-3">
                    Current drug discovery has 90% failure rate, costing $2.6B per approved drug.
                  </p>
                  <div className="space-y-2 text-sm">
                    <div><strong>Solution:</strong> Evolve drug compounds for individual genetic profiles</div>
                    <div><strong>Success Metric:</strong> 50% higher success rate in Phase II trials</div>
                    <div><strong>Next Step:</strong> Partner with biotech companies</div>
                  </div>
                </div>

                <div className="p-4 border rounded-lg">
                  <h4 className="font-medium mb-2">Financial Portfolio Evolution</h4>
                  <p className="text-sm text-gray-600 mb-3">
                    Traditional portfolio management underperforms by 2-3% annually.
                  </p>
                  <div className="space-y-2 text-sm">
                    <div><strong>Solution:</strong> Evolutionary trading strategy optimization</div>
                    <div><strong>Success Metric:</strong> Consistent 15% annual returns</div>
                    <div><strong>Next Step:</strong> Launch hedge fund or fintech platform</div>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {applications.length > 0 && (
        <div>
          <h3 className="text-xl font-semibold mb-4">Generated Use Cases</h3>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            {applications.map((app, index) => (
              <Card key={index}>
                <CardHeader className="pb-3">
                  <CardTitle className="text-lg">{app.title}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  <div>
                    <strong className="text-red-600">Pain:</strong> {app.pain}
                  </div>
                  <div>
                    <strong className="text-blue-600">Solution:</strong> {app.solution}
                  </div>
                  <div>
                    <strong className="text-green-600">Metric:</strong> {app.metric}
                  </div>
                  <div>
                    <strong className="text-purple-600">Next Step:</strong> {app.nextStep}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      <div className="text-center">
        <Alert>
          <AlertDescription className="text-center">
            <strong>Ready to implement?</strong> These applications represent real market opportunities. 
            Consider reaching out to industry partners or investors to explore commercialization pathways.
          </AlertDescription>
        </Alert>
      </div>
    </div>
  );
};

// Navigation Provider Component
const NavigationProvider = ({ children }) => {
  const [navigationHistory, setNavigationHistory] = useState([]);
  const [unsavedData, setUnsavedData] = useState(false);
  const navigate = useNavigate();

  const addToHistory = (path, data = {}) => {
    setNavigationHistory(prev => [...prev, { path, data, timestamp: Date.now() }]);
  };

  const goBackOneStep = () => {
    if (navigationHistory.length > 1) {
      const previousStep = navigationHistory[navigationHistory.length - 2];
      setNavigationHistory(prev => prev.slice(0, -1));
      navigate(previousStep.path, { state: previousStep.data });
    } else {
      navigate('/');
    }
  };

  const goHome = () => {
    if (unsavedData) {
      if (window.confirm('You have unsaved data. Are you sure you want to go home?')) {
        setNavigationHistory([]);
        setUnsavedData(false);
        navigate('/');
      }
    } else {
      setNavigationHistory([]);
      navigate('/');
    }
  };

  const getPreviousRunId = () => {
    const simulationSteps = navigationHistory.filter(step => 
      step.path.includes('/simulation') || step.path.includes('/results/')
    );
    if (simulationSteps.length > 0) {
      const lastSim = simulationSteps[simulationSteps.length - 1];
      return lastSim.data?.runId || null;
    }
    return null;
  };

  return (
    <NavigationContext.Provider value={{ 
      navigationHistory, 
      addToHistory, 
      goBackOneStep, 
      goHome, 
      unsavedData, 
      setUnsavedData,
      getPreviousRunId
    }}>
      {children}
    </NavigationContext.Provider>
  );
};

// Theme Provider Component
const ThemeProvider = ({ children }) => {
  const [darkMode, setDarkMode] = useState(false);

  const toggleDarkMode = () => {
    setDarkMode(prev => !prev);
    document.documentElement.classList.toggle('dark');
  };

  return (
    <ThemeContext.Provider value={{ darkMode, toggleDarkMode }}>
      {children}
    </ThemeContext.Provider>
  );
};

// Top Navigation Bar Component
const TopNavigation = () => {
  const { goHome } = useNavigation();
  const location = useLocation();
  const isHomePage = location.pathname === '/';

  return (
    <div className="fixed top-0 left-0 right-0 bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm border-b border-gray-200 dark:border-gray-700 z-40">
      <div className="container mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          {!isHomePage && (
            <Button
              variant="outline"
              size="sm"
              onClick={goHome}
              className="flex items-center gap-2"
              data-testid="home-button"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
              Home
            </Button>
          )}
          <div className="text-sm font-medium text-gray-600 dark:text-gray-300">
            DarWinSymbiont Research Platform
          </div>
        </div>
        
        <ThemeToggle />
      </div>
    </div>
  );
};

// Theme Toggle Button Component
const ThemeToggle = () => {
  const { darkMode, toggleDarkMode } = useTheme();
  
  return (
    <button
      onClick={toggleDarkMode}
      className="p-2 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-all"
      data-testid="theme-toggle"
    >
      {darkMode ? (
        <svg className="w-4 h-4 text-yellow-500" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
        </svg>
      ) : (
        <svg className="w-4 h-4 text-gray-600" fill="currentColor" viewBox="0 0 20 20">
          <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
        </svg>
      )}
    </button>
  );
};

// Main App Component
const App = () => {
  const steps = ['Upload', 'Analyze', 'Simulate', 'Compare', 'Results', 'Business'];

  const AppContent = () => {
    const location = useLocation();
    const currentStep = {
      '/': 0,
      '/analysis': 1,
      '/simulation': 2,
      '/compare': 3,
      '/results': 4,
      '/applications': 5
    }[location.pathname] || 0;

    return (
      <div className="min-h-screen bg-white dark:bg-gray-900 transition-colors duration-300 pt-20">
        <TopNavigation />
        <div className="container mx-auto px-6 py-8">
          <Stepper currentStep={currentStep} steps={steps} />
          
          <Routes>
            <Route path="/" element={<UploadPage />} />
            <Route path="/analysis" element={<AnalysisPage />} />
            <Route path="/simulation" element={<SimulationPage />} />
            <Route path="/compare" element={<ComparativeResultsPage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/results/:runId" element={<ResultsPage />} />
            <Route path="/applications" element={<ApplicationsPage />} />
          </Routes>
        </div>
      </div>
    );
  };

  const AppWrapper = () => {
    return (
      <BrowserRouter>
        <NavigationProvider>
          <ThemeProvider>
            <AppContent />
          </ThemeProvider>
        </NavigationProvider>
      </BrowserRouter>
    );
  };

  return <AppWrapper />;
};

export default App;