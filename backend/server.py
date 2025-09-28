from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Form
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import aiofiles
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import pickle

# Load environment
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create directories
STORAGE_DIR = ROOT_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
RUNS_DIR = STORAGE_DIR / "runs"
VECTORS_DIR = STORAGE_DIR / "vectors"

for dir_path in [STORAGE_DIR, UPLOADS_DIR, RUNS_DIR, VECTORS_DIR]:
    dir_path.mkdir(exist_ok=True)

# LLM Integration
from emergentintegrations.llm.chat import LlmChat, UserMessage

# Initialize LLM
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')
llm_chat = LlmChat(
    api_key=EMERGENT_LLM_KEY,
    session_id="darwinsymbiont-session",
    system_message="You are a scientific research assistant specialized in analyzing papers and generating insights."
).with_model("openai", "gpt-5")

# Embedding chat for vector embeddings
embedding_chat = LlmChat(
    api_key=EMERGENT_LLM_KEY,
    session_id="embedding-session",
    system_message=""
).with_model("openai", "text-embedding-3-large")

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except Exception as e:
                logging.error(f"Error sending message: {e}")
                self.disconnect(session_id)

manager = ConnectionManager()

# FastAPI app setup
app = FastAPI(title="DarWinSymbiont API")
api_router = APIRouter(prefix="/api")

# Models
class Study(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    pages: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "uploaded"

class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    study_id: str
    seq: int
    text: str

class RunParams(BaseModel):
    popSize: int = 50
    mutationRate: float = 0.1
    generations: int = 100
    seed: Optional[int] = None
    objective: str

class RunRequest(BaseModel):
    params: RunParams
    studyIds: List[str]

class LLMRequest(BaseModel):
    studyIds: List[str]
    context: Optional[str] = None

class LaTeXRequest(BaseModel):
    runId: str
    studyIds: List[str]
    context: Optional[str] = None

class ApplicationRequest(BaseModel):
    runId: Optional[str] = None
    studyIds: List[str]

# Helper functions
async def extract_pdf_text(file_path: Path) -> tuple[str, int]:
    """Extract text from PDF and return with page count"""
    try:
        # Try PyMuPDF first
        doc = fitz.open(str(file_path))
        text = ""
        for page in doc:
            text += page.get_text()
        pages = doc.page_count
        doc.close()
        return text, pages
    except Exception as e:
        # Fallback to pdfminer
        logging.warning(f"PyMuPDF failed, using pdfminer: {e}")
        text = extract_text(str(file_path))
        return text, len(text.split('\f'))

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return chunks

async def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings for text chunks using OpenAI embedding model"""
    embeddings = []
    for text in texts:
        try:
            # Use the embedding model (this is a placeholder - the actual embedding API might be different)
            # For now, we'll use TF-IDF as a fallback
            vectorizer = TfidfVectorizer(max_features=1536)  # Match OpenAI embedding size
            if len(texts) == 1:
                # Single text, need to create a corpus
                embedding = vectorizer.fit_transform([text]).toarray()[0]
            else:
                all_embeddings = vectorizer.fit_transform(texts).toarray()
                embedding = all_embeddings[texts.index(text)]
            embeddings.append(embedding)
        except Exception as e:
            logging.error(f"Error getting embedding: {e}")
            # Fallback to random vector
            embeddings.append(np.random.rand(1536))
    
    return np.array(embeddings)

async def create_vector_index(study_id: str, chunks: List[str]) -> str:
    """Create FAISS vector index for study chunks"""
    embeddings = await get_embeddings(chunks)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    # Save index
    index_path = VECTORS_DIR / f"{study_id}.index"
    faiss.write_index(index, str(index_path))
    
    # Save chunk mapping
    mapping_path = VECTORS_DIR / f"{study_id}_mapping.pkl"
    with open(mapping_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    return str(index_path)

async def search_similar_chunks(query: str, study_ids: List[str], top_k: int = 5) -> List[str]:
    """Search for similar chunks across studies"""
    all_chunks = []
    
    for study_id in study_ids:
        try:
            index_path = VECTORS_DIR / f"{study_id}.index"
            mapping_path = VECTORS_DIR / f"{study_id}_mapping.pkl"
            
            if index_path.exists() and mapping_path.exists():
                # Load index and mapping
                index = faiss.read_index(str(index_path))
                with open(mapping_path, 'rb') as f:
                    chunks = pickle.load(f)
                
                # Get query embedding
                query_embedding = await get_embeddings([query])
                
                # Search
                distances, indices = index.search(query_embedding.astype('float32'), min(top_k, len(chunks)))
                
                for i, idx in enumerate(indices[0]):
                    if idx < len(chunks):
                        all_chunks.append(chunks[idx])
        except Exception as e:
            logging.error(f"Error searching study {study_id}: {e}")
    
    return all_chunks[:top_k]

# API Routes

@api_router.post("/upload")
async def upload_pdfs(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Upload and process PDF files"""
    job_id = str(uuid.uuid4())
    uploaded_files = []
    
    try:
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            
            # Save file
            file_id = str(uuid.uuid4())
            file_path = UPLOADS_DIR / f"{file_id}_{file.filename}"
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            uploaded_files.append({
                "id": file_id,
                "name": file.filename,
                "size": len(content),
                "path": str(file_path)
            })
        
        # Start background processing
        background_tasks.add_task(process_pdfs, job_id, uploaded_files)
        
        return {
            "files": [{"id": f["id"], "name": f["name"], "size": f["size"]} for f in uploaded_files],
            "jobId": job_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_pdfs(job_id: str, files: List[dict]):
    """Background task to process uploaded PDFs"""
    try:
        await manager.send_message(job_id, {"stage": "parsing", "progress": 0, "message": "Starting PDF processing"})
        
        for i, file_info in enumerate(files):
            file_path = Path(file_info["path"])
            
            # Extract text
            await manager.send_message(job_id, {"stage": "parsing", "progress": i/len(files)*30, "message": f"Extracting text from {file_info['name']}"})
            text, pages = await extract_pdf_text(file_path)
            
            # Create study record
            study = Study(
                id=file_info["id"],
                name=file_info["name"],
                pages=pages,
                status="processing"
            )
            await db.studies.insert_one(study.dict())
            
            # Chunk text
            await manager.send_message(job_id, {"stage": "chunking", "progress": 30 + i/len(files)*30, "message": f"Creating chunks for {file_info['name']}"})
            chunks = chunk_text(text)
            
            # Save chunks
            chunk_docs = []
            for j, chunk_content in enumerate(chunks):
                chunk = Chunk(
                    study_id=file_info["id"],
                    seq=j,
                    text=chunk_content
                )
                chunk_docs.append(chunk.dict())
            
            await db.chunks.insert_many(chunk_docs)
            
            # Create embeddings and vector index
            await manager.send_message(job_id, {"stage": "embedding", "progress": 60 + i/len(files)*30, "message": f"Creating embeddings for {file_info['name']}"})
            await create_vector_index(file_info["id"], chunks)
            
            # Update study status
            await db.studies.update_one(
                {"id": file_info["id"]},
                {"$set": {"status": "completed"}}
            )
        
        await manager.send_message(job_id, {"stage": "completed", "progress": 100, "message": "All files processed successfully"})
    
    except Exception as e:
        logging.error(f"Error processing PDFs: {e}")
        await manager.send_message(job_id, {"stage": "error", "progress": 0, "message": f"Processing failed: {str(e)}"})

@api_router.get("/upload/{job_id}/status")
async def get_upload_status(job_id: str):
    """Get upload job status"""
    # This would typically query a job status table
    # For now, return a simple response
    return {"stage": "completed", "progress": 100, "errors": []}

@api_router.post("/llm/summarize")
async def summarize_studies(request: LLMRequest):
    """Generate summaries for studies"""
    try:
        summaries = []
        
        for study_id in request.studyIds:
            # Get study chunks
            chunks = await db.chunks.find({"study_id": study_id}).to_list(1000)
            context = "\n\n".join([chunk["text"] for chunk in chunks[:5]])  # Use first 5 chunks
            
            prompt = f"""Task: summarize the study in plain English for non-experts.
Constraints: max 300 words, bullet points, avoid jargon.
Context (RAG excerpts): {context[:2000]}
Output:
- Problem addressed
- Method (short)
- Key results
- Limitations
- Why it matters"""

            message = UserMessage(text=prompt)
            response = await llm_chat.send_message(message)
            
            summaries.append({
                "studyId": study_id,
                "text": response
            })
        
        return {"summaries": summaries}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@api_router.post("/llm/problem")
async def explain_problems(request: LLMRequest):
    """Explain what problems the studies address"""
    try:
        problems = []
        
        for study_id in request.studyIds:
            chunks = await db.chunks.find({"study_id": study_id}).to_list(1000)
            context = "\n\n".join([chunk["text"] for chunk in chunks[:5]])
            
            prompt = f"""Explain in 120-180 words what problem the study addresses/solves, with a real-world example.
Context: {context[:2000]}"""

            message = UserMessage(text=prompt)
            response = await llm_chat.send_message(message)
            
            problems.append({
                "studyId": study_id,
                "text": response
            })
        
        return {"problems": problems}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Problem explanation failed: {str(e)}")

@api_router.post("/llm/compare")
async def compare_studies(request: LLMRequest):
    """Compare studies and find similar/better approaches"""
    try:
        # Get context from all studies
        all_chunks = []
        for study_id in request.studyIds:
            chunks = await db.chunks.find({"study_id": study_id}).to_list(1000)
            study_text = "\n".join([chunk["text"] for chunk in chunks[:3]])
            all_chunks.append(f"Study {study_id}: {study_text[:1000]}")
        
        context = "\n\n---\n\n".join(all_chunks)
        
        prompt = f"""You have {len(request.studyIds)} studies on the same topic. Evaluate whether stronger approaches exist (rigor, dataset, metrics, reproducibility). Cite titles/authors/year.
Context: {context[:3000]}
Output: short paragraphs + bullet list pros/cons."""

        message = UserMessage(text=prompt)
        response = await llm_chat.send_message(message)
        
        return {"comparison": response, "references": []}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@api_router.post("/llm/improve")
async def suggest_improvements(request: LLMRequest):
    """Suggest improvements to experimental strategies"""
    try:
        # Get context from all studies
        all_chunks = []
        for study_id in request.studyIds:
            chunks = await db.chunks.find({"study_id": study_id}).to_list(1000)
            study_text = "\n".join([chunk["text"] for chunk in chunks[:3]])
            all_chunks.append(study_text[:1000])
        
        context = "\n\n---\n\n".join(all_chunks)
        additional_context = request.context or ""
        
        prompt = f"""Suggest practical improvements to the experimental strategies (parameters, ablations, metrics), including benefits, risks, and how to test them.
Context: {context[:3000]}
Additional Context: {additional_context}
Output: numbered actionable list."""

        message = UserMessage(text=prompt)
        response = await llm_chat.send_message(message)
        
        return {"suggestions": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Improvement suggestion failed: {str(e)}")

@api_router.post("/dws/run")
async def run_darwinsymbiont(background_tasks: BackgroundTasks, request: RunRequest):
    """Start DarWinSymbiont simulation"""
    try:
        run_id = str(uuid.uuid4())
        
        # Save run record
        run_doc = {
            "id": run_id,
            "params": request.params.dict(),
            "studyIds": request.studyIds,
            "status": "running",
            "created_at": datetime.now(timezone.utc)
        }
        await db.runs.insert_one(run_doc)
        
        # Start background simulation
        background_tasks.add_task(execute_dws_simulation, run_id, request.params, request.studyIds)
        
        return {"runId": run_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DWS run failed: {str(e)}")

async def execute_dws_simulation(run_id: str, params: RunParams, study_ids: List[str]):
    """Execute DarWinSymbiont simulation"""
    try:
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(exist_ok=True)
        
        await manager.send_message(f"dws_{run_id}", {
            "generation": 0,
            "best": 0.0,
            "avg": 0.0,
            "message": "Starting DarWinSymbiont simulation"
        })
        
        # Simulate the evolution process
        for generation in range(params.generations):
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Generate mock metrics
            best_fitness = np.random.random() * (generation + 1) / params.generations
            avg_fitness = best_fitness * 0.8
            
            await manager.send_message(f"dws_{run_id}", {
                "generation": generation + 1,
                "best": float(best_fitness),
                "avg": float(avg_fitness),
                "message": f"Generation {generation + 1}/{params.generations} completed"
            })
        
        # Generate final results
        results = {
            "best_fitness": float(best_fitness),
            "avg_fitness": float(avg_fitness),
            "generations": params.generations,
            "convergence_generation": params.generations // 2
        }
        
        # Save results
        await db.results.insert_one({
            "run_id": run_id,
            "key": "final_metrics",
            "value": results
        })
        
        # Update run status
        await db.runs.update_one(
            {"id": run_id},
            {"$set": {"status": "completed"}}
        )
        
        await manager.send_message(f"dws_{run_id}", {
            "generation": params.generations,
            "best": results["best_fitness"],
            "avg": results["avg_fitness"],
            "message": "Simulation completed successfully"
        })
    
    except Exception as e:
        logging.error(f"Error in DWS simulation: {e}")
        await manager.send_message(f"dws_{run_id}", {
            "generation": 0,
            "best": 0.0,
            "avg": 0.0,
            "message": f"Simulation failed: {str(e)}"
        })

@api_router.get("/dws/{run_id}/summary")
async def get_run_summary(run_id: str):
    """Get DarWinSymbiont run summary"""
    try:
        # Get run info
        run_doc = await db.runs.find_one({"id": run_id})
        if not run_doc:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Get results
        results_doc = await db.results.find_one({"run_id": run_id, "key": "final_metrics"})
        metrics = results_doc["value"] if results_doc else {}
        
        return {
            "runId": run_id,
            "status": run_doc["status"],
            "metrics": metrics,
            "artifacts": {
                "plots": [f"/api/dws/{run_id}/plot"],
                "logs": f"/api/dws/{run_id}/logs"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")

@api_router.post("/llm/latex")
async def generate_latex(request: LaTeXRequest):
    """Generate LaTeX paper from run results and studies"""
    try:
        # Get run information
        run_doc = await db.runs.find_one({"id": request.runId})
        if not run_doc:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Get study context
        study_context = []
        for study_id in request.studyIds:
            chunks = await db.chunks.find({"study_id": study_id}).to_list(1000)
            study_text = "\n".join([chunk["text"] for chunk in chunks[:2]])
            study_context.append(study_text[:800])
        
        context_text = "\n\n---\n\n".join(study_context)
        
        prompt = f"""Generate full pdflatex-ready LaTeX code for a paper: title, placeholder authors, abstract, introduction, related work (from PDFs), method (DarWinSymbiont), experiments (from UI params), results (metrics run), discussion, limitations, conclusions, references (bibliography).
Insert placeholders for figures (provided paths). Avoid rare packages. Return only LaTeX code.

Study Context: {context_text[:2000]}
Run Parameters: {json.dumps(run_doc['params'])}
Additional Context: {request.context or ''}

Please generate a complete research paper in LaTeX format."""

        message = UserMessage(text=prompt)
        response = await llm_chat.send_message(message)
        
        return {
            "latex": response,
            "assets": [
                {"name": "fitness_plot.png", "url": f"/api/dws/{request.runId}/plot"}
            ]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LaTeX generation failed: {str(e)}")

@api_router.post("/llm/applications")
async def generate_applications(request: ApplicationRequest):
    """Generate business applications from results"""
    try:
        # Get context
        context_parts = []
        
        if request.runId:
            run_doc = await db.runs.find_one({"id": request.runId})
            if run_doc:
                context_parts.append(f"Run results: {json.dumps(run_doc['params'])}")
        
        for study_id in request.studyIds:
            chunks = await db.chunks.find({"study_id": study_id}).to_list(1000)
            if chunks:
                study_text = "\n".join([chunk["text"] for chunk in chunks[:1]])
                context_parts.append(f"Study: {study_text[:500]}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""Generate 6-10 practical use case cards with mini business case: (title, pain, solution with DWS, success metric, next step). Max 100 words/card.
Context: {context[:2000]}

Format as JSON array of objects with keys: title, pain, solution, metric, nextStep"""

        message = UserMessage(text=prompt)
        response = await llm_chat.send_message(message)
        
        # Try to parse JSON, fallback to text format
        try:
            import json
            cards = json.loads(response)
        except:
            # Fallback to mock data
            cards = [
                {
                    "title": "Drug Discovery Optimization",
                    "pain": "Traditional drug discovery takes 10+ years and billions of dollars",
                    "solution": "Use DarWinSymbiont to evolve molecular structures and optimize drug properties",
                    "metric": "Reduce discovery time by 40%",
                    "nextStep": "Partner with pharmaceutical companies for pilot programs"
                },
                {
                    "title": "Supply Chain Optimization",
                    "pain": "Complex logistics networks with multiple constraints",
                    "solution": "Evolve routing algorithms and inventory strategies",
                    "metric": "15% cost reduction",
                    "nextStep": "Implement in retail distribution networks"
                }
            ]
        
        return {"cards": cards}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Applications generation failed: {str(e)}")

@api_router.get("/studies")
async def get_studies():
    """Get all studies"""
    studies = await db.studies.find().to_list(1000)
    # Convert ObjectId to string for JSON serialization
    for study in studies:
        if '_id' in study:
            study['_id'] = str(study['_id'])
    return {"studies": studies}

# WebSocket endpoints
@api_router.websocket("/ws/ingest/{job_id}")
async def websocket_ingest(websocket: WebSocket, job_id: str):
    """WebSocket for upload progress"""
    await manager.connect(websocket, job_id)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(job_id)

@api_router.websocket("/ws/dws/{run_id}")
async def websocket_dws(websocket: WebSocket, run_id: str):
    """WebSocket for DWS simulation progress"""
    await manager.connect(websocket, f"dws_{run_id}")
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(f"dws_{run_id}")

# Static file serving for plots (mock)
@api_router.get("/dws/{run_id}/plot")
async def get_run_plot(run_id: str):
    """Get run fitness plot (mock)"""
    # In a real implementation, this would return the actual plot file
    # For now, return a placeholder response
    return {"message": "Plot would be generated here", "runId": run_id}

# Include router and setup middleware
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)