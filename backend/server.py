"""
EMERGENT: AI-Powered Optimization Platform - Backend Server
"""
import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routes
from routes import problem, analysis


# Global database client
db_client = None
database = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global db_client, database
    
    # Startup
    mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017/emergent_platform")
    db_client = AsyncIOMotorClient(mongo_url)
    database = db_client.get_database()
    
    print(f"✅ Connected to MongoDB: {mongo_url}")
    print(f"✅ Database: {database.name}")
    
    yield
    
    # Shutdown
    if db_client:
        db_client.close()
        print("✅ Closed MongoDB connection")


# Create FastAPI app
app = FastAPI(
    title="EMERGENT: AI-Powered Optimization Platform",
    description="Problem-aware evolutionary algorithm platform with LLM analysis",
    version="1.0.0",
    lifespan=lifespan
)


# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to inject database
async def get_database() -> AsyncIOMotorDatabase:
    """Get database instance"""
    return database


# Include routers with database dependency
problem.get_db = get_database
analysis.get_database = get_database

app.include_router(problem.router)
app.include_router(analysis.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "EMERGENT: AI-Powered Optimization Platform API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    db_status = "connected" if database is not None else "disconnected"
    return {
        "status": "healthy",
        "database": db_status,
        "llm": "emergent_universal_key"
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("BACKEND_PORT", 8001))
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
