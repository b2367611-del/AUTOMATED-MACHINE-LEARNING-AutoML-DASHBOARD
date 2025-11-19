from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from pathlib import Path

from app.routers import upload, train, evaluate, download, quality
from app.core.config import Settings

# Initialize settings
settings = Settings()

# Create FastAPI app
app = FastAPI(
    title="AutoML Platform",
    description="Automated Machine Learning Platform with FastAPI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Include routers
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(train.router, prefix="/api/train", tags=["training"])
app.include_router(evaluate.router, prefix="/api/evaluate", tags=["evaluation"])
app.include_router(download.router, prefix="/api/download", tags=["download"])
app.include_router(quality.router, prefix="/api/quality", tags=["quality"])

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )