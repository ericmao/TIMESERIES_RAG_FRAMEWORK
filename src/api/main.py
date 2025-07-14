"""
FastAPI Application for the Time Series RAG Framework
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import asyncio
import uvicorn
from datetime import datetime
import json

from ..agents.master_agent import MasterAgent
from ..agents.forecasting_agent import ForecastingAgent
from ..agents.anomaly_detection_agent import AnomalyDetectionAgent
from ..agents.classification_agent import ClassificationAgent
from ..utils.prompt_manager import PromptManager
from ..config.config import get_config
from ..utils.logger import get_logger

# Initialize FastAPI app
app = FastAPI(
    title="Time Series RAG Framework API",
    description="A comprehensive API for time series analysis using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
master_agent: Optional[MasterAgent] = None
prompt_manager: Optional[PromptManager] = None
logger = get_logger("api")

# Pydantic models for request/response
class TimeSeriesData(BaseModel):
    """Time series data model"""
    ds: List[str] = Field(..., description="Date/time values")
    y: List[float] = Field(..., description="Time series values")

class AnalysisRequest(BaseModel):
    """Analysis request model"""
    data: TimeSeriesData = Field(..., description="Time series data")
    task_type: Optional[str] = Field(None, description="Type of analysis task")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Analysis parameters")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")

class ForecastingRequest(BaseModel):
    """Forecasting request model"""
    data: TimeSeriesData = Field(..., description="Time series data")
    forecast_horizon: Optional[int] = Field(24, description="Forecast horizon")
    method: Optional[str] = Field("prophet", description="Forecasting method")

class AnomalyDetectionRequest(BaseModel):
    """Anomaly detection request model"""
    data: TimeSeriesData = Field(..., description="Time series data")
    method: Optional[str] = Field("combined", description="Detection method")
    threshold: Optional[float] = Field(0.95, description="Detection threshold")
    window_size: Optional[int] = Field(10, description="Window size for analysis")

class ClassificationRequest(BaseModel):
    """Classification request model"""
    data: TimeSeriesData = Field(..., description="Time series data")
    method: Optional[str] = Field("comprehensive", description="Classification method")
    classification_type: Optional[str] = Field("pattern", description="Type of classification")

class PromptRequest(BaseModel):
    """Prompt management request model"""
    title: str = Field(..., description="Prompt title")
    description: str = Field(..., description="Prompt description")
    template: str = Field(..., description="Prompt template")
    agent_type: str = Field(..., description="Agent type")
    task_type: str = Field(..., description="Task type")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Template parameters")
    tags: Optional[List[str]] = Field(default=[], description="Prompt tags")

class AgentStatus(BaseModel):
    """Agent status model"""
    agent_id: str
    agent_type: str
    is_initialized: bool
    total_requests: int
    successful_requests: int
    avg_response_time: float

class SystemStatus(BaseModel):
    """System status model"""
    status: str
    timestamp: str
    agents: Dict[str, AgentStatus]
    prompt_manager: Dict[str, Any]

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global master_agent, prompt_manager
    
    try:
        logger.info("Starting Time Series RAG Framework API...")
        
        # Initialize prompt manager
        prompt_manager = PromptManager()
        logger.info("Prompt manager initialized")
        
        # Initialize master agent
        master_agent = MasterAgent()
        await master_agent.initialize()
        logger.info("Master agent initialized")
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global master_agent
    
    try:
        if master_agent:
            await master_agent.cleanup()
            logger.info("Master agent cleaned up")
        
        logger.info("API shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

# Dependency functions
async def get_master_agent() -> MasterAgent:
    """Get the master agent instance"""
    if not master_agent:
        raise HTTPException(status_code=503, detail="Master agent not initialized")
    return master_agent

async def get_prompt_manager() -> PromptManager:
    """Get the prompt manager instance"""
    if not prompt_manager:
        raise HTTPException(status_code=503, detail="Prompt manager not initialized")
    return prompt_manager

# Health check endpoint
@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# System status endpoint
@app.get("/status", response_model=SystemStatus)
async def get_system_status(
    agent: MasterAgent = Depends(get_master_agent),
    pm: PromptManager = Depends(get_prompt_manager)
):
    """Get system status"""
    try:
        agent_status = await agent.get_agent_status()
        prompt_stats = pm.get_statistics()
        
        return SystemStatus(
            status="operational",
            timestamp=datetime.now().isoformat(),
            agents=agent_status.get("specialized_agents", {}),
            prompt_manager=prompt_stats
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Comprehensive analysis endpoint
@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_timeseries(
    request: AnalysisRequest,
    agent: MasterAgent = Depends(get_master_agent)
):
    """Perform comprehensive time series analysis"""
    try:
        # Convert data to DataFrame format
        data = {
            "ds": request.data.ds,
            "y": request.data.y
        }
        
        # Prepare request
        analysis_request = {
            "data": data,
            "task_type": request.task_type or "comprehensive_analysis",
            **request.parameters
        }
        
        # Process request
        response = await agent.process_request(analysis_request, request.context)
        
        if not response.success:
            raise HTTPException(status_code=400, detail=response.message)
        
        return response.data
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Forecasting endpoint
@app.post("/forecast", response_model=Dict[str, Any])
async def forecast_timeseries(
    request: ForecastingRequest,
    agent: MasterAgent = Depends(get_master_agent)
):
    """Perform time series forecasting"""
    try:
        # Convert data to DataFrame format
        data = {
            "ds": request.data.ds,
            "y": request.data.y
        }
        
        # Prepare request
        forecast_request = {
            "data": data,
            "task_type": "forecasting",
            "forecast_horizon": request.forecast_horizon,
            "method": request.method
        }
        
        # Process request
        response = await agent.process_request(forecast_request)
        
        if not response.success:
            raise HTTPException(status_code=400, detail=response.message)
        
        return response.data
        
    except Exception as e:
        logger.error(f"Forecasting failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Anomaly detection endpoint
@app.post("/detect-anomalies", response_model=Dict[str, Any])
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    agent: MasterAgent = Depends(get_master_agent)
):
    """Detect anomalies in time series data"""
    try:
        # Convert data to DataFrame format
        data = {
            "ds": request.data.ds,
            "y": request.data.y
        }
        
        # Prepare request
        anomaly_request = {
            "data": data,
            "task_type": "anomaly_detection",
            "method": request.method,
            "threshold": request.threshold,
            "window_size": request.window_size
        }
        
        # Process request
        response = await agent.process_request(anomaly_request)
        
        if not response.success:
            raise HTTPException(status_code=400, detail=response.message)
        
        return response.data
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Classification endpoint
@app.post("/classify", response_model=Dict[str, Any])
async def classify_timeseries(
    request: ClassificationRequest,
    agent: MasterAgent = Depends(get_master_agent)
):
    """Classify time series patterns"""
    try:
        # Convert data to DataFrame format
        data = {
            "ds": request.data.ds,
            "y": request.data.y
        }
        
        # Prepare request
        classification_request = {
            "data": data,
            "task_type": "classification",
            "method": request.method,
            "classification_type": request.classification_type
        }
        
        # Process request
        response = await agent.process_request(classification_request)
        
        if not response.success:
            raise HTTPException(status_code=400, detail=response.message)
        
        return response.data
        
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Prompt management endpoints
@app.post("/prompts", response_model=Dict[str, str])
async def create_prompt(
    request: PromptRequest,
    pm: PromptManager = Depends(get_prompt_manager)
):
    """Create a new prompt template"""
    try:
        prompt_id = pm.create_prompt(
            title=request.title,
            description=request.description,
            template=request.template,
            agent_type=request.agent_type,
            task_type=request.task_type,
            parameters=request.parameters,
            tags=request.tags
        )
        
        return {"prompt_id": prompt_id}
        
    except Exception as e:
        logger.error(f"Failed to create prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompts", response_model=List[Dict[str, Any]])
async def list_prompts(
    agent_type: Optional[str] = None,
    task_type: Optional[str] = None,
    tags: Optional[str] = None,
    pm: PromptManager = Depends(get_prompt_manager)
):
    """List prompt templates with optional filtering"""
    try:
        if agent_type:
            prompts = pm.get_prompts_by_agent(agent_type)
        elif task_type:
            prompts = pm.get_prompts_by_task(task_type)
        elif tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
            prompts = pm.get_prompts_by_tags(tag_list)
        else:
            prompts = list(pm.prompts_cache.values())
        
        return [prompt.__dict__ for prompt in prompts]
        
    except Exception as e:
        logger.error(f"Failed to list prompts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompts/{prompt_id}", response_model=Dict[str, Any])
async def get_prompt(
    prompt_id: str,
    pm: PromptManager = Depends(get_prompt_manager)
):
    """Get a specific prompt template"""
    try:
        prompt = pm.get_prompt(prompt_id)
        if not prompt:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        return prompt.__dict__
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prompt {prompt_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/prompts/{prompt_id}")
async def update_prompt(
    prompt_id: str,
    request: PromptRequest,
    pm: PromptManager = Depends(get_prompt_manager)
):
    """Update a prompt template"""
    try:
        success = pm.update_prompt(
            prompt_id=prompt_id,
            title=request.title,
            description=request.description,
            template=request.template,
            parameters=request.parameters,
            tags=request.tags
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        return {"message": "Prompt updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update prompt {prompt_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/prompts/{prompt_id}")
async def delete_prompt(
    prompt_id: str,
    pm: PromptManager = Depends(get_prompt_manager)
):
    """Delete a prompt template"""
    try:
        success = pm.delete_prompt(prompt_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Prompt not found")
        
        return {"message": "Prompt deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete prompt {prompt_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompts/optimize", response_model=Dict[str, Any])
async def optimize_prompts(
    pm: PromptManager = Depends(get_prompt_manager)
):
    """Get prompt optimization suggestions"""
    try:
        return pm.optimize_prompts()
        
    except Exception as e:
        logger.error(f"Failed to optimize prompts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@app.get("/agents", response_model=Dict[str, Any])
async def get_agents_status(
    agent: MasterAgent = Depends(get_master_agent)
):
    """Get status of all agents"""
    try:
        return await agent.get_agent_status()
        
    except Exception as e:
        logger.error(f"Failed to get agents status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/render-prompt", response_model=Dict[str, str])
async def render_prompt(
    prompt_id: str,
    parameters: Dict[str, Any],
    pm: PromptManager = Depends(get_prompt_manager)
):
    """Render a prompt template with parameters"""
    try:
        rendered = pm.render_prompt(prompt_id, parameters)
        
        if not rendered:
            raise HTTPException(status_code=404, detail="Prompt not found or rendering failed")
        
        return {"rendered_prompt": rendered}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to render prompt {prompt_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Main function for running the server
def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server"""
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    run_server() 