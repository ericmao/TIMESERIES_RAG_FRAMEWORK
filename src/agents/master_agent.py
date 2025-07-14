"""
Master Agent for orchestrating specialized agents in the Time Series RAG Framework
"""
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json

from .base_agent import BaseAgent, AgentResponse
from .forecasting_agent import ForecastingAgent
from .anomaly_detection_agent import AnomalyDetectionAgent
from .classification_agent import ClassificationAgent
from ..config.config import get_config
from ..utils.logger import get_logger

class MasterAgent(BaseAgent):
    """
    Master agent that orchestrates specialized agents for comprehensive time series analysis.
    
    This agent acts as a coordinator, routing requests to appropriate specialized agents
    and combining their responses for comprehensive analysis.
    """
    
    def __init__(self, agent_id: str = "master", model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type="master",
            model_name=model_name or get_config().model.master_agent_model,
            config=config
        )
        self.logger = get_logger(f"master_agent_{agent_id}")
        
        # Initialize specialized agents
        self.specialized_agents = {}
        self.agent_tasks = {}
        
        # Task routing configuration
        self.task_routing = {
            "forecast": "forecasting",
            "predict": "forecasting",
            "anomaly": "anomaly_detection",
            "detect_anomaly": "anomaly_detection",
            "classify": "classification",
            "trend": "trend_analysis",
            "seasonality": "seasonality_analysis",
            "comprehensive": "comprehensive_analysis"
        }
    
    async def initialize(self) -> bool:
        """Initialize master agent and all specialized agents"""
        try:
            # Initialize base agent
            base_initialized = await super().initialize()
            if not base_initialized:
                return False
            
            # Initialize specialized agents
            await self._initialize_specialized_agents()
            
            self.logger.info(f"Master agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize master agent: {str(e)}")
            return False
    
    async def _initialize_specialized_agents(self) -> None:
        """Initialize all specialized agents"""
        try:
            # Initialize forecasting agent
            self.specialized_agents["forecasting"] = ForecastingAgent(
                agent_id="forecasting_001",
                config=self.config
            )
            await self.specialized_agents["forecasting"].initialize()
            
            # Initialize anomaly detection agent
            self.specialized_agents["anomaly_detection"] = AnomalyDetectionAgent(
                agent_id="anomaly_001",
                config=self.config
            )
            await self.specialized_agents["anomaly_detection"].initialize()
            
            # Initialize classification agent
            self.specialized_agents["classification"] = ClassificationAgent(
                agent_id="classification_001",
                config=self.config
            )
            await self.specialized_agents["classification"].initialize()
            
            # TODO: Initialize other agents as they are implemented
            # self.specialized_agents["trend_analysis"] = TrendAnalysisAgent(...)
            # self.specialized_agents["seasonality_analysis"] = SeasonalityAnalysisAgent(...)
            
            self.logger.info(f"Initialized {len(self.specialized_agents)} specialized agents")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize specialized agents: {str(e)}")
            raise
    
    async def _process_request_internal(
        self,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        relevant_prompts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process requests by routing to appropriate specialized agents
        
        Args:
            request: The request to process
            context: Additional context information
            relevant_prompts: Retrieved prompts for guidance
            
        Returns:
            Dict with comprehensive analysis results
        """
        try:
            # Determine task type and route to appropriate agent
            task_type = self._determine_task_type(request)
            
            if task_type == "comprehensive_analysis":
                return await self._perform_comprehensive_analysis(request, context, relevant_prompts)
            elif task_type in self.specialized_agents:
                return await self._route_to_specialized_agent(task_type, request, context, relevant_prompts)
            else:
                return {"error": f"Unsupported task type: {task_type}"}
                
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return {"error": f"Processing failed: {str(e)}"}
    
    def _determine_task_type(self, request: Dict[str, Any]) -> str:
        """Determine the type of task from the request"""
        # Check explicit task type
        if "task_type" in request:
            return request["task_type"]
        
        # Check for task keywords in the request
        request_text = str(request).lower()
        
        for keyword, task_type in self.task_routing.items():
            if keyword in request_text:
                return task_type
        
        # Default to comprehensive analysis if no specific task identified
        return "comprehensive_analysis"
    
    async def _route_to_specialized_agent(
        self,
        agent_type: str,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        relevant_prompts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Route request to a specialized agent"""
        try:
            if agent_type not in self.specialized_agents:
                return {"error": f"Agent type {agent_type} not available"}
            
            agent = self.specialized_agents[agent_type]
            response = await agent.process_request(request, context)
            
            return {
                "task_type": agent_type,
                "agent_id": agent.agent_id,
                "results": response.data,
                "confidence": response.confidence,
                "execution_time": response.execution_time
            }
            
        except Exception as e:
            self.logger.error(f"Error routing to {agent_type} agent: {str(e)}")
            return {"error": f"Routing failed: {str(e)}"}
    
    async def _perform_comprehensive_analysis(
        self,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        relevant_prompts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis using multiple agents"""
        try:
            results = {}
            execution_times = {}
            
            # Run forecasting analysis
            if "forecasting" in self.specialized_agents:
                forecast_result = await self._route_to_specialized_agent(
                    "forecasting", request, context, relevant_prompts
                )
                results["forecasting"] = forecast_result
                execution_times["forecasting"] = forecast_result.get("execution_time", 0)
            
            # Run anomaly detection analysis
            if "anomaly_detection" in self.specialized_agents:
                anomaly_result = await self._route_to_specialized_agent(
                    "anomaly_detection", request, context, relevant_prompts
                )
                results["anomaly_detection"] = anomaly_result
                execution_times["anomaly_detection"] = anomaly_result.get("execution_time", 0)
            
            # Run classification analysis
            if "classification" in self.specialized_agents:
                classification_result = await self._route_to_specialized_agent(
                    "classification", request, context, relevant_prompts
                )
                results["classification"] = classification_result
                execution_times["classification"] = classification_result.get("execution_time", 0)
            
            # TODO: Add other analyses as agents are implemented
            # if "trend_analysis" in self.specialized_agents:
            #     trend_result = await self._route_to_specialized_agent(...)
            #     results["trend_analysis"] = trend_result
            
            # Combine results
            combined_result = {
                "comprehensive_analysis": True,
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "execution_times": execution_times,
                "total_execution_time": sum(execution_times.values()),
                "used_prompts": relevant_prompts[:3]  # Top 3 prompts used
            }
            
            return combined_result
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {"error": f"Comprehensive analysis failed: {str(e)}"}
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {
            "master_agent": {
                "agent_id": self.agent_id,
                "is_initialized": self.is_initialized,
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "avg_response_time": self.avg_response_time
            },
            "specialized_agents": {}
        }
        
        for agent_type, agent in self.specialized_agents.items():
            status["specialized_agents"][agent_type] = agent.get_status()
        
        return status
    
    async def cleanup(self) -> None:
        """Cleanup all agents"""
        try:
            # Cleanup specialized agents
            for agent in self.specialized_agents.values():
                await agent.cleanup()
            
            # Cleanup master agent
            await super().cleanup()
            
            self.logger.info(f"Master agent {self.agent_id} cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 