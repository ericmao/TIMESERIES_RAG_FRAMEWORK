"""
Base Agent class for the Time Series RAG Framework
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from pathlib import Path

from ..config.config import get_config
from ..utils.logger import get_logger

@dataclass
class AgentResponse:
    """Response structure for agent interactions"""
    success: bool
    message: str
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    execution_time: float = 0.0

class BaseAgent(ABC):
    """
    Base class for all agents in the Time Series RAG Framework
    
    This class provides the foundation for specialized agents that handle
    different aspects of time series analysis including forecasting,
    anomaly detection, classification, and trend analysis.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base agent
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of agent (forecasting, anomaly_detection, etc.)
            model_name: Name of the language model to use
            config: Additional configuration parameters
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = get_config()
        self.logger = get_logger(f"{agent_type}_agent_{agent_id}")
        
        # Initialize model components
        self.model_name = model_name or self.config.model.base_lm_model
        self.tokenizer = None
        self.model = None
        self.embedding_model = None
        
        # RAG components
        self.prompt_pool = []
        self.vector_db = None
        
        # Agent state
        self.is_initialized = False
        self.conversation_history = []
        self.task_history = []
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.avg_response_time = 0.0
        
        # Load configuration overrides
        if config:
            self._update_config(config)
    
    async def initialize(self) -> bool:
        """
        Initialize the agent by loading models and setting up components
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info(f"Initializing {self.agent_type} agent: {self.agent_id}")
            
            # Load tokenizer and model
            await self._load_models()
            
            # Initialize RAG components
            await self._initialize_rag()
            
            # Load prompt pool
            await self._load_prompt_pool()
            
            self.is_initialized = True
            self.logger.info(f"Successfully initialized {self.agent_type} agent: {self.agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent {self.agent_id}: {str(e)}")
            return False
    
    async def _load_models(self) -> None:
        """Load language models and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load language model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.config.model.embedding_model)
            
            self.logger.info(f"Successfully loaded models for agent {self.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            raise
    
    async def _initialize_rag(self) -> None:
        """Initialize RAG components"""
        try:
            # Initialize vector database
            import chromadb
            self.vector_db = chromadb.PersistentClient(
                path=self.config.rag.vector_db_path
            )
            
            # Create or get collection
            try:
                self.collection = self.vector_db.get_collection(
                    name=self.config.rag.collection_name
                )
            except:
                self.collection = self.vector_db.create_collection(
                    name=self.config.rag.collection_name
                )
            
            self.logger.info(f"Successfully initialized RAG components for agent {self.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG components: {str(e)}")
            raise
    
    async def _load_prompt_pool(self) -> None:
        """Load prompt pool from storage"""
        try:
            prompt_path = Path(self.config.rag.prompt_pool_path) / f"{self.agent_type}_prompts.json"
            
            if prompt_path.exists():
                with open(prompt_path, 'r') as f:
                    self.prompt_pool = json.load(f)
                self.logger.info(f"Loaded {len(self.prompt_pool)} prompts for agent {self.agent_id}")
            else:
                self.logger.info(f"No existing prompt pool found for agent {self.agent_id}")
                
        except Exception as e:
            self.logger.warning(f"Failed to load prompt pool: {str(e)}")
            self.prompt_pool = []
    
    async def process_request(
        self,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a request using the agent's specialized capabilities
        
        Args:
            request: The request to process
            context: Additional context information
            
        Returns:
            AgentResponse: The response from the agent
        """
        start_time = time.time()
        self.total_requests += 1
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Validate request
            if not self._validate_request(request):
                return AgentResponse(
                    success=False,
                    message="Invalid request format",
                    execution_time=time.time() - start_time
                )
            
            # Retrieve relevant prompts
            relevant_prompts = await self._retrieve_relevant_prompts(request)
            
            # Process the request
            result = await self._process_request_internal(request, context, relevant_prompts)
            
            # Update conversation history
            self.conversation_history.append({
                'request': request,
                'response': result,
                'timestamp': time.time()
            })
            
            self.successful_requests += 1
            execution_time = time.time() - start_time
            self.avg_response_time = (
                (self.avg_response_time * (self.total_requests - 1) + execution_time) 
                / self.total_requests
            )
            
            return AgentResponse(
                success=True,
                message="Request processed successfully",
                data=result,
                execution_time=execution_time,
                confidence=result.get('confidence', 0.0) if isinstance(result, dict) else 0.0
            )
            
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return AgentResponse(
                success=False,
                message=f"Error processing request: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    @abstractmethod
    async def _process_request_internal(
        self,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        relevant_prompts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Internal method to process requests - must be implemented by subclasses
        
        Args:
            request: The request to process
            context: Additional context
            relevant_prompts: Retrieved relevant prompts
            
        Returns:
            Dict containing the processing results
        """
        pass
    
    def _validate_request(self, request: Dict[str, Any]) -> bool:
        """
        Validate the incoming request format
        
        Args:
            request: The request to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_fields = ['task_type', 'data']
        
        if not isinstance(request, dict):
            return False
        
        for field in required_fields:
            if field not in request:
                return False
        
        return True
    
    async def _retrieve_relevant_prompts(
        self,
        request: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant prompts from the prompt pool using RAG
        
        Args:
            request: The request to find relevant prompts for
            
        Returns:
            List of relevant prompts
        """
        try:
            # Create query embedding
            query_text = f"{request.get('task_type', '')} {request.get('description', '')}"
            query_embedding = self.embedding_model.encode(query_text)
            
            # Search in vector database
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=self.config.rag.top_k
            )
            
            relevant_prompts = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    relevant_prompts.append({
                        'prompt': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else 0.0
                    })
            
            return relevant_prompts
            
        except Exception as e:
            self.logger.warning(f"Failed to retrieve prompts: {str(e)}")
            return []
    
    def _update_config(self, config: Dict[str, Any]) -> None:
        """Update agent configuration"""
        if isinstance(config, dict):
            for key, value in config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        else:
            # Handle case where config is a dataclass or other object
            for key in dir(config):
                if not key.startswith('_') and hasattr(self, key):
                    value = getattr(config, key)
                    setattr(self, key, value)
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and performance metrics"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'is_initialized': self.is_initialized,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'success_rate': self.successful_requests / max(self.total_requests, 1),
            'avg_response_time': self.avg_response_time,
            'conversation_history_length': len(self.conversation_history)
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources when agent is no longer needed"""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            if self.embedding_model:
                del self.embedding_model
            
            self.logger.info(f"Cleaned up resources for agent {self.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}") 