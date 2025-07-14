"""
Prompt Management System for the Time Series RAG Framework
"""
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import hashlib
import numpy as np
from dataclasses import dataclass, asdict

from ..config.config import get_config
from .logger import get_logger

@dataclass
class PromptTemplate:
    """Prompt template structure"""
    id: str
    title: str
    description: str
    template: str
    agent_type: str
    task_type: str
    parameters: Dict[str, Any]
    tags: List[str]
    performance_metrics: Dict[str, float]
    created_at: str
    updated_at: str
    usage_count: int = 0
    success_rate: float = 0.0

class PromptManager:
    """
    Manages prompt templates for the RAG framework.
    
    Features:
    - Prompt storage and retrieval
    - Performance tracking
    - Template optimization
    - Version control
    - Tag-based organization
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.config = get_config()
        self.logger = get_logger("prompt_manager")
        
        # Storage configuration
        self.storage_path = Path(storage_path or self.config.rag.prompt_pool_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.prompts_cache = {}
        self.embeddings_cache = {}
        
        # Performance tracking
        self.performance_history = []
        
        # Load existing prompts
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Load all prompt templates from storage"""
        try:
            for prompt_file in self.storage_path.glob("*.json"):
                try:
                    with open(prompt_file, 'r') as f:
                        prompt_data = json.load(f)
                        prompt = PromptTemplate(**prompt_data)
                        self.prompts_cache[prompt.id] = prompt
                except Exception as e:
                    self.logger.warning(f"Failed to load prompt from {prompt_file}: {str(e)}")
            
            self.logger.info(f"Loaded {len(self.prompts_cache)} prompt templates")
            
        except Exception as e:
            self.logger.error(f"Failed to load prompts: {str(e)}")
    
    def create_prompt(
        self,
        title: str,
        description: str,
        template: str,
        agent_type: str,
        task_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create a new prompt template
        
        Args:
            title: Prompt title
            description: Prompt description
            template: Prompt template string
            agent_type: Type of agent this prompt is for
            task_type: Type of task this prompt handles
            parameters: Template parameters
            tags: Tags for organization
            
        Returns:
            Prompt ID
        """
        try:
            # Generate unique ID
            prompt_id = self._generate_prompt_id(title, agent_type, task_type)
            
            # Create prompt template
            prompt = PromptTemplate(
                id=prompt_id,
                title=title,
                description=description,
                template=template,
                agent_type=agent_type,
                task_type=task_type,
                parameters=parameters or {},
                tags=tags or [],
                performance_metrics={},
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            
            # Save to storage
            self._save_prompt(prompt)
            
            # Add to cache
            self.prompts_cache[prompt_id] = prompt
            
            self.logger.info(f"Created prompt template: {prompt_id}")
            return prompt_id
            
        except Exception as e:
            self.logger.error(f"Failed to create prompt: {str(e)}")
            raise
    
    def get_prompt(self, prompt_id: str) -> Optional[PromptTemplate]:
        """Get a prompt template by ID"""
        return self.prompts_cache.get(prompt_id)
    
    def get_prompts_by_agent(self, agent_type: str) -> List[PromptTemplate]:
        """Get all prompts for a specific agent type"""
        return [
            prompt for prompt in self.prompts_cache.values()
            if prompt.agent_type == agent_type
        ]
    
    def get_prompts_by_task(self, task_type: str) -> List[PromptTemplate]:
        """Get all prompts for a specific task type"""
        return [
            prompt for prompt in self.prompts_cache.values()
            if prompt.task_type == task_type
        ]
    
    def get_prompts_by_tags(self, tags: List[str]) -> List[PromptTemplate]:
        """Get prompts that match any of the given tags"""
        return [
            prompt for prompt in self.prompts_cache.values()
            if any(tag in prompt.tags for tag in tags)
        ]
    
    def search_prompts(self, query: str) -> List[PromptTemplate]:
        """Search prompts by title, description, or content"""
        query_lower = query.lower()
        results = []
        
        for prompt in self.prompts_cache.values():
            if (query_lower in prompt.title.lower() or
                query_lower in prompt.description.lower() or
                query_lower in prompt.template.lower()):
                results.append(prompt)
        
        return results
    
    def update_prompt(
        self,
        prompt_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        template: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Update an existing prompt template"""
        try:
            prompt = self.prompts_cache.get(prompt_id)
            if not prompt:
                return False
            
            # Update fields
            if title is not None:
                prompt.title = title
            if description is not None:
                prompt.description = description
            if template is not None:
                prompt.template = template
            if parameters is not None:
                prompt.parameters = parameters
            if tags is not None:
                prompt.tags = tags
            
            prompt.updated_at = datetime.now().isoformat()
            
            # Save to storage
            self._save_prompt(prompt)
            
            self.logger.info(f"Updated prompt template: {prompt_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update prompt {prompt_id}: {str(e)}")
            return False
    
    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt template"""
        try:
            prompt = self.prompts_cache.get(prompt_id)
            if not prompt:
                return False
            
            # Remove from cache
            del self.prompts_cache[prompt_id]
            
            # Remove from storage
            prompt_file = self.storage_path / f"{prompt_id}.json"
            if prompt_file.exists():
                prompt_file.unlink()
            
            self.logger.info(f"Deleted prompt template: {prompt_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete prompt {prompt_id}: {str(e)}")
            return False
    
    def render_prompt(
        self,
        prompt_id: str,
        parameters: Dict[str, Any]
    ) -> Optional[str]:
        """
        Render a prompt template with given parameters
        
        Args:
            prompt_id: ID of the prompt template
            parameters: Parameters to substitute in the template
            
        Returns:
            Rendered prompt string
        """
        try:
            prompt = self.get_prompt(prompt_id)
            if not prompt:
                return None
            
            # Simple template rendering
            rendered = prompt.template
            
            for key, value in parameters.items():
                placeholder = f"{{{key}}}"
                if placeholder in rendered:
                    rendered = rendered.replace(placeholder, str(value))
            
            return rendered
            
        except Exception as e:
            self.logger.error(f"Failed to render prompt {prompt_id}: {str(e)}")
            return None
    
    def record_performance(
        self,
        prompt_id: str,
        success: bool,
        execution_time: float,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record performance metrics for a prompt"""
        try:
            prompt = self.get_prompt(prompt_id)
            if not prompt:
                return
            
            # Update usage count
            prompt.usage_count += 1
            
            # Update success rate
            if not hasattr(prompt, 'success_count'):
                prompt.success_count = 0
            
            if success:
                prompt.success_count += 1
            
            prompt.success_rate = prompt.success_count / prompt.usage_count
            
            # Update performance metrics
            if additional_metrics:
                prompt.performance_metrics.update(additional_metrics)
            
            # Track execution time
            if 'execution_times' not in prompt.performance_metrics:
                prompt.performance_metrics['execution_times'] = []
            
            prompt.performance_metrics['execution_times'].append(execution_time)
            
            # Keep only recent execution times
            if len(prompt.performance_metrics['execution_times']) > 100:
                prompt.performance_metrics['execution_times'] = prompt.performance_metrics['execution_times'][-100:]
            
            # Calculate average execution time
            prompt.performance_metrics['avg_execution_time'] = np.mean(prompt.performance_metrics['execution_times'])
            
            # Save updated prompt
            self._save_prompt(prompt)
            
        except Exception as e:
            self.logger.error(f"Failed to record performance for {prompt_id}: {str(e)}")
    
    def get_best_prompts(
        self,
        agent_type: str,
        task_type: str,
        limit: int = 5
    ) -> List[PromptTemplate]:
        """Get the best performing prompts for a given agent and task"""
        try:
            # Get relevant prompts
            relevant_prompts = [
                prompt for prompt in self.prompts_cache.values()
                if prompt.agent_type == agent_type and prompt.task_type == task_type
            ]
            
            # Sort by performance (success rate and usage count)
            sorted_prompts = sorted(
                relevant_prompts,
                key=lambda p: (p.success_rate, p.usage_count),
                reverse=True
            )
            
            return sorted_prompts[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get best prompts: {str(e)}")
            return []
    
    def optimize_prompts(self) -> Dict[str, Any]:
        """Analyze and suggest prompt optimizations"""
        try:
            optimization_results = {
                "low_performing_prompts": [],
                "unused_prompts": [],
                "suggestions": []
            }
            
            # Find low performing prompts
            for prompt in self.prompts_cache.values():
                if prompt.usage_count > 10 and prompt.success_rate < 0.5:
                    optimization_results["low_performing_prompts"].append({
                        "id": prompt.id,
                        "title": prompt.title,
                        "success_rate": prompt.success_rate,
                        "usage_count": prompt.usage_count
                    })
                
                # Find unused prompts
                if prompt.usage_count == 0:
                    optimization_results["unused_prompts"].append({
                        "id": prompt.id,
                        "title": prompt.title,
                        "created_at": prompt.created_at
                    })
            
            # Generate suggestions
            if optimization_results["low_performing_prompts"]:
                optimization_results["suggestions"].append(
                    "Consider reviewing and updating low-performing prompts"
                )
            
            if optimization_results["unused_prompts"]:
                optimization_results["suggestions"].append(
                    "Consider removing or updating unused prompts"
                )
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Failed to optimize prompts: {str(e)}")
            return {}
    
    def _generate_prompt_id(self, title: str, agent_type: str, task_type: str) -> str:
        """Generate a unique prompt ID"""
        content = f"{title}_{agent_type}_{task_type}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _save_prompt(self, prompt: PromptTemplate) -> None:
        """Save prompt template to storage"""
        try:
            prompt_file = self.storage_path / f"{prompt.id}.json"
            with open(prompt_file, 'w') as f:
                json.dump(asdict(prompt), f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save prompt {prompt.id}: {str(e)}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prompt management statistics"""
        try:
            total_prompts = len(self.prompts_cache)
            agent_types = set(p.agent_type for p in self.prompts_cache.values())
            task_types = set(p.task_type for p in self.prompts_cache.values())
            
            # Calculate average performance
            avg_success_rate = np.mean([p.success_rate for p in self.prompts_cache.values() if p.usage_count > 0])
            avg_usage_count = np.mean([p.usage_count for p in self.prompts_cache.values()])
            
            return {
                "total_prompts": total_prompts,
                "agent_types": list(agent_types),
                "task_types": list(task_types),
                "avg_success_rate": avg_success_rate,
                "avg_usage_count": avg_usage_count,
                "storage_path": str(self.storage_path)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {str(e)}")
            return {} 