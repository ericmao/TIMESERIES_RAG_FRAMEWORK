"""
ForecastingAgent for the Time Series RAG Framework
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from .base_agent import BaseAgent, AgentResponse
from ..config.config import get_config
from ..utils.logger import get_logger

class ForecastingAgent(BaseAgent):
    """
    Specialized agent for time series forecasting tasks.
    """
    def __init__(self, agent_id: str, model_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type="forecasting",
            model_name=model_name or get_config().model.forecasting_agent_model,
            config=config
        )
        self.logger = get_logger(f"forecasting_agent_{agent_id}")

    async def _process_request_internal(
        self,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        relevant_prompts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a forecasting request.
        Args:
            request: Dict with 'data' (time series) and other params
            context: Optional context
            relevant_prompts: List of retrieved prompts
        Returns:
            Dict with forecast results
        """
        # Extract data
        data = request.get('data')
        if data is None:
            return {"error": "No data provided"}
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame(data)
            if 'ds' not in df.columns or 'y' not in df.columns:
                return {"error": "Data must contain 'ds' (date) and 'y' (value) columns"}
        except Exception as e:
            return {"error": f"Invalid data format: {str(e)}"}
        
        # Use Prophet for forecasting
        try:
            from prophet import Prophet
            model = Prophet()
            model.fit(df)
            forecast_horizon = request.get('forecast_horizon', get_config().timeseries.forecast_horizon)
            future = model.make_future_dataframe(periods=forecast_horizon)
            forecast = model.predict(future)
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon).to_dict(orient='records')
            return {
                "forecast": result,
                "confidence": 0.95,  # Placeholder
                "used_prompts": relevant_prompts[:3]  # Show top 3 prompts used
            }
        except Exception as e:
            return {"error": f"Forecasting failed: {str(e)}"}
