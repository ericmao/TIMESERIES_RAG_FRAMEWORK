"""
CRF-based Anomaly Detection Agent for Time Series RAG Framework
"""
from typing import List, Optional, Dict, Any
import sklearn_crfsuite
from .base_agent import BaseAgent
from ..config.config import get_config
from ..utils.logger import get_logger

class CRFAnomalyDetectionAgent(BaseAgent):
    """
    Anomaly detection agent using Conditional Random Fields (CRF).
    """
    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type="crf_anomaly_detection",
            model_name="sklearn-crfsuite",
            config=config
        )
        self.logger = get_logger(f"crf_anomaly_detection_agent_{agent_id}")
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True
        )

    def fit(self, X_seq: List[List[Dict[str, Any]]], y_seq: List[List[str]]):
        """
        Fit CRF to labeled sequences.
        X_seq: list of list of feature dicts
        y_seq: list of list of labels (e.g., 'normal', 'anomaly')
        """
        self.model.fit(X_seq, y_seq)
        self.logger.info(f"Fitted CRF on {len(X_seq)} sequences.")

    def detect_anomalies(self, X_seq: List[List[Dict[str, Any]]]) -> List[List[str]]:
        """
        Predict anomaly labels for each event in the sequence.
        Returns a list of label sequences.
        """
        y_pred = self.model.predict(X_seq)
        return y_pred

    async def _process_request_internal(
        self,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        relevant_prompts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process CRF anomaly detection request
        """
        try:
            # Extract feature sequences from request
            X_seq = request.get('feature_sequences', [])
            
            if not X_seq:
                return {
                    'success': False,
                    'message': 'No feature sequences provided for CRF analysis',
                    'predictions': []
                }
            
            # Predict anomalies
            predictions = self.detect_anomalies(X_seq)
            
            # Count anomalies
            anomaly_count = sum(1 for seq in predictions for label in seq if label == 'anomaly')
            
            return {
                'success': True,
                'message': f'Detected {anomaly_count} anomalies using CRF',
                'predictions': predictions,
                'model_type': 'crf',
                'anomaly_count': anomaly_count
            }
            
        except Exception as e:
            self.logger.error(f"CRF processing error: {str(e)}")
            return {
                'success': False,
                'message': f'CRF processing failed: {str(e)}',
                'predictions': []
            } 