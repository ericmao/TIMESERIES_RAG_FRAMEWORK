"""
HMM-based Anomaly Detection Agent for Time Series RAG Framework
"""
from typing import List, Optional, Dict, Any
import numpy as np
from hmmlearn import hmm
from .base_agent import BaseAgent
from ..config.config import get_config
from ..utils.logger import get_logger

class HMMAnomalyDetectionAgent(BaseAgent):
    """
    Anomaly detection agent using Hidden Markov Models (HMM).
    """
    def __init__(self, agent_id: str, n_components: int = 5, model_type: str = 'multinomial', config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id=agent_id,
            agent_type="hmm_anomaly_detection",
            model_name="hmmlearn",
            config=config
        )
        self.logger = get_logger(f"hmm_anomaly_detection_agent_{agent_id}")
        self.n_components = n_components
        self.model_type = model_type
        self.model = None

    def fit(self, sequences: List[List[int]]):
        """
        Fit HMM to a list of sequences (each sequence is a list of integer states).
        """
        lengths = [len(seq) for seq in sequences]
        X = np.concatenate([np.array(seq) for seq in sequences]).reshape(-1, 1)
        if self.model_type == 'multinomial':
            self.model = hmm.MultinomialHMM(n_components=self.n_components, n_iter=100)
        else:
            self.model = hmm.GaussianHMM(n_components=self.n_components, n_iter=100)
        self.model.fit(X, lengths)
        self.logger.info(f"Fitted HMM with {self.n_components} components on {len(sequences)} sequences.")

    def detect_anomalies(self, sequences: List[List[int]], threshold: float = -50.0) -> List[Dict[str, Any]]:
        """
        Detect anomalies in sequences using log likelihood under the HMM.
        Returns a list of anomaly dicts with sequence index and log likelihood.
        """
        anomalies = []
        for idx, seq in enumerate(sequences):
            X = np.array(seq).reshape(-1, 1)
            log_likelihood = self.model.score(X)
            if log_likelihood < threshold:
                anomalies.append({
                    "sequence_index": idx,
                    "log_likelihood": log_likelihood,
                    "method": "hmm"
                })
        return anomalies

    async def _process_request_internal(
        self,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        relevant_prompts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process HMM anomaly detection request
        """
        try:
            # Extract sequences from request
            sequences = request.get('sequences', [])
            threshold = request.get('threshold', -50.0)
            
            if not sequences:
                return {
                    'success': False,
                    'message': 'No sequences provided for HMM analysis',
                    'anomalies': []
                }
            
            # Fit model if not already fitted
            if self.model is None:
                self.fit(sequences)
            
            # Detect anomalies
            anomalies = self.detect_anomalies(sequences, threshold)
            
            return {
                'success': True,
                'message': f'Detected {len(anomalies)} anomalies using HMM',
                'anomalies': anomalies,
                'model_type': 'hmm',
                'threshold': threshold
            }
            
        except Exception as e:
            self.logger.error(f"HMM processing error: {str(e)}")
            return {
                'success': False,
                'message': f'HMM processing failed: {str(e)}',
                'anomalies': []
            } 