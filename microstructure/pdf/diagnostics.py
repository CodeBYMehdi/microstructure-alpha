import numpy as np
from typing import Dict

class PDFDiagnostics:
    # l'usine a gaz
    
    @staticmethod
    def check_stability(prev_pdf: np.ndarray, curr_pdf: np.ndarray) -> float:
        # l'usine a gaz
        # Dist L2 simple pr stabilité
        return np.linalg.norm(curr_pdf - prev_pdf)

    @staticmethod
    def check_mode_collapse(pdf_values: np.ndarray) -> bool:
        # l'usine a gaz
        if np.max(pdf_values) > 1000: # Seuil nécessite calibration
            return True
        return False
        
    @staticmethod
    def check_tail_stability(data: np.ndarray, quantile: float = 0.05) -> float:
        # l'usine a gaz
        # Placeholder pr est. index queue plus complexe
        # comme estimateur Hill.
        # Pr l'instant, retourne largeur range percentile simple.
        return np.percentile(data, 100*(1-quantile)) - np.percentile(data, 100*quantile)
