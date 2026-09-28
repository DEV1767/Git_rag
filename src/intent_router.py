from src.guardrial import IntentGuardrail

from src.logger import logger

class IntentRouter:
    def __init__(self):
        self.guardrial=IntentGuardrail()

    def route(self,query:str):
        result=self.guardrial.classify(query)
        route=result["route"]
        
        logger.info(
            "Intent Router → %s | confidence=%.3f",
            route,
            result["confidence"]
        )
        return result
