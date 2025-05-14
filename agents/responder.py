from agents.base_agent import BaseSecurityAgent
from crewai import Agent
import asyncio

class IncidentResponderAgent(BaseSecurityAgent):
    def _create_agent(self):
        return Agent(
            role='Incident Responder',
            goal='Develop and execute incident response plans',
            backstory="""You are a skilled incident responder with extensive experience in 
            handling security incidents. Your expertise includes incident containment, 
            eradication, and recovery procedures.""",
            verbose=True,
            llm=self.llm
        )
    
    async def process(self, input_data):
        prompt = f"""As an Incident Responder, develop a response plan for the following security incident:
        Incident Details: {input_data}
        
        Please provide:
        1. Immediate Actions
           - Containment steps
           - Evidence preservation
           - Communication plan
        
        2. Investigation Steps
           - Evidence collection
           - Timeline analysis
           - Root cause identification
        
        3. Recovery Plan
           - System restoration
           - Security hardening
           - Monitoring enhancement
        
        4. Lessons Learned
           - Process improvements
           - Training recommendations
           - Documentation updates
        
        Response Plan:"""
        return await self._call_llm(prompt) 