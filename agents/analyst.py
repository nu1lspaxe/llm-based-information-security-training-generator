from agents.base_agent import BaseSecurityAgent
from crewai import Agent
import asyncio

class SecurityAnalystAgent(BaseSecurityAgent):
    def _create_agent(self):
        return Agent(
            role='Security Analyst',
            goal='Analyze security threats and provide detailed technical assessments',
            backstory="""You are an experienced security analyst with deep technical expertise 
            in threat analysis, vulnerability assessment, and security monitoring. Your strength 
            lies in identifying and analyzing security threats and providing actionable insights.""",
            verbose=True,
            llm=self.llm
        )
    
    async def process(self, input_data):
        prompt = f"""As a Security Analyst, analyze the following security context and provide a detailed assessment:
        Context: {input_data}
        
        Please provide:
        1. Threat Analysis
           - Attack vectors
           - Vulnerabilities
           - Exploitation methods
        
        2. Technical Assessment
           - System impact
           - Data exposure
           - Security controls
        
        3. Detection Methods
           - Log analysis
           - Monitoring recommendations
           - Alert triggers
        
        4. Mitigation Strategies
           - Technical controls
           - Configuration changes
           - Security patches
        
        Technical Assessment:"""
        return await self._call_llm(prompt) 