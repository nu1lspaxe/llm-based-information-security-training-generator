from agents.base_agent import BaseSecurityAgent
from crewai import Agent
import asyncio

class CISOAgent(BaseSecurityAgent):
    def _create_agent(self):
        return Agent(
            role='CISO',
            goal='Provide strategic security guidance and risk management',
            backstory="""You are a Chief Information Security Officer with extensive experience 
            in security strategy, risk management, and compliance. Your expertise includes 
            aligning security initiatives with business objectives and managing security programs.""",
            verbose=True,
            llm=self.llm
        )
    
    async def process(self, input_data):
        prompt = f"""As a CISO, provide strategic guidance for the following security situation:
        Situation: {input_data}
        
        Please provide:
        1. Risk Assessment
           - Business impact
           - Risk level
           - Compliance implications
        
        2. Strategic Recommendations
           - Policy updates
           - Resource allocation
           - Program enhancements
        
        3. Stakeholder Communication
           - Executive summary
           - Board presentation
           - Team messaging
        
        4. Long-term Strategy
           - Security roadmap
           - Investment priorities
           - Capability development
        
        Strategic Guidance:"""
        return await self._call_llm(prompt)
