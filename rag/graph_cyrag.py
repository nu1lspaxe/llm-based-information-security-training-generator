from neo4j import GraphDatabase
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import asyncio
import logging
import torch
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class GraphCyRAG:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, llm):
        # Load environment variables
        load_dotenv()
        
        # Verify Neo4j credentials
        if not neo4j_password:
            neo4j_password = os.getenv("NEO4J_PASSWORD")
            if not neo4j_password:
                raise ValueError("NEO4J_PASSWORD environment variable is not set")
        
        if not neo4j_uri:
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        
        if not neo4j_user:
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        
        logger.info(f"Attempting to connect to Neo4j at {neo4j_uri} as user {neo4j_user}")
        
        try:
            # Test connection first
            self.driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password),
                max_connection_lifetime=30
            )
            
            # Verify connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                if not result.single():
                    raise RuntimeError("Failed to verify Neo4j connection")
            
            logger.info("Successfully connected to Neo4j")
            
            self.llm = llm
            self._init_vectorstore()
            self._create_initial_schema()  # Create initial schema
            self._initialize_sample_data()  # Initialize sample data
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            logger.error("Please verify your Neo4j credentials and ensure the database is running")
            raise RuntimeError(f"Neo4j connection failed: {str(e)}")
    
    def _init_vectorstore(self):
        try:
            # Initialize the sentence transformer model directly
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Create a custom embedding function
            def embed_function(texts):
                if isinstance(texts, str):
                    texts = [texts]
                embeddings = model.encode(texts, convert_to_tensor=True)
                return embeddings.cpu().numpy()
            
            # Initialize vector store with the custom embedding function
            self.vectorstore = Chroma(
                collection_name="security_docs",
                embedding_function=embed_function,
                persist_directory="./chroma_db"
            )
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise RuntimeError(f"Failed to initialize vector store: {str(e)}")
    
    def _initialize_sample_data(self):
        """Initialize sample data in Neo4j"""
        try:
            with self.driver.session() as session:
                # Create sample CWE
                session.run("""
                MERGE (w:CWE {id: 'CWE-89', name: 'SQL Injection'})
                """)
                
                # Create sample CVE
                session.run("""
                MERGE (c:CVE {id: 'CVE-2023-1234', name: 'SQL Injection Vulnerability'})
                WITH c
                MATCH (w:CWE {id: 'CWE-89'})
                MERGE (c)-[:RELATED_TO]->(w)
                """)
                
                # Create sample CAPEC
                session.run("""
                MERGE (a:CAPEC {id: 'CAPEC-66', name: 'SQL Injection'})
                WITH a
                MATCH (w:CWE {id: 'CWE-89'})
                MERGE (w)-[:MAPPED_TO]->(a)
                """)
                
                # Create sample ATT&CK
                session.run("""
                MERGE (t:ATTCK {id: 'T1190', name: 'Exploit Public-Facing Application'})
                WITH t
                MATCH (a:CAPEC {id: 'CAPEC-66'})
                MERGE (a)-[:ALIGNED_TO]->(t)
                """)
                
                logger.info("Successfully initialized sample data")
        except Exception as e:
            logger.error(f"Error initializing sample data: {str(e)}")
            raise
    
    async def fetch_threat_path(self, cwe_id: str) -> List[dict]:
        query = """
        MATCH (w:CWE {id: $cwe_id})
        OPTIONAL MATCH path=(c:CVE)-[:RELATED_TO]->(w)-[:MAPPED_TO]->(a:CAPEC)-[:ALIGNED_TO]->(t:ATTCK)
        RETURN path
        """
        try:
            loop = asyncio.get_event_loop()
            with self.driver.session() as session:
                result = await loop.run_in_executor(
                    None,
                    lambda: session.run(query, cwe_id=cwe_id)
                )
                paths = [record["path"] for record in result if record["path"] is not None]
                
                if not paths:
                    logger.warning(f"No paths found for CWE ID: {cwe_id}")
                    return []
                
                return paths
        except Exception as e:
            logger.error(f"Error fetching threat path: {str(e)}")
            return []
    
    async def get_attack_context(self, threat_type: str) -> str:
        try:
            # Use the LLM directly for context generation
            prompt = f"""Generate a detailed analysis of {threat_type} vulnerability:
            1. Common attack vectors
            2. Potential impact
            3. Mitigation strategies
            4. Detection methods
            
            Analysis:"""
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.llm(prompt)
            )
        except Exception as e:
            logger.error(f"Error generating attack context: {str(e)}")
            return f"Error analyzing {threat_type}: {str(e)}"
    
    async def close(self):
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.driver.close)
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {str(e)}")
    
    def _create_initial_schema(self):
        """Create initial schema if it doesn't exist"""
        try:
            with self.driver.session() as session:
                # Create constraints
                session.run("CREATE CONSTRAINT cve_id IF NOT EXISTS FOR (c:CVE) REQUIRE c.id IS UNIQUE")
                session.run("CREATE CONSTRAINT cwe_id IF NOT EXISTS FOR (w:CWE) REQUIRE w.id IS UNIQUE")
                session.run("CREATE CONSTRAINT capec_id IF NOT EXISTS FOR (a:CAPEC) REQUIRE a.id IS UNIQUE")
                session.run("CREATE CONSTRAINT attck_id IF NOT EXISTS FOR (t:ATTCK) REQUIRE t.id IS UNIQUE")
                
                logger.info("Successfully created initial schema")
        except Exception as e:
            logger.error(f"Error creating initial schema: {str(e)}")
            raise
