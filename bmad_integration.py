#!/usr/bin/env python3
"""
BMAD-CrewAI Integration Module
==============================
Bridges BMAD orchestration with existing Stellar Connect infrastructure
"""

import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path
from crewai import Agent, Task, Crew
from src.stellar_crew import (
    retrieval_agent,
    data_extraction_agent,
    content_generation_agent,
    run_crew,
    create_general_query_tasks
)
from bmad_orchestration import BMADOrchestrator, BMADRole
import json

# ============================================================================
# BMAD-CrewAI Bridge
# ============================================================================

class BMADCrewAIBridge:
    """Bridge between BMAD orchestration and existing CrewAI agents"""

    def __init__(self, config_path: str = "bmad_config.yaml"):
        """Initialize bridge with configuration"""
        self.config = self._load_config(config_path)
        self.bmad_orchestrator = BMADOrchestrator()
        self.existing_agents = {
            "retrieval": retrieval_agent,
            "extraction": data_extraction_agent,
            "generation": content_generation_agent
        }

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load BMAD configuration from YAML"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def create_hybrid_crew(
        self,
        bmad_agents: List[BMADRole],
        include_existing: bool = True
    ) -> Crew:
        """Create a crew combining BMAD and existing agents"""

        agents = []
        tasks = []

        # Add BMAD agents
        for role in bmad_agents:
            agent = self.bmad_orchestrator.agents.get(role)
            if agent:
                agents.append(agent)

        # Optionally add existing CrewAI agents
        if include_existing:
            agents.extend(self.existing_agents.values())

        return Crew(
            agents=agents,
            tasks=tasks,  # Tasks will be added dynamically
            process="sequential",
            verbose=True
        )

    def execute_bmad_enhanced_query(self, query: str) -> str:
        """Execute a query using BMAD-enhanced pipeline"""

        print(f"\n{'='*60}")
        print("BMAD-Enhanced Query Processing")
        print(f"Query: {query}")
        print(f"{'='*60}\n")

        # Step 1: Business Analyst extracts requirements
        ba_task = Task(
            description=f"""Analyze this user query and extract requirements: {query}
            Identify the type of information needed, success criteria, and any constraints.""",
            expected_output="Structured requirements for the query",
            agent=self.bmad_orchestrator.agents[BMADRole.BUSINESS_ANALYST]
        )

        # Step 2: Sales Specialist provides domain context
        ss_task = Task(
            description="""Based on the requirements, provide sales domain context.
            What sales patterns, best practices, or insights are relevant?""",
            expected_output="Sales domain insights and context",
            agent=self.bmad_orchestrator.agents[BMADRole.SALES_SPECIALIST]
        )

        # Step 3: Use existing retrieval agent for data
        retrieval_task = Task(
            description=f"""Retrieve relevant information for: {query}
            Use both vector search and knowledge graph as needed.""",
            expected_output="Retrieved context from knowledge base",
            agent=retrieval_agent
        )

        # Step 4: Generate final response
        generation_task = Task(
            description="""Synthesize all information into a comprehensive response.
            Ensure it addresses the requirements and includes sales insights.""",
            expected_output="Final synthesized response",
            agent=content_generation_agent
        )

        # Create and execute crew
        crew = Crew(
            agents=[
                self.bmad_orchestrator.agents[BMADRole.BUSINESS_ANALYST],
                self.bmad_orchestrator.agents[BMADRole.SALES_SPECIALIST],
                retrieval_agent,
                content_generation_agent
            ],
            tasks=[ba_task, ss_task, retrieval_task, generation_task],
            process="sequential",
            verbose=True
        )

        result = crew.kickoff()
        return str(result)

# ============================================================================
# Advanced BMAD Workflows
# ============================================================================

class AdvancedBMADWorkflows:
    """Advanced workflows combining multiple BMAD agents"""

    def __init__(self):
        self.bridge = BMADCrewAIBridge()
        self.orchestrator = BMADOrchestrator()

    def sales_deal_analysis(self, deal_info: str) -> Dict[str, Any]:
        """Comprehensive deal analysis using all BMAD agents"""

        results = {}

        # Business Analysis
        ba_crew = Crew(
            agents=[self.orchestrator.agents[BMADRole.BUSINESS_ANALYST]],
            tasks=[Task(
                description=f"Extract key deal components from: {deal_info}",
                expected_output="Structured deal analysis",
                agent=self.orchestrator.agents[BMADRole.BUSINESS_ANALYST]
            )],
            process="sequential",
            verbose=False
        )
        results["business_analysis"] = str(ba_crew.kickoff())

        # Sales Strategy
        ss_crew = Crew(
            agents=[self.orchestrator.agents[BMADRole.SALES_SPECIALIST]],
            tasks=[Task(
                description=f"Provide strategic recommendations for: {deal_info}",
                expected_output="Sales strategy and tactics",
                agent=self.orchestrator.agents[BMADRole.SALES_SPECIALIST]
            )],
            process="sequential",
            verbose=False
        )
        results["sales_strategy"] = str(ss_crew.kickoff())

        # Technical Requirements
        sa_crew = Crew(
            agents=[self.orchestrator.agents[BMADRole.SOLUTION_ARCHITECT]],
            tasks=[Task(
                description=f"Identify system enhancements to support: {deal_info}",
                expected_output="Technical requirements and design",
                agent=self.orchestrator.agents[BMADRole.SOLUTION_ARCHITECT]
            )],
            process="sequential",
            verbose=False
        )
        results["technical_requirements"] = str(sa_crew.kickoff())

        return results

    def continuous_improvement_cycle(self) -> Dict[str, Any]:
        """Execute continuous improvement cycle"""

        # QA Health Check
        qa_task = Task(
            description="Perform system health check and quality assessment",
            expected_output="System health report with recommendations",
            agent=self.orchestrator.agents[BMADRole.QA_TESTER]
        )

        # Sales Performance Analysis
        sales_task = Task(
            description="Analyze recent sales performance and identify patterns",
            expected_output="Sales performance insights",
            agent=self.orchestrator.agents[BMADRole.SALES_SPECIALIST]
        )

        # Improvement Planning
        pm_task = Task(
            description="Based on health and performance data, plan improvements",
            expected_output="Prioritized improvement backlog",
            agent=self.orchestrator.agents[BMADRole.PROJECT_MANAGER]
        )

        crew = Crew(
            agents=[
                self.orchestrator.agents[BMADRole.QA_TESTER],
                self.orchestrator.agents[BMADRole.SALES_SPECIALIST],
                self.orchestrator.agents[BMADRole.PROJECT_MANAGER]
            ],
            tasks=[qa_task, sales_task, pm_task],
            process="sequential",
            verbose=True
        )

        return {"improvement_cycle": str(crew.kickoff())}

# ============================================================================
# BMAD Dashboard Integration
# ============================================================================

class BMADDashboardAdapter:
    """Adapter for integrating BMAD with Streamlit dashboard"""

    def __init__(self):
        self.bridge = BMADCrewAIBridge()
        self.workflows = AdvancedBMADWorkflows()

    def process_chat_query(self, query: str, mode: str = "standard") -> str:
        """Process chat query with selected BMAD mode"""

        if mode == "bmad_enhanced":
            # Use full BMAD enhancement
            return self.bridge.execute_bmad_enhanced_query(query)
        elif mode == "sales_focused":
            # Focus on sales specialist
            return self._sales_focused_query(query)
        elif mode == "technical":
            # Focus on technical agents
            return self._technical_query(query)
        else:
            # Standard CrewAI processing
            tasks = create_general_query_tasks(query)
            return str(run_crew(tasks))

    def _sales_focused_query(self, query: str) -> str:
        """Process query with sales focus"""
        task = Task(
            description=f"Provide sales-focused analysis for: {query}",
            expected_output="Sales insights and recommendations",
            agent=self.bridge.bmad_orchestrator.agents[BMADRole.SALES_SPECIALIST]
        )

        crew = Crew(
            agents=[self.bridge.bmad_orchestrator.agents[BMADRole.SALES_SPECIALIST]],
            tasks=[task],
            process="sequential",
            verbose=False
        )

        return str(crew.kickoff())

    def _technical_query(self, query: str) -> str:
        """Process query with technical focus"""
        tasks = [
            Task(
                description=f"Analyze technical aspects of: {query}",
                expected_output="Technical analysis",
                agent=self.bridge.bmad_orchestrator.agents[BMADRole.SOLUTION_ARCHITECT]
            ),
            Task(
                description="Provide implementation details",
                expected_output="Implementation guide",
                agent=self.bridge.bmad_orchestrator.agents[BMADRole.DEVELOPER]
            )
        ]

        crew = Crew(
            agents=[
                self.bridge.bmad_orchestrator.agents[BMADRole.SOLUTION_ARCHITECT],
                self.bridge.bmad_orchestrator.agents[BMADRole.DEVELOPER]
            ],
            tasks=tasks,
            process="sequential",
            verbose=False
        )

        return str(crew.kickoff())

    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard display"""
        return {
            "bmad_agents": {
                "total": len(self.bridge.bmad_orchestrator.agents),
                "active": sum(1 for a in self.bridge.bmad_orchestrator.agents.values()),
                "roles": list(self.bridge.bmad_orchestrator.agents.keys())
            },
            "existing_agents": {
                "total": len(self.bridge.existing_agents),
                "types": list(self.bridge.existing_agents.keys())
            },
            "workflows_available": [
                "requirement_analysis",
                "system_implementation",
                "sales_optimization",
                "continuous_improvement"
            ]
        }

# ============================================================================
# Example Usage and Testing
# ============================================================================

def test_bmad_integration():
    """Test BMAD integration with examples"""

    print("\n" + "="*60)
    print("BMAD-CrewAI Integration Test")
    print("="*60)

    # Initialize components
    bridge = BMADCrewAIBridge()
    adapter = BMADDashboardAdapter()

    # Test 1: Basic BMAD-enhanced query
    print("\nTest 1: BMAD-Enhanced Query")
    print("-"*40)
    test_query = "What are the top objections in estate planning sales?"

    # Note: Uncomment to execute
    # result = adapter.process_chat_query(test_query, mode="bmad_enhanced")
    # print(f"Result: {result[:500]}...")  # First 500 chars

    # Test 2: Get metrics
    print("\nTest 2: Agent Metrics")
    print("-"*40)
    metrics = adapter.get_agent_metrics()
    print(json.dumps(metrics, indent=2, default=str))

    # Test 3: Sales-focused query
    print("\nTest 3: Sales-Focused Query")
    print("-"*40)
    sales_query = "How can we improve trust-building in initial calls?"

    # Note: Uncomment to execute
    # result = adapter.process_chat_query(sales_query, mode="sales_focused")
    # print(f"Result: {result[:500]}...")

    print("\n" + "="*60)
    print("Integration test complete!")
    print("="*60)

if __name__ == "__main__":
    test_bmad_integration()