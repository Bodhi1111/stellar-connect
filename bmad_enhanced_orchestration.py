#!/usr/bin/env python3
"""
Enhanced BMAD-METHOD Orchestration for Stellar Connect
======================================================
Integrates BMAD methodology with existing sophisticated agent framework
including reasoning engine, specialist agents, and advanced orchestration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import json

from crewai import Agent, Task, Crew, Process
from src.config import CONFIG, init_settings
from src.agent_tools import vector_tool, kg_tool, extraction_tool

# Import existing sophisticated agents
from agents.reasoning.reasoning_engine import EstateReasoningEngine, ReasoningConfig, ExecutionMode
from agents.reasoning.gatekeeper import EstateGatekeeper
from agents.reasoning.planner import EstatePlanner
from agents.reasoning.auditor import EstateAuditor
from agents.reasoning.strategist import EstateStrategist
from agents.specialists.base_specialist import BaseSpecialist, SpecialistExpertise, TaskPriority
from agents.specialists.estate_librarian import EstateLibrarian
from agents.specialists.trust_sales_analyst import TrustSalesAnalyst
from agents.specialists.sales_specialist import SalesSpecialist
from agents.specialists.market_scout import MarketScout
from agents.specialists.similar_case_finder import SimilarCaseFinder
from agents.specialists.rebuttal_library import RebuttalLibrary

# Initialize settings
init_settings()

# ============================================================================
# Enhanced BMAD Role Definitions
# ============================================================================

class BMADRole(str, Enum):
    """Enhanced BMAD roles that integrate with existing architecture"""
    BUSINESS_ANALYST = "business_analyst"
    PROJECT_MANAGER = "project_manager"
    SOLUTION_ARCHITECT = "solution_architect"
    DEVELOPER = "developer"
    QA_TESTER = "qa_tester"
    SALES_SPECIALIST = "sales_specialist"

    # Integration with existing roles
    ESTATE_REASONER = "estate_reasoner"
    GATEKEEPER = "gatekeeper"
    PLANNER = "planner"
    AUDITOR = "auditor"
    STRATEGIST = "strategist"

class WorkflowType(str, Enum):
    """Types of BMAD workflows"""
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    SYSTEM_IMPLEMENTATION = "system_implementation"
    SALES_OPTIMIZATION = "sales_optimization"
    COGNITIVE_ANALYSIS = "cognitive_analysis"
    SPECIALIST_COORDINATION = "specialist_coordination"
    QUALITY_ASSURANCE = "quality_assurance"

# ============================================================================
# Enhanced Agent Configurations
# ============================================================================

@dataclass
class EnhancedBMADConfig:
    """Enhanced configuration integrating with existing architecture"""
    use_reasoning_engine: bool = True
    use_specialist_team: bool = True
    execution_mode: ExecutionMode = ExecutionMode.STANDARD
    max_parallel_agents: int = 6
    quality_threshold: float = 0.85
    enable_self_correction: bool = True
    memory_retention_days: int = 90

class BMADAgentType(str, Enum):
    """Types of BMAD agents"""
    CREW_AI = "crew_ai"
    REASONING = "reasoning"
    SPECIALIST = "specialist"
    HYBRID = "hybrid"

@dataclass
class BMADAgentConfig:
    """Configuration for BMAD agents"""
    role: BMADRole
    agent_type: BMADAgentType
    name: str
    goal: str
    backstory: str
    tools: List[Any] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    expertise: Optional[SpecialistExpertise] = None
    reasoning_components: List[str] = field(default_factory=list)
    collaboration_rules: Dict[str, Any] = field(default_factory=dict)
    quality_gates: Dict[str, float] = field(default_factory=dict)

# ============================================================================
# Enhanced BMAD Orchestrator
# ============================================================================

class EnhancedBMADOrchestrator:
    """Enhanced orchestrator integrating BMAD with existing sophisticated architecture"""

    def __init__(self, config: EnhancedBMADConfig = None):
        """Initialize enhanced orchestrator"""
        self.config = config or EnhancedBMADConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.reasoning_engine = EstateReasoningEngine() if self.config.use_reasoning_engine else None
        self.specialist_agents = self._initialize_specialist_agents()
        self.bmad_agents = self._initialize_bmad_agents()
        self.workflow_registry = self._initialize_workflows()

        # State management
        self.active_workflows: Dict[str, Dict] = {}
        self.agent_metrics: Dict[str, Dict] = {}
        self.collaboration_history: List[Dict] = []

    def _initialize_specialist_agents(self) -> Dict[str, BaseSpecialist]:
        """Initialize existing specialist agents"""
        if not self.config.use_specialist_team:
            return {}

        return {
            "estate_librarian": EstateLibrarian(),
            "trust_sales_analyst": TrustSalesAnalyst(),
            "sales_specialist": SalesSpecialist(),
            "market_scout": MarketScout(),
            "similar_case_finder": SimilarCaseFinder(),
            "rebuttal_library": RebuttalLibrary()
        }

    def _initialize_bmad_agents(self) -> Dict[BMADRole, Agent]:
        """Initialize BMAD CrewAI agents with enhanced capabilities"""

        agents = {}

        # Business Analyst with Estate Planning expertise
        agents[BMADRole.BUSINESS_ANALYST] = Agent(
            role="Estate Planning Business Analyst",
            goal="Extract and validate business requirements for estate planning solutions, integrating with reasoning engine for comprehensive analysis",
            backstory="""You are a senior business analyst specializing in estate planning and financial services.
            You work closely with the reasoning engine to validate requirements through the gatekeeper,
            plan analysis through the planner, and ensure quality through the auditor. You understand
            complex trust structures, tax implications, and client relationship dynamics.""",
            tools=[vector_tool, extraction_tool],
            llm=f"ollama/{CONFIG.GENERATIVE_MODEL}",
            verbose=True,
            allow_delegation=True,
            max_iter=5
        )

        # Project Manager with workflow orchestration
        agents[BMADRole.PROJECT_MANAGER] = Agent(
            role="Estate Planning Project Manager",
            goal="Orchestrate multi-agent workflows, coordinate specialist teams, and ensure delivery of estate planning solutions",
            backstory="""You are an experienced project manager specializing in estate planning technology solutions.
            You coordinate between the reasoning engine, specialist agents, and BMAD team members.
            You understand the complexity of estate planning processes and can break down complex
            client requirements into manageable tasks across multiple expert agents.""",
            tools=[],
            llm=f"ollama/{CONFIG.GENERATIVE_MODEL}",
            verbose=True,
            allow_delegation=True,
            max_iter=5
        )

        # Solution Architect with estate planning tech expertise
        agents[BMADRole.SOLUTION_ARCHITECT] = Agent(
            role="Estate Planning Solution Architect",
            goal="Design scalable estate planning technology solutions, integrate with knowledge graphs, and ensure best practices",
            backstory="""You are a solution architect with deep expertise in estate planning technology,
            knowledge graphs, and multi-agent systems. You design solutions that integrate seamlessly
            with the reasoning engine and specialist agents. You understand trust law, tax regulations,
            and how to model complex estate planning relationships in technology systems.""",
            tools=[kg_tool],
            llm=f"ollama/{CONFIG.GENERATIVE_MODEL}",
            verbose=True,
            allow_delegation=False,
            max_iter=5
        )

        # Developer with estate planning domain knowledge
        agents[BMADRole.DEVELOPER] = Agent(
            role="Estate Planning Solutions Developer",
            goal="Implement estate planning features, integrate with specialist agents, and ensure code quality",
            backstory="""You are a senior developer specializing in estate planning technology solutions.
            You implement features that integrate with the reasoning engine, specialist agents, and
            knowledge systems. You understand trust law requirements, compliance needs, and how to
            build reliable systems for high-stakes financial planning scenarios.""",
            tools=[extraction_tool],
            llm=f"ollama/{CONFIG.GENERATIVE_MODEL}",
            verbose=True,
            allow_delegation=False,
            max_iter=5
        )

        # QA Tester with estate planning compliance focus
        agents[BMADRole.QA_TESTER] = Agent(
            role="Estate Planning QA Specialist",
            goal="Ensure quality, compliance, and reliability of estate planning solutions and agent outputs",
            backstory="""You are a QA specialist with deep understanding of estate planning compliance,
            fiduciary responsibilities, and system reliability requirements. You test not just functionality
            but also compliance with estate planning regulations, accuracy of trust structures, and
            reliability of agent reasoning processes.""",
            tools=[vector_tool],
            llm=f"ollama/{CONFIG.GENERATIVE_MODEL}",
            verbose=True,
            allow_delegation=False,
            max_iter=5
        )

        # Sales Specialist integrating with existing sales agents
        agents[BMADRole.SALES_SPECIALIST] = Agent(
            role="Elite Estate Planning Sales Strategist",
            goal="Optimize sales processes, analyze conversion patterns, and coordinate with specialist sales agents",
            backstory="""You are an elite sales strategist specializing in high-net-worth estate planning.
            You work closely with the trust sales analyst, similar case finder, and rebuttal library
            to optimize sales processes. You understand the psychology of wealthy clients, complex
            family dynamics, and how to position sophisticated estate planning solutions.""",
            tools=[vector_tool, kg_tool, extraction_tool],
            llm=f"ollama/{CONFIG.GENERATIVE_MODEL}",
            verbose=True,
            allow_delegation=True,
            max_iter=5
        )

        return agents

    def _initialize_workflows(self) -> Dict[WorkflowType, Dict]:
        """Initialize workflow templates"""
        return {
            WorkflowType.COGNITIVE_ANALYSIS: {
                "name": "Cognitive Estate Analysis",
                "description": "Deep cognitive analysis using reasoning engine",
                "phases": ["gatekeeper", "planner", "specialist_execution", "auditor", "strategist"],
                "agents": [BMADRole.BUSINESS_ANALYST, BMADRole.SALES_SPECIALIST],
                "reasoning_required": True
            },

            WorkflowType.SPECIALIST_COORDINATION: {
                "name": "Specialist Team Coordination",
                "description": "Coordinate multiple specialist agents for comprehensive analysis",
                "phases": ["coordination", "parallel_execution", "synthesis", "validation"],
                "agents": [BMADRole.PROJECT_MANAGER, BMADRole.SALES_SPECIALIST],
                "specialist_required": True
            },

            WorkflowType.SALES_OPTIMIZATION: {
                "name": "Estate Planning Sales Optimization",
                "description": "Optimize sales processes using all available intelligence",
                "phases": ["analysis", "strategy", "implementation", "testing"],
                "agents": [BMADRole.SALES_SPECIALIST, BMADRole.BUSINESS_ANALYST, BMADRole.QA_TESTER],
                "reasoning_required": True,
                "specialist_required": True
            },

            WorkflowType.SYSTEM_IMPLEMENTATION: {
                "name": "System Feature Implementation",
                "description": "Full-cycle feature implementation with quality assurance",
                "phases": ["design", "development", "testing", "deployment"],
                "agents": [BMADRole.SOLUTION_ARCHITECT, BMADRole.DEVELOPER, BMADRole.QA_TESTER, BMADRole.PROJECT_MANAGER],
                "reasoning_required": False
            }
        }

    async def execute_cognitive_workflow(self, query: str) -> Dict[str, Any]:
        """Execute cognitive analysis workflow using reasoning engine"""

        if not self.reasoning_engine:
            raise ValueError("Reasoning engine not initialized")

        workflow_id = str(uuid.uuid4())
        self.active_workflows[workflow_id] = {
            "type": WorkflowType.COGNITIVE_ANALYSIS,
            "query": query,
            "start_time": datetime.now(),
            "status": "running"
        }

        try:
            # Step 1: Gatekeeper validation
            validation = await self.reasoning_engine.gatekeeper.validate_query(query)

            if validation.severity.value >= 3:  # High severity issues
                return {
                    "workflow_id": workflow_id,
                    "status": "validation_failed",
                    "validation_issues": validation.issues,
                    "recommendations": validation.recommendations
                }

            # Step 2: Analysis planning
            plan = await self.reasoning_engine.planner.create_analysis_plan(query)

            # Step 3: Execute with specialist coordination
            results = await self._execute_specialist_coordination(query, plan)

            # Step 4: Audit results
            audit = await self.reasoning_engine.auditor.audit_analysis(results)

            # Step 5: Strategic synthesis
            synthesis = await self.reasoning_engine.strategist.synthesize_results(results, audit)

            # Update workflow status
            self.active_workflows[workflow_id]["status"] = "completed"
            self.active_workflows[workflow_id]["end_time"] = datetime.now()

            return {
                "workflow_id": workflow_id,
                "status": "success",
                "validation": validation,
                "plan": plan,
                "results": results,
                "audit": audit,
                "synthesis": synthesis
            }

        except Exception as e:
            self.active_workflows[workflow_id]["status"] = "failed"
            self.active_workflows[workflow_id]["error"] = str(e)
            self.logger.error(f"Cognitive workflow failed: {e}")

            return {
                "workflow_id": workflow_id,
                "status": "error",
                "error": str(e)
            }

    async def _execute_specialist_coordination(self, query: str, plan: Any) -> Dict[str, Any]:
        """Coordinate specialist agents based on analysis plan"""

        results = {}

        # Determine which specialists to engage based on query type
        specialist_assignments = self._assign_specialists(query, plan)

        # Execute specialist tasks in parallel
        specialist_tasks = []
        for specialist_name, task_config in specialist_assignments.items():
            if specialist_name in self.specialist_agents:
                specialist = self.specialist_agents[specialist_name]
                task = specialist.execute_task(task_config)
                specialist_tasks.append((specialist_name, task))

        # Await all specialist results
        for specialist_name, task in specialist_tasks:
            try:
                result = await task
                results[specialist_name] = result
            except Exception as e:
                results[specialist_name] = {"error": str(e)}
                self.logger.error(f"Specialist {specialist_name} failed: {e}")

        return results

    def _assign_specialists(self, query: str, plan: Any) -> Dict[str, Dict]:
        """Assign specialists based on query analysis"""

        assignments = {}

        # Estate Librarian for document retrieval
        if "document" in query.lower() or "retrieve" in query.lower():
            assignments["estate_librarian"] = {
                "query": query,
                "task_type": "document_retrieval",
                "priority": TaskPriority.HIGH
            }

        # Trust Sales Analyst for deal analysis
        if any(word in query.lower() for word in ["sales", "deal", "conversion", "trust"]):
            assignments["trust_sales_analyst"] = {
                "query": query,
                "task_type": "sales_analysis",
                "priority": TaskPriority.HIGH
            }

        # Market Scout for competitive intelligence
        if any(word in query.lower() for word in ["market", "competitor", "pricing"]):
            assignments["market_scout"] = {
                "query": query,
                "task_type": "market_analysis",
                "priority": TaskPriority.MEDIUM
            }

        # Similar Case Finder for precedent analysis
        if any(word in query.lower() for word in ["similar", "case", "precedent", "example"]):
            assignments["similar_case_finder"] = {
                "query": query,
                "task_type": "case_matching",
                "priority": TaskPriority.MEDIUM
            }

        # Rebuttal Library for objection handling
        if any(word in query.lower() for word in ["objection", "concern", "rebuttal", "handle"]):
            assignments["rebuttal_library"] = {
                "query": query,
                "task_type": "objection_analysis",
                "priority": TaskPriority.HIGH
            }

        return assignments

    def create_enhanced_crew(self, workflow_type: WorkflowType, agents: List[BMADRole] = None) -> Crew:
        """Create enhanced crew with specified workflow type"""

        workflow_config = self.workflow_registry.get(workflow_type)
        if not workflow_config:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        # Select agents for workflow
        if agents is None:
            agents = workflow_config["agents"]

        crew_agents = [self.bmad_agents[role] for role in agents if role in self.bmad_agents]

        return Crew(
            agents=crew_agents,
            tasks=[],  # Tasks will be added dynamically
            process=Process.sequential,
            verbose=True,
            memory=True  # Enable crew memory
        )

    def execute_bmad_workflow(self, workflow_type: WorkflowType, request: str, agents: List[BMADRole] = None) -> str:
        """Execute BMAD workflow with enhanced capabilities"""

        print(f"\n{'='*80}")
        print(f"Enhanced BMAD-METHOD Orchestrator")
        print(f"Workflow: {workflow_type.value}")
        print(f"Request: {request}")
        print(f"{'='*80}\n")

        workflow_config = self.workflow_registry.get(workflow_type)
        if not workflow_config:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        # Create tasks based on workflow type
        tasks = self._create_workflow_tasks(workflow_type, request, agents)

        # Create and execute crew
        crew = self.create_enhanced_crew(workflow_type, agents)
        crew.tasks = tasks

        # Execute with timing
        start_time = datetime.now()
        result = crew.kickoff()
        end_time = datetime.now()

        # Record metrics
        workflow_id = str(uuid.uuid4())
        self.agent_metrics[workflow_id] = {
            "workflow_type": workflow_type.value,
            "duration": (end_time - start_time).total_seconds(),
            "agents_used": len(crew.agents),
            "tasks_completed": len(tasks),
            "timestamp": end_time.isoformat()
        }

        return str(result)

    def _create_workflow_tasks(self, workflow_type: WorkflowType, request: str, agents: List[BMADRole]) -> List[Task]:
        """Create tasks based on workflow type"""

        tasks = []

        if workflow_type == WorkflowType.SALES_OPTIMIZATION:
            # Sales optimization workflow
            if BMADRole.SALES_SPECIALIST in agents:
                tasks.append(Task(
                    description=f"""Analyze estate planning sales patterns and optimization opportunities for: {request}
                    Integrate insights from specialist agents including trust sales analyst and rebuttal library.
                    Identify specific conversion bottlenecks and improvement strategies.""",
                    expected_output="Comprehensive sales optimization analysis with actionable recommendations",
                    agent=self.bmad_agents[BMADRole.SALES_SPECIALIST]
                ))

            if BMADRole.BUSINESS_ANALYST in agents:
                tasks.append(Task(
                    description="""Validate sales optimization requirements and define success metrics.
                    Ensure alignment with business objectives and compliance requirements.
                    Document acceptance criteria for optimization initiatives.""",
                    expected_output="Validated requirements document with success metrics",
                    agent=self.bmad_agents[BMADRole.BUSINESS_ANALYST]
                ))

        elif workflow_type == WorkflowType.COGNITIVE_ANALYSIS:
            # Cognitive analysis workflow
            if BMADRole.BUSINESS_ANALYST in agents:
                tasks.append(Task(
                    description=f"""Perform cognitive analysis of: {request}
                    Use reasoning engine components for validation, planning, and analysis.
                    Integrate insights from specialist agents as needed.""",
                    expected_output="Comprehensive cognitive analysis with reasoning chain",
                    agent=self.bmad_agents[BMADRole.BUSINESS_ANALYST]
                ))

        elif workflow_type == WorkflowType.SYSTEM_IMPLEMENTATION:
            # System implementation workflow
            if BMADRole.SOLUTION_ARCHITECT in agents:
                tasks.append(Task(
                    description=f"""Design technical solution for: {request}
                    Consider integration with reasoning engine and specialist agents.
                    Define scalable architecture and implementation approach.""",
                    expected_output="Technical design document with implementation plan",
                    agent=self.bmad_agents[BMADRole.SOLUTION_ARCHITECT]
                ))

            if BMADRole.DEVELOPER in agents:
                tasks.append(Task(
                    description="""Implement the designed solution with proper integration.
                    Ensure compatibility with existing agent framework.
                    Include comprehensive error handling and logging.""",
                    expected_output="Production-ready implementation with documentation",
                    agent=self.bmad_agents[BMADRole.DEVELOPER]
                ))

            if BMADRole.QA_TESTER in agents:
                tasks.append(Task(
                    description="""Test implementation for functionality, compliance, and reliability.
                    Validate integration with reasoning engine and specialist agents.
                    Ensure estate planning compliance requirements are met.""",
                    expected_output="Comprehensive test report with quality assessment",
                    agent=self.bmad_agents[BMADRole.QA_TESTER]
                ))

        return tasks

    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive status of enhanced orchestrator"""

        return {
            "config": {
                "reasoning_engine_enabled": self.config.use_reasoning_engine,
                "specialist_team_enabled": self.config.use_specialist_team,
                "execution_mode": self.config.execution_mode.value,
                "quality_threshold": self.config.quality_threshold
            },
            "agents": {
                "bmad_agents": {
                    role.value: {
                        "name": agent.role,
                        "goal": agent.goal,
                        "tools": len(agent.tools)
                    }
                    for role, agent in self.bmad_agents.items()
                },
                "specialist_agents": {
                    name: {
                        "type": type(agent).__name__,
                        "expertise": getattr(agent, 'expertise', 'Unknown'),
                        "status": "active"
                    }
                    for name, agent in self.specialist_agents.items()
                }
            },
            "workflows": {
                "available": [wf.value for wf in WorkflowType],
                "active": len(self.active_workflows),
                "completed": len(self.agent_metrics)
            },
            "performance": {
                "total_workflows": len(self.agent_metrics),
                "avg_duration": sum(m["duration"] for m in self.agent_metrics.values()) / len(self.agent_metrics) if self.agent_metrics else 0,
                "collaboration_events": len(self.collaboration_history)
            }
        }

# ============================================================================
# Enhanced Integration Layer
# ============================================================================

class EnhancedBMADIntegration:
    """Enhanced integration layer for Streamlit dashboard"""

    def __init__(self, config: EnhancedBMADConfig = None):
        """Initialize enhanced integration"""
        self.orchestrator = EnhancedBMADOrchestrator(config)

    def process_enhanced_query(self, query: str, mode: str = "cognitive") -> str:
        """Process query with enhanced BMAD capabilities"""

        if mode == "cognitive":
            # Use cognitive analysis workflow
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.orchestrator.execute_cognitive_workflow(query)
                )
                return json.dumps(result, indent=2, default=str)
            finally:
                loop.close()

        elif mode == "sales_optimization":
            # Use sales optimization workflow
            return self.orchestrator.execute_bmad_workflow(
                WorkflowType.SALES_OPTIMIZATION,
                query,
                [BMADRole.SALES_SPECIALIST, BMADRole.BUSINESS_ANALYST]
            )

        elif mode == "specialist_coordination":
            # Use specialist coordination
            return self.orchestrator.execute_bmad_workflow(
                WorkflowType.SPECIALIST_COORDINATION,
                query,
                [BMADRole.PROJECT_MANAGER, BMADRole.SALES_SPECIALIST]
            )

        else:
            # Standard BMAD workflow
            return self.orchestrator.execute_bmad_workflow(
                WorkflowType.COGNITIVE_ANALYSIS,
                query,
                [BMADRole.BUSINESS_ANALYST, BMADRole.SALES_SPECIALIST]
            )

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard display"""
        return self.orchestrator.get_enhanced_status()

# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Demonstrate enhanced BMAD orchestration"""

    print("\n" + "="*80)
    print("Enhanced BMAD-METHOD Orchestration for Stellar Connect")
    print("="*80)

    # Initialize enhanced orchestrator
    config = EnhancedBMADConfig(
        use_reasoning_engine=True,
        use_specialist_team=True,
        execution_mode=ExecutionMode.STANDARD
    )

    integration = EnhancedBMADIntegration(config)

    # Example 1: Cognitive analysis
    print("\nExample 1: Cognitive Analysis Mode")
    print("-" * 40)
    query = "What are the most effective strategies for overcoming price objections in high-net-worth estate planning?"

    # Note: Uncomment to execute
    # result = integration.process_enhanced_query(query, mode="cognitive")
    # print(f"Result: {result[:500]}...")

    # Example 2: Get dashboard metrics
    print("\nExample 2: Enhanced Dashboard Metrics")
    print("-" * 40)
    metrics = integration.get_dashboard_metrics()
    print(json.dumps(metrics, indent=2, default=str))

    print("\n" + "="*80)
    print("Enhanced BMAD orchestration ready for integration!")
    print("="*80)

if __name__ == "__main__":
    main()