#!/usr/bin/env python3
"""
BMAD-METHOD Multi-Agent Orchestration System for Stellar Connect
================================================================
Business-Minded Autonomous Development orchestrator that coordinates
specialized agents for comprehensive sales intelligence and automation.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel, Field
import json
from datetime import datetime
from src.config import CONFIG, init_settings
from src.agent_tools import vector_tool, kg_tool, extraction_tool

# Initialize settings
init_settings()

# ============================================================================
# BMAD Agent Role Definitions
# ============================================================================

class BMADRole(str, Enum):
    """Core BMAD-METHOD agent roles"""
    BUSINESS_ANALYST = "business_analyst"
    PROJECT_MANAGER = "project_manager"
    SOLUTION_ARCHITECT = "solution_architect"
    DEVELOPER = "developer"
    QA_TESTER = "qa_tester"
    SALES_SPECIALIST = "sales_specialist"

class AgentCapability(str, Enum):
    """Agent capabilities for task routing"""
    REQUIREMENT_GATHERING = "requirement_gathering"
    TASK_PLANNING = "task_planning"
    SYSTEM_DESIGN = "system_design"
    CODE_IMPLEMENTATION = "code_implementation"
    QUALITY_ASSURANCE = "quality_assurance"
    SALES_ANALYSIS = "sales_analysis"

# ============================================================================
# BMAD Agent Configuration Models
# ============================================================================

class BMADAgentConfig(BaseModel):
    """Configuration for a BMAD agent"""
    role: BMADRole
    name: str
    goal: str
    backstory: str
    capabilities: List[AgentCapability]
    tools: List[Any] = Field(default_factory=list)
    max_iter: int = Field(default=5)
    verbose: bool = Field(default=True)
    allow_delegation: bool = Field(default=True)

    class Config:
        arbitrary_types_allowed = True

class TaskHandoff(BaseModel):
    """Task handoff between BMAD agents"""
    from_agent: BMADRole
    to_agent: BMADRole
    task_id: str
    context: Dict[str, Any]
    deliverables: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)

# ============================================================================
# BMAD Agent Factory
# ============================================================================

class BMADAgentFactory:
    """Factory for creating specialized BMAD agents"""

    @staticmethod
    def create_business_analyst() -> Agent:
        """Create Business Analyst agent for requirement extraction"""
        config = BMADAgentConfig(
            role=BMADRole.BUSINESS_ANALYST,
            name="Strategic Business Analyst",
            goal="Extract and clarify business requirements, validate sales use cases, and ensure alignment with strategic objectives",
            backstory="""You are a seasoned business analyst specializing in sales enablement and CRM systems.
            With 15 years of experience in estate planning and financial services, you understand the nuances
            of consultative selling and can translate complex business needs into actionable requirements.
            You excel at identifying patterns in successful deals and understanding what drives conversion.""",
            capabilities=[
                AgentCapability.REQUIREMENT_GATHERING,
                AgentCapability.SALES_ANALYSIS
            ],
            tools=[vector_tool, extraction_tool]
        )

        return Agent(
            role=config.name,
            goal=config.goal,
            backstory=config.backstory,
            tools=config.tools,
            llm=f"ollama/{CONFIG.GENERATIVE_MODEL}",
            max_iter=config.max_iter,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation
        )

    @staticmethod
    def create_project_manager() -> Agent:
        """Create Project Manager agent for orchestration and tracking"""
        config = BMADAgentConfig(
            role=BMADRole.PROJECT_MANAGER,
            name="Agile Project Manager",
            goal="Define deliverables, break down milestones, coordinate agent activities, and ensure timely delivery",
            backstory="""You are an experienced project manager with expertise in agile methodologies and
            multi-agent system coordination. You've managed numerous sales enablement projects and understand
            how to balance speed with quality. You excel at breaking complex projects into manageable sprints
            and ensuring smooth handoffs between team members.""",
            capabilities=[
                AgentCapability.TASK_PLANNING
            ],
            tools=[]
        )

        return Agent(
            role=config.name,
            goal=config.goal,
            backstory=config.backstory,
            tools=config.tools,
            llm=f"ollama/{CONFIG.GENERATIVE_MODEL}",
            max_iter=config.max_iter,
            verbose=config.verbose,
            allow_delegation=True  # PM can delegate to other agents
        )

    @staticmethod
    def create_solution_architect() -> Agent:
        """Create Solution Architect agent for system design"""
        config = BMADAgentConfig(
            role=BMADRole.SOLUTION_ARCHITECT,
            name="Senior Solution Architect",
            goal="Design scalable system architecture, select optimal tech stack, and enforce best practices",
            backstory="""You are a solution architect with deep expertise in RAG systems, multi-agent
            architectures, and sales automation platforms. You've designed systems that process millions
            of sales interactions and understand how to balance performance, scalability, and maintainability.
            You're passionate about clean architecture and design patterns.""",
            capabilities=[
                AgentCapability.SYSTEM_DESIGN
            ],
            tools=[kg_tool]  # Can analyze system structure via knowledge graph
        )

        return Agent(
            role=config.name,
            goal=config.goal,
            backstory=config.backstory,
            tools=config.tools,
            llm=f"ollama/{CONFIG.GENERATIVE_MODEL}",
            max_iter=config.max_iter,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation
        )

    @staticmethod
    def create_developer() -> Agent:
        """Create Developer agent for implementation"""
        config = BMADAgentConfig(
            role=BMADRole.DEVELOPER,
            name="Full-Stack Developer",
            goal="Implement features, write integration-ready code, and ensure proper documentation",
            backstory="""You are a senior full-stack developer specializing in Python, React, and AI/ML
            integrations. You've built numerous sales automation tools and understand the importance of
            clean, maintainable code. You follow TDD principles and ensure all code is production-ready
            with comprehensive error handling and logging.""",
            capabilities=[
                AgentCapability.CODE_IMPLEMENTATION
            ],
            tools=[extraction_tool]  # Can generate structured code artifacts
        )

        return Agent(
            role=config.name,
            goal=config.goal,
            backstory=config.backstory,
            tools=config.tools,
            llm=f"ollama/{CONFIG.GENERATIVE_MODEL}",
            max_iter=config.max_iter,
            verbose=config.verbose,
            allow_delegation=False  # Developer focuses on implementation
        )

    @staticmethod
    def create_qa_tester() -> Agent:
        """Create QA/Tester agent for quality assurance"""
        config = BMADAgentConfig(
            role=BMADRole.QA_TESTER,
            name="Senior QA Engineer",
            goal="Review code quality, create comprehensive test cases, and ensure system reliability",
            backstory="""You are a meticulous QA engineer with expertise in test automation, security
            testing, and performance optimization. You've tested critical financial systems and understand
            the importance of data accuracy in sales contexts. You think like a hacker to find edge cases
            and potential vulnerabilities.""",
            capabilities=[
                AgentCapability.QUALITY_ASSURANCE
            ],
            tools=[vector_tool]  # Can search for test scenarios
        )

        return Agent(
            role=config.name,
            goal=config.goal,
            backstory=config.backstory,
            tools=config.tools,
            llm=f"ollama/{CONFIG.GENERATIVE_MODEL}",
            max_iter=config.max_iter,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation
        )

    @staticmethod
    def create_sales_specialist() -> Agent:
        """Create Sales Specialist agent for domain expertise"""
        config = BMADAgentConfig(
            role=BMADRole.SALES_SPECIALIST,
            name="Elite Sales Strategist",
            goal="Apply sales domain knowledge, analyze pipeline insights, and optimize playbooks",
            backstory="""You are a top-performing sales strategist with 20 years in estate planning
            and financial services. You've closed over $50M in deals and trained hundreds of advisors.
            You understand the psychology of high-net-worth clients and can identify winning patterns
            in sales conversations. You're data-driven and constantly optimize sales processes.""",
            capabilities=[
                AgentCapability.SALES_ANALYSIS,
                AgentCapability.REQUIREMENT_GATHERING
            ],
            tools=[vector_tool, kg_tool, extraction_tool]  # Full access to analyze sales data
        )

        return Agent(
            role=config.name,
            goal=config.goal,
            backstory=config.backstory,
            tools=config.tools,
            llm=f"ollama/{CONFIG.GENERATIVE_MODEL}",
            max_iter=config.max_iter,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation
        )

# ============================================================================
# BMAD Orchestration Engine
# ============================================================================

class BMADOrchestrator:
    """Main orchestration engine for BMAD agents"""

    def __init__(self):
        """Initialize BMAD orchestrator with all agents"""
        self.agents = {
            BMADRole.BUSINESS_ANALYST: BMADAgentFactory.create_business_analyst(),
            BMADRole.PROJECT_MANAGER: BMADAgentFactory.create_project_manager(),
            BMADRole.SOLUTION_ARCHITECT: BMADAgentFactory.create_solution_architect(),
            BMADRole.DEVELOPER: BMADAgentFactory.create_developer(),
            BMADRole.QA_TESTER: BMADAgentFactory.create_qa_tester(),
            BMADRole.SALES_SPECIALIST: BMADAgentFactory.create_sales_specialist()
        }

        self.handoff_history: List[TaskHandoff] = []

    def create_requirement_analysis_workflow(self, user_request: str) -> Crew:
        """Create workflow for requirement analysis and validation"""

        # Phase 1: Business Analysis
        requirement_extraction = Task(
            description=f"""Analyze the user request: '{user_request}'
            Extract all functional and non-functional requirements.
            Identify success criteria and key performance indicators.
            Document any assumptions or constraints.""",
            expected_output="Comprehensive requirements document with prioritized features",
            agent=self.agents[BMADRole.BUSINESS_ANALYST]
        )

        # Phase 2: Sales Domain Validation
        sales_validation = Task(
            description="""Review the requirements from a sales perspective.
            Validate against common sales patterns and best practices.
            Identify potential impact on conversion rates and deal velocity.
            Suggest enhancements based on successful sales strategies.""",
            expected_output="Sales-validated requirements with strategic recommendations",
            agent=self.agents[BMADRole.SALES_SPECIALIST]
        )

        # Phase 3: Project Planning
        project_planning = Task(
            description="""Create a detailed project plan for the requirements.
            Break down into epics, stories, and tasks.
            Estimate effort and define sprint boundaries.
            Identify dependencies and critical path.""",
            expected_output="Agile project plan with sprint breakdown and timeline",
            agent=self.agents[BMADRole.PROJECT_MANAGER]
        )

        return Crew(
            agents=[
                self.agents[BMADRole.BUSINESS_ANALYST],
                self.agents[BMADRole.SALES_SPECIALIST],
                self.agents[BMADRole.PROJECT_MANAGER]
            ],
            tasks=[requirement_extraction, sales_validation, project_planning],
            process=Process.sequential,
            verbose=True
        )

    def create_implementation_workflow(self, requirements: str) -> Crew:
        """Create workflow for system implementation"""

        # Phase 1: Architecture Design
        system_design = Task(
            description=f"""Design the technical architecture for: {requirements}
            Define system components and interfaces.
            Select appropriate technologies and frameworks.
            Create data flow and integration diagrams.""",
            expected_output="Technical architecture document with design patterns",
            agent=self.agents[BMADRole.SOLUTION_ARCHITECT]
        )

        # Phase 2: Development
        implementation = Task(
            description="""Implement the designed solution.
            Write clean, documented, production-ready code.
            Follow SOLID principles and design patterns.
            Include comprehensive error handling and logging.""",
            expected_output="Complete implementation with documentation",
            agent=self.agents[BMADRole.DEVELOPER]
        )

        # Phase 3: Quality Assurance
        testing = Task(
            description="""Test the implementation thoroughly.
            Create unit, integration, and end-to-end test cases.
            Perform security and performance testing.
            Document any issues or recommendations.""",
            expected_output="Test report with coverage metrics and recommendations",
            agent=self.agents[BMADRole.QA_TESTER]
        )

        return Crew(
            agents=[
                self.agents[BMADRole.SOLUTION_ARCHITECT],
                self.agents[BMADRole.DEVELOPER],
                self.agents[BMADRole.QA_TESTER]
            ],
            tasks=[system_design, implementation, testing],
            process=Process.sequential,
            verbose=True
        )

    def create_sales_analysis_workflow(self, analysis_request: str) -> Crew:
        """Create workflow for sales analysis and optimization"""

        # Phase 1: Data Analysis
        data_analysis = Task(
            description=f"""Analyze sales data for: {analysis_request}
            Extract patterns from successful and failed deals.
            Identify key conversion factors and bottlenecks.
            Calculate relevant metrics and KPIs.""",
            expected_output="Data analysis report with insights and patterns",
            agent=self.agents[BMADRole.SALES_SPECIALIST]
        )

        # Phase 2: Requirements for Improvement
        improvement_requirements = Task(
            description="""Based on the analysis, define improvement requirements.
            Specify what system changes could enhance sales performance.
            Prioritize based on potential impact and effort.""",
            expected_output="Prioritized list of system improvements",
            agent=self.agents[BMADRole.BUSINESS_ANALYST]
        )

        # Phase 3: Implementation Planning
        implementation_plan = Task(
            description="""Create an implementation plan for the improvements.
            Define quick wins vs. long-term initiatives.
            Estimate resources and timeline.""",
            expected_output="Phased implementation roadmap",
            agent=self.agents[BMADRole.PROJECT_MANAGER]
        )

        return Crew(
            agents=[
                self.agents[BMADRole.SALES_SPECIALIST],
                self.agents[BMADRole.BUSINESS_ANALYST],
                self.agents[BMADRole.PROJECT_MANAGER]
            ],
            tasks=[data_analysis, improvement_requirements, implementation_plan],
            process=Process.sequential,
            verbose=True
        )

    def execute_workflow(self, workflow_type: str, request: str) -> str:
        """Execute a specific BMAD workflow"""

        print(f"\n{'='*60}")
        print(f"BMAD-METHOD Orchestrator Starting")
        print(f"Workflow: {workflow_type}")
        print(f"Request: {request}")
        print(f"{'='*60}\n")

        # Select and create appropriate workflow
        if workflow_type == "requirements":
            crew = self.create_requirement_analysis_workflow(request)
        elif workflow_type == "implementation":
            crew = self.create_implementation_workflow(request)
        elif workflow_type == "sales_analysis":
            crew = self.create_sales_analysis_workflow(request)
        else:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        # Execute the crew
        result = crew.kickoff()

        # Record handoff
        self.handoff_history.append(TaskHandoff(
            from_agent=BMADRole.PROJECT_MANAGER,
            to_agent=BMADRole.BUSINESS_ANALYST,
            task_id=f"workflow_{workflow_type}_{datetime.now().isoformat()}",
            context={"request": request, "result": str(result)},
            deliverables=[str(result)]
        ))

        return result

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all BMAD agents"""
        return {
            "agents": {
                role.value: {
                    "name": agent.role,
                    "goal": agent.goal,
                    "tools": len(agent.tools) if hasattr(agent, 'tools') else 0,
                    "status": "active"
                }
                for role, agent in self.agents.items()
            },
            "handoff_history": len(self.handoff_history),
            "last_handoff": self.handoff_history[-1].dict() if self.handoff_history else None
        }

# ============================================================================
# BMAD Workflow Templates
# ============================================================================

class BMADWorkflowTemplates:
    """Pre-defined workflow templates for common scenarios"""

    @staticmethod
    def sales_pipeline_optimization() -> Dict[str, Any]:
        """Template for optimizing sales pipeline"""
        return {
            "name": "Sales Pipeline Optimization",
            "description": "Analyze and optimize the entire sales pipeline",
            "phases": [
                {
                    "phase": "Discovery",
                    "agent": BMADRole.SALES_SPECIALIST,
                    "tasks": [
                        "Analyze current pipeline metrics",
                        "Identify bottlenecks and drop-off points",
                        "Review successful deal patterns"
                    ]
                },
                {
                    "phase": "Requirements",
                    "agent": BMADRole.BUSINESS_ANALYST,
                    "tasks": [
                        "Document optimization requirements",
                        "Define success metrics",
                        "Prioritize improvements"
                    ]
                },
                {
                    "phase": "Design",
                    "agent": BMADRole.SOLUTION_ARCHITECT,
                    "tasks": [
                        "Design system enhancements",
                        "Define integration points",
                        "Create data flow diagrams"
                    ]
                },
                {
                    "phase": "Implementation",
                    "agent": BMADRole.DEVELOPER,
                    "tasks": [
                        "Implement pipeline tracking",
                        "Build analytics dashboards",
                        "Create automation workflows"
                    ]
                },
                {
                    "phase": "Testing",
                    "agent": BMADRole.QA_TESTER,
                    "tasks": [
                        "Test data accuracy",
                        "Validate metrics calculation",
                        "Performance testing"
                    ]
                }
            ]
        }

    @staticmethod
    def client_segmentation_analysis() -> Dict[str, Any]:
        """Template for client segmentation and targeting"""
        return {
            "name": "Client Segmentation Analysis",
            "description": "Analyze and segment clients for targeted strategies",
            "phases": [
                {
                    "phase": "Analysis",
                    "agent": BMADRole.SALES_SPECIALIST,
                    "tasks": [
                        "Analyze client demographics",
                        "Identify high-value segments",
                        "Map segment-specific needs"
                    ]
                },
                {
                    "phase": "Strategy",
                    "agent": BMADRole.BUSINESS_ANALYST,
                    "tasks": [
                        "Define segment strategies",
                        "Create targeting criteria",
                        "Document personalization rules"
                    ]
                },
                {
                    "phase": "Implementation",
                    "agent": BMADRole.DEVELOPER,
                    "tasks": [
                        "Build segmentation engine",
                        "Implement targeting logic",
                        "Create personalization features"
                    ]
                }
            ]
        }

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Example usage of BMAD orchestrator"""

    # Initialize orchestrator
    orchestrator = BMADOrchestrator()

    # Example 1: Requirement Analysis
    print("\n" + "="*60)
    print("Example 1: Requirement Analysis Workflow")
    print("="*60)

    requirement_request = """
    We need to improve our estate planning sales conversion rate.
    Analyze current patterns and suggest system enhancements to
    increase deal closure by 25% in the next quarter.
    """

    # Note: Uncomment to execute
    # result = orchestrator.execute_workflow("requirements", requirement_request)
    # print(f"Result: {result}")

    # Example 2: Get agent status
    print("\n" + "="*60)
    print("BMAD Agent Status")
    print("="*60)

    status = orchestrator.get_agent_status()
    print(json.dumps(status, indent=2, default=str))

    # Example 3: Show workflow template
    print("\n" + "="*60)
    print("Sales Pipeline Optimization Template")
    print("="*60)

    template = BMADWorkflowTemplates.sales_pipeline_optimization()
    print(json.dumps(template, indent=2))

if __name__ == "__main__":
    main()