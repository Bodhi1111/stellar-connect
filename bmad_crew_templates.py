#!/usr/bin/env python3
"""
BMAD CrewAI Agent Templates for Stellar Connect
===============================================
Production-ready CrewAI agent templates with sophisticated collaboration
and integration with existing reasoning engine and specialist agents.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from crewai.agent import Agent as CrewAIAgent
from crewai.task import Task as CrewAITask

from src.config import CONFIG, init_settings
from src.agent_tools import vector_tool, kg_tool, extraction_tool

# Initialize settings
init_settings()

# ============================================================================
# Enhanced Collaboration Framework
# ============================================================================

class CollaborationMode(str, Enum):
    """Modes of agent collaboration"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    REASONING_ENHANCED = "reasoning_enhanced"
    SPECIALIST_COORDINATED = "specialist_coordinated"

class HandoffProtocol(str, Enum):
    """Protocols for agent handoffs"""
    DIRECT = "direct"              # Direct task output passing
    VALIDATED = "validated"        # Quality-validated handoff
    SYNTHESIZED = "synthesized"    # Multi-source synthesis
    REASONED = "reasoned"         # Reasoning-engine mediated

@dataclass
class CollaborationConfig:
    """Configuration for agent collaboration"""
    mode: CollaborationMode
    handoff_protocol: HandoffProtocol
    quality_threshold: float = 0.85
    timeout_seconds: int = 300
    retry_attempts: int = 3
    validation_required: bool = True

class AgentCollaborationManager:
    """Manages sophisticated agent collaboration patterns"""

    def __init__(self):
        self.collaboration_history: List[Dict] = []
        self.performance_metrics: Dict[str, Dict] = {}

    def create_collaboration_task(
        self,
        description: str,
        expected_output: str,
        agent: CrewAIAgent,
        collaboration_config: CollaborationConfig,
        context_tasks: List[CrewAITask] = None,
        validation_agents: List[CrewAIAgent] = None
    ) -> CrewAITask:
        """Create task with enhanced collaboration capabilities"""

        # Enhance description with collaboration instructions
        enhanced_description = f"""
        {description}

        COLLABORATION INSTRUCTIONS:
        - Mode: {collaboration_config.mode.value}
        - Handoff Protocol: {collaboration_config.handoff_protocol.value}
        - Quality Threshold: {collaboration_config.quality_threshold}
        - Validation Required: {collaboration_config.validation_required}

        INTEGRATION REQUIREMENTS:
        - Integrate insights from reasoning engine components when available
        - Coordinate with specialist agents for domain expertise
        - Ensure output quality meets or exceeds threshold requirements
        - Follow established handoff protocols for seamless collaboration
        """

        task = Task(
            description=enhanced_description,
            expected_output=expected_output,
            agent=agent,
            context=context_tasks or []
        )

        return task

    def record_collaboration(self, agent_from: str, agent_to: str, task_id: str, success: bool, metrics: Dict):
        """Record collaboration event for analysis"""
        self.collaboration_history.append({
            "timestamp": datetime.now().isoformat(),
            "from_agent": agent_from,
            "to_agent": agent_to,
            "task_id": task_id,
            "success": success,
            "metrics": metrics
        })

# ============================================================================
# BMAD Agent Templates
# ============================================================================

class BMADAgentTemplates:
    """Production-ready BMAD agent templates with sophisticated integration"""

    def __init__(self):
        self.collaboration_manager = AgentCollaborationManager()
        self.llm_config = f"ollama/{CONFIG.GENERATIVE_MODEL}"

    def create_business_analyst_agent(self) -> CrewAIAgent:
        """Create sophisticated Business Analyst agent with reasoning integration"""

        return Agent(
            role="Senior Estate Planning Business Analyst",
            goal="""Extract, validate, and synthesize business requirements for estate planning solutions.
            Integrate with reasoning engine for comprehensive analysis and coordinate with specialist
            agents for domain-specific validation. Ensure all requirements meet fiduciary standards.""",

            backstory="""You are a senior business analyst with 15+ years in estate planning and wealth management.
            You specialize in extracting complex requirements from client conversations and translating them
            into actionable business specifications. You work closely with the reasoning engine's gatekeeper
            for validation, planner for analysis strategy, and auditor for quality assurance.

            Your expertise includes:
            - Trust and estate law requirements analysis
            - Fiduciary responsibility specification
            - Regulatory compliance validation
            - Complex family dynamics assessment
            - High-net-worth client needs analysis

            You collaborate seamlessly with specialist agents including the estate librarian for documentation
            research, trust sales analyst for business impact assessment, and similar case finder for
            precedent analysis. Your output is always structured, validated, and ready for implementation.""",

            tools=[vector_tool, extraction_tool],
            llm=self.llm_config,
            verbose=True,
            allow_delegation=True,
            max_iter=7,
            max_execution_time=300
        )

    def create_project_manager_agent(self) -> CrewAIAgent:
        """Create sophisticated Project Manager agent with orchestration capabilities"""

        return Agent(
            role="Estate Planning Technology Project Manager",
            goal="""Orchestrate complex multi-agent workflows, coordinate between reasoning engine
            and specialist agents, manage parallel execution, and ensure timely delivery of
            estate planning technology solutions.""",

            backstory="""You are an experienced project manager specializing in estate planning technology
            and multi-agent system orchestration. You have deep understanding of how to coordinate
            complex workflows involving reasoning engines, specialist agents, and development teams.

            Your expertise includes:
            - Multi-agent workflow design and optimization
            - Parallel task coordination and synchronization
            - Resource allocation and performance optimization
            - Risk management in complex estate planning projects
            - Stakeholder communication and expectation management

            You excel at coordinating between the reasoning engine components (gatekeeper, planner,
            auditor, strategist) and specialist agents (estate librarian, trust sales analyst,
            market scout, etc.). You understand the dependencies, timing requirements, and quality
            gates necessary for successful estate planning solution delivery.""",

            tools=[],  # PM focuses on coordination, not direct tools
            llm=self.llm_config,
            verbose=True,
            allow_delegation=True,
            max_iter=10,
            max_execution_time=600
        )

    def create_solution_architect_agent(self) -> CrewAIAgent:
        """Create sophisticated Solution Architect agent with system design expertise"""

        return Agent(
            role="Senior Estate Planning Solution Architect",
            goal="""Design scalable, secure, and compliant estate planning technology solutions.
            Architect integration patterns for reasoning engines, specialist agents, and knowledge
            systems. Ensure solutions meet performance, security, and regulatory requirements.""",

            backstory="""You are a solution architect with deep expertise in estate planning technology,
            multi-agent systems, and financial services architecture. You design solutions that integrate
            seamlessly with sophisticated reasoning engines and specialist agent teams.

            Your expertise includes:
            - Multi-agent system architecture and design patterns
            - Knowledge graph design for complex estate planning relationships
            - Vector database optimization for semantic search
            - Real-time collaboration and communication patterns
            - Security and compliance architecture for financial services
            - Performance optimization and scalability planning

            You work closely with the reasoning engine strategist for architectural recommendations
            and coordinate with specialist agents to understand integration requirements. Your designs
            always consider the sophisticated interplay between cognitive reasoning, domain expertise,
            and system performance requirements.""",

            tools=[kg_tool],  # Architect uses knowledge graph for system analysis
            llm=self.llm_config,
            verbose=True,
            allow_delegation=False,
            max_iter=6,
            max_execution_time=400
        )

    def create_developer_agent(self) -> CrewAIAgent:
        """Create sophisticated Developer agent with estate planning domain expertise"""

        return Agent(
            role="Senior Estate Planning Solutions Developer",
            goal="""Implement sophisticated estate planning features with seamless integration to
            reasoning engines and specialist agents. Build reliable, secure, and performant
            solutions that meet complex estate planning requirements.""",

            backstory="""You are a senior developer specializing in estate planning technology solutions
            and multi-agent system integration. You have deep understanding of both the technical
            requirements and the domain complexities of estate planning.

            Your expertise includes:
            - Asynchronous multi-agent communication patterns
            - Integration with reasoning engines and cognitive pipelines
            - Estate planning domain modeling and data structures
            - Security and compliance implementation for financial services
            - Performance optimization for concurrent agent operations
            - Error handling and reliability for mission-critical systems

            You build solutions that integrate seamlessly with the reasoning engine's cognitive
            pipeline and coordinate effectively with specialist agents. Your code is always
            production-ready, well-documented, and designed for the high-stakes environment
            of estate planning and wealth management.""",

            tools=[extraction_tool],  # Developer uses structured extraction for implementation
            llm=self.llm_config,
            verbose=True,
            allow_delegation=False,
            max_iter=8,
            max_execution_time=500
        )

    def create_qa_tester_agent(self) -> CrewAIAgent:
        """Create sophisticated QA/Tester agent with compliance expertise"""

        return Agent(
            role="Senior Estate Planning QA and Compliance Specialist",
            goal="""Ensure quality, compliance, and reliability of estate planning solutions
            and multi-agent workflows. Validate reasoning accuracy, specialist coordination,
            and regulatory compliance across all system components.""",

            backstory="""You are a QA specialist with deep expertise in estate planning compliance,
            multi-agent system testing, and financial services quality assurance. You understand
            the critical importance of accuracy and compliance in estate planning technology.

            Your expertise includes:
            - Multi-agent workflow testing and validation
            - Estate planning compliance and regulatory testing
            - Reasoning engine accuracy and reliability testing
            - Performance testing under concurrent load
            - Security testing for financial services systems
            - End-to-end integration testing across complex agent networks

            You work closely with the reasoning engine auditor to validate cognitive accuracy
            and coordinate with all specialist agents to ensure their outputs meet quality
            standards. Your testing is comprehensive, covering not just functionality but
            also compliance, security, and reliability requirements critical to estate planning.""",

            tools=[vector_tool],  # QA uses vector search for test case research
            llm=self.llm_config,
            verbose=True,
            allow_delegation=False,
            max_iter=6,
            max_execution_time=400
        )

    def create_sales_specialist_agent(self) -> CrewAIAgent:
        """Create sophisticated Sales Specialist agent with coordination capabilities"""

        return Agent(
            role="Elite Estate Planning Sales Intelligence Coordinator",
            goal="""Optimize estate planning sales processes through sophisticated analysis and
            coordination with specialist sales agents. Leverage reasoning engine insights and
            specialist intelligence to maximize conversion and client satisfaction.""",

            backstory="""You are an elite sales strategist with 20+ years in estate planning and
            wealth management. You excel at coordinating multiple specialist agents to create
            comprehensive sales intelligence and optimization strategies.

            Your expertise includes:
            - High-net-worth client psychology and relationship management
            - Complex estate planning sales process optimization
            - Multi-agent sales intelligence coordination
            - Trust and estate product positioning and pricing
            - Objection handling and relationship building strategies
            - Performance analysis and conversion optimization

            You coordinate closely with specialist agents including:
            - Trust Sales Analyst for deal analysis and conversion metrics
            - Rebuttal Library for objection handling optimization
            - Similar Case Finder for precedent-based sales strategies
            - Market Scout for competitive positioning
            - Estate Librarian for client research and background analysis

            You also leverage the reasoning engine's planner for strategic sales planning,
            auditor for sales process validation, and strategist for comprehensive sales
            strategy synthesis.""",

            tools=[vector_tool, kg_tool, extraction_tool],  # Full access for comprehensive analysis
            llm=self.llm_config,
            verbose=True,
            allow_delegation=True,
            max_iter=8,
            max_execution_time=500
        )

# ============================================================================
# Sophisticated Workflow Templates
# ============================================================================

class BMADWorkflowTemplates:
    """Sophisticated workflow templates integrating with existing architecture"""

    def __init__(self):
        self.agent_templates = BMADAgentTemplates()
        self.collaboration_manager = AgentCollaborationManager()

    def create_cognitive_estate_analysis_workflow(self, query: str) -> Crew:
        """Create workflow integrating reasoning engine with BMAD agents"""

        # Create agents
        business_analyst = self.agent_templates.create_business_analyst_agent()
        sales_specialist = self.agent_templates.create_sales_specialist_agent()

        # Configure collaboration
        collaboration_config = CollaborationConfig(
            mode=CollaborationMode.REASONING_ENHANCED,
            handoff_protocol=HandoffProtocol.VALIDATED,
            quality_threshold=0.90
        )

        # Phase 1: Requirement Analysis with Reasoning Integration
        requirement_task = self.collaboration_manager.create_collaboration_task(
            description=f"""Analyze the estate planning query: '{query}'

            REASONING ENGINE INTEGRATION:
            1. Work with gatekeeper component to validate query quality and scope
            2. Collaborate with planner component to develop comprehensive analysis strategy
            3. Coordinate with auditor component for requirement validation

            SPECIALIST COORDINATION:
            - Estate Librarian: Research relevant documentation and precedents
            - Trust Sales Analyst: Assess business and sales implications
            - Similar Case Finder: Identify relevant precedents and case studies

            Extract comprehensive business requirements including:
            - Functional requirements for estate planning solution
            - Compliance and regulatory requirements
            - Client relationship and family dynamics considerations
            - Technical requirements for system implementation
            - Success criteria and acceptance criteria""",

            expected_output="""Comprehensive requirements document including:
            1. Validated and structured business requirements
            2. Compliance and regulatory analysis
            3. Technical specifications
            4. Success criteria and quality gates
            5. Integration requirements with reasoning engine and specialists""",

            agent=business_analyst,
            collaboration_config=collaboration_config
        )

        # Phase 2: Sales Intelligence Coordination
        sales_intelligence_task = self.collaboration_manager.create_collaboration_task(
            description=f"""Develop comprehensive sales intelligence for: '{query}'

            SPECIALIST AGENT COORDINATION:
            1. Trust Sales Analyst: Analyze conversion patterns and sales metrics
            2. Rebuttal Library: Identify potential objections and response strategies
            3. Market Scout: Gather competitive intelligence and market positioning
            4. Similar Case Finder: Research successful precedents and strategies

            REASONING ENGINE INTEGRATION:
            - Strategic planning with planner component
            - Quality validation with auditor component
            - Strategic synthesis with strategist component

            Develop comprehensive sales strategy including:
            - Client engagement and relationship building approach
            - Product positioning and value proposition
            - Objection handling and response strategies
            - Competitive differentiation and positioning
            - Conversion optimization recommendations""",

            expected_output="""Comprehensive sales intelligence report including:
            1. Detailed sales strategy and approach
            2. Client engagement and relationship strategies
            3. Objection handling playbook
            4. Competitive positioning analysis
            5. Conversion optimization recommendations
            6. Integration with specialist agent insights""",

            agent=sales_specialist,
            collaboration_config=collaboration_config,
            context_tasks=[requirement_task]
        )

        # Create and return crew
        return Crew(
            agents=[business_analyst, sales_specialist],
            tasks=[requirement_task, sales_intelligence_task],
            process=Process.sequential,
            verbose=True,
            memory=True
        )

    def create_system_implementation_workflow(self, requirements: str) -> Crew:
        """Create full implementation workflow with all BMAD agents"""

        # Create all agents
        solution_architect = self.agent_templates.create_solution_architect_agent()
        developer = self.agent_templates.create_developer_agent()
        qa_tester = self.agent_templates.create_qa_tester_agent()
        project_manager = self.agent_templates.create_project_manager_agent()

        collaboration_config = CollaborationConfig(
            mode=CollaborationMode.HYBRID,
            handoff_protocol=HandoffProtocol.VALIDATED
        )

        # Phase 1: Architecture Design
        architecture_task = self.collaboration_manager.create_collaboration_task(
            description=f"""Design comprehensive technical architecture for: {requirements}

            INTEGRATION REQUIREMENTS:
            - Seamless integration with reasoning engine components
            - Coordination interfaces for specialist agents
            - Real-time collaboration and communication patterns
            - Performance optimization for concurrent operations

            ARCHITECTURE CONSIDERATIONS:
            - Multi-agent communication protocols
            - Data consistency and transaction management
            - Security and compliance for estate planning data
            - Scalability for high-volume operations
            - Error handling and recovery mechanisms""",

            expected_output="""Detailed technical architecture including:
            1. System component design and interfaces
            2. Multi-agent integration patterns
            3. Data flow and communication protocols
            4. Security and compliance architecture
            5. Performance and scalability specifications""",

            agent=solution_architect,
            collaboration_config=collaboration_config
        )

        # Phase 2: Implementation
        implementation_task = self.collaboration_manager.create_collaboration_task(
            description="""Implement the designed architecture with focus on:

            INTEGRATION IMPLEMENTATION:
            - Asynchronous communication with reasoning engine
            - Coordination protocols for specialist agents
            - Real-time data synchronization and consistency
            - Performance monitoring and optimization

            IMPLEMENTATION REQUIREMENTS:
            - Production-ready code with comprehensive error handling
            - Estate planning compliance and security measures
            - Comprehensive logging and monitoring
            - Integration testing and validation""",

            expected_output="""Production-ready implementation including:
            1. Complete codebase with integration layers
            2. Configuration and deployment scripts
            3. Comprehensive documentation
            4. Integration testing suite
            5. Performance monitoring implementation""",

            agent=developer,
            collaboration_config=collaboration_config,
            context_tasks=[architecture_task]
        )

        # Phase 3: Quality Assurance
        qa_task = self.collaboration_manager.create_collaboration_task(
            description="""Comprehensive testing and validation including:

            MULTI-AGENT TESTING:
            - Reasoning engine integration accuracy
            - Specialist agent coordination reliability
            - End-to-end workflow validation
            - Performance under concurrent load

            COMPLIANCE TESTING:
            - Estate planning regulatory compliance
            - Security and data protection validation
            - Fiduciary responsibility adherence
            - Client confidentiality protection""",

            expected_output="""Comprehensive test report including:
            1. Functional testing results and coverage
            2. Integration testing validation
            3. Performance and scalability analysis
            4. Compliance and security assessment
            5. Quality recommendations and improvements""",

            agent=qa_tester,
            collaboration_config=collaboration_config,
            context_tasks=[implementation_task]
        )

        # Phase 4: Project Coordination
        coordination_task = self.collaboration_manager.create_collaboration_task(
            description="""Coordinate final delivery and deployment including:

            ORCHESTRATION RESPONSIBILITIES:
            - Validate all deliverables meet requirements
            - Coordinate deployment and go-live activities
            - Ensure stakeholder communication and training
            - Establish monitoring and maintenance procedures""",

            expected_output="""Project completion report including:
            1. Deliverable validation and sign-off
            2. Deployment plan and procedures
            3. Training and documentation package
            4. Monitoring and maintenance framework
            5. Success metrics and performance baselines""",

            agent=project_manager,
            collaboration_config=collaboration_config,
            context_tasks=[qa_task]
        )

        return Crew(
            agents=[solution_architect, developer, qa_tester, project_manager],
            tasks=[architecture_task, implementation_task, qa_task, coordination_task],
            process=Process.sequential,
            verbose=True,
            memory=True
        )

# ============================================================================
# Integration Helper Functions
# ============================================================================

def create_bmad_agent_registry() -> Dict[str, CrewAIAgent]:
    """Create registry of all BMAD agents for easy access"""

    templates = BMADAgentTemplates()

    return {
        "business_analyst": templates.create_business_analyst_agent(),
        "project_manager": templates.create_project_manager_agent(),
        "solution_architect": templates.create_solution_architect_agent(),
        "developer": templates.create_developer_agent(),
        "qa_tester": templates.create_qa_tester_agent(),
        "sales_specialist": templates.create_sales_specialist_agent()
    }

def execute_bmad_workflow(workflow_type: str, query: str) -> str:
    """Execute BMAD workflow with sophisticated integration"""

    templates = BMADWorkflowTemplates()

    if workflow_type == "cognitive_analysis":
        crew = templates.create_cognitive_estate_analysis_workflow(query)
    elif workflow_type == "system_implementation":
        crew = templates.create_system_implementation_workflow(query)
    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")

    print(f"\n{'='*80}")
    print(f"BMAD Sophisticated Workflow Execution")
    print(f"Type: {workflow_type}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")

    result = crew.kickoff()
    return str(result)

# ============================================================================
# Example Usage and Testing
# ============================================================================

def main():
    """Demonstrate sophisticated BMAD agent templates"""

    print("\n" + "="*80)
    print("BMAD Sophisticated Agent Templates for Stellar Connect")
    print("="*80)

    # Create agent registry
    agents = create_bmad_agent_registry()

    print(f"\nCreated {len(agents)} sophisticated BMAD agents:")
    for role, agent in agents.items():
        print(f"- {role}: {agent.role}")

    # Example workflow execution
    print("\nExample Workflow Configuration:")
    print("-" * 40)

    templates = BMADWorkflowTemplates()

    # Note: Uncomment to execute
    # query = "How can we optimize trust sales for high-net-worth clients with complex family dynamics?"
    # result = execute_bmad_workflow("cognitive_analysis", query)
    # print(f"Result preview: {result[:200]}...")

    print("\n" + "="*80)
    print("Sophisticated BMAD templates ready for production deployment!")
    print("="*80)

if __name__ == "__main__":
    main()