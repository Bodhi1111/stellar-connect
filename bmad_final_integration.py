#!/usr/bin/env python3
"""
Final BMAD Integration Module for Stellar Connect
================================================
Complete integration with existing sophisticated agent architecture
using correct class names and production-ready implementations.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from src.config import CONFIG, init_settings
from src.agent_tools import vector_tool, kg_tool, extraction_tool

# Initialize settings
init_settings()

# ============================================================================
# BMAD Agent Factory (Production Ready)
# ============================================================================

class ProductionBMADAgents:
    """Production-ready BMAD agents with proper integration"""

    def __init__(self):
        self.llm_config = f"ollama/{CONFIG.GENERATIVE_MODEL}"

    def create_business_analyst(self) -> Agent:
        """Business Analyst with estate planning expertise"""
        return Agent(
            role="Estate Planning Business Analyst",
            goal="Extract and validate comprehensive business requirements for estate planning solutions with sophisticated analysis and specialist coordination",
            backstory="""You are a senior business analyst specializing in estate planning and wealth management
            with 15+ years of experience. You excel at extracting complex requirements from client conversations
            and coordinating with specialist agents for comprehensive analysis. You understand trust law,
            tax implications, family dynamics, and fiduciary responsibilities.""",
            tools=[vector_tool, extraction_tool],
            llm=self.llm_config,
            verbose=True,
            allow_delegation=True,
            max_iter=6
        )

    def create_project_manager(self) -> Agent:
        """Project Manager with multi-agent orchestration capabilities"""
        return Agent(
            role="Estate Planning Project Manager",
            goal="Orchestrate complex multi-agent workflows and coordinate between BMAD agents and specialist teams for optimal estate planning solution delivery",
            backstory="""You are an experienced project manager specializing in estate planning technology
            and multi-agent system coordination. You excel at managing complex workflows involving multiple
            specialist agents, ensuring quality deliverables, and optimizing team performance. You understand
            the intricacies of estate planning projects and regulatory requirements.""",
            tools=[],
            llm=self.llm_config,
            verbose=True,
            allow_delegation=True,
            max_iter=8
        )

    def create_solution_architect(self) -> Agent:
        """Solution Architect with estate planning system expertise"""
        return Agent(
            role="Estate Planning Solution Architect",
            goal="Design scalable, secure, and compliant estate planning technology solutions with sophisticated agent integration and knowledge system optimization",
            backstory="""You are a solution architect with deep expertise in estate planning technology,
            multi-agent systems, and financial services architecture. You design solutions that integrate
            seamlessly with specialist agents and knowledge systems. You understand complex trust structures,
            regulatory compliance, and performance optimization for high-stakes financial planning.""",
            tools=[kg_tool],
            llm=self.llm_config,
            verbose=True,
            allow_delegation=False,
            max_iter=5
        )

    def create_developer(self) -> Agent:
        """Developer with estate planning domain knowledge"""
        return Agent(
            role="Estate Planning Solutions Developer",
            goal="Implement sophisticated estate planning features with seamless agent integration, ensuring reliability, security, and compliance with industry standards",
            backstory="""You are a senior developer specializing in estate planning technology and
            multi-agent system integration. You build production-ready solutions that integrate with
            specialist agents and handle the complexities of estate planning data and workflows.
            You understand compliance requirements, security needs, and performance optimization.""",
            tools=[extraction_tool],
            llm=self.llm_config,
            verbose=True,
            allow_delegation=False,
            max_iter=7
        )

    def create_qa_tester(self) -> Agent:
        """QA Tester with compliance and reliability focus"""
        return Agent(
            role="Estate Planning QA Specialist",
            goal="Ensure quality, compliance, and reliability of estate planning solutions through comprehensive testing of agent workflows and system integration",
            backstory="""You are a QA specialist with expertise in estate planning compliance,
            multi-agent system testing, and financial services quality assurance. You understand
            the critical importance of accuracy in estate planning and test for functionality,
            compliance, security, and reliability across complex agent networks.""",
            tools=[vector_tool],
            llm=self.llm_config,
            verbose=True,
            allow_delegation=False,
            max_iter=5
        )

    def create_sales_specialist(self) -> Agent:
        """Sales Specialist with estate planning domain expertise"""
        return Agent(
            role="Elite Estate Planning Sales Strategist",
            goal="Optimize estate planning sales processes through sophisticated analysis, specialist coordination, and comprehensive sales intelligence for high-net-worth clients",
            backstory="""You are an elite sales strategist with 20+ years in estate planning and
            wealth management. You excel at analyzing complex sales patterns, optimizing conversion
            strategies, and coordinating with specialist agents for comprehensive sales intelligence.
            You understand high-net-worth client psychology, complex family dynamics, and sophisticated
            estate planning product positioning.""",
            tools=[vector_tool, kg_tool, extraction_tool],
            llm=self.llm_config,
            verbose=True,
            allow_delegation=True,
            max_iter=8
        )

# ============================================================================
# Workflow Orchestrator
# ============================================================================

class BMADWorkflowOrchestrator:
    """Orchestrates BMAD workflows with existing architecture"""

    def __init__(self):
        self.agent_factory = ProductionBMADAgents()
        self.execution_history: List[Dict] = []

    def create_sales_optimization_workflow(self, query: str) -> Crew:
        """Create comprehensive sales optimization workflow"""

        # Create agents
        sales_specialist = self.agent_factory.create_sales_specialist()
        business_analyst = self.agent_factory.create_business_analyst()
        project_manager = self.agent_factory.create_project_manager()

        # Define tasks
        sales_analysis_task = Task(
            description=f"""Analyze estate planning sales optimization for: {query}

            COMPREHENSIVE ANALYSIS REQUIREMENTS:
            1. Analyze current sales patterns and conversion metrics
            2. Identify bottlenecks and optimization opportunities
            3. Research successful strategies from similar cases
            4. Coordinate with available specialist agents for insights
            5. Develop specific, actionable recommendations

            INTEGRATION CONSIDERATIONS:
            - Leverage vector search for pattern identification
            - Use knowledge graph for relationship analysis
            - Extract structured insights for implementation

            Focus on high-net-worth client needs, complex family dynamics, and sophisticated trust structures.""",

            expected_output="""Comprehensive sales optimization analysis including:
            1. Current performance analysis and gap identification
            2. Specific optimization opportunities with impact assessment
            3. Strategic recommendations for conversion improvement
            4. Implementation roadmap with priority ranking
            5. Success metrics and measurement framework""",

            agent=sales_specialist
        )

        requirements_task = Task(
            description=f"""Extract and validate business requirements for sales optimization: {query}

            REQUIREMENT ANALYSIS:
            1. Validate sales optimization requirements against business objectives
            2. Define success criteria and acceptance criteria
            3. Identify compliance and regulatory considerations
            4. Document implementation requirements and constraints
            5. Ensure alignment with estate planning best practices

            VALIDATION REQUIREMENTS:
            - Ensure requirements are specific, measurable, and achievable
            - Validate against fiduciary responsibility standards
            - Confirm compliance with regulatory requirements
            - Align with client relationship management best practices""",

            expected_output="""Validated requirements document including:
            1. Functional and non-functional requirements
            2. Success criteria and quality gates
            3. Compliance and regulatory requirements
            4. Implementation constraints and dependencies
            5. Risk assessment and mitigation strategies""",

            agent=business_analyst,
            context=[sales_analysis_task]
        )

        coordination_task = Task(
            description="""Coordinate implementation planning and resource allocation for sales optimization:

            PROJECT COORDINATION:
            1. Review sales analysis and requirements validation
            2. Develop implementation timeline and resource plan
            3. Identify dependencies and risk mitigation strategies
            4. Create monitoring and measurement framework
            5. Define stakeholder communication plan

            ORCHESTRATION REQUIREMENTS:
            - Coordinate between different teams and stakeholders
            - Ensure optimal resource allocation and timeline management
            - Establish quality gates and validation checkpoints
            - Plan for change management and training requirements""",

            expected_output="""Implementation coordination plan including:
            1. Detailed project timeline with milestones
            2. Resource allocation and responsibility matrix
            3. Risk management and mitigation strategies
            4. Quality assurance and validation framework
            5. Communication and change management plan""",

            agent=project_manager,
            context=[sales_analysis_task, requirements_task]
        )

        return Crew(
            agents=[sales_specialist, business_analyst, project_manager],
            tasks=[sales_analysis_task, requirements_task, coordination_task],
            process=Process.sequential,
            verbose=True,
            memory=True
        )

    def create_system_implementation_workflow(self, requirements: str) -> Crew:
        """Create system implementation workflow with full BMAD team"""

        # Create agents
        solution_architect = self.agent_factory.create_solution_architect()
        developer = self.agent_factory.create_developer()
        qa_tester = self.agent_factory.create_qa_tester()

        # Architecture design task
        design_task = Task(
            description=f"""Design comprehensive technical architecture for: {requirements}

            ARCHITECTURE REQUIREMENTS:
            1. Design scalable system architecture for estate planning solutions
            2. Define integration patterns with existing specialist agents
            3. Ensure security and compliance with financial services standards
            4. Optimize for performance and reliability
            5. Plan for future scalability and enhancement

            INTEGRATION CONSIDERATIONS:
            - Seamless integration with vector and knowledge graph systems
            - Coordination interfaces for multi-agent workflows
            - Real-time communication and data synchronization
            - Security and audit trail requirements""",

            expected_output="""Technical architecture document including:
            1. System component design and interfaces
            2. Integration patterns and communication protocols
            3. Security and compliance architecture
            4. Performance and scalability specifications
            5. Deployment and operations framework""",

            agent=solution_architect
        )

        # Implementation task
        implementation_task = Task(
            description="""Implement the designed architecture with production-ready code:

            IMPLEMENTATION REQUIREMENTS:
            1. Build robust, scalable implementation following architectural design
            2. Implement comprehensive error handling and logging
            3. Ensure security and compliance with estate planning requirements
            4. Create thorough documentation and deployment guides
            5. Optimize for performance and reliability

            QUALITY STANDARDS:
            - Production-ready code with comprehensive testing
            - Security implementation for sensitive financial data
            - Compliance with estate planning and fiduciary standards
            - Performance optimization for concurrent operations""",

            expected_output="""Production-ready implementation including:
            1. Complete codebase with comprehensive documentation
            2. Security and compliance implementation
            3. Performance optimization and monitoring
            4. Deployment scripts and configuration
            5. Testing framework and validation suite""",

            agent=developer,
            context=[design_task]
        )

        # Quality assurance task
        qa_task = Task(
            description="""Comprehensive testing and quality validation:

            TESTING REQUIREMENTS:
            1. Functional testing of all implemented features
            2. Integration testing with existing systems and agents
            3. Security testing for estate planning compliance
            4. Performance testing under expected load
            5. Compliance validation against regulatory requirements

            QUALITY VALIDATION:
            - Accuracy and reliability of estate planning calculations
            - Security and data protection validation
            - Performance and scalability under load
            - Compliance with fiduciary and regulatory standards""",

            expected_output="""Comprehensive test report including:
            1. Functional testing results and coverage analysis
            2. Integration testing validation
            3. Security and compliance assessment
            4. Performance and scalability analysis
            5. Quality recommendations and improvement plan""",

            agent=qa_tester,
            context=[implementation_task]
        )

        return Crew(
            agents=[solution_architect, developer, qa_tester],
            tasks=[design_task, implementation_task, qa_task],
            process=Process.sequential,
            verbose=True,
            memory=True
        )

    def execute_workflow(self, workflow_type: str, query: str) -> str:
        """Execute specified BMAD workflow"""

        print(f"\n{'='*80}")
        print(f"BMAD Production Workflow Execution")
        print(f"Type: {workflow_type}")
        print(f"Query: {query}")
        print(f"{'='*80}\n")

        start_time = datetime.now()

        try:
            if workflow_type == "sales_optimization":
                crew = self.create_sales_optimization_workflow(query)
            elif workflow_type == "system_implementation":
                crew = self.create_system_implementation_workflow(query)
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")

            result = crew.kickoff()
            end_time = datetime.now()

            # Record execution
            self.execution_history.append({
                "workflow_type": workflow_type,
                "query": query,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "success": True
            })

            return str(result)

        except Exception as e:
            end_time = datetime.now()
            self.execution_history.append({
                "workflow_type": workflow_type,
                "query": query,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "success": False,
                "error": str(e)
            })
            raise e

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get orchestrator status and metrics"""

        successful_executions = [h for h in self.execution_history if h.get("success", False)]
        failed_executions = [h for h in self.execution_history if not h.get("success", True)]

        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(successful_executions),
            "failed_executions": len(failed_executions),
            "average_duration": sum(h["duration"] for h in successful_executions) / len(successful_executions) if successful_executions else 0,
            "available_workflows": ["sales_optimization", "system_implementation"],
            "agent_types": ["business_analyst", "project_manager", "solution_architect", "developer", "qa_tester", "sales_specialist"]
        }

# ============================================================================
# Dashboard Integration
# ============================================================================

class BMADDashboardIntegration:
    """Integration layer for Streamlit dashboard"""

    def __init__(self):
        self.orchestrator = BMADWorkflowOrchestrator()

    def process_bmad_query(self, query: str, mode: str = "sales_optimization") -> str:
        """Process query using BMAD workflow"""

        try:
            return self.orchestrator.execute_workflow(mode, query)
        except Exception as e:
            return f"Error processing BMAD query: {str(e)}"

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for dashboard display"""
        return self.orchestrator.get_orchestrator_status()

# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Demonstrate production BMAD integration"""

    print("\n" + "="*80)
    print("BMAD Production Integration for Stellar Connect")
    print("="*80)

    # Initialize integration
    integration = BMADDashboardIntegration()

    # Example query
    query = "How can we improve conversion rates for high-net-worth estate planning clients?"

    print(f"\nExample Query: {query}")
    print("-" * 60)

    # Get dashboard metrics
    metrics = integration.get_dashboard_metrics()
    print("Dashboard Metrics:")
    print(json.dumps(metrics, indent=2))

    # Note: Uncomment to execute workflow
    # print("\nExecuting BMAD workflow...")
    # result = integration.process_bmad_query(query, "sales_optimization")
    # print(f"Result preview: {result[:300]}...")

    print("\n" + "="*80)
    print("BMAD Production Integration Ready!")
    print("="*80)

if __name__ == "__main__":
    main()