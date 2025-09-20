"""
Estate Planner Agent for Stellar Connect
Phase 2 Week 3: Cognitive Pipeline Components

The Estate Planner creates comprehensive multi-step analysis plans for complex estate planning
queries. It orchestrates specialist agents and ensures systematic, thorough analysis of all
relevant aspects of estate planning scenarios.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from .gatekeeper import QueryValidation, QueryType


class AnalysisType(Enum):
    """Types of analysis that can be performed."""
    DOCUMENT_RESEARCH = "document_research"
    MARKET_INTELLIGENCE = "market_intelligence"
    SALES_OPTIMIZATION = "sales_optimization"
    CONVERSION_ANALYSIS = "conversion_analysis"
    TAX_ANALYSIS = "tax_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    COMPLIANCE_CHECK = "compliance_check"
    SCENARIO_MODELING = "scenario_modeling"


class TaskPriority(Enum):
    """Priority levels for analysis tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(Enum):
    """Status of analysis tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AnalysisTask:
    """Represents a single analysis task in the plan."""
    task_id: str
    analysis_type: AnalysisType
    assigned_specialist: str
    description: str
    priority: TaskPriority
    estimated_duration: int  # in seconds
    dependencies: List[str] = field(default_factory=list)
    input_requirements: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class AnalysisPlan:
    """Comprehensive analysis plan for an estate planning query."""
    plan_id: str
    query_type: QueryType
    total_tasks: int
    estimated_duration: int  # in seconds
    tasks: List[AnalysisTask]
    execution_order: List[str]  # Task IDs in execution order
    parallel_groups: List[List[str]] = field(default_factory=list)  # Tasks that can run in parallel
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    fallback_strategies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    execution_trace: List[str] = field(default_factory=list)


class EstatePlanner:
    """
    Estate Planner Agent for creating comprehensive analysis plans.

    Responsibilities:
    - Analyze validated queries to determine required analysis steps
    - Create optimized execution plans with proper task sequencing
    - Assign appropriate specialist agents to each analysis task
    - Handle dependencies and parallel execution opportunities
    - Provide fallback strategies for failed analyses
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Specialist capabilities mapping
        self.specialist_capabilities = {
            "estate_librarian": [
                AnalysisType.DOCUMENT_RESEARCH,
                AnalysisType.COMPLIANCE_CHECK
            ],
            "trust_sales_analyst": [
                AnalysisType.CONVERSION_ANALYSIS,
                AnalysisType.SALES_OPTIMIZATION
            ],
            "market_scout": [
                AnalysisType.MARKET_INTELLIGENCE,
                AnalysisType.SCENARIO_MODELING
            ],
            "sales_specialist": [
                AnalysisType.SALES_OPTIMIZATION,
                AnalysisType.CONVERSION_ANALYSIS
            ]
        }

        # Query type to analysis mapping
        self.query_analysis_mapping = {
            QueryType.TRUST_STRUCTURE: [
                AnalysisType.DOCUMENT_RESEARCH,
                AnalysisType.TAX_ANALYSIS,
                AnalysisType.COMPLIANCE_CHECK,
                AnalysisType.SCENARIO_MODELING
            ],
            QueryType.TAX_OPTIMIZATION: [
                AnalysisType.TAX_ANALYSIS,
                AnalysisType.DOCUMENT_RESEARCH,
                AnalysisType.SCENARIO_MODELING,
                AnalysisType.COMPLIANCE_CHECK
            ],
            QueryType.ASSET_PROTECTION: [
                AnalysisType.RISK_ASSESSMENT,
                AnalysisType.DOCUMENT_RESEARCH,
                AnalysisType.COMPLIANCE_CHECK,
                AnalysisType.SCENARIO_MODELING
            ],
            QueryType.SUCCESSION_PLANNING: [
                AnalysisType.DOCUMENT_RESEARCH,
                AnalysisType.TAX_ANALYSIS,
                AnalysisType.SCENARIO_MODELING,
                AnalysisType.MARKET_INTELLIGENCE
            ],
            QueryType.CHARITABLE_GIVING: [
                AnalysisType.TAX_ANALYSIS,
                AnalysisType.COMPLIANCE_CHECK,
                AnalysisType.SCENARIO_MODELING
            ],
            QueryType.BUSINESS_TRANSITION: [
                AnalysisType.MARKET_INTELLIGENCE,
                AnalysisType.TAX_ANALYSIS,
                AnalysisType.RISK_ASSESSMENT,
                AnalysisType.SCENARIO_MODELING
            ],
            QueryType.GENERAL_PLANNING: [
                AnalysisType.DOCUMENT_RESEARCH,
                AnalysisType.SCENARIO_MODELING
            ]
        }

        # Task dependencies - which analyses should come before others
        self.task_dependencies = {
            AnalysisType.SCENARIO_MODELING: [
                AnalysisType.DOCUMENT_RESEARCH,
                AnalysisType.TAX_ANALYSIS
            ],
            AnalysisType.COMPLIANCE_CHECK: [
                AnalysisType.DOCUMENT_RESEARCH
            ],
            AnalysisType.SALES_OPTIMIZATION: [
                AnalysisType.MARKET_INTELLIGENCE,
                AnalysisType.CONVERSION_ANALYSIS
            ]
        }

    async def create_plan(self, validation: QueryValidation) -> AnalysisPlan:
        """
        Create a comprehensive analysis plan based on validated query.

        Args:
            validation: QueryValidation result from the gatekeeper

        Returns:
            AnalysisPlan with optimized task sequence and assignments
        """
        self.logger.info(f"Creating analysis plan for {validation.query_type.value} query")

        plan_id = str(uuid.uuid4())

        # Step 1: Determine required analyses
        required_analyses = self._determine_required_analyses(validation)

        # Step 2: Create analysis tasks
        tasks = await self._create_analysis_tasks(required_analyses, validation)

        # Step 3: Optimize execution order
        execution_order, parallel_groups = self._optimize_execution_order(tasks)

        # Step 4: Calculate total duration
        total_duration = self._calculate_total_duration(tasks, parallel_groups)

        # Step 5: Define success criteria
        success_criteria = self._define_success_criteria(validation.query_type)

        # Step 6: Create fallback strategies
        fallback_strategies = self._create_fallback_strategies(required_analyses)

        plan = AnalysisPlan(
            plan_id=plan_id,
            query_type=validation.query_type,
            total_tasks=len(tasks),
            estimated_duration=total_duration,
            tasks=tasks,
            execution_order=execution_order,
            parallel_groups=parallel_groups,
            success_criteria=success_criteria,
            fallback_strategies=fallback_strategies
        )

        self.logger.info(f"Created plan {plan_id} with {len(tasks)} tasks, "
                        f"estimated duration: {total_duration}s")

        return plan

    def _determine_required_analyses(self, validation: QueryValidation) -> List[AnalysisType]:
        """Determine which types of analysis are required for the query."""
        base_analyses = self.query_analysis_mapping.get(validation.query_type, [])
        required_analyses = base_analyses.copy()

        # Add sales-focused analyses if this seems like a sales-related query
        if 'sales' in validation.validated_query.lower() or 'client' in validation.validated_query.lower():
            if AnalysisType.SALES_OPTIMIZATION not in required_analyses:
                required_analyses.append(AnalysisType.SALES_OPTIMIZATION)
            if AnalysisType.CONVERSION_ANALYSIS not in required_analyses:
                required_analyses.append(AnalysisType.CONVERSION_ANALYSIS)

        # Add market intelligence for high-value scenarios
        if 'monetary_amounts' in validation.extracted_entities:
            amounts = validation.extracted_entities['monetary_amounts']
            # Check for high-value indicators
            for amount in amounts:
                if 'million' in amount.lower() or 'billion' in amount.lower():
                    if AnalysisType.MARKET_INTELLIGENCE not in required_analyses:
                        required_analyses.append(AnalysisType.MARKET_INTELLIGENCE)
                    break

        return required_analyses

    async def _create_analysis_tasks(self, analyses: List[AnalysisType],
                                   validation: QueryValidation) -> List[AnalysisTask]:
        """Create detailed analysis tasks for each required analysis type."""
        tasks = []

        for analysis_type in analyses:
            # Find best specialist for this analysis
            assigned_specialist = self._assign_specialist(analysis_type)

            # Determine task priority
            priority = self._determine_task_priority(analysis_type, validation.query_type)

            # Create task
            task = AnalysisTask(
                task_id=str(uuid.uuid4()),
                analysis_type=analysis_type,
                assigned_specialist=assigned_specialist,
                description=self._generate_task_description(analysis_type, validation),
                priority=priority,
                estimated_duration=self._estimate_task_duration(analysis_type),
                input_requirements=self._define_input_requirements(analysis_type),
                expected_outputs=self._define_expected_outputs(analysis_type)
            )

            # Add dependencies
            if analysis_type in self.task_dependencies:
                for dep_analysis in self.task_dependencies[analysis_type]:
                    # Find the task ID for the dependency
                    for existing_task in tasks:
                        if existing_task.analysis_type == dep_analysis:
                            task.dependencies.append(existing_task.task_id)
                            break

            tasks.append(task)

        return tasks

    def _assign_specialist(self, analysis_type: AnalysisType) -> str:
        """Assign the most appropriate specialist for an analysis type."""
        # Find specialists capable of this analysis
        capable_specialists = []
        for specialist, capabilities in self.specialist_capabilities.items():
            if analysis_type in capabilities:
                capable_specialists.append(specialist)

        if not capable_specialists:
            # Default assignment for unsupported analysis types
            return "estate_librarian"  # Most general specialist

        # For now, return the first capable specialist
        # In a more sophisticated system, this could consider workload, performance, etc.
        return capable_specialists[0]

    def _determine_task_priority(self, analysis_type: AnalysisType, query_type: QueryType) -> TaskPriority:
        """Determine the priority of a task based on analysis and query type."""
        # Critical priorities
        if analysis_type == AnalysisType.COMPLIANCE_CHECK:
            return TaskPriority.CRITICAL

        # High priorities for tax-related queries
        if query_type == QueryType.TAX_OPTIMIZATION and analysis_type == AnalysisType.TAX_ANALYSIS:
            return TaskPriority.HIGH

        # High priorities for trust structure queries
        if query_type == QueryType.TRUST_STRUCTURE and analysis_type == AnalysisType.DOCUMENT_RESEARCH:
            return TaskPriority.HIGH

        # Medium priority for scenario modeling (usually dependent on other analyses)
        if analysis_type == AnalysisType.SCENARIO_MODELING:
            return TaskPriority.MEDIUM

        # Default to medium priority
        return TaskPriority.MEDIUM

    def _generate_task_description(self, analysis_type: AnalysisType,
                                 validation: QueryValidation) -> str:
        """Generate a descriptive task description."""
        descriptions = {
            AnalysisType.DOCUMENT_RESEARCH: f"Research relevant estate planning documents and precedents for {validation.query_type.value}",
            AnalysisType.MARKET_INTELLIGENCE: "Gather current market trends and competitive intelligence",
            AnalysisType.SALES_OPTIMIZATION: "Analyze sales opportunities and optimization strategies",
            AnalysisType.CONVERSION_ANALYSIS: "Evaluate conversion potential and improvement recommendations",
            AnalysisType.TAX_ANALYSIS: "Perform comprehensive tax impact analysis",
            AnalysisType.RISK_ASSESSMENT: "Assess potential risks and mitigation strategies",
            AnalysisType.COMPLIANCE_CHECK: "Verify compliance with current regulations and requirements",
            AnalysisType.SCENARIO_MODELING: "Model various scenarios and their outcomes"
        }

        return descriptions.get(analysis_type, f"Perform {analysis_type.value} analysis")

    def _estimate_task_duration(self, analysis_type: AnalysisType) -> int:
        """Estimate task duration in seconds."""
        durations = {
            AnalysisType.DOCUMENT_RESEARCH: 300,  # 5 minutes
            AnalysisType.MARKET_INTELLIGENCE: 240,  # 4 minutes
            AnalysisType.SALES_OPTIMIZATION: 180,  # 3 minutes
            AnalysisType.CONVERSION_ANALYSIS: 180,  # 3 minutes
            AnalysisType.TAX_ANALYSIS: 360,  # 6 minutes
            AnalysisType.RISK_ASSESSMENT: 300,  # 5 minutes
            AnalysisType.COMPLIANCE_CHECK: 240,  # 4 minutes
            AnalysisType.SCENARIO_MODELING: 420  # 7 minutes
        }

        return durations.get(analysis_type, 240)  # Default 4 minutes

    def _define_input_requirements(self, analysis_type: AnalysisType) -> List[str]:
        """Define input requirements for each analysis type."""
        requirements = {
            AnalysisType.DOCUMENT_RESEARCH: ["query_entities", "query_type"],
            AnalysisType.MARKET_INTELLIGENCE: ["query_entities", "asset_values"],
            AnalysisType.SALES_OPTIMIZATION: ["client_profile", "conversation_context"],
            AnalysisType.CONVERSION_ANALYSIS: ["client_interactions", "sales_stage"],
            AnalysisType.TAX_ANALYSIS: ["asset_values", "jurisdiction", "entity_structure"],
            AnalysisType.RISK_ASSESSMENT: ["asset_details", "family_structure"],
            AnalysisType.COMPLIANCE_CHECK: ["proposed_structure", "jurisdiction"],
            AnalysisType.SCENARIO_MODELING: ["baseline_analysis", "variables"]
        }

        return requirements.get(analysis_type, ["query_context"])

    def _define_expected_outputs(self, analysis_type: AnalysisType) -> List[str]:
        """Define expected outputs for each analysis type."""
        outputs = {
            AnalysisType.DOCUMENT_RESEARCH: ["relevant_documents", "precedent_cases", "legal_references"],
            AnalysisType.MARKET_INTELLIGENCE: ["market_trends", "competitive_analysis", "opportunities"],
            AnalysisType.SALES_OPTIMIZATION: ["optimization_recommendations", "next_best_actions"],
            AnalysisType.CONVERSION_ANALYSIS: ["conversion_metrics", "improvement_areas"],
            AnalysisType.TAX_ANALYSIS: ["tax_implications", "optimization_strategies", "compliance_requirements"],
            AnalysisType.RISK_ASSESSMENT: ["risk_factors", "mitigation_strategies", "risk_scores"],
            AnalysisType.COMPLIANCE_CHECK: ["compliance_status", "required_actions", "regulatory_updates"],
            AnalysisType.SCENARIO_MODELING: ["scenario_outcomes", "probability_analysis", "recommendations"]
        }

        return outputs.get(analysis_type, ["analysis_results"])

    def _optimize_execution_order(self, tasks: List[AnalysisTask]) -> Tuple[List[str], List[List[str]]]:
        """Optimize the execution order of tasks considering dependencies."""
        # Create dependency graph
        task_map = {task.task_id: task for task in tasks}

        # Topological sort for dependency resolution
        visited = set()
        temp_visited = set()
        execution_order = []

        def visit(task_id: str):
            if task_id in temp_visited:
                # Circular dependency detected - skip for now
                return
            if task_id in visited:
                return

            temp_visited.add(task_id)

            # Visit dependencies first
            task = task_map[task_id]
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    visit(dep_id)

            temp_visited.remove(task_id)
            visited.add(task_id)
            execution_order.append(task_id)

        # Visit all tasks
        for task in tasks:
            if task.task_id not in visited:
                visit(task.task_id)

        # Identify parallel execution opportunities
        parallel_groups = self._identify_parallel_groups(tasks, execution_order)

        return execution_order, parallel_groups

    def _identify_parallel_groups(self, tasks: List[AnalysisTask],
                                execution_order: List[str]) -> List[List[str]]:
        """Identify groups of tasks that can be executed in parallel."""
        task_map = {task.task_id: task for task in tasks}
        parallel_groups = []
        current_group = []

        for task_id in execution_order:
            task = task_map[task_id]

            # Check if this task can run in parallel with current group
            can_parallel = True
            for group_task_id in current_group:
                group_task = task_map[group_task_id]

                # Tasks can't run in parallel if one depends on the other
                if (task_id in group_task.dependencies or
                    group_task_id in task.dependencies):
                    can_parallel = False
                    break

            if can_parallel and current_group:
                current_group.append(task_id)
            else:
                if current_group:
                    parallel_groups.append(current_group)
                current_group = [task_id]

        # Add the last group
        if current_group:
            parallel_groups.append(current_group)

        return parallel_groups

    def _calculate_total_duration(self, tasks: List[AnalysisTask],
                                parallel_groups: List[List[str]]) -> int:
        """Calculate total estimated duration considering parallel execution."""
        task_map = {task.task_id: task for task in tasks}
        total_duration = 0

        for group in parallel_groups:
            # For parallel groups, take the maximum duration
            group_duration = max(task_map[task_id].estimated_duration for task_id in group)
            total_duration += group_duration

        return total_duration

    def _define_success_criteria(self, query_type: QueryType) -> Dict[str, Any]:
        """Define success criteria for the analysis plan."""
        base_criteria = {
            "minimum_completed_tasks": 0.8,  # 80% of tasks must complete
            "required_critical_tasks": True,  # All critical priority tasks must complete
            "maximum_duration_multiplier": 2.0  # Don't exceed 2x estimated duration
        }

        # Add query-specific criteria
        query_specific = {
            QueryType.TAX_OPTIMIZATION: {
                "required_analyses": [AnalysisType.TAX_ANALYSIS],
                "minimum_confidence_score": 0.85
            },
            QueryType.TRUST_STRUCTURE: {
                "required_analyses": [AnalysisType.DOCUMENT_RESEARCH, AnalysisType.COMPLIANCE_CHECK],
                "minimum_confidence_score": 0.80
            },
            QueryType.ASSET_PROTECTION: {
                "required_analyses": [AnalysisType.RISK_ASSESSMENT],
                "minimum_confidence_score": 0.75
            }
        }

        criteria = base_criteria.copy()
        criteria.update(query_specific.get(query_type, {}))
        return criteria

    def _create_fallback_strategies(self, analyses: List[AnalysisType]) -> List[str]:
        """Create fallback strategies for failed analyses."""
        strategies = [
            "Retry failed tasks with simplified parameters",
            "Use alternative specialists for critical analyses",
            "Proceed with partial results if minimum requirements are met",
            "Generate recommendations based on available analysis results"
        ]

        # Add analysis-specific fallbacks
        if AnalysisType.TAX_ANALYSIS in analyses:
            strategies.append("Use general tax guidelines if detailed analysis fails")

        if AnalysisType.SCENARIO_MODELING in analyses:
            strategies.append("Provide single-scenario analysis if multi-scenario modeling fails")

        return strategies

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the planner."""
        return {
            "status": "healthy",
            "component": "estate_planner",
            "specialist_count": len(self.specialist_capabilities),
            "query_type_mappings": len(self.query_analysis_mapping),
            "analysis_types": len(AnalysisType),
            "last_check": datetime.now().isoformat()
        }


# Example usage and testing
async def main():
    """Example usage of Estate Planner."""
    from .gatekeeper import EstateGatekeeper, QueryValidation, QueryType

    planner = EstatePlanner()

    # Mock validation result
    validation = QueryValidation(
        is_valid=True,
        query_type=QueryType.TRUST_STRUCTURE,
        confidence_score=0.85,
        validated_query="I want to create a trust for my $2 million estate to benefit my wife and two children",
        extracted_entities={
            'monetary_amounts': ['$2 million'],
            'family_members': ['wife', 'children'],
            'legal_entities': ['trust']
        }
    )

    print("Creating analysis plan...")
    plan = await planner.create_plan(validation)

    print(f"\nPlan ID: {plan.plan_id}")
    print(f"Query Type: {plan.query_type.value}")
    print(f"Total Tasks: {plan.total_tasks}")
    print(f"Estimated Duration: {plan.estimated_duration}s")

    print("\nTasks:")
    for task in plan.tasks:
        print(f"  - {task.analysis_type.value}: {task.assigned_specialist} "
              f"({task.priority.value}, {task.estimated_duration}s)")
        if task.dependencies:
            print(f"    Dependencies: {len(task.dependencies)} tasks")

    print(f"\nParallel Groups: {len(plan.parallel_groups)}")
    for i, group in enumerate(plan.parallel_groups):
        print(f"  Group {i+1}: {len(group)} tasks")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())