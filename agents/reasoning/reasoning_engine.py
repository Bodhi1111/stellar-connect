"""
Estate Reasoning Engine for Stellar Connect
Phase 2 Week 3: Cognitive Pipeline Components

The unified reasoning engine orchestrates the complete cognitive pipeline:
Gatekeeper → Planner → Specialist Execution → Auditor → Strategist

This creates a comprehensive, self-correcting AI reasoning system for estate planning.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from .gatekeeper import EstateGatekeeper, QueryValidation, ValidationSeverity
from .planner import EstatePlanner, AnalysisPlan, AnalysisTask, TaskStatus
from .auditor import EstateAuditor, AuditResult, AuditSeverity
from .strategist import EstateStrategist, SynthesisResult


class ReasoningStatus(Enum):
    """Status of reasoning process."""
    PENDING = "pending"
    VALIDATING = "validating"
    PLANNING = "planning"
    EXECUTING = "executing"
    AUDITING = "auditing"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_CLARIFICATION = "needs_clarification"


class ExecutionMode(Enum):
    """Execution modes for the reasoning engine."""
    STANDARD = "standard"      # Full pipeline
    FAST = "fast"             # Skip some validation steps
    THOROUGH = "thorough"     # Extra validation and analysis
    DEBUG = "debug"           # Full logging and intermediate results


@dataclass
class ReasoningConfig:
    """Configuration for the reasoning engine."""
    execution_mode: ExecutionMode = ExecutionMode.STANDARD
    max_clarification_rounds: int = 3
    enable_self_correction: bool = True
    max_correction_attempts: int = 2
    parallel_execution: bool = True
    timeout_seconds: int = 300
    quality_threshold: float = 0.7
    confidence_threshold: float = 0.6


@dataclass
class ReasoningResult:
    """Complete result of the reasoning process."""
    reasoning_id: str
    status: ReasoningStatus
    query: str
    synthesis: Optional[SynthesisResult] = None
    validation: Optional[QueryValidation] = None
    plan: Optional[AnalysisPlan] = None
    audit: Optional[AuditResult] = None
    execution_results: Dict[str, Any] = field(default_factory=dict)
    clarifying_questions: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    confidence_score: float = 0.0
    reasoning_chain: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class EstateReasoningEngine:
    """
    Unified Estate Reasoning Engine orchestrating the complete cognitive pipeline.

    Architecture:
    1. Gatekeeper: Validates and preprocesses queries
    2. Planner: Creates optimized analysis plans
    3. Specialists: Execute domain-specific analyses
    4. Auditor: Quality control and self-correction
    5. Strategist: Strategic synthesis and insights

    Features:
    - Self-correcting pipeline with quality gates
    - Parallel task execution for performance
    - Comprehensive error handling and recovery
    - Multiple execution modes (standard, fast, thorough, debug)
    - Adaptive clarification and refinement
    """

    def __init__(self, config: Optional[ReasoningConfig] = None):
        self.config = config or ReasoningConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize cognitive components
        self.gatekeeper = EstateGatekeeper()
        self.planner = EstatePlanner()
        self.auditor = EstateAuditor()
        self.strategist = EstateStrategist()

        # Initialize specialist agents (placeholder - would import actual specialists)
        self.specialists = self._initialize_specialists()

        # Reasoning session tracking
        self.active_sessions: Dict[str, ReasoningResult] = {}

    def _initialize_specialists(self) -> Dict[str, Any]:
        """Initialize specialist agents."""
        try:
            from ..specialists.estate_librarian import EstateLibrarianAgent
            from ..specialists.trust_sales_analyst import TrustSalesAnalystAgent
            from ..specialists.market_scout import MarketScoutAgent
            from ..specialists.sales_specialist import SalesSpecialistAgent

            return {
                "estate_librarian": EstateLibrarianAgent(),
                "trust_sales_analyst": TrustSalesAnalystAgent() if hasattr(locals(), 'TrustSalesAnalystAgent') else None,
                "market_scout": MarketScoutAgent() if hasattr(locals(), 'MarketScoutAgent') else None,
                "sales_specialist": SalesSpecialistAgent() if hasattr(locals(), 'SalesSpecialistAgent') else None
            }
        except ImportError as e:
            self.logger.warning(f"Could not import all specialists: {e}")
            # Return estate librarian only since we know it exists
            try:
                from ..specialists.estate_librarian import EstateLibrarianAgent
                return {
                    "estate_librarian": EstateLibrarianAgent(),
                    "trust_sales_analyst": None,
                    "market_scout": None,
                    "sales_specialist": None
                }
            except ImportError:
                self.logger.error("Could not import any specialist agents")
                return {
                    "estate_librarian": None,
                    "trust_sales_analyst": None,
                    "market_scout": None,
                    "sales_specialist": None
                }

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> ReasoningResult:
        """
        Process a complete estate planning query through the reasoning pipeline.

        Args:
            query: The estate planning query to process
            context: Optional context information (client data, previous interactions, etc.)

        Returns:
            ReasoningResult with complete analysis and recommendations
        """
        reasoning_id = str(uuid.uuid4())
        start_time = datetime.now()

        result = ReasoningResult(
            reasoning_id=reasoning_id,
            status=ReasoningStatus.PENDING,
            query=query
        )

        self.active_sessions[reasoning_id] = result

        try:
            self.logger.info(f"Starting reasoning process {reasoning_id} for query: {query[:100]}...")

            # Step 1: Query Validation and Preprocessing
            result.status = ReasoningStatus.VALIDATING
            result.reasoning_chain.append("Starting query validation")

            validation = await self.gatekeeper.validate(query)
            result.validation = validation

            if not validation.is_valid:
                if validation.clarifying_questions:
                    result.status = ReasoningStatus.NEEDS_CLARIFICATION
                    result.clarifying_questions = validation.clarifying_questions
                    result.reasoning_chain.append("Query needs clarification")
                    return result
                else:
                    result.status = ReasoningStatus.FAILED
                    result.error_message = "Query validation failed with critical issues"
                    result.reasoning_chain.append("Query validation failed")
                    return result

            result.reasoning_chain.append(f"Query validated: {validation.query_type.value}")

            # Step 2: Analysis Planning
            result.status = ReasoningStatus.PLANNING
            result.reasoning_chain.append("Creating analysis plan")

            plan = await self.planner.create_plan(validation)
            result.plan = plan

            result.reasoning_chain.append(f"Plan created with {len(plan.tasks)} tasks")

            # Step 3: Execute Analysis Plan
            result.status = ReasoningStatus.EXECUTING
            result.reasoning_chain.append("Executing analysis plan")

            execution_results = await self._execute_plan(plan, context)
            result.execution_results = execution_results

            result.reasoning_chain.append(f"Executed {len(execution_results)} analyses")

            # Step 4: Quality Audit
            result.status = ReasoningStatus.AUDITING
            result.reasoning_chain.append("Performing quality audit")

            audit = await self.auditor.audit(plan, execution_results)
            result.audit = audit

            result.reasoning_chain.append(f"Audit complete: {'PASS' if audit.passes_quality_check else 'FAIL'}")

            # Step 5: Self-Correction (if needed and enabled)
            if (not audit.passes_quality_check and
                self.config.enable_self_correction and
                self.config.max_correction_attempts > 0):

                result.reasoning_chain.append("Applying self-correction")
                execution_results = await self._apply_corrections(plan, execution_results, audit)
                result.execution_results = execution_results

                # Re-audit after corrections
                audit = await self.auditor.audit(plan, execution_results)
                result.audit = audit

                result.reasoning_chain.append("Self-correction completed")

            # Step 6: Strategic Synthesis
            result.status = ReasoningStatus.SYNTHESIZING
            result.reasoning_chain.append("Synthesizing strategic insights")

            synthesis = await self.strategist.synthesize(audit, execution_results)
            result.synthesis = synthesis

            result.reasoning_chain.append("Strategic synthesis completed")

            # Step 7: Final Assessment
            result.status = ReasoningStatus.COMPLETED
            result.confidence_score = synthesis.overall_confidence
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()

            result.reasoning_chain.append(f"Reasoning completed in {result.execution_time:.1f}s")

            self.logger.info(f"Reasoning process {reasoning_id} completed successfully. "
                           f"Confidence: {result.confidence_score:.2f}, "
                           f"Time: {result.execution_time:.1f}s")

        except Exception as e:
            result.status = ReasoningStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now()
            result.execution_time = (result.completed_at - start_time).total_seconds()
            result.reasoning_chain.append(f"Reasoning failed: {str(e)}")

            self.logger.error(f"Reasoning process {reasoning_id} failed: {str(e)}")

        finally:
            # Clean up session
            if reasoning_id in self.active_sessions:
                del self.active_sessions[reasoning_id]

        return result

    async def _execute_plan(self, plan: AnalysisPlan, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the analysis plan using specialist agents."""
        results = {}

        if self.config.parallel_execution:
            # Execute tasks in parallel groups
            for group in plan.parallel_groups:
                group_tasks = []
                for task_id in group:
                    task = next((t for t in plan.tasks if t.task_id == task_id), None)
                    if task:
                        group_tasks.append(self._execute_task(task, context, results))

                # Wait for all tasks in the group to complete
                if group_tasks:
                    group_results = await asyncio.gather(*group_tasks, return_exceptions=True)

                    # Process results
                    for i, task_result in enumerate(group_results):
                        task_id = group[i]
                        if isinstance(task_result, Exception):
                            self.logger.error(f"Task {task_id} failed: {task_result}")
                            results[task_id] = {"error": str(task_result), "status": "failed"}
                        else:
                            results[task_id] = task_result
        else:
            # Execute tasks sequentially
            for task_id in plan.execution_order:
                task = next((t for t in plan.tasks if t.task_id == task_id), None)
                if task:
                    try:
                        task_result = await self._execute_task(task, context, results)
                        results[task_id] = task_result
                    except Exception as e:
                        self.logger.error(f"Task {task_id} failed: {e}")
                        results[task_id] = {"error": str(e), "status": "failed"}

        return results

    async def _execute_task(self, task: AnalysisTask, context: Optional[Dict[str, Any]],
                          previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single analysis task."""
        task.start_time = datetime.now()
        task.status = TaskStatus.IN_PROGRESS

        self.logger.info(f"Executing task {task.task_id}: {task.analysis_type.value}")

        try:
            # Get assigned specialist
            specialist = self.specialists.get(task.assigned_specialist)

            if not specialist:
                # Mock execution for demonstration
                return await self._mock_task_execution(task, context, previous_results)

            # Execute task with specialist
            # In real implementation: result = await specialist.execute_task(task, context, previous_results)
            result = await self._mock_task_execution(task, context, previous_results)

            task.status = TaskStatus.COMPLETED
            task.completion_time = datetime.now()
            task.result = result

            return result

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            task.completion_time = datetime.now()
            raise

    async def _mock_task_execution(self, task: AnalysisTask, context: Optional[Dict[str, Any]],
                                 previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Mock task execution for demonstration purposes."""
        # Simulate processing time
        await asyncio.sleep(0.1)

        # Generate mock results based on analysis type
        mock_results = {
            "document_research": {
                "relevant_documents": ["Trust Agreement Template", "Estate Planning Checklist"],
                "legal_references": ["IRC Section 671", "State Trust Code 123.45"],
                "confidence_score": 0.85,
                "recommendations": ["Use revocable trust structure", "Include succession planning"],
                "insights": ["Trust structure provides estate tax benefits"]
            },
            "market_intelligence": {
                "market_trends": ["Increasing demand for digital estate planning", "Tax law changes affecting trusts"],
                "competitive_analysis": ["Traditional firms slow to adopt technology", "Opportunity for digital solutions"],
                "confidence_score": 0.78,
                "opportunities": ["Digital transformation market", "Automated compliance tools"]
            },
            "sales_optimization": {
                "recommendations": ["Focus on high-net-worth clients", "Emphasize tax savings"],
                "next_actions": ["Schedule follow-up consultation", "Prepare trust documentation"],
                "confidence_score": 0.82,
                "optimization_score": 0.75
            },
            "tax_analysis": {
                "tax_implications": ["Potential estate tax savings of $200,000", "Annual gift tax exclusion opportunities"],
                "strategies": ["Lifetime exemption utilization", "Charitable giving deductions"],
                "compliance_status": "Compliant with current tax regulations",
                "confidence_score": 0.88,
                "disclaimers": ["Tax analysis for informational purposes only"]
            },
            "risk_assessment": {
                "risk_factors": ["Market volatility affecting asset values", "Potential liquidity constraints"],
                "mitigation_strategies": ["Diversification recommendations", "Liquidity planning"],
                "confidence_score": 0.79,
                "risk_score": 0.35
            }
        }

        analysis_type = task.analysis_type.value
        return mock_results.get(analysis_type, {
            "analysis_results": f"Completed {analysis_type} analysis",
            "confidence_score": 0.75,
            "recommendations": [f"Recommendation based on {analysis_type}"]
        })

    async def _apply_corrections(self, plan: AnalysisPlan, results: Dict[str, Any],
                               audit: AuditResult) -> Dict[str, Any]:
        """Apply corrections based on audit findings."""
        corrected_results = results.copy()

        # Identify tasks that need correction
        critical_findings = [f for f in audit.findings if f.severity == AuditSeverity.CRITICAL]

        for finding in critical_findings:
            if finding.affected_component in results:
                # Re-execute the affected task
                task = next((t for t in plan.tasks if t.task_id == finding.affected_component), None)
                if task:
                    try:
                        self.logger.info(f"Re-executing task {task.task_id} due to critical finding")
                        corrected_result = await self._execute_task(task, None, corrected_results)
                        corrected_results[task.task_id] = corrected_result
                    except Exception as e:
                        self.logger.error(f"Correction failed for task {task.task_id}: {e}")

        return corrected_results

    async def get_clarification_response(self, reasoning_id: str,
                                       clarification: str) -> ReasoningResult:
        """Process clarification response and continue reasoning."""
        # This would implement the clarification handling logic
        # For now, return a placeholder
        return ReasoningResult(
            reasoning_id=reasoning_id,
            status=ReasoningStatus.FAILED,
            query=clarification,
            error_message="Clarification handling not yet implemented"
        )

    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of the reasoning engine."""
        health_checks = {}

        # Check individual components
        try:
            health_checks["gatekeeper"] = await self.gatekeeper.health_check()
        except Exception as e:
            health_checks["gatekeeper"] = {"status": "unhealthy", "error": str(e)}

        try:
            health_checks["planner"] = await self.planner.health_check()
        except Exception as e:
            health_checks["planner"] = {"status": "unhealthy", "error": str(e)}

        try:
            health_checks["auditor"] = await self.auditor.health_check()
        except Exception as e:
            health_checks["auditor"] = {"status": "unhealthy", "error": str(e)}

        try:
            health_checks["strategist"] = await self.strategist.health_check()
        except Exception as e:
            health_checks["strategist"] = {"status": "unhealthy", "error": str(e)}

        # Overall health assessment
        all_healthy = all(check.get("status") == "healthy" for check in health_checks.values())

        return {
            "status": "healthy" if all_healthy else "degraded",
            "component": "estate_reasoning_engine",
            "components": health_checks,
            "active_sessions": len(self.active_sessions),
            "configuration": {
                "execution_mode": self.config.execution_mode.value,
                "self_correction_enabled": self.config.enable_self_correction,
                "parallel_execution": self.config.parallel_execution
            },
            "last_check": datetime.now().isoformat()
        }

    def get_session_status(self, reasoning_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active reasoning session."""
        session = self.active_sessions.get(reasoning_id)
        if not session:
            return None

        return {
            "reasoning_id": reasoning_id,
            "status": session.status.value,
            "query": session.query,
            "execution_time": (datetime.now() - session.created_at).total_seconds(),
            "reasoning_chain": session.reasoning_chain,
            "confidence_score": session.confidence_score
        }


# Example usage and testing
async def main():
    """Example usage of Estate Reasoning Engine."""
    # Configure engine
    config = ReasoningConfig(
        execution_mode=ExecutionMode.STANDARD,
        enable_self_correction=True,
        parallel_execution=True
    )

    engine = EstateReasoningEngine(config)

    # Test queries
    test_queries = [
        "I need to create a trust for my $5 million estate to benefit my spouse and two children while minimizing estate taxes",
        "How can I protect my business assets from potential creditors while planning for succession to my daughter?",
        "What are the tax implications of gifting my vacation home to my children?",
        "I want to set up a charitable foundation - what are my options?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test Query {i}: {query}")
        print('='*60)

        try:
            result = await engine.process_query(query)

            print(f"\nResult ID: {result.reasoning_id}")
            print(f"Status: {result.status.value}")
            print(f"Execution Time: {result.execution_time:.2f}s")
            print(f"Confidence: {result.confidence_score:.2f}")

            if result.status == ReasoningStatus.NEEDS_CLARIFICATION:
                print("\nClarifying Questions:")
                for question in result.clarifying_questions:
                    print(f"  - {question}")

            elif result.status == ReasoningStatus.COMPLETED:
                print(f"\nPrimary Recommendation:")
                print(f"  {result.synthesis.primary_recommendation}")

                print(f"\nKey Findings:")
                for finding in result.synthesis.key_findings[:3]:
                    print(f"  - {finding}")

                print(f"\nNext Actions:")
                for action in result.synthesis.next_actions[:3]:
                    print(f"  - {action}")

                print(f"\nBusiness Value Score: {result.synthesis.business_value_score:.2f}")
                print(f"Success Probability: {result.synthesis.success_probability:.2f}")

            elif result.status == ReasoningStatus.FAILED:
                print(f"\nError: {result.error_message}")

            print(f"\nReasoning Chain:")
            for step in result.reasoning_chain:
                print(f"  → {step}")

        except Exception as e:
            print(f"Error processing query: {e}")

    # Health check
    print(f"\n{'='*60}")
    print("System Health Check")
    print('='*60)

    health = await engine.health_check()
    print(f"Overall Status: {health['status']}")
    print(f"Active Sessions: {health['active_sessions']}")

    for component, status in health['components'].items():
        print(f"  {component}: {status.get('status', 'unknown')}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())