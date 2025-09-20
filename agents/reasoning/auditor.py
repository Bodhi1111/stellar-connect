"""
Estate Auditor Agent for Stellar Connect
Phase 2 Week 3: Cognitive Pipeline Components

The Estate Auditor performs quality control, validation, and self-correction on analysis
results. It ensures accuracy, completeness, and reliability of estate planning recommendations
before they are presented to users.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics

from .planner import AnalysisPlan, AnalysisTask, TaskStatus


class AuditSeverity(Enum):
    """Severity levels for audit findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AuditCategory(Enum):
    """Categories of audit checks."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"


@dataclass
class AuditFinding:
    """Represents an audit finding or issue."""
    finding_id: str
    category: AuditCategory
    severity: AuditSeverity
    description: str
    affected_component: str
    recommendation: str
    auto_correctable: bool = False
    confidence_impact: float = 0.0  # Impact on overall confidence (-1.0 to 1.0)


@dataclass
class QualityMetrics:
    """Quality metrics for analysis results."""
    accuracy_score: float
    completeness_score: float
    consistency_score: float
    confidence_score: float
    reliability_score: float
    overall_quality_score: float
    total_findings: int
    critical_findings: int
    execution_time: float
    success_rate: float


@dataclass
class AuditResult:
    """Result of the audit process."""
    audit_id: str
    passes_quality_check: bool
    quality_metrics: QualityMetrics
    findings: List[AuditFinding]
    corrections_applied: List[str]
    confidence_score: float
    recommendation: str
    audit_timestamp: datetime = field(default_factory=datetime.now)


class EstateAuditor:
    """
    Estate Auditor Agent for quality control and validation.

    Responsibilities:
    - Validate accuracy of analysis results
    - Check completeness of required analyses
    - Ensure consistency across different analysis outputs
    - Verify compliance with estate planning standards
    - Assess performance and reliability metrics
    - Apply automatic corrections where possible
    - Generate quality reports and recommendations
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Quality thresholds
        self.quality_thresholds = {
            "minimum_accuracy": 0.75,
            "minimum_completeness": 0.80,
            "minimum_consistency": 0.70,
            "minimum_confidence": 0.65,
            "minimum_overall_quality": 0.70,
            "maximum_critical_findings": 2,
            "minimum_success_rate": 0.75
        }

        # Expected data structures for different analysis types
        self.expected_result_structures = {
            "document_research": {
                "required_fields": ["relevant_documents", "legal_references", "confidence_score"],
                "optional_fields": ["precedent_cases", "risk_factors", "recommendations"]
            },
            "market_intelligence": {
                "required_fields": ["market_trends", "competitive_analysis", "confidence_score"],
                "optional_fields": ["opportunities", "threats", "market_size"]
            },
            "sales_optimization": {
                "required_fields": ["recommendations", "next_actions", "confidence_score"],
                "optional_fields": ["optimization_score", "implementation_timeline"]
            },
            "tax_analysis": {
                "required_fields": ["tax_implications", "strategies", "compliance_status", "confidence_score"],
                "optional_fields": ["savings_potential", "risk_assessment"]
            }
        }

        # Common validation patterns
        self.validation_patterns = {
            "monetary_amounts": r'\$[\d,]+(?:\.\d{2})?',
            "percentages": r'\d+(?:\.\d+)?%',
            "dates": r'\d{1,2}\/\d{1,2}\/\d{4}|\d{4}-\d{2}-\d{2}',
            "confidence_scores": r'0\.\d+|1\.0'
        }

    async def audit(self, plan: AnalysisPlan, results: Dict[str, Any]) -> AuditResult:
        """
        Perform comprehensive audit of analysis results.

        Args:
            plan: The original analysis plan
            results: Dictionary of results from specialist agents

        Returns:
            AuditResult with quality assessment and recommendations
        """
        self.logger.info(f"Starting audit for plan {plan.plan_id}")

        audit_id = f"audit_{plan.plan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        findings = []
        corrections_applied = []

        # Step 1: Validate plan execution
        execution_findings = await self._audit_plan_execution(plan)
        findings.extend(execution_findings)

        # Step 2: Check result structure and completeness
        structure_findings = await self._audit_result_structures(plan, results)
        findings.extend(structure_findings)

        # Step 3: Validate data accuracy
        accuracy_findings = await self._audit_data_accuracy(results)
        findings.extend(accuracy_findings)

        # Step 4: Check consistency across results
        consistency_findings = await self._audit_consistency(results)
        findings.extend(consistency_findings)

        # Step 5: Verify compliance requirements
        compliance_findings = await self._audit_compliance(plan, results)
        findings.extend(compliance_findings)

        # Step 6: Assess performance metrics
        performance_findings = await self._audit_performance(plan, results)
        findings.extend(performance_findings)

        # Step 7: Apply automatic corrections
        corrections_applied = await self._apply_automatic_corrections(findings, results)

        # Step 8: Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(plan, results, findings)

        # Step 9: Determine overall quality assessment
        passes_quality_check = self._assess_overall_quality(quality_metrics, findings)

        # Step 10: Generate recommendations
        recommendation = self._generate_recommendation(quality_metrics, findings, passes_quality_check)

        audit_result = AuditResult(
            audit_id=audit_id,
            passes_quality_check=passes_quality_check,
            quality_metrics=quality_metrics,
            findings=findings,
            corrections_applied=corrections_applied,
            confidence_score=quality_metrics.confidence_score,
            recommendation=recommendation
        )

        self.logger.info(f"Audit complete. Quality check: {'PASS' if passes_quality_check else 'FAIL'}, "
                        f"Overall score: {quality_metrics.overall_quality_score:.2f}")

        return audit_result

    async def _audit_plan_execution(self, plan: AnalysisPlan) -> List[AuditFinding]:
        """Audit the execution of the analysis plan."""
        findings = []

        # Check completion rates
        completed_tasks = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in plan.tasks if t.status == TaskStatus.FAILED]
        completion_rate = len(completed_tasks) / len(plan.tasks) if plan.tasks else 0

        if completion_rate < self.quality_thresholds["minimum_success_rate"]:
            findings.append(AuditFinding(
                finding_id=f"execution_001",
                category=AuditCategory.PERFORMANCE,
                severity=AuditSeverity.HIGH,
                description=f"Low completion rate: {completion_rate:.2f} ({len(completed_tasks)}/{len(plan.tasks)} tasks)",
                affected_component="plan_execution",
                recommendation="Investigate failed tasks and consider retry or alternative approaches",
                confidence_impact=-0.3
            ))

        # Check for critical task failures
        critical_failed = [t for t in failed_tasks if t.priority.value == "critical"]
        if critical_failed:
            findings.append(AuditFinding(
                finding_id=f"execution_002",
                category=AuditCategory.RELIABILITY,
                severity=AuditSeverity.CRITICAL,
                description=f"{len(critical_failed)} critical tasks failed",
                affected_component="critical_tasks",
                recommendation="Critical tasks must be resolved before proceeding",
                confidence_impact=-0.5
            ))

        # Check execution time vs estimates
        actual_duration = sum((t.completion_time - t.start_time).total_seconds()
                            for t in completed_tasks if t.start_time and t.completion_time)

        if actual_duration > plan.estimated_duration * 2:
            findings.append(AuditFinding(
                finding_id=f"execution_003",
                category=AuditCategory.PERFORMANCE,
                severity=AuditSeverity.MEDIUM,
                description=f"Execution time exceeded estimates by {(actual_duration/plan.estimated_duration - 1)*100:.0f}%",
                affected_component="performance",
                recommendation="Review task complexity and estimation accuracy",
                confidence_impact=-0.1
            ))

        return findings

    async def _audit_result_structures(self, plan: AnalysisPlan, results: Dict[str, Any]) -> List[AuditFinding]:
        """Audit the structure and completeness of results."""
        findings = []

        for task in plan.tasks:
            if task.status != TaskStatus.COMPLETED:
                continue

            analysis_type = task.analysis_type.value
            task_result = results.get(task.task_id, {})

            # Check if result exists
            if not task_result:
                findings.append(AuditFinding(
                    finding_id=f"structure_{task.task_id}_001",
                    category=AuditCategory.COMPLETENESS,
                    severity=AuditSeverity.HIGH,
                    description=f"Missing results for completed task: {analysis_type}",
                    affected_component=task.task_id,
                    recommendation="Ensure all completed tasks produce results",
                    confidence_impact=-0.2
                ))
                continue

            # Check required fields
            expected_structure = self.expected_result_structures.get(analysis_type)
            if expected_structure:
                required_fields = expected_structure["required_fields"]
                missing_fields = [field for field in required_fields if field not in task_result]

                if missing_fields:
                    findings.append(AuditFinding(
                        finding_id=f"structure_{task.task_id}_002",
                        category=AuditCategory.COMPLETENESS,
                        severity=AuditSeverity.MEDIUM,
                        description=f"Missing required fields in {analysis_type}: {missing_fields}",
                        affected_component=task.task_id,
                        recommendation=f"Ensure {analysis_type} results include all required fields",
                        auto_correctable=True,
                        confidence_impact=-0.1
                    ))

        return findings

    async def _audit_data_accuracy(self, results: Dict[str, Any]) -> List[AuditFinding]:
        """Audit the accuracy of data in results."""
        findings = []

        for task_id, result in results.items():
            if not isinstance(result, dict):
                continue

            # Check confidence scores
            confidence_score = result.get("confidence_score")
            if confidence_score is not None:
                try:
                    conf_value = float(confidence_score)
                    if conf_value < 0 or conf_value > 1:
                        findings.append(AuditFinding(
                            finding_id=f"accuracy_{task_id}_001",
                            category=AuditCategory.ACCURACY,
                            severity=AuditSeverity.MEDIUM,
                            description=f"Invalid confidence score: {confidence_score} (should be 0-1)",
                            affected_component=task_id,
                            recommendation="Ensure confidence scores are between 0 and 1",
                            auto_correctable=True,
                            confidence_impact=-0.1
                        ))
                except (ValueError, TypeError):
                    findings.append(AuditFinding(
                        finding_id=f"accuracy_{task_id}_002",
                        category=AuditCategory.ACCURACY,
                        severity=AuditSeverity.MEDIUM,
                        description=f"Non-numeric confidence score: {confidence_score}",
                        affected_component=task_id,
                        recommendation="Confidence scores must be numeric values",
                        auto_correctable=True,
                        confidence_impact=-0.1
                    ))

            # Check for empty or null critical fields
            critical_fields = ["recommendations", "analysis", "insights"]
            for field in critical_fields:
                if field in result:
                    value = result[field]
                    if not value or (isinstance(value, (list, dict)) and len(value) == 0):
                        findings.append(AuditFinding(
                            finding_id=f"accuracy_{task_id}_003",
                            category=AuditCategory.COMPLETENESS,
                            severity=AuditSeverity.MEDIUM,
                            description=f"Empty critical field: {field}",
                            affected_component=task_id,
                            recommendation=f"Ensure {field} contains meaningful content",
                            confidence_impact=-0.15
                        ))

        return findings

    async def _audit_consistency(self, results: Dict[str, Any]) -> List[AuditFinding]:
        """Audit consistency across different analysis results."""
        findings = []

        # Extract confidence scores from all results
        confidence_scores = []
        for result in results.values():
            if isinstance(result, dict) and "confidence_score" in result:
                try:
                    conf = float(result["confidence_score"])
                    if 0 <= conf <= 1:
                        confidence_scores.append(conf)
                except (ValueError, TypeError):
                    pass

        # Check for extreme variance in confidence scores
        if len(confidence_scores) > 1:
            std_dev = statistics.stdev(confidence_scores)
            mean_conf = statistics.mean(confidence_scores)

            if std_dev > 0.3:  # High variance threshold
                findings.append(AuditFinding(
                    finding_id="consistency_001",
                    category=AuditCategory.CONSISTENCY,
                    severity=AuditSeverity.MEDIUM,
                    description=f"High variance in confidence scores (std: {std_dev:.2f}, mean: {mean_conf:.2f})",
                    affected_component="confidence_scores",
                    recommendation="Review analyses with extreme confidence scores for potential issues",
                    confidence_impact=-0.1
                ))

        # Check for contradictory recommendations
        all_recommendations = []
        for result in results.values():
            if isinstance(result, dict):
                recs = result.get("recommendations", [])
                if isinstance(recs, list):
                    all_recommendations.extend(recs)
                elif isinstance(recs, str):
                    all_recommendations.append(recs)

        # Simple contradiction detection (this could be more sophisticated)
        contradiction_keywords = [
            ("recommend", "not recommend"),
            ("should", "should not"),
            ("increase", "decrease"),
            ("buy", "sell")
        ]

        for pos_keyword, neg_keyword in contradiction_keywords:
            pos_count = sum(1 for rec in all_recommendations if pos_keyword in str(rec).lower())
            neg_count = sum(1 for rec in all_recommendations if neg_keyword in str(rec).lower())

            if pos_count > 0 and neg_count > 0:
                findings.append(AuditFinding(
                    finding_id="consistency_002",
                    category=AuditCategory.CONSISTENCY,
                    severity=AuditSeverity.HIGH,
                    description=f"Potential contradictory recommendations: {pos_keyword} vs {neg_keyword}",
                    affected_component="recommendations",
                    recommendation="Review recommendations for potential conflicts",
                    confidence_impact=-0.2
                ))

        return findings

    async def _audit_compliance(self, plan: AnalysisPlan, results: Dict[str, Any]) -> List[AuditFinding]:
        """Audit compliance with estate planning standards and requirements."""
        findings = []

        # Check if compliance check was performed for trust-related queries
        if "trust" in plan.query_type.value:
            compliance_task = None
            for task in plan.tasks:
                if "compliance" in task.analysis_type.value:
                    compliance_task = task
                    break

            if not compliance_task:
                findings.append(AuditFinding(
                    finding_id="compliance_001",
                    category=AuditCategory.COMPLIANCE,
                    severity=AuditSeverity.HIGH,
                    description="Compliance check not performed for trust-related query",
                    affected_component="compliance_analysis",
                    recommendation="Always include compliance analysis for trust structures",
                    confidence_impact=-0.25
                ))
            elif compliance_task.status != TaskStatus.COMPLETED:
                findings.append(AuditFinding(
                    finding_id="compliance_002",
                    category=AuditCategory.COMPLIANCE,
                    severity=AuditSeverity.HIGH,
                    description="Compliance check task failed or incomplete",
                    affected_component="compliance_analysis",
                    recommendation="Ensure compliance analysis completes successfully",
                    confidence_impact=-0.3
                ))

        # Check for required disclaimers in tax analysis
        for task_id, result in results.items():
            if isinstance(result, dict):
                task = next((t for t in plan.tasks if t.task_id == task_id), None)
                if task and "tax" in task.analysis_type.value:
                    # Look for tax disclaimer
                    disclaimers = result.get("disclaimers", [])
                    has_tax_disclaimer = any("tax" in str(disclaimer).lower() for disclaimer in disclaimers)

                    if not has_tax_disclaimer:
                        findings.append(AuditFinding(
                            finding_id=f"compliance_{task_id}_001",
                            category=AuditCategory.COMPLIANCE,
                            severity=AuditSeverity.MEDIUM,
                            description="Tax analysis missing required disclaimer",
                            affected_component=task_id,
                            recommendation="Include appropriate tax advisory disclaimers",
                            auto_correctable=True,
                            confidence_impact=-0.05
                        ))

        return findings

    async def _audit_performance(self, plan: AnalysisPlan, results: Dict[str, Any]) -> List[AuditFinding]:
        """Audit performance metrics of the analysis."""
        findings = []

        # Calculate actual vs estimated duration
        completed_tasks = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]

        for task in completed_tasks:
            if task.start_time and task.completion_time:
                actual_duration = (task.completion_time - task.start_time).total_seconds()
                estimated_duration = task.estimated_duration

                if actual_duration > estimated_duration * 3:  # 3x threshold
                    findings.append(AuditFinding(
                        finding_id=f"performance_{task.task_id}_001",
                        category=AuditCategory.PERFORMANCE,
                        severity=AuditSeverity.LOW,
                        description=f"Task took {actual_duration/estimated_duration:.1f}x longer than estimated",
                        affected_component=task.task_id,
                        recommendation="Review task complexity and estimation accuracy",
                        confidence_impact=-0.05
                    ))

        # Check result size and complexity
        for task_id, result in results.items():
            if isinstance(result, dict):
                result_size = len(str(result))
                if result_size < 100:  # Very small result
                    findings.append(AuditFinding(
                        finding_id=f"performance_{task_id}_002",
                        category=AuditCategory.COMPLETENESS,
                        severity=AuditSeverity.MEDIUM,
                        description=f"Result appears unusually small ({result_size} chars)",
                        affected_component=task_id,
                        recommendation="Verify that analysis produced sufficient detail",
                        confidence_impact=-0.1
                    ))

        return findings

    async def _apply_automatic_corrections(self, findings: List[AuditFinding],
                                         results: Dict[str, Any]) -> List[str]:
        """Apply automatic corrections for correctable findings."""
        corrections_applied = []

        for finding in findings:
            if not finding.auto_correctable:
                continue

            # Apply confidence score corrections
            if "confidence_score" in finding.description:
                for task_id, result in results.items():
                    if isinstance(result, dict) and "confidence_score" in result:
                        try:
                            conf = float(result["confidence_score"])
                            if conf < 0:
                                result["confidence_score"] = 0.0
                                corrections_applied.append(f"Corrected negative confidence score in {task_id}")
                            elif conf > 1:
                                result["confidence_score"] = 1.0
                                corrections_applied.append(f"Corrected confidence score > 1 in {task_id}")
                        except (ValueError, TypeError):
                            result["confidence_score"] = 0.5  # Default value
                            corrections_applied.append(f"Corrected non-numeric confidence score in {task_id}")

            # Add missing disclaimers for tax analysis
            if "tax analysis missing required disclaimer" in finding.description:
                task_id = finding.affected_component
                if task_id in results and isinstance(results[task_id], dict):
                    disclaimers = results[task_id].get("disclaimers", [])
                    disclaimers.append("This analysis is for informational purposes only and does not constitute tax advice. Consult a qualified tax professional for specific guidance.")
                    results[task_id]["disclaimers"] = disclaimers
                    corrections_applied.append(f"Added tax disclaimer to {task_id}")

        return corrections_applied

    def _calculate_quality_metrics(self, plan: AnalysisPlan, results: Dict[str, Any],
                                 findings: List[AuditFinding]) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        # Count findings by severity
        critical_findings = len([f for f in findings if f.severity == AuditSeverity.CRITICAL])
        high_findings = len([f for f in findings if f.severity == AuditSeverity.HIGH])
        medium_findings = len([f for f in findings if f.severity == AuditSeverity.MEDIUM])

        # Calculate component scores
        accuracy_score = max(0.0, 1.0 - (critical_findings * 0.3 + high_findings * 0.2 + medium_findings * 0.1))

        completed_tasks = len([t for t in plan.tasks if t.status == TaskStatus.COMPLETED])
        completeness_score = completed_tasks / len(plan.tasks) if plan.tasks else 0

        # Consistency score based on variance in confidence scores
        confidence_scores = []
        for result in results.values():
            if isinstance(result, dict) and "confidence_score" in result:
                try:
                    conf = float(result["confidence_score"])
                    if 0 <= conf <= 1:
                        confidence_scores.append(conf)
                except (ValueError, TypeError):
                    pass

        if confidence_scores:
            mean_confidence = statistics.mean(confidence_scores)
            variance = statistics.variance(confidence_scores) if len(confidence_scores) > 1 else 0
            consistency_score = max(0.0, 1.0 - variance)
            overall_confidence = mean_confidence
        else:
            consistency_score = 0.5
            overall_confidence = 0.5

        # Reliability score based on task success rate
        reliability_score = completeness_score

        # Calculate overall quality score
        weights = {
            "accuracy": 0.3,
            "completeness": 0.25,
            "consistency": 0.2,
            "reliability": 0.25
        }

        overall_quality_score = (
            accuracy_score * weights["accuracy"] +
            completeness_score * weights["completeness"] +
            consistency_score * weights["consistency"] +
            reliability_score * weights["reliability"]
        )

        # Calculate execution time
        execution_time = 0
        for task in plan.tasks:
            if task.start_time and task.completion_time:
                execution_time += (task.completion_time - task.start_time).total_seconds()

        return QualityMetrics(
            accuracy_score=accuracy_score,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            confidence_score=overall_confidence,
            reliability_score=reliability_score,
            overall_quality_score=overall_quality_score,
            total_findings=len(findings),
            critical_findings=critical_findings,
            execution_time=execution_time,
            success_rate=completeness_score
        )

    def _assess_overall_quality(self, metrics: QualityMetrics, findings: List[AuditFinding]) -> bool:
        """Assess whether the overall quality meets standards."""
        # Critical failures
        if metrics.critical_findings > self.quality_thresholds["maximum_critical_findings"]:
            return False

        # Minimum thresholds
        if metrics.overall_quality_score < self.quality_thresholds["minimum_overall_quality"]:
            return False

        if metrics.completeness_score < self.quality_thresholds["minimum_completeness"]:
            return False

        if metrics.confidence_score < self.quality_thresholds["minimum_confidence"]:
            return False

        return True

    def _generate_recommendation(self, metrics: QualityMetrics, findings: List[AuditFinding],
                               passes_quality_check: bool) -> str:
        """Generate recommendation based on audit results."""
        if passes_quality_check:
            if metrics.overall_quality_score > 0.9:
                return "Excellent quality - results ready for presentation"
            elif metrics.overall_quality_score > 0.8:
                return "Good quality - results acceptable with minor improvements"
            else:
                return "Acceptable quality - consider addressing medium-priority findings"
        else:
            critical_count = metrics.critical_findings
            if critical_count > 0:
                return f"Quality check failed - {critical_count} critical issues must be resolved"
            else:
                return "Quality check failed - overall quality below minimum threshold"

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the auditor."""
        return {
            "status": "healthy",
            "component": "estate_auditor",
            "quality_thresholds": len(self.quality_thresholds),
            "validation_patterns": len(self.validation_patterns),
            "expected_structures": len(self.expected_result_structures),
            "last_check": datetime.now().isoformat()
        }


# Example usage and testing
async def main():
    """Example usage of Estate Auditor."""
    from .planner import AnalysisPlan, AnalysisTask, TaskStatus, TaskPriority, AnalysisType
    from .gatekeeper import QueryType

    auditor = EstateAuditor()

    # Mock analysis plan
    plan = AnalysisPlan(
        plan_id="test_plan_001",
        query_type=QueryType.TRUST_STRUCTURE,
        total_tasks=3,
        estimated_duration=600,
        tasks=[
            AnalysisTask(
                task_id="task_001",
                analysis_type=AnalysisType.DOCUMENT_RESEARCH,
                assigned_specialist="estate_librarian",
                description="Research trust documents",
                priority=TaskPriority.HIGH,
                estimated_duration=300,
                status=TaskStatus.COMPLETED,
                start_time=datetime.now(),
                completion_time=datetime.now()
            ),
            AnalysisTask(
                task_id="task_002",
                analysis_type=AnalysisType.TAX_ANALYSIS,
                assigned_specialist="trust_sales_analyst",
                description="Analyze tax implications",
                priority=TaskPriority.CRITICAL,
                estimated_duration=300,
                status=TaskStatus.COMPLETED,
                start_time=datetime.now(),
                completion_time=datetime.now()
            )
        ],
        execution_order=["task_001", "task_002"]
    )

    # Mock results
    results = {
        "task_001": {
            "relevant_documents": ["Trust Agreement Template", "State Trust Laws"],
            "legal_references": ["IRC Section 671", "State Code 123.45"],
            "confidence_score": 0.85,
            "recommendations": ["Use revocable trust structure", "Include spendthrift provisions"]
        },
        "task_002": {
            "tax_implications": ["Estate tax savings", "Gift tax considerations"],
            "strategies": ["Annual exclusion gifts", "Lifetime exemption usage"],
            "compliance_status": "Compliant with current regulations",
            "confidence_score": 0.78
        }
    }

    print("Performing audit...")
    audit_result = await auditor.audit(plan, results)

    print(f"\nAudit ID: {audit_result.audit_id}")
    print(f"Quality Check: {'PASS' if audit_result.passes_quality_check else 'FAIL'}")
    print(f"Overall Quality Score: {audit_result.quality_metrics.overall_quality_score:.2f}")
    print(f"Confidence Score: {audit_result.confidence_score:.2f}")

    print(f"\nQuality Metrics:")
    print(f"  Accuracy: {audit_result.quality_metrics.accuracy_score:.2f}")
    print(f"  Completeness: {audit_result.quality_metrics.completeness_score:.2f}")
    print(f"  Consistency: {audit_result.quality_metrics.consistency_score:.2f}")
    print(f"  Reliability: {audit_result.quality_metrics.reliability_score:.2f}")

    if audit_result.findings:
        print(f"\nFindings ({len(audit_result.findings)} total):")
        for finding in audit_result.findings:
            print(f"  - {finding.severity.value.upper()}: {finding.description}")

    if audit_result.corrections_applied:
        print(f"\nCorrections Applied:")
        for correction in audit_result.corrections_applied:
            print(f"  - {correction}")

    print(f"\nRecommendation: {audit_result.recommendation}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())