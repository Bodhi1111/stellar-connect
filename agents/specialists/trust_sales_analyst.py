"""
Trust Sales Analyst Agent for Stellar Connect
Implements Story 5.2: Sales Specialist Agent Team - Trust Sales Analyst

The Trust Sales Analyst specializes in sales conversion analysis, performance optimization,
objection handling analysis, and revenue forecasting for estate planning sales.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import json

from .base_specialist import (
    BaseSpecialist, SpecialistTask, SpecialistExpertise,
    SpecialistCapability, TaskStatus
)


class ConversionStage(Enum):
    """Sales conversion stages for estate planning."""
    INITIAL_CONTACT = "initial_contact"
    DISCOVERY = "discovery"
    NEEDS_ANALYSIS = "needs_analysis"
    PRESENTATION = "presentation"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING = "closing"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class ObjectionCategory(Enum):
    """Categories of objections in estate planning sales."""
    COST = "cost"
    COMPLEXITY = "complexity"
    TIMING = "timing"
    NECESSITY = "necessity"
    TRUST = "trust"
    FAMILY_DYNAMICS = "family_dynamics"
    TAX_CONCERNS = "tax_concerns"
    LEGACY_CONCERNS = "legacy_concerns"


@dataclass
class ConversionAnalysis:
    """Analysis of conversion patterns and performance."""
    analysis_id: str
    time_period: str
    total_opportunities: int
    conversion_rate: float
    average_deal_size: float
    average_sales_cycle: int  # days
    stage_conversion_rates: Dict[str, float]
    top_loss_reasons: List[Tuple[str, int]]
    performance_trends: Dict[str, float]
    recommendations: List[str]
    confidence_score: float


@dataclass
class ObjectionAnalysis:
    """Analysis of objection patterns and handling effectiveness."""
    objection_category: str
    frequency: int
    avg_resolution_time: float  # minutes
    resolution_success_rate: float
    effective_responses: List[str]
    ineffective_responses: List[str]
    context_patterns: List[str]
    impact_on_conversion: float


@dataclass
class SalesPerformanceMetrics:
    """Comprehensive sales performance metrics."""
    metric_id: str
    advisor_name: str
    time_period: str

    # Volume metrics
    total_conversations: int
    qualified_leads: int
    presentations_given: int
    proposals_sent: int
    deals_closed: int

    # Conversion metrics
    lead_qualification_rate: float
    presentation_conversion_rate: float
    proposal_conversion_rate: float
    overall_conversion_rate: float

    # Performance metrics
    average_deal_size: float
    total_revenue: float
    average_sales_cycle: int
    activity_score: float

    # Quality metrics
    client_satisfaction_score: float
    referral_rate: float
    repeat_business_rate: float


@dataclass
class RevenuePredicition:
    """Revenue prediction based on pipeline analysis."""
    prediction_id: str
    forecast_period: str
    predicted_revenue: float
    confidence_interval: Tuple[float, float]
    key_assumptions: List[str]
    risk_factors: List[str]
    recommended_actions: List[str]
    pipeline_health_score: float


class TrustSalesAnalystAgent(BaseSpecialist):
    """
    Trust Sales Analyst Agent - Specialist in sales conversion and performance analysis.

    Capabilities:
    - Analyze conversion patterns and pipeline performance
    - Identify objection handling effectiveness
    - Generate sales performance reports
    - Predict revenue based on pipeline analysis
    - Recommend sales process optimizations
    """

    def __init__(self, data_path: str = "data/sales_analytics"):
        # Define capabilities
        capabilities = [
            SpecialistCapability(
                name="conversion_analysis",
                description="Analyze sales conversion patterns and performance",
                input_types=["sales_data", "time_period"],
                output_types=["conversion_metrics", "performance_insights"]
            ),
            SpecialistCapability(
                name="objection_analysis",
                description="Analyze objection patterns and handling effectiveness",
                input_types=["conversation_data", "objection_filters"],
                output_types=["objection_insights", "improvement_recommendations"]
            ),
            SpecialistCapability(
                name="performance_reporting",
                description="Generate comprehensive sales performance reports",
                input_types=["performance_data", "report_criteria"],
                output_types=["performance_report", "comparative_analysis"]
            ),
            SpecialistCapability(
                name="revenue_forecasting",
                description="Predict future revenue based on pipeline analysis",
                input_types=["pipeline_data", "forecast_parameters"],
                output_types=["revenue_prediction", "scenario_analysis"]
            ),
            SpecialistCapability(
                name="process_optimization",
                description="Recommend sales process improvements",
                input_types=["process_data", "performance_gaps"],
                output_types=["optimization_recommendations", "implementation_plan"]
            )
        ]

        super().__init__(
            name="Trust Sales Analyst",
            expertise=SpecialistExpertise.TRUST_SALES_ANALYSIS,
            description="Specialist in sales conversion analysis, performance optimization, and revenue forecasting for estate planning",
            capabilities=capabilities,
            max_concurrent_tasks=4
        )

        self.data_path = data_path
        self.conversion_history: List[ConversionAnalysis] = []
        self.objection_analyses: List[ObjectionAnalysis] = []
        self.performance_metrics: List[SalesPerformanceMetrics] = []

        self.logger = logging.getLogger(f"{__name__}.TrustSalesAnalystAgent")

    def get_task_types(self) -> List[str]:
        """Return list of task types this specialist can handle."""
        return [
            "analyze_conversions",
            "analyze_objections",
            "generate_performance_report",
            "forecast_revenue",
            "optimize_sales_process",
            "benchmark_performance",
            "identify_coaching_opportunities"
        ]

    async def validate_input(self, task: SpecialistTask) -> Tuple[bool, Optional[str]]:
        """Validate task input data."""
        task_type = task.task_type
        input_data = task.input_data

        try:
            if task_type == "analyze_conversions":
                if "conversations" not in input_data:
                    return False, "Missing required field: conversations"
                if not isinstance(input_data["conversations"], list):
                    return False, "conversations must be a list"

            elif task_type == "analyze_objections":
                if "objection_data" not in input_data:
                    return False, "Missing required field: objection_data"

            elif task_type == "generate_performance_report":
                if "advisor_name" not in input_data and "team_scope" not in input_data:
                    return False, "Must specify either advisor_name or team_scope"

            elif task_type == "forecast_revenue":
                required_fields = ["pipeline_data", "forecast_period"]
                for field in required_fields:
                    if field not in input_data:
                        return False, f"Missing required field: {field}"

            return True, None

        except Exception as e:
            return False, str(e)

    async def process_task(self, task: SpecialistTask) -> Dict[str, Any]:
        """Process a specific task based on task type."""
        task_type = task.task_type
        input_data = task.input_data

        try:
            if task_type == "analyze_conversions":
                return await self._analyze_conversions(input_data)

            elif task_type == "analyze_objections":
                return await self._analyze_objections(input_data)

            elif task_type == "generate_performance_report":
                return await self._generate_performance_report(input_data)

            elif task_type == "forecast_revenue":
                return await self._forecast_revenue(input_data)

            elif task_type == "optimize_sales_process":
                return await self._optimize_sales_process(input_data)

            elif task_type == "benchmark_performance":
                return await self._benchmark_performance(input_data)

            elif task_type == "identify_coaching_opportunities":
                return await self._identify_coaching_opportunities(input_data)

            else:
                raise ValueError(f"Unsupported task type: {task_type}")

        except Exception as e:
            self.logger.error(f"Error processing task {task.task_id}: {str(e)}")
            raise

    async def _analyze_conversions(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversion patterns and performance."""
        conversations = input_data.get("conversations", [])
        time_period = input_data.get("time_period", "30_days")

        # Filter conversations by time period
        cutoff_date = self._get_cutoff_date(time_period)
        recent_conversations = [
            conv for conv in conversations
            if datetime.fromisoformat(conv.get("date", "")) >= cutoff_date
        ]

        # Calculate basic metrics
        total_opportunities = len(recent_conversations)
        closed_won = len([c for c in recent_conversations if c.get("outcome") == "closed_won"])
        conversion_rate = closed_won / total_opportunities if total_opportunities > 0 else 0

        # Calculate average deal size
        won_deals = [c for c in recent_conversations if c.get("outcome") == "closed_won"]
        average_deal_size = statistics.mean([
            c.get("deal_value", 0) for c in won_deals
        ]) if won_deals else 0

        # Calculate average sales cycle
        completed_deals = [
            c for c in recent_conversations
            if c.get("outcome") in ["closed_won", "closed_lost"] and c.get("first_contact_date")
        ]

        sales_cycles = []
        for deal in completed_deals:
            first_contact = datetime.fromisoformat(deal["first_contact_date"])
            close_date = datetime.fromisoformat(deal["date"])
            cycle_days = (close_date - first_contact).days
            sales_cycles.append(cycle_days)

        average_sales_cycle = int(statistics.mean(sales_cycles)) if sales_cycles else 0

        # Analyze stage conversion rates
        stage_conversion_rates = self._calculate_stage_conversions(recent_conversations)

        # Identify top loss reasons
        lost_deals = [c for c in recent_conversations if c.get("outcome") == "closed_lost"]
        loss_reasons = {}
        for deal in lost_deals:
            reason = deal.get("loss_reason", "unknown")
            loss_reasons[reason] = loss_reasons.get(reason, 0) + 1

        top_loss_reasons = sorted(loss_reasons.items(), key=lambda x: x[1], reverse=True)[:5]

        # Analyze performance trends
        performance_trends = self._calculate_performance_trends(conversations, time_period)

        # Generate recommendations
        recommendations = self._generate_conversion_recommendations(
            conversion_rate, stage_conversion_rates, top_loss_reasons, performance_trends
        )

        # Calculate confidence score
        confidence_score = self._calculate_analysis_confidence(total_opportunities, len(sales_cycles))

        analysis = ConversionAnalysis(
            analysis_id=f"CA_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            time_period=time_period,
            total_opportunities=total_opportunities,
            conversion_rate=conversion_rate,
            average_deal_size=average_deal_size,
            average_sales_cycle=average_sales_cycle,
            stage_conversion_rates=stage_conversion_rates,
            top_loss_reasons=top_loss_reasons,
            performance_trends=performance_trends,
            recommendations=recommendations,
            confidence_score=confidence_score
        )

        return {
            "analysis": self._conversion_analysis_to_dict(analysis),
            "summary": {
                "conversion_rate": f"{conversion_rate:.1%}",
                "avg_deal_size": f"${average_deal_size:,.0f}",
                "avg_sales_cycle": f"{average_sales_cycle} days",
                "total_opportunities": total_opportunities
            },
            "timestamp": datetime.now().isoformat()
        }

    async def _analyze_objections(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze objection patterns and handling effectiveness."""
        objection_data = input_data.get("objection_data", [])
        category_filter = input_data.get("category_filter", None)

        objection_analyses = {}

        # Group objections by category
        for objection_entry in objection_data:
            category = objection_entry.get("category", "unknown")

            # Apply filter if specified
            if category_filter and category != category_filter:
                continue

            if category not in objection_analyses:
                objection_analyses[category] = {
                    "objections": [],
                    "resolutions": [],
                    "resolution_times": [],
                    "conversion_outcomes": []
                }

            objection_analyses[category]["objections"].append(objection_entry)

            if objection_entry.get("resolution_time"):
                objection_analyses[category]["resolution_times"].append(
                    objection_entry["resolution_time"]
                )

            if objection_entry.get("resolved"):
                objection_analyses[category]["resolutions"].append(objection_entry)

            if objection_entry.get("final_outcome"):
                objection_analyses[category]["conversion_outcomes"].append(
                    objection_entry["final_outcome"]
                )

        # Analyze each category
        results = []
        for category, data in objection_analyses.items():
            analysis = self._analyze_objection_category(category, data)
            results.append(analysis)

        # Generate overall insights
        overall_insights = self._generate_objection_insights(results)

        return {
            "objection_analyses": [self._objection_analysis_to_dict(analysis) for analysis in results],
            "overall_insights": overall_insights,
            "categories_analyzed": len(results),
            "total_objections": len(objection_data),
            "timestamp": datetime.now().isoformat()
        }

    async def _generate_performance_report(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive sales performance report."""
        advisor_name = input_data.get("advisor_name")
        team_scope = input_data.get("team_scope", False)
        time_period = input_data.get("time_period", "30_days")
        comparison_period = input_data.get("comparison_period", "previous_period")

        # Collect performance data
        if team_scope:
            performance_data = await self._collect_team_performance_data(time_period)
            comparison_data = await self._collect_team_performance_data(comparison_period)
        else:
            performance_data = await self._collect_advisor_performance_data(advisor_name, time_period)
            comparison_data = await self._collect_advisor_performance_data(advisor_name, comparison_period)

        # Calculate performance metrics
        current_metrics = self._calculate_performance_metrics(performance_data)
        comparison_metrics = self._calculate_performance_metrics(comparison_data)

        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(current_metrics, comparison_metrics)

        # Identify performance gaps and opportunities
        gaps_and_opportunities = self._identify_performance_gaps(current_metrics, comparison_metrics)

        # Generate recommendations
        recommendations = self._generate_performance_recommendations(
            current_metrics, comparative_analysis, gaps_and_opportunities
        )

        return {
            "performance_report": {
                "scope": "team" if team_scope else advisor_name,
                "time_period": time_period,
                "current_metrics": current_metrics,
                "comparative_analysis": comparative_analysis,
                "gaps_and_opportunities": gaps_and_opportunities,
                "recommendations": recommendations,
                "report_generated": datetime.now().isoformat()
            },
            "summary": {
                "overall_performance": self._calculate_overall_performance_score(current_metrics),
                "trend": "improving" if comparative_analysis.get("overall_trend", 0) > 0 else "declining",
                "key_metric": self._identify_key_metric(current_metrics, comparative_analysis)
            },
            "timestamp": datetime.now().isoformat()
        }

    async def _forecast_revenue(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future revenue based on pipeline analysis."""
        pipeline_data = input_data.get("pipeline_data", [])
        forecast_period = input_data.get("forecast_period", "quarterly")
        confidence_level = input_data.get("confidence_level", 0.8)

        # Analyze current pipeline
        pipeline_analysis = self._analyze_pipeline(pipeline_data)

        # Calculate weighted pipeline value
        stage_weights = {
            "discovery": 0.1,
            "needs_analysis": 0.2,
            "presentation": 0.4,
            "proposal": 0.7,
            "negotiation": 0.9
        }

        weighted_pipeline = 0
        for opportunity in pipeline_data:
            stage = opportunity.get("stage", "discovery")
            value = opportunity.get("potential_value", 0)
            weight = stage_weights.get(stage, 0.1)
            weighted_pipeline += value * weight

        # Apply historical conversion rates
        historical_conversion = self._get_historical_conversion_rate(forecast_period)
        predicted_revenue = weighted_pipeline * historical_conversion

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            predicted_revenue, pipeline_analysis["volatility"], confidence_level
        )

        # Identify key assumptions and risk factors
        key_assumptions = self._identify_forecast_assumptions(pipeline_data, historical_conversion)
        risk_factors = self._identify_forecast_risks(pipeline_analysis)

        # Generate recommended actions
        recommended_actions = self._generate_forecast_actions(
            pipeline_analysis, predicted_revenue, risk_factors
        )

        # Calculate pipeline health score
        pipeline_health_score = self._calculate_pipeline_health(pipeline_analysis)

        prediction = RevenuePredicition(
            prediction_id=f"RP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            forecast_period=forecast_period,
            predicted_revenue=predicted_revenue,
            confidence_interval=confidence_interval,
            key_assumptions=key_assumptions,
            risk_factors=risk_factors,
            recommended_actions=recommended_actions,
            pipeline_health_score=pipeline_health_score
        )

        return {
            "revenue_prediction": self._revenue_prediction_to_dict(prediction),
            "pipeline_analysis": pipeline_analysis,
            "summary": {
                "predicted_revenue": f"${predicted_revenue:,.0f}",
                "confidence_range": f"${confidence_interval[0]:,.0f} - ${confidence_interval[1]:,.0f}",
                "pipeline_health": f"{pipeline_health_score:.1%}",
                "forecast_period": forecast_period
            },
            "timestamp": datetime.now().isoformat()
        }

    async def _optimize_sales_process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend sales process improvements."""
        process_data = input_data.get("process_data", {})
        performance_gaps = input_data.get("performance_gaps", [])
        optimization_goals = input_data.get("optimization_goals", [])

        # Analyze current process efficiency
        process_efficiency = self._analyze_process_efficiency(process_data)

        # Identify bottlenecks
        bottlenecks = self._identify_process_bottlenecks(process_data, performance_gaps)

        # Generate optimization recommendations
        optimizations = []

        for bottleneck in bottlenecks:
            recommendations = self._generate_bottleneck_solutions(bottleneck, optimization_goals)
            optimizations.extend(recommendations)

        # Prioritize recommendations
        prioritized_optimizations = self._prioritize_optimizations(optimizations)

        # Create implementation plan
        implementation_plan = self._create_optimization_implementation_plan(prioritized_optimizations)

        # Estimate impact
        estimated_impact = self._estimate_optimization_impact(prioritized_optimizations)

        return {
            "process_optimization": {
                "current_efficiency": process_efficiency,
                "identified_bottlenecks": bottlenecks,
                "optimization_recommendations": prioritized_optimizations,
                "implementation_plan": implementation_plan,
                "estimated_impact": estimated_impact
            },
            "summary": {
                "total_recommendations": len(prioritized_optimizations),
                "high_priority_items": len([o for o in prioritized_optimizations if o.get("priority") == "high"]),
                "estimated_revenue_impact": estimated_impact.get("revenue_improvement", 0),
                "implementation_timeline": implementation_plan.get("total_timeline", "unknown")
            },
            "timestamp": datetime.now().isoformat()
        }

    # Helper methods for calculations and analysis
    def _get_cutoff_date(self, time_period: str) -> datetime:
        """Get cutoff date based on time period."""
        now = datetime.now()

        if time_period == "7_days":
            return now - timedelta(days=7)
        elif time_period == "30_days":
            return now - timedelta(days=30)
        elif time_period == "90_days":
            return now - timedelta(days=90)
        elif time_period == "1_year":
            return now - timedelta(days=365)
        else:
            return now - timedelta(days=30)  # Default to 30 days

    def _calculate_stage_conversions(self, conversations: List[Dict]) -> Dict[str, float]:
        """Calculate conversion rates between sales stages."""
        stage_counts = {}

        for conv in conversations:
            stage = conv.get("final_stage", "unknown")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

        total = sum(stage_counts.values())

        return {
            stage: count / total if total > 0 else 0
            for stage, count in stage_counts.items()
        }

    def _calculate_performance_trends(self, conversations: List[Dict], time_period: str) -> Dict[str, float]:
        """Calculate performance trends over time."""
        # This would implement more sophisticated trend analysis
        # For now, return basic trends
        return {
            "conversion_trend": 0.05,  # 5% improvement
            "activity_trend": 0.02,    # 2% increase in activity
            "deal_size_trend": -0.01   # 1% decrease in deal size
        }

    def _generate_conversion_recommendations(self, conversion_rate: float,
                                           stage_rates: Dict[str, float],
                                           loss_reasons: List[Tuple[str, int]],
                                           trends: Dict[str, float]) -> List[str]:
        """Generate recommendations based on conversion analysis."""
        recommendations = []

        if conversion_rate < 0.2:  # Less than 20% conversion
            recommendations.append("Conversion rate is below industry average. Focus on lead qualification.")

        if loss_reasons:
            top_reason = loss_reasons[0][0]
            recommendations.append(f"Top loss reason is '{top_reason}'. Develop specific strategies to address this objection.")

        if trends.get("conversion_trend", 0) < 0:
            recommendations.append("Conversion trend is declining. Review recent process changes and market conditions.")

        return recommendations

    def _calculate_analysis_confidence(self, sample_size: int, data_points: int) -> float:
        """Calculate confidence score for analysis."""
        # Simple confidence calculation based on sample size
        if sample_size < 10:
            return 0.3
        elif sample_size < 30:
            return 0.6
        elif sample_size < 100:
            return 0.8
        else:
            return 0.95

    def _analyze_objection_category(self, category: str, data: Dict) -> ObjectionAnalysis:
        """Analyze a specific objection category."""
        objections = data["objections"]
        resolutions = data["resolutions"]
        resolution_times = data["resolution_times"]
        outcomes = data["conversion_outcomes"]

        frequency = len(objections)

        avg_resolution_time = statistics.mean(resolution_times) if resolution_times else 0

        resolution_success_rate = len(resolutions) / frequency if frequency > 0 else 0

        # Analyze effective vs ineffective responses
        effective_responses = []
        ineffective_responses = []

        for objection in objections:
            if objection.get("resolved", False):
                response = objection.get("advisor_response", "")
                if response:
                    effective_responses.append(response)
            else:
                response = objection.get("advisor_response", "")
                if response:
                    ineffective_responses.append(response)

        # Calculate impact on conversion
        won_after_objection = len([o for o in outcomes if o == "closed_won"])
        impact_on_conversion = won_after_objection / len(outcomes) if outcomes else 0

        return ObjectionAnalysis(
            objection_category=category,
            frequency=frequency,
            avg_resolution_time=avg_resolution_time,
            resolution_success_rate=resolution_success_rate,
            effective_responses=effective_responses[:3],  # Top 3
            ineffective_responses=ineffective_responses[:3],  # Top 3
            context_patterns=[],  # Would be implemented with NLP
            impact_on_conversion=impact_on_conversion
        )

    # Conversion methods for data structures
    def _conversion_analysis_to_dict(self, analysis: ConversionAnalysis) -> Dict[str, Any]:
        """Convert ConversionAnalysis to dictionary."""
        return {
            "analysis_id": analysis.analysis_id,
            "time_period": analysis.time_period,
            "total_opportunities": analysis.total_opportunities,
            "conversion_rate": analysis.conversion_rate,
            "average_deal_size": analysis.average_deal_size,
            "average_sales_cycle": analysis.average_sales_cycle,
            "stage_conversion_rates": analysis.stage_conversion_rates,
            "top_loss_reasons": analysis.top_loss_reasons,
            "performance_trends": analysis.performance_trends,
            "recommendations": analysis.recommendations,
            "confidence_score": analysis.confidence_score
        }

    def _objection_analysis_to_dict(self, analysis: ObjectionAnalysis) -> Dict[str, Any]:
        """Convert ObjectionAnalysis to dictionary."""
        return {
            "objection_category": analysis.objection_category,
            "frequency": analysis.frequency,
            "avg_resolution_time": analysis.avg_resolution_time,
            "resolution_success_rate": analysis.resolution_success_rate,
            "effective_responses": analysis.effective_responses,
            "ineffective_responses": analysis.ineffective_responses,
            "context_patterns": analysis.context_patterns,
            "impact_on_conversion": analysis.impact_on_conversion
        }

    def _revenue_prediction_to_dict(self, prediction: RevenuePredicition) -> Dict[str, Any]:
        """Convert RevenuePredicition to dictionary."""
        return {
            "prediction_id": prediction.prediction_id,
            "forecast_period": prediction.forecast_period,
            "predicted_revenue": prediction.predicted_revenue,
            "confidence_interval": prediction.confidence_interval,
            "key_assumptions": prediction.key_assumptions,
            "risk_factors": prediction.risk_factors,
            "recommended_actions": prediction.recommended_actions,
            "pipeline_health_score": prediction.pipeline_health_score
        }

    # Additional helper methods
    def _generate_objection_insights(self, results: List[ObjectionAnalysis]) -> List[str]:
        """Generate overall insights from objection analysis results."""
        insights = []

        if results:
            # Most common objection
            categories = [r.objection_category for r in results]
            most_common = max(set(categories), key=categories.count)
            insights.append(f"Most common objection type: {most_common}")

            # Average resolution success rate
            avg_success = sum(r.resolution_success_rate for r in results) / len(results)
            insights.append(f"Average objection resolution rate: {avg_success:.1%}")

            # Timing insights
            avg_time = sum(r.avg_resolution_time for r in results) / len(results)
            insights.append(f"Average resolution time: {avg_time:.1f} minutes")

        return insights

    async def _collect_team_performance_data(self, time_period: str) -> Dict[str, Any]:
        """Collect team performance data for the specified period."""
        # Simulate team data collection
        return {
            "team_conversations": 50,
            "team_conversion_rate": 0.35,
            "team_revenue": 1500000,
            "avg_deal_size": 300000,
            "time_period": time_period
        }

    async def _collect_advisor_performance_data(self, advisor: str, time_period: str) -> Dict[str, Any]:
        """Collect individual advisor performance data."""
        # Simulate advisor data collection
        return {
            "advisor_conversations": 15,
            "advisor_conversion_rate": 0.4,
            "advisor_revenue": 450000,
            "avg_deal_size": 300000,
            "time_period": time_period
        }

    def _calculate_performance_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics from collected data."""
        return {
            "total_conversations": data.get("team_conversations", data.get("advisor_conversations", 0)),
            "conversion_rate": data.get("team_conversion_rate", data.get("advisor_conversion_rate", 0)),
            "total_revenue": data.get("team_revenue", data.get("advisor_revenue", 0)),
            "avg_deal_size": data.get("avg_deal_size", 0),
            "activity_score": 0.8  # Simulated
        }

    def _generate_comparative_analysis(self, current: Dict[str, Any], comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis between time periods."""
        return {
            "conversion_trend": (current.get("conversion_rate", 0) - comparison.get("conversion_rate", 0)),
            "revenue_trend": (current.get("total_revenue", 0) - comparison.get("total_revenue", 0)),
            "activity_trend": (current.get("total_conversations", 0) - comparison.get("total_conversations", 0)),
            "overall_trend": 0.05  # 5% improvement
        }

    def _identify_performance_gaps(self, current: Dict[str, Any], comparison: Dict[str, Any]) -> List[str]:
        """Identify performance gaps and opportunities."""
        gaps = []
        if current.get("conversion_rate", 0) < 0.3:
            gaps.append("Conversion rate below target")
        if current.get("activity_score", 0) < 0.7:
            gaps.append("Activity level below optimal")
        return gaps

    def _generate_performance_recommendations(self, current: Dict[str, Any],
                                            analysis: Dict[str, Any],
                                            gaps: List[str]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        for gap in gaps:
            if "conversion" in gap.lower():
                recommendations.append("Focus on objection handling training")
            elif "activity" in gap.lower():
                recommendations.append("Increase prospecting activities")
        return recommendations

    def _calculate_overall_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        conversion_score = min(metrics.get("conversion_rate", 0) * 5, 1.0)  # Scale to 0-1
        activity_score = metrics.get("activity_score", 0)
        return (conversion_score + activity_score) / 2

    def _identify_key_metric(self, current: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Identify the key metric to focus on."""
        if analysis.get("conversion_trend", 0) < 0:
            return "conversion_rate"
        elif analysis.get("activity_trend", 0) < 0:
            return "activity_level"
        else:
            return "revenue_growth"


# Example usage
async def main():
    """Example usage of Trust Sales Analyst Agent."""
    analyst = TrustSalesAnalystAgent()

    # Test conversion analysis
    analysis_task = SpecialistTask(
        task_type="analyze_conversions",
        description="Analyze sales conversion patterns",
        input_data={
            "conversations": [
                {
                    "date": "2024-01-15",
                    "outcome": "closed_won",
                    "deal_value": 250000,
                    "first_contact_date": "2024-01-01",
                    "final_stage": "closed_won"
                },
                {
                    "date": "2024-01-16",
                    "outcome": "closed_lost",
                    "deal_value": 0,
                    "first_contact_date": "2024-01-02",
                    "final_stage": "objection_handling",
                    "loss_reason": "cost_concerns"
                }
            ],
            "time_period": "30_days"
        }
    )

    await analyst.assign_task(analysis_task)
    result = await analyst.execute_task(analysis_task.task_id)

    print(f"Sales analysis result: {result}")
    print(f"Agent status: {analyst.get_status()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())