"""
Sales Specialist Agent for Stellar Connect
Extends Story 5.2: Sales Specialist Agent Team - Sales Specialist

The Sales Specialist focuses on direct sales execution, coaching recommendations,
conversation optimization, and real-time sales support for estate planning advisors.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import re

from .base_specialist import (
    BaseSpecialist, SpecialistTask, SpecialistExpertise,
    SpecialistCapability, TaskStatus
)


class SalesStage(Enum):
    """Sales conversation stages for tracking and optimization."""
    RAPPORT_BUILDING = "rapport_building"
    DISCOVERY = "discovery"
    NEEDS_ASSESSMENT = "needs_assessment"
    SOLUTION_PRESENTATION = "solution_presentation"
    OBJECTION_HANDLING = "objection_handling"
    TRIAL_CLOSE = "trial_close"
    FINAL_CLOSE = "final_close"
    FOLLOW_UP = "follow_up"


class CoachingArea(Enum):
    """Areas where sales coaching can be provided."""
    QUESTIONING_TECHNIQUE = "questioning_technique"
    ACTIVE_LISTENING = "active_listening"
    OBJECTION_HANDLING = "objection_handling"
    CLOSING_TECHNIQUES = "closing_techniques"
    RAPPORT_BUILDING = "rapport_building"
    PRESENTATION_SKILLS = "presentation_skills"
    TIME_MANAGEMENT = "time_management"
    FOLLOW_UP_DISCIPLINE = "follow_up_discipline"


class SalesMethodology(Enum):
    """Sales methodologies for estate planning."""
    CONSULTATIVE_SELLING = "consultative_selling"
    SOLUTION_SELLING = "solution_selling"
    CHALLENGER_SALE = "challenger_sale"
    SPIN_SELLING = "spin_selling"
    TRUST_BASED_SELLING = "trust_based_selling"


@dataclass
class SalesRecommendation:
    """Real-time sales recommendation during conversations."""
    recommendation_id: str
    recommendation_type: str  # question, response, action, technique
    title: str
    content: str
    timing: str  # immediate, next_opportunity, follow_up
    confidence_score: float
    context: Dict[str, Any]
    expected_outcome: str
    alternative_approaches: List[str]
    generated_at: datetime


@dataclass
class CoachingInsight:
    """Coaching insight based on conversation analysis."""
    insight_id: str
    coaching_area: CoachingArea
    current_performance: str
    improvement_opportunity: str
    specific_recommendations: List[str]
    skill_level_assessment: str  # beginner, intermediate, advanced, expert
    priority_level: str  # low, medium, high, critical
    practice_exercises: List[str]
    success_metrics: List[str]
    timeline_for_improvement: str


@dataclass
class ConversationOptimization:
    """Optimization suggestions for conversation flow."""
    optimization_id: str
    conversation_stage: SalesStage
    current_approach: str
    suggested_improvements: List[str]
    timing_adjustments: List[str]
    language_recommendations: List[str]
    emotional_intelligence_tips: List[str]
    expected_impact: str
    implementation_difficulty: str


@dataclass
class SalesPlaybook:
    """Dynamic sales playbook entry."""
    playbook_id: str
    scenario_name: str
    client_profile: Dict[str, Any]
    situation_triggers: List[str]
    recommended_approach: str
    key_talking_points: List[str]
    common_objections: List[str]
    success_metrics: List[str]
    real_examples: List[str]
    methodology: SalesMethodology
    difficulty_level: str


class SalesSpecialistAgent(BaseSpecialist):
    """
    Sales Specialist Agent - Expert in direct sales execution and coaching.

    Capabilities:
    - Provide real-time sales recommendations during conversations
    - Generate coaching insights and improvement plans
    - Optimize conversation flow and techniques
    - Create dynamic sales playbooks
    - Analyze sales performance and provide targeted feedback
    """

    def __init__(self, sales_knowledge_base: str = "data/sales_knowledge"):
        # Define capabilities
        capabilities = [
            SpecialistCapability(
                name="real_time_sales_support",
                description="Provide real-time recommendations during sales conversations",
                input_types=["conversation_context", "client_profile", "current_stage"],
                output_types=["sales_recommendations", "next_best_actions"]
            ),
            SpecialistCapability(
                name="coaching_analysis",
                description="Analyze performance and provide coaching insights",
                input_types=["conversation_transcript", "performance_data"],
                output_types=["coaching_insights", "improvement_plan"]
            ),
            SpecialistCapability(
                name="conversation_optimization",
                description="Optimize conversation flow and techniques",
                input_types=["conversation_analysis", "outcome_data"],
                output_types=["optimization_recommendations", "technique_improvements"]
            ),
            SpecialistCapability(
                name="playbook_generation",
                description="Create dynamic sales playbooks for specific scenarios",
                input_types=["scenario_data", "client_characteristics"],
                output_types=["custom_playbook", "scenario_guidance"]
            ),
            SpecialistCapability(
                name="performance_coaching",
                description="Provide targeted performance coaching and skill development",
                input_types=["performance_metrics", "skill_assessments"],
                output_types=["coaching_plan", "skill_development_roadmap"]
            )
        ]

        super().__init__(
            name="Sales Specialist",
            expertise=SpecialistExpertise.SALES_ENABLEMENT,
            description="Expert in direct sales execution, real-time coaching, and performance optimization for estate planning",
            capabilities=capabilities,
            max_concurrent_tasks=4
        )

        self.sales_knowledge_base = sales_knowledge_base

        # Sales frameworks and methodologies
        self.sales_frameworks = {
            SalesMethodology.CONSULTATIVE_SELLING: {
                "stages": ["rapport", "discovery", "solution_design", "presentation", "close"],
                "key_principles": ["listen_first", "understand_needs", "solve_problems", "build_trust"],
                "techniques": ["open_questions", "active_listening", "needs_analysis", "solution_mapping"]
            },
            SalesMethodology.SPIN_SELLING: {
                "stages": ["situation", "problem", "implication", "need_payoff"],
                "key_principles": ["question_progression", "pain_identification", "value_demonstration"],
                "techniques": ["situation_questions", "problem_questions", "implication_questions", "need_payoff_questions"]
            },
            SalesMethodology.TRUST_BASED_SELLING: {
                "stages": ["credibility", "reliability", "intimacy", "self_orientation"],
                "key_principles": ["expertise_demonstration", "consistent_delivery", "safe_space", "client_focus"],
                "techniques": ["credential_establishment", "promise_keeping", "confidentiality", "client_advocacy"]
            }
        }

        # Common objections and response frameworks
        self.objection_frameworks = {
            "cost": {
                "acknowledge": "I understand cost is a significant consideration...",
                "isolate": "If we could address the cost concern, would you move forward?",
                "respond": ["value_demonstration", "roi_calculation", "cost_of_inaction"],
                "close": "Given the value we've discussed, shall we proceed?"
            },
            "timing": {
                "acknowledge": "Timing is important for any major decision...",
                "isolate": "Besides timing, are there other concerns?",
                "respond": ["urgency_creation", "cost_of_delay", "limited_availability"],
                "close": "What would need to happen to move forward this quarter?"
            },
            "complexity": {
                "acknowledge": "Estate planning can seem complex initially...",
                "isolate": "If we simplify the process, would that address your concern?",
                "respond": ["process_breakdown", "expert_guidance", "step_by_step"],
                "close": "Let me show you how simple we make this process..."
            }
        }

        self.logger = logging.getLogger(f"{__name__}.SalesSpecialistAgent")

    def get_task_types(self) -> List[str]:
        """Return list of task types this specialist can handle."""
        return [
            "provide_sales_recommendations",
            "analyze_conversation_performance",
            "generate_coaching_insights",
            "optimize_conversation_flow",
            "create_sales_playbook",
            "assess_sales_skills",
            "recommend_closing_strategy",
            "handle_objection_coaching"
        ]

    async def validate_input(self, task: SpecialistTask) -> Tuple[bool, Optional[str]]:
        """Validate task input data."""
        task_type = task.task_type
        input_data = task.input_data

        try:
            if task_type == "provide_sales_recommendations":
                required_fields = ["conversation_context", "current_stage"]
                for field in required_fields:
                    if field not in input_data:
                        return False, f"Missing required field: {field}"

            elif task_type == "analyze_conversation_performance":
                if "conversation_transcript" not in input_data:
                    return False, "Missing required field: conversation_transcript"

            elif task_type == "generate_coaching_insights":
                if "performance_data" not in input_data:
                    return False, "Missing required field: performance_data"

            elif task_type == "create_sales_playbook":
                if "scenario_description" not in input_data:
                    return False, "Missing required field: scenario_description"

            elif task_type == "assess_sales_skills":
                if "conversation_examples" not in input_data and "performance_metrics" not in input_data:
                    return False, "Must provide either conversation_examples or performance_metrics"

            return True, None

        except Exception as e:
            return False, str(e)

    async def process_task(self, task: SpecialistTask) -> Dict[str, Any]:
        """Process a specific task based on task type."""
        task_type = task.task_type
        input_data = task.input_data

        try:
            if task_type == "provide_sales_recommendations":
                return await self._provide_sales_recommendations(input_data)

            elif task_type == "analyze_conversation_performance":
                return await self._analyze_conversation_performance(input_data)

            elif task_type == "generate_coaching_insights":
                return await self._generate_coaching_insights(input_data)

            elif task_type == "optimize_conversation_flow":
                return await self._optimize_conversation_flow(input_data)

            elif task_type == "create_sales_playbook":
                return await self._create_sales_playbook(input_data)

            elif task_type == "assess_sales_skills":
                return await self._assess_sales_skills(input_data)

            elif task_type == "recommend_closing_strategy":
                return await self._recommend_closing_strategy(input_data)

            elif task_type == "handle_objection_coaching":
                return await self._handle_objection_coaching(input_data)

            else:
                raise ValueError(f"Unsupported task type: {task_type}")

        except Exception as e:
            self.logger.error(f"Error processing task {task.task_id}: {str(e)}")
            raise

    async def _provide_sales_recommendations(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Provide real-time sales recommendations during conversations."""
        conversation_context = input_data.get("conversation_context", {})
        current_stage = input_data.get("current_stage", "discovery")
        client_profile = input_data.get("client_profile", {})
        time_constraint = input_data.get("time_constraint", "normal")

        recommendations = []

        # Analyze current conversation context
        context_analysis = self._analyze_conversation_context(conversation_context, current_stage)

        # Generate stage-specific recommendations
        stage_recommendations = self._generate_stage_recommendations(
            current_stage, context_analysis, client_profile
        )
        recommendations.extend(stage_recommendations)

        # Generate technique-specific recommendations
        technique_recommendations = self._generate_technique_recommendations(
            conversation_context, context_analysis
        )
        recommendations.extend(technique_recommendations)

        # Generate timing-sensitive recommendations
        if time_constraint == "limited":
            urgent_recommendations = self._generate_urgent_recommendations(
                current_stage, context_analysis
            )
            recommendations.extend(urgent_recommendations)

        # Prioritize recommendations
        prioritized_recommendations = self._prioritize_recommendations(recommendations)

        # Generate next best actions
        next_actions = self._generate_next_best_actions(
            current_stage, prioritized_recommendations[:3]
        )

        return {
            "sales_recommendations": {
                "immediate_recommendations": [self._recommendation_to_dict(rec) for rec in prioritized_recommendations[:3]],
                "additional_recommendations": [self._recommendation_to_dict(rec) for rec in prioritized_recommendations[3:]],
                "next_best_actions": next_actions,
                "context_analysis": context_analysis,
                "conversation_stage": current_stage,
                "recommendation_count": len(recommendations)
            },
            "summary": {
                "priority_recommendations": len([r for r in recommendations if r.confidence_score > 0.8]),
                "immediate_actions": len([r for r in recommendations if r.timing == "immediate"]),
                "stage_optimization": context_analysis.get("stage_optimization_score", 0),
                "conversation_health": context_analysis.get("conversation_health", "good")
            },
            "timestamp": datetime.now().isoformat()
        }

    async def _analyze_conversation_performance(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversation performance and provide feedback."""
        conversation_transcript = input_data.get("conversation_transcript", "")
        outcome = input_data.get("outcome", "unknown")
        client_feedback = input_data.get("client_feedback", {})
        analysis_focus = input_data.get("analysis_focus", ["overall", "techniques", "opportunities"])

        performance_analysis = {}

        # Overall conversation analysis
        if "overall" in analysis_focus:
            performance_analysis["overall"] = self._analyze_overall_performance(
                conversation_transcript, outcome
            )

        # Technique analysis
        if "techniques" in analysis_focus:
            performance_analysis["techniques"] = self._analyze_sales_techniques(
                conversation_transcript
            )

        # Missed opportunities analysis
        if "opportunities" in analysis_focus:
            performance_analysis["missed_opportunities"] = self._identify_missed_opportunities(
                conversation_transcript, outcome
            )

        # Stage progression analysis
        performance_analysis["stage_progression"] = self._analyze_stage_progression(
            conversation_transcript
        )

        # Generate performance score
        performance_score = self._calculate_performance_score(performance_analysis)

        # Generate improvement recommendations
        improvement_recommendations = self._generate_performance_improvements(
            performance_analysis, outcome
        )

        return {
            "performance_analysis": {
                "performance_score": performance_score,
                "detailed_analysis": performance_analysis,
                "improvement_recommendations": improvement_recommendations,
                "conversation_outcome": outcome,
                "analysis_date": datetime.now().isoformat()
            },
            "summary": {
                "overall_score": performance_score.get("overall", 0),
                "strengths": len(performance_analysis.get("overall", {}).get("strengths", [])),
                "improvement_areas": len(improvement_recommendations),
                "technique_effectiveness": performance_score.get("techniques", 0)
            },
            "timestamp": datetime.now().isoformat()
        }

    async def _generate_coaching_insights(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate coaching insights and improvement plans."""
        performance_data = input_data.get("performance_data", {})
        skill_focus_areas = input_data.get("skill_focus_areas", [])
        coaching_goals = input_data.get("coaching_goals", [])
        timeline = input_data.get("timeline", "30_days")

        coaching_insights = []

        # Analyze each coaching area
        for area in CoachingArea:
            if skill_focus_areas and area not in skill_focus_areas:
                continue

            insight = self._analyze_coaching_area(area, performance_data)
            if insight:
                coaching_insights.append(insight)

        # Prioritize coaching areas
        prioritized_insights = self._prioritize_coaching_insights(coaching_insights, coaching_goals)

        # Create development plan
        development_plan = self._create_development_plan(prioritized_insights, timeline)

        # Generate practice recommendations
        practice_recommendations = self._generate_practice_recommendations(prioritized_insights)

        return {
            "coaching_insights": {
                "prioritized_insights": [self._coaching_insight_to_dict(insight) for insight in prioritized_insights],
                "development_plan": development_plan,
                "practice_recommendations": practice_recommendations,
                "coaching_timeline": timeline,
                "focus_areas": len(prioritized_insights)
            },
            "summary": {
                "critical_areas": len([i for i in coaching_insights if i.priority_level == "critical"]),
                "high_priority_areas": len([i for i in coaching_insights if i.priority_level == "high"]),
                "skill_level_distribution": self._calculate_skill_distribution(coaching_insights),
                "estimated_improvement_timeline": development_plan.get("total_timeline", timeline)
            },
            "timestamp": datetime.now().isoformat()
        }

    async def _create_sales_playbook(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create dynamic sales playbook for specific scenarios."""
        scenario_description = input_data.get("scenario_description", "")
        client_characteristics = input_data.get("client_characteristics", {})
        methodology_preference = input_data.get("methodology", SalesMethodology.CONSULTATIVE_SELLING)
        complexity_level = input_data.get("complexity_level", "intermediate")

        # Analyze scenario requirements
        scenario_analysis = self._analyze_scenario_requirements(
            scenario_description, client_characteristics
        )

        # Generate playbook content
        playbook = SalesPlaybook(
            playbook_id=f"SP_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            scenario_name=scenario_analysis["scenario_name"],
            client_profile=client_characteristics,
            situation_triggers=scenario_analysis["triggers"],
            recommended_approach=self._generate_approach_strategy(
                scenario_analysis, methodology_preference
            ),
            key_talking_points=self._generate_talking_points(
                scenario_analysis, client_characteristics
            ),
            common_objections=self._identify_likely_objections(
                scenario_analysis, client_characteristics
            ),
            success_metrics=self._define_success_metrics(scenario_analysis),
            real_examples=self._find_relevant_examples(scenario_analysis),
            methodology=methodology_preference,
            difficulty_level=complexity_level
        )

        # Generate implementation guidance
        implementation_guidance = self._generate_implementation_guidance(playbook)

        # Create practice scenarios
        practice_scenarios = self._create_practice_scenarios(playbook)

        return {
            "sales_playbook": {
                "playbook": self._playbook_to_dict(playbook),
                "implementation_guidance": implementation_guidance,
                "practice_scenarios": practice_scenarios,
                "methodology_framework": self.sales_frameworks[methodology_preference],
                "customization_level": "high"
            },
            "summary": {
                "scenario_complexity": complexity_level,
                "talking_points": len(playbook.key_talking_points),
                "objections_covered": len(playbook.common_objections),
                "practice_scenarios": len(practice_scenarios),
                "methodology": methodology_preference.value
            },
            "timestamp": datetime.now().isoformat()
        }

    # Helper methods for analysis and generation
    def _analyze_conversation_context(self, context: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """Analyze current conversation context for recommendations."""
        return {
            "stage_alignment": self._assess_stage_alignment(context, stage),
            "client_engagement": self._assess_client_engagement(context),
            "conversation_health": self._assess_conversation_health(context),
            "stage_optimization_score": self._calculate_stage_optimization(context, stage),
            "next_stage_readiness": self._assess_next_stage_readiness(context, stage)
        }

    def _generate_stage_recommendations(self, stage: str, analysis: Dict[str, Any],
                                      client_profile: Dict[str, Any]) -> List[SalesRecommendation]:
        """Generate recommendations specific to current sales stage."""
        recommendations = []

        if stage == "discovery":
            recommendations.extend(self._generate_discovery_recommendations(analysis, client_profile))
        elif stage == "needs_assessment":
            recommendations.extend(self._generate_needs_recommendations(analysis, client_profile))
        elif stage == "solution_presentation":
            recommendations.extend(self._generate_presentation_recommendations(analysis, client_profile))
        elif stage == "objection_handling":
            recommendations.extend(self._generate_objection_recommendations(analysis, client_profile))
        elif stage == "closing":
            recommendations.extend(self._generate_closing_recommendations(analysis, client_profile))

        return recommendations

    def _generate_discovery_recommendations(self, analysis: Dict[str, Any],
                                          client_profile: Dict[str, Any]) -> List[SalesRecommendation]:
        """Generate discovery stage recommendations."""
        recommendations = []

        # Example discovery recommendations
        if analysis.get("client_engagement", "medium") == "low":
            recommendations.append(SalesRecommendation(
                recommendation_id=f"DR_{datetime.now().strftime('%H%M%S')}_001",
                recommendation_type="question",
                title="Re-engage with Personal Question",
                content="What initially brought you to consider estate planning at this time?",
                timing="immediate",
                confidence_score=0.85,
                context={"stage": "discovery", "engagement": "low"},
                expected_outcome="Increase client engagement and openness",
                alternative_approaches=["Ask about family situation", "Inquire about recent life changes"],
                generated_at=datetime.now()
            ))

        if "family_structure" not in client_profile:
            recommendations.append(SalesRecommendation(
                recommendation_id=f"DR_{datetime.now().strftime('%H%M%S')}_002",
                recommendation_type="question",
                title="Explore Family Structure",
                content="Tell me about your family - who are the important people in your life?",
                timing="next_opportunity",
                confidence_score=0.92,
                context={"stage": "discovery", "missing_info": "family_structure"},
                expected_outcome="Understand beneficiary structure and dynamics",
                alternative_approaches=["Ask about children specifically", "Inquire about spouse/partner"],
                generated_at=datetime.now()
            ))

        return recommendations

    def _analyze_overall_performance(self, transcript: str, outcome: str) -> Dict[str, Any]:
        """Analyze overall conversation performance."""
        return {
            "conversation_flow": self._assess_conversation_flow(transcript),
            "client_engagement_level": self._measure_engagement_level(transcript),
            "information_gathering": self._assess_information_gathering(transcript),
            "value_demonstration": self._assess_value_demonstration(transcript),
            "outcome_alignment": self._assess_outcome_alignment(transcript, outcome),
            "strengths": self._identify_conversation_strengths(transcript),
            "weaknesses": self._identify_conversation_weaknesses(transcript)
        }

    def _analyze_coaching_area(self, area: CoachingArea, performance_data: Dict[str, Any]) -> Optional[CoachingInsight]:
        """Analyze specific coaching area and generate insights."""
        # This would implement detailed analysis for each coaching area
        # For now, return a sample insight

        if area == CoachingArea.QUESTIONING_TECHNIQUE:
            return CoachingInsight(
                insight_id=f"CI_{area.value}_{datetime.now().strftime('%H%M%S')}",
                coaching_area=area,
                current_performance="Uses mostly closed-ended questions, limiting client sharing",
                improvement_opportunity="Increase use of open-ended questions to encourage deeper responses",
                specific_recommendations=[
                    "Replace 'Do you...' questions with 'What...', 'How...', 'Why...'",
                    "Practice the 3-second pause after asking questions",
                    "Use follow-up questions to go deeper: 'Tell me more about that...'"
                ],
                skill_level_assessment="intermediate",
                priority_level="high",
                practice_exercises=[
                    "Record practice conversations and count open vs. closed questions",
                    "Role-play scenarios focusing only on questioning technique",
                    "Study SPIN selling question framework"
                ],
                success_metrics=[
                    "80% open-ended questions in discovery phase",
                    "Average 3+ follow-up questions per topic",
                    "Client talk time >60% of conversation"
                ],
                timeline_for_improvement="2-3 weeks with focused practice"
            )

        return None

    # Data conversion methods
    def _recommendation_to_dict(self, recommendation: SalesRecommendation) -> Dict[str, Any]:
        """Convert SalesRecommendation to dictionary."""
        return {
            "recommendation_id": recommendation.recommendation_id,
            "recommendation_type": recommendation.recommendation_type,
            "title": recommendation.title,
            "content": recommendation.content,
            "timing": recommendation.timing,
            "confidence_score": recommendation.confidence_score,
            "context": recommendation.context,
            "expected_outcome": recommendation.expected_outcome,
            "alternative_approaches": recommendation.alternative_approaches,
            "generated_at": recommendation.generated_at.isoformat()
        }

    def _coaching_insight_to_dict(self, insight: CoachingInsight) -> Dict[str, Any]:
        """Convert CoachingInsight to dictionary."""
        return {
            "insight_id": insight.insight_id,
            "coaching_area": insight.coaching_area.value,
            "current_performance": insight.current_performance,
            "improvement_opportunity": insight.improvement_opportunity,
            "specific_recommendations": insight.specific_recommendations,
            "skill_level_assessment": insight.skill_level_assessment,
            "priority_level": insight.priority_level,
            "practice_exercises": insight.practice_exercises,
            "success_metrics": insight.success_metrics,
            "timeline_for_improvement": insight.timeline_for_improvement
        }

    def _playbook_to_dict(self, playbook: SalesPlaybook) -> Dict[str, Any]:
        """Convert SalesPlaybook to dictionary."""
        return {
            "playbook_id": playbook.playbook_id,
            "scenario_name": playbook.scenario_name,
            "client_profile": playbook.client_profile,
            "situation_triggers": playbook.situation_triggers,
            "recommended_approach": playbook.recommended_approach,
            "key_talking_points": playbook.key_talking_points,
            "common_objections": playbook.common_objections,
            "success_metrics": playbook.success_metrics,
            "real_examples": playbook.real_examples,
            "methodology": playbook.methodology.value,
            "difficulty_level": playbook.difficulty_level
        }

    # Additional helper methods
    def _assess_stage_alignment(self, context: Dict[str, Any], stage: str) -> float:
        """Assess how well the conversation aligns with the current stage."""
        duration = context.get("duration_minutes", 0)

        # Stage-specific duration expectations
        stage_durations = {
            "discovery": (10, 25),
            "needs_assessment": (15, 30),
            "presentation": (20, 40),
            "objection_handling": (5, 20),
            "closing": (5, 15)
        }

        expected_range = stage_durations.get(stage, (10, 30))
        if expected_range[0] <= duration <= expected_range[1]:
            return 0.9
        else:
            return 0.6

    def _assess_client_engagement(self, context: Dict[str, Any]) -> str:
        """Assess client engagement level."""
        responses = context.get("client_responses", [])
        if "engaged" in responses or "detailed" in responses:
            return "high"
        elif "short" in responses or "hesitant" in responses:
            return "low"
        else:
            return "medium"

    def _assess_conversation_health(self, context: Dict[str, Any]) -> str:
        """Assess overall conversation health."""
        engagement = self._assess_client_engagement(context)
        duration = context.get("duration_minutes", 0)

        if engagement == "high" and duration > 15:
            return "excellent"
        elif engagement == "medium" and duration > 10:
            return "good"
        else:
            return "needs_improvement"

    def _calculate_stage_optimization(self, context: Dict[str, Any], stage: str) -> float:
        """Calculate stage optimization score."""
        alignment = self._assess_stage_alignment(context, stage)
        engagement = 0.8 if self._assess_client_engagement(context) == "high" else 0.5
        return (alignment + engagement) / 2

    def _assess_next_stage_readiness(self, context: Dict[str, Any], current_stage: str) -> bool:
        """Assess if ready to move to next stage."""
        info_gathered = context.get("information_gathered", [])
        engagement = self._assess_client_engagement(context)

        stage_requirements = {
            "discovery": ["basic_demographics", "family_structure"],
            "needs_assessment": ["primary_concerns", "goals"],
            "presentation": ["solution_fit", "value_alignment"],
            "objection_handling": ["objection_addressed"],
            "closing": ["buying_signals"]
        }

        required = stage_requirements.get(current_stage, [])
        has_required = len(set(info_gathered) & set(required)) >= len(required) * 0.7

        return has_required and engagement != "low"

    def _prioritize_recommendations(self, recommendations: List[SalesRecommendation]) -> List[SalesRecommendation]:
        """Prioritize recommendations by timing and confidence."""
        def priority_score(rec):
            timing_weights = {"immediate": 3, "next_opportunity": 2, "follow_up": 1}
            timing_score = timing_weights.get(rec.timing, 1)
            return timing_score * rec.confidence_score

        return sorted(recommendations, key=priority_score, reverse=True)

    def _generate_next_best_actions(self, stage: str, recommendations: List[SalesRecommendation]) -> List[str]:
        """Generate next best actions based on stage and recommendations."""
        actions = []

        for rec in recommendations[:2]:  # Top 2 recommendations
            if rec.recommendation_type == "question":
                actions.append(f"Ask: {rec.content}")
            elif rec.recommendation_type == "action":
                actions.append(f"Action: {rec.content}")
            else:
                actions.append(rec.content)

        return actions

    def _analyze_overall_performance(self, transcript: str, outcome: str) -> Dict[str, Any]:
        """Analyze overall conversation performance."""
        return {
            "conversation_flow": self._assess_conversation_flow(transcript),
            "client_engagement_level": self._measure_engagement_level(transcript),
            "information_gathering": self._assess_information_gathering(transcript),
            "value_demonstration": self._assess_value_demonstration(transcript),
            "outcome_alignment": self._assess_outcome_alignment(transcript, outcome),
            "strengths": self._identify_conversation_strengths(transcript),
            "weaknesses": self._identify_conversation_weaknesses(transcript)
        }

    def _assess_conversation_flow(self, transcript: str) -> float:
        """Assess conversation flow quality."""
        # Simple heuristic based on transcript structure
        return 0.8 if len(transcript) > 500 else 0.6

    def _measure_engagement_level(self, transcript: str) -> float:
        """Measure client engagement level from transcript."""
        client_responses = transcript.count("Client:")
        advisor_responses = transcript.count("Advisor:")

        if advisor_responses == 0:
            return 0.5

        ratio = client_responses / advisor_responses
        return min(ratio * 0.5 + 0.3, 1.0)  # Scale to 0.3-1.0

    def _assess_information_gathering(self, transcript: str) -> float:
        """Assess information gathering effectiveness."""
        question_indicators = ["?", "tell me", "what", "how", "why"]
        question_count = sum(transcript.lower().count(indicator) for indicator in question_indicators)
        return min(question_count * 0.1, 1.0)

    def _assess_value_demonstration(self, transcript: str) -> float:
        """Assess value demonstration in conversation."""
        value_indicators = ["benefit", "value", "help", "protect", "save", "advantage"]
        value_count = sum(transcript.lower().count(indicator) for indicator in value_indicators)
        return min(value_count * 0.1, 1.0)

    def _assess_outcome_alignment(self, transcript: str, outcome: str) -> float:
        """Assess alignment between conversation and outcome."""
        if outcome == "closed_won":
            return 0.9 if "next step" in transcript.lower() else 0.7
        elif outcome == "closed_lost":
            return 0.6
        else:
            return 0.5

    def _identify_conversation_strengths(self, transcript: str) -> List[str]:
        """Identify conversation strengths."""
        strengths = []
        if "thank you" in transcript.lower():
            strengths.append("Good rapport building")
        if transcript.count("?") > 5:
            strengths.append("Strong questioning technique")
        if "understand" in transcript.lower():
            strengths.append("Active listening demonstrated")
        return strengths

    def _identify_conversation_weaknesses(self, transcript: str) -> List[str]:
        """Identify conversation weaknesses."""
        weaknesses = []
        if transcript.count("?") < 3:
            weaknesses.append("Limited questioning")
        if "objection" in transcript.lower() and "address" not in transcript.lower():
            weaknesses.append("Objections not properly addressed")
        return weaknesses

    def _generate_technique_recommendations(self, conversation_context: Dict[str, Any],
                                          analysis: Dict[str, Any]) -> List['SalesRecommendation']:
        """Generate technique-specific recommendations."""
        recommendations = []

        # Check for questioning technique issues
        if analysis.get("questioning_score", 0.7) < 0.6:
            recommendations.append(SalesRecommendation(
                recommendation_id=f"tech_rec_{datetime.now().timestamp()}",
                recommendation_type="technique",
                title="Improve Questioning Technique",
                content="Use more open-ended questions to gather deeper insights",
                timing="immediate",
                confidence_score=0.8,
                context={"analysis": analysis, "conversation_context": conversation_context},
                expected_outcome="improved_discovery",
                alternative_approaches=["Use probing questions", "Try reflective questions"],
                generated_at=datetime.now()
            ))

        # Check for listening skills
        if conversation_context.get("client_responses") == ["short", "hesitant"]:
            recommendations.append(SalesRecommendation(
                recommendation_id=f"listen_rec_{datetime.now().timestamp()}",
                recommendation_type="technique",
                title="Enhance Active Listening",
                content="Demonstrate active listening with reflective statements",
                timing="immediate",
                confidence_score=0.85,
                context={"client_responses": conversation_context.get("client_responses")},
                expected_outcome="improved_engagement",
                alternative_approaches=["Paraphrase client statements", "Ask clarifying questions"],
                generated_at=datetime.now()
            ))

        return recommendations

    def _generate_urgent_recommendations(self, current_stage: str,
                                       analysis: Dict[str, Any]) -> List['SalesRecommendation']:
        """Generate urgent recommendations for time-constrained situations."""
        recommendations = []

        if current_stage == "discovery" and analysis.get("engagement_level") == "low":
            recommendations.append(SalesRecommendation(
                recommendation_id=f"urgent_rec_{datetime.now().timestamp()}",
                recommendation_type="action",
                title="Pattern Interrupt Required",
                content="Use a pattern interrupt to regain attention",
                timing="immediate",
                confidence_score=0.9,
                context={"stage": current_stage, "engagement_level": analysis.get("engagement_level")},
                expected_outcome="immediate_engagement",
                alternative_approaches=["Change topic", "Ask surprising question", "Share relevant story"],
                generated_at=datetime.now()
            ))

        if current_stage in ["solution_presentation", "trial_close"]:
            recommendations.append(SalesRecommendation(
                recommendation_id=f"priority_rec_{datetime.now().timestamp()}",
                recommendation_type="strategy",
                title="Focus on High-Value Proposition",
                content="Focus on highest value proposition immediately",
                timing="immediate",
                confidence_score=0.85,
                context={"stage": current_stage, "time_constraint": "limited"},
                expected_outcome="increased_urgency",
                alternative_approaches=["Present ROI calculations", "Share success stories", "Highlight key benefits"],
                generated_at=datetime.now()
            ))

        return recommendations

    def _analyze_sales_techniques(self, conversation_transcript: str) -> Dict[str, Any]:
        """Analyze sales techniques used in conversation."""
        analysis = {
            "techniques_identified": [],
            "effectiveness_scores": {},
            "recommendations": [],
            "strengths": [],
            "weaknesses": []
        }

        # Analyze questioning technique
        question_count = conversation_transcript.count("?")
        if question_count > 5:
            analysis["techniques_identified"].append("active_questioning")
            analysis["effectiveness_scores"]["questioning"] = 0.8
            analysis["strengths"].append("Good use of questioning")
        else:
            analysis["weaknesses"].append("Limited questioning technique")
            analysis["effectiveness_scores"]["questioning"] = 0.4

        # Analyze listening skills
        if "understand" in conversation_transcript.lower() or "hear you" in conversation_transcript.lower():
            analysis["techniques_identified"].append("active_listening")
            analysis["effectiveness_scores"]["listening"] = 0.85
            analysis["strengths"].append("Demonstrates active listening")
        else:
            analysis["weaknesses"].append("Limited active listening demonstrated")
            analysis["effectiveness_scores"]["listening"] = 0.5

        # Analyze rapport building
        if "thank you" in conversation_transcript.lower() or "appreciate" in conversation_transcript.lower():
            analysis["techniques_identified"].append("rapport_building")
            analysis["effectiveness_scores"]["rapport"] = 0.7
            analysis["strengths"].append("Good rapport building")

        # Generate technique recommendations
        if analysis["effectiveness_scores"].get("questioning", 0) < 0.6:
            analysis["recommendations"].append("Increase use of open-ended questions")
        if analysis["effectiveness_scores"].get("listening", 0) < 0.6:
            analysis["recommendations"].append("Demonstrate more active listening techniques")

        analysis["overall_technique_score"] = sum(analysis["effectiveness_scores"].values()) / max(len(analysis["effectiveness_scores"]), 1)

        return analysis

    def _identify_missed_opportunities(self, conversation_transcript: str, outcome: str) -> Dict[str, Any]:
        """Identify missed opportunities in sales conversation."""
        missed_opportunities = {
            "opportunities": [],
            "severity_scores": {},
            "recovery_strategies": [],
            "impact_assessment": {}
        }

        # Check for missed discovery opportunities
        if conversation_transcript.count("?") < 5:
            missed_opportunities["opportunities"].append({
                "type": "insufficient_discovery",
                "description": "Limited questioning to understand client needs",
                "severity": "high"
            })
            missed_opportunities["severity_scores"]["discovery"] = 0.8
            missed_opportunities["recovery_strategies"].append("Increase discovery questions in future conversations")

        # Check for missed objection handling
        if "objection" in conversation_transcript.lower() and "address" not in conversation_transcript.lower():
            missed_opportunities["opportunities"].append({
                "type": "unaddressed_objections",
                "description": "Client objections not properly addressed",
                "severity": "critical"
            })
            missed_opportunities["severity_scores"]["objection_handling"] = 0.9
            missed_opportunities["recovery_strategies"].append("Develop objection handling skills")

        # Check for missed closing opportunities
        if outcome == "no_close" and "next step" not in conversation_transcript.lower():
            missed_opportunities["opportunities"].append({
                "type": "missed_close",
                "description": "No clear next steps established",
                "severity": "medium"
            })
            missed_opportunities["severity_scores"]["closing"] = 0.6
            missed_opportunities["recovery_strategies"].append("Always establish clear next steps")

        # Check for missed rapport building
        if "thank you" not in conversation_transcript.lower():
            missed_opportunities["opportunities"].append({
                "type": "limited_rapport",
                "description": "Limited rapport building detected",
                "severity": "low"
            })
            missed_opportunities["severity_scores"]["rapport"] = 0.3

        # Calculate overall impact
        if missed_opportunities["severity_scores"]:
            avg_severity = sum(missed_opportunities["severity_scores"].values()) / len(missed_opportunities["severity_scores"])
            missed_opportunities["impact_assessment"] = {
                "overall_severity": avg_severity,
                "total_opportunities": len(missed_opportunities["opportunities"]),
                "potential_revenue_impact": avg_severity * 1000  # Example calculation
            }

        return missed_opportunities

    def _analyze_stage_progression(self, conversation_transcript: str) -> Dict[str, Any]:
        """Analyze how well the conversation progressed through sales stages."""
        stage_analysis = {
            "stages_completed": [],
            "current_stage": "unknown",
            "progression_score": 0.0,
            "stage_effectiveness": {},
            "bottlenecks": [],
            "recommendations": []
        }

        # Detect completed stages based on conversation content
        if "hello" in conversation_transcript.lower() or "thank you" in conversation_transcript.lower():
            stage_analysis["stages_completed"].append("rapport_building")
            stage_analysis["stage_effectiveness"]["rapport_building"] = 0.7

        if conversation_transcript.count("?") > 3:
            stage_analysis["stages_completed"].append("discovery")
            stage_analysis["stage_effectiveness"]["discovery"] = 0.8

        if "need" in conversation_transcript.lower() or "problem" in conversation_transcript.lower():
            stage_analysis["stages_completed"].append("needs_assessment")
            stage_analysis["stage_effectiveness"]["needs_assessment"] = 0.6

        if "solution" in conversation_transcript.lower() or "recommend" in conversation_transcript.lower():
            stage_analysis["stages_completed"].append("solution_presentation")
            stage_analysis["stage_effectiveness"]["solution_presentation"] = 0.7

        if "objection" in conversation_transcript.lower():
            stage_analysis["stages_completed"].append("objection_handling")
            stage_analysis["stage_effectiveness"]["objection_handling"] = 0.5

        if "next step" in conversation_transcript.lower() or "follow up" in conversation_transcript.lower():
            stage_analysis["stages_completed"].append("trial_close")
            stage_analysis["stage_effectiveness"]["trial_close"] = 0.8

        # Determine current stage
        if stage_analysis["stages_completed"]:
            stage_analysis["current_stage"] = stage_analysis["stages_completed"][-1]
        else:
            stage_analysis["current_stage"] = "rapport_building"

        # Calculate progression score
        total_stages = 6  # Total possible stages
        completed_stages = len(stage_analysis["stages_completed"])
        stage_analysis["progression_score"] = completed_stages / total_stages

        # Identify bottlenecks
        if "discovery" not in stage_analysis["stages_completed"]:
            stage_analysis["bottlenecks"].append("insufficient_discovery")
            stage_analysis["recommendations"].append("Focus on asking more discovery questions")

        if "objection_handling" in stage_analysis["stages_completed"] and stage_analysis["stage_effectiveness"]["objection_handling"] < 0.6:
            stage_analysis["bottlenecks"].append("poor_objection_handling")
            stage_analysis["recommendations"].append("Improve objection handling techniques")

        if completed_stages < 3:
            stage_analysis["bottlenecks"].append("slow_progression")
            stage_analysis["recommendations"].append("Accelerate conversation through stages")

        return stage_analysis

    def _calculate_performance_score(self, performance_analysis: Dict[str, Any]) -> float:
        """Calculate overall performance score based on analysis results."""
        scores = []
        weights = {}

        # Extract scores from different analysis components
        if "techniques" in performance_analysis:
            technique_score = performance_analysis["techniques"].get("overall_technique_score", 0.5)
            scores.append(technique_score)
            weights["techniques"] = 0.3

        if "missed_opportunities" in performance_analysis:
            missed_opps = performance_analysis["missed_opportunities"]
            if missed_opps.get("impact_assessment"):
                # Invert severity to make it a positive score (lower severity = higher score)
                opportunity_score = 1.0 - missed_opps["impact_assessment"].get("overall_severity", 0.5)
                scores.append(opportunity_score)
                weights["opportunities"] = 0.25

        if "stage_progression" in performance_analysis:
            progression_score = performance_analysis["stage_progression"].get("progression_score", 0.5)
            scores.append(progression_score)
            weights["progression"] = 0.35

        # If no analysis components found, return default score
        if not scores:
            return 0.5

        # Calculate weighted average if weights are available
        if weights and len(weights) == len(scores):
            total_weight = sum(weights.values())
            weighted_score = sum(score * weight for score, weight in zip(scores, weights.values())) / total_weight
            return max(0.0, min(1.0, weighted_score))  # Clamp between 0 and 1
        else:
            # Simple average fallback
            return sum(scores) / len(scores)

    def _generate_performance_improvements(self, performance_analysis: Dict[str, Any], outcome: str) -> List[str]:
        """Generate specific performance improvement recommendations."""
        improvements = []

        # Check technique performance
        if "techniques" in performance_analysis:
            techniques = performance_analysis["techniques"]
            overall_score = techniques.get("overall_technique_score", 0.5)

            if overall_score < 0.6:
                improvements.append("Focus on improving overall sales techniques")

            if "questioning" in techniques.get("effectiveness_scores", {}):
                if techniques["effectiveness_scores"]["questioning"] < 0.6:
                    improvements.append("Practice asking more open-ended questions during discovery")

            if "listening" in techniques.get("effectiveness_scores", {}):
                if techniques["effectiveness_scores"]["listening"] < 0.6:
                    improvements.append("Demonstrate active listening with reflective statements")

        # Check stage progression
        if "stage_progression" in performance_analysis:
            progression = performance_analysis["stage_progression"]
            if progression.get("progression_score", 0) < 0.5:
                improvements.append("Work on advancing conversations through all sales stages")

            if "slow_progression" in progression.get("bottlenecks", []):
                improvements.append("Accelerate conversation flow between stages")

        # Check missed opportunities
        if "missed_opportunities" in performance_analysis:
            missed_opps = performance_analysis["missed_opportunities"]
            for opportunity in missed_opps.get("opportunities", []):
                if opportunity.get("severity") == "critical":
                    improvements.append(f"Address critical issue: {opportunity.get('description')}")
                elif opportunity.get("severity") == "high":
                    improvements.append(f"Improve: {opportunity.get('description')}")

        # Outcome-specific improvements
        if outcome == "no_close":
            improvements.append("Always establish clear next steps before ending conversations")
        elif outcome == "closed_lost":
            improvements.append("Review objection handling techniques and follow-up strategies")

        # Ensure we have at least some general recommendations
        if not improvements:
            improvements.extend([
                "Continue practicing active listening techniques",
                "Focus on building stronger rapport with clients",
                "Ask more discovery questions to understand client needs"
            ])

        return improvements[:5]  # Limit to top 5 recommendations


# Example usage
async def main():
    """Example usage of Sales Specialist Agent."""
    sales_specialist = SalesSpecialistAgent()

    # Test real-time sales recommendations
    recommendation_task = SpecialistTask(
        task_type="provide_sales_recommendations",
        description="Provide sales recommendations for discovery stage",
        input_data={
            "conversation_context": {
                "duration_minutes": 15,
                "client_responses": ["short", "hesitant"],
                "information_gathered": ["basic_demographics"],
                "engagement_level": "low"
            },
            "current_stage": "discovery",
            "client_profile": {
                "age": 65,
                "estimated_net_worth": "high",
                "family_situation": "unknown"
            },
            "time_constraint": "normal"
        }
    )

    await sales_specialist.assign_task(recommendation_task)
    result = await sales_specialist.execute_task(recommendation_task.task_id)

    print(f"Sales specialist recommendations: {result}")
    print(f"Agent status: {sales_specialist.get_status()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())