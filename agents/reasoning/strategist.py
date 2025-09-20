"""
Estate Strategist Agent for Stellar Connect
Phase 2 Week 3: Cognitive Pipeline Components

The Estate Strategist synthesizes insights from multiple analysis results to generate
strategic recommendations with causal reasoning and business impact assessment.
This is the final synthesis step that transforms raw analysis into actionable intelligence.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics
import re

from .auditor import AuditResult, QualityMetrics


class InsightType(Enum):
    """Types of strategic insights."""
    OPPORTUNITY = "opportunity"
    RISK = "risk"
    RECOMMENDATION = "recommendation"
    CAUSAL_RELATIONSHIP = "causal_relationship"
    TREND = "trend"
    SYNTHESIS = "synthesis"
    WARNING = "warning"
    OPTIMIZATION = "optimization"


class ConfidenceLevel(Enum):
    """Confidence levels for insights."""
    VERY_HIGH = "very_high"  # 0.9-1.0
    HIGH = "high"           # 0.75-0.89
    MEDIUM = "medium"       # 0.5-0.74
    LOW = "low"            # 0.25-0.49
    VERY_LOW = "very_low"   # 0.0-0.24


class BusinessImpact(Enum):
    """Business impact levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class CausalRelationship:
    """Represents a causal relationship between factors."""
    cause: str
    effect: str
    strength: float  # 0-1 scale
    evidence: List[str]
    confidence: float
    supporting_analyses: List[str] = field(default_factory=list)


@dataclass
class StrategicInsight:
    """Represents a strategic insight derived from analysis."""
    insight_id: str
    insight_type: InsightType
    title: str
    description: str
    confidence: ConfidenceLevel
    business_impact: BusinessImpact
    supporting_evidence: List[str]
    source_analyses: List[str]
    actionable_recommendations: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    timeline: Optional[str] = None
    estimated_value: Optional[str] = None


@dataclass
class SynthesisResult:
    """Result of strategic synthesis process."""
    synthesis_id: str
    primary_recommendation: str
    strategic_insights: List[StrategicInsight]
    causal_relationships: List[CausalRelationship]
    executive_summary: str
    key_findings: List[str]
    next_actions: List[str]
    risk_assessment: str
    opportunity_analysis: str
    overall_confidence: float
    business_value_score: float
    implementation_complexity: str
    success_probability: float
    generated_at: datetime = field(default_factory=datetime.now)


class EstateStrategist:
    """
    Estate Strategist Agent for strategic synthesis and causal reasoning.

    Responsibilities:
    - Synthesize insights from multiple specialist analyses
    - Identify causal relationships and dependencies
    - Generate strategic recommendations with business impact
    - Assess opportunities and risks holistically
    - Provide executive-level summaries and next actions
    - Calculate business value and success probabilities
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Causal reasoning patterns
        self.causal_indicators = {
            "strong_causation": [
                r"due to", r"because of", r"results in", r"leads to",
                r"causes", r"triggers", r"drives", r"creates"
            ],
            "correlation": [
                r"associated with", r"related to", r"correlated with",
                r"linked to", r"connected to"
            ],
            "temporal": [
                r"after", r"before", r"following", r"preceding",
                r"subsequently", r"then", r"next"
            ]
        }

        # Business impact indicators
        self.impact_keywords = {
            BusinessImpact.CRITICAL: [
                "critical", "essential", "mandatory", "required",
                "compliance", "legal", "regulatory"
            ],
            BusinessImpact.HIGH: [
                "significant", "substantial", "major", "important",
                "revenue", "cost savings", "efficiency"
            ],
            BusinessImpact.MEDIUM: [
                "moderate", "useful", "beneficial", "improvement",
                "enhancement", "optimization"
            ],
            BusinessImpact.LOW: [
                "minor", "small", "incremental", "slight",
                "marginal", "limited"
            ]
        }

        # Value estimation patterns
        self.value_patterns = {
            "monetary": r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|k|K))?',
            "percentage": r'\d+(?:\.\d+)?%',
            "time_savings": r'\d+\s*(?:hours?|days?|weeks?|months?)',
            "efficiency": r'\d+(?:\.\d+)?x\s*(?:faster|improvement|increase)'
        }

    async def synthesize(self, audit_result: AuditResult, results: Dict[str, Any]) -> SynthesisResult:
        """
        Synthesize strategic insights from audited analysis results.

        Args:
            audit_result: Quality-assured audit results
            results: Dictionary of analysis results from specialists

        Returns:
            SynthesisResult with strategic insights and recommendations
        """
        self.logger.info(f"Starting strategic synthesis for audit {audit_result.audit_id}")

        synthesis_id = f"synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Step 1: Extract and categorize insights from all analyses
        raw_insights = await self._extract_insights(results)

        # Step 2: Identify causal relationships
        causal_relationships = await self._identify_causal_relationships(results, raw_insights)

        # Step 3: Generate strategic insights
        strategic_insights = await self._generate_strategic_insights(raw_insights, causal_relationships)

        # Step 4: Synthesize primary recommendation
        primary_recommendation = await self._synthesize_primary_recommendation(strategic_insights, audit_result)

        # Step 5: Create executive summary
        executive_summary = await self._create_executive_summary(strategic_insights, audit_result)

        # Step 6: Extract key findings
        key_findings = await self._extract_key_findings(strategic_insights, results)

        # Step 7: Generate next actions
        next_actions = await self._generate_next_actions(strategic_insights)

        # Step 8: Assess risks and opportunities
        risk_assessment = await self._assess_risks(strategic_insights)
        opportunity_analysis = await self._analyze_opportunities(strategic_insights)

        # Step 9: Calculate confidence and business metrics
        overall_confidence = self._calculate_overall_confidence(strategic_insights, audit_result)
        business_value_score = self._calculate_business_value(strategic_insights)
        implementation_complexity = self._assess_implementation_complexity(strategic_insights)
        success_probability = self._calculate_success_probability(strategic_insights, audit_result)

        synthesis_result = SynthesisResult(
            synthesis_id=synthesis_id,
            primary_recommendation=primary_recommendation,
            strategic_insights=strategic_insights,
            causal_relationships=causal_relationships,
            executive_summary=executive_summary,
            key_findings=key_findings,
            next_actions=next_actions,
            risk_assessment=risk_assessment,
            opportunity_analysis=opportunity_analysis,
            overall_confidence=overall_confidence,
            business_value_score=business_value_score,
            implementation_complexity=implementation_complexity,
            success_probability=success_probability
        )

        self.logger.info(f"Synthesis complete. Confidence: {overall_confidence:.2f}, "
                        f"Business Value: {business_value_score:.2f}, "
                        f"Success Probability: {success_probability:.2f}")

        return synthesis_result

    async def _extract_insights(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from all analysis results."""
        insights = []

        for task_id, result in results.items():
            if not isinstance(result, dict):
                continue

            # Extract recommendations
            recommendations = result.get("recommendations", [])
            if isinstance(recommendations, list):
                for rec in recommendations:
                    insights.append({
                        "type": "recommendation",
                        "content": str(rec),
                        "source": task_id,
                        "confidence": result.get("confidence_score", 0.5)
                    })

            # Extract insights field
            result_insights = result.get("insights", [])
            if isinstance(result_insights, list):
                for insight in result_insights:
                    insights.append({
                        "type": "insight",
                        "content": str(insight),
                        "source": task_id,
                        "confidence": result.get("confidence_score", 0.5)
                    })

            # Extract opportunities
            opportunities = result.get("opportunities", [])
            if isinstance(opportunities, list):
                for opp in opportunities:
                    insights.append({
                        "type": "opportunity",
                        "content": str(opp),
                        "source": task_id,
                        "confidence": result.get("confidence_score", 0.5)
                    })

            # Extract risks
            risks = result.get("risks", result.get("risk_factors", []))
            if isinstance(risks, list):
                for risk in risks:
                    insights.append({
                        "type": "risk",
                        "content": str(risk),
                        "source": task_id,
                        "confidence": result.get("confidence_score", 0.5)
                    })

        return insights

    async def _identify_causal_relationships(self, results: Dict[str, Any],
                                           insights: List[Dict[str, Any]]) -> List[CausalRelationship]:
        """Identify causal relationships in the analysis results."""
        relationships = []

        # Analyze text for causal indicators
        all_text = []
        for result in results.values():
            if isinstance(result, dict):
                for value in result.values():
                    if isinstance(value, (str, list)):
                        all_text.append(str(value))

        combined_text = " ".join(all_text).lower()

        # Look for strong causal patterns
        for insight in insights:
            content = insight["content"].lower()

            # Check for causal indicators
            for pattern_type, patterns in self.causal_indicators.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        # Extract potential cause and effect
                        start_pos = max(0, match.start() - 50)
                        end_pos = min(len(content), match.end() + 50)
                        context = content[start_pos:end_pos]

                        # Simple causal relationship extraction
                        parts = context.split(pattern)
                        if len(parts) >= 2:
                            potential_cause = parts[0].strip()[-30:]  # Last 30 chars before pattern
                            potential_effect = parts[1].strip()[:30]  # First 30 chars after pattern

                            if len(potential_cause) > 5 and len(potential_effect) > 5:
                                strength = 0.8 if pattern_type == "strong_causation" else 0.6
                                relationships.append(CausalRelationship(
                                    cause=potential_cause,
                                    effect=potential_effect,
                                    strength=strength,
                                    evidence=[content],
                                    confidence=insight["confidence"],
                                    supporting_analyses=[insight["source"]]
                                ))

        # Remove duplicates and weak relationships
        filtered_relationships = []
        for rel in relationships:
            if rel.strength > 0.5 and rel.confidence > 0.4:
                # Check for duplicates
                is_duplicate = False
                for existing in filtered_relationships:
                    if (existing.cause.lower() in rel.cause.lower() or
                        rel.cause.lower() in existing.cause.lower()) and \
                       (existing.effect.lower() in rel.effect.lower() or
                        rel.effect.lower() in existing.effect.lower()):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    filtered_relationships.append(rel)

        return filtered_relationships[:5]  # Limit to top 5 relationships

    async def _generate_strategic_insights(self, raw_insights: List[Dict[str, Any]],
                                         causal_relationships: List[CausalRelationship]) -> List[StrategicInsight]:
        """Generate strategic insights from raw insights and causal relationships."""
        strategic_insights = []
        insight_counter = 1

        # Group insights by type and confidence
        insight_groups = {}
        for insight in raw_insights:
            insight_type = insight["type"]
            if insight_type not in insight_groups:
                insight_groups[insight_type] = []
            insight_groups[insight_type].append(insight)

        # Generate strategic insights for each group
        for insight_type, group_insights in insight_groups.items():
            # Only process high-confidence insights
            high_conf_insights = [i for i in group_insights if i["confidence"] > 0.6]

            if not high_conf_insights:
                continue

            # Determine insight type enum
            if insight_type == "recommendation":
                strategic_type = InsightType.RECOMMENDATION
            elif insight_type == "opportunity":
                strategic_type = InsightType.OPPORTUNITY
            elif insight_type == "risk":
                strategic_type = InsightType.RISK
            else:
                strategic_type = InsightType.SYNTHESIS

            # Create consolidated insight
            avg_confidence = statistics.mean([i["confidence"] for i in high_conf_insights])
            confidence_level = self._confidence_to_level(avg_confidence)

            # Determine business impact
            combined_content = " ".join([i["content"] for i in high_conf_insights])
            business_impact = self._assess_business_impact(combined_content)

            # Extract actionable recommendations
            actionable_recs = []
            if strategic_type == InsightType.RECOMMENDATION:
                actionable_recs = [i["content"] for i in high_conf_insights]

            # Generate title and description
            title = self._generate_insight_title(strategic_type, high_conf_insights)
            description = self._generate_insight_description(high_conf_insights)

            strategic_insight = StrategicInsight(
                insight_id=f"insight_{insight_counter:03d}",
                insight_type=strategic_type,
                title=title,
                description=description,
                confidence=confidence_level,
                business_impact=business_impact,
                supporting_evidence=[i["content"] for i in high_conf_insights],
                source_analyses=list(set([i["source"] for i in high_conf_insights])),
                actionable_recommendations=actionable_recs,
                estimated_value=self._extract_value_estimate(combined_content)
            )

            strategic_insights.append(strategic_insight)
            insight_counter += 1

        # Add causal relationship insights
        for rel in causal_relationships:
            strategic_insight = StrategicInsight(
                insight_id=f"insight_{insight_counter:03d}",
                insight_type=InsightType.CAUSAL_RELATIONSHIP,
                title=f"Causal Relationship: {rel.cause[:20]}... → {rel.effect[:20]}...",
                description=f"Analysis indicates that {rel.cause} leads to {rel.effect} with {rel.strength:.0%} strength",
                confidence=self._confidence_to_level(rel.confidence),
                business_impact=BusinessImpact.MEDIUM,
                supporting_evidence=rel.evidence,
                source_analyses=rel.supporting_analyses
            )
            strategic_insights.append(strategic_insight)
            insight_counter += 1

        return strategic_insights

    def _confidence_to_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level enum."""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _assess_business_impact(self, content: str) -> BusinessImpact:
        """Assess business impact level based on content."""
        content_lower = content.lower()

        # Score impact based on keywords
        impact_scores = {}
        for impact_level, keywords in self.impact_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            impact_scores[impact_level] = score

        # Find highest scoring impact level
        max_score = max(impact_scores.values())
        if max_score == 0:
            return BusinessImpact.MEDIUM  # Default

        for impact_level, score in impact_scores.items():
            if score == max_score:
                return impact_level

        return BusinessImpact.MEDIUM

    def _extract_value_estimate(self, content: str) -> Optional[str]:
        """Extract value estimates from content."""
        for pattern_type, pattern in self.value_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                return f"{pattern_type}: {matches[0]}"
        return None

    def _generate_insight_title(self, insight_type: InsightType,
                              insights: List[Dict[str, Any]]) -> str:
        """Generate a title for the strategic insight."""
        titles = {
            InsightType.RECOMMENDATION: "Strategic Recommendation",
            InsightType.OPPORTUNITY: "Business Opportunity",
            InsightType.RISK: "Risk Assessment",
            InsightType.SYNTHESIS: "Key Insight"
        }

        base_title = titles.get(insight_type, "Strategic Insight")

        # Try to extract key concepts for more specific title
        all_content = " ".join([i["content"] for i in insights]).lower()

        key_concepts = {
            "trust": "Trust Strategy",
            "tax": "Tax Optimization",
            "estate": "Estate Planning",
            "asset": "Asset Protection",
            "succession": "Succession Planning",
            "market": "Market Analysis",
            "sales": "Sales Strategy"
        }

        for concept, title_modifier in key_concepts.items():
            if concept in all_content:
                return f"{title_modifier} - {base_title}"

        return base_title

    def _generate_insight_description(self, insights: List[Dict[str, Any]]) -> str:
        """Generate a description for the strategic insight."""
        if len(insights) == 1:
            return insights[0]["content"]

        # Summarize multiple insights
        return f"Analysis of {len(insights)} related findings reveals " + \
               insights[0]["content"][:100] + "..."

    async def _synthesize_primary_recommendation(self, insights: List[StrategicInsight],
                                               audit_result: AuditResult) -> str:
        """Synthesize the primary strategic recommendation."""
        # Find highest impact, highest confidence recommendations
        recommendations = [i for i in insights if i.insight_type == InsightType.RECOMMENDATION]

        if not recommendations:
            return "Based on the analysis, continue with current strategy while monitoring key performance indicators."

        # Score recommendations by impact and confidence
        scored_recs = []
        for rec in recommendations:
            impact_score = {
                BusinessImpact.CRITICAL: 5,
                BusinessImpact.HIGH: 4,
                BusinessImpact.MEDIUM: 3,
                BusinessImpact.LOW: 2,
                BusinessImpact.MINIMAL: 1
            }.get(rec.business_impact, 3)

            confidence_score = {
                ConfidenceLevel.VERY_HIGH: 5,
                ConfidenceLevel.HIGH: 4,
                ConfidenceLevel.MEDIUM: 3,
                ConfidenceLevel.LOW: 2,
                ConfidenceLevel.VERY_LOW: 1
            }.get(rec.confidence, 3)

            total_score = impact_score * confidence_score
            scored_recs.append((rec, total_score))

        # Select top recommendation
        if scored_recs:
            top_rec = max(scored_recs, key=lambda x: x[1])[0]
            if top_rec.actionable_recommendations:
                return top_rec.actionable_recommendations[0]
            else:
                return top_rec.description

        return "Implement a comprehensive estate planning strategy based on the identified opportunities and risk factors."

    async def _create_executive_summary(self, insights: List[StrategicInsight],
                                      audit_result: AuditResult) -> str:
        """Create an executive summary of the strategic analysis."""
        # Count insights by type and impact
        insight_counts = {}
        high_impact_count = 0

        for insight in insights:
            insight_type = insight.insight_type.value
            insight_counts[insight_type] = insight_counts.get(insight_type, 0) + 1

            if insight.business_impact in [BusinessImpact.CRITICAL, BusinessImpact.HIGH]:
                high_impact_count += 1

        # Calculate overall quality
        quality_score = audit_result.quality_metrics.overall_quality_score

        summary = f"Strategic analysis identified {len(insights)} key insights across estate planning considerations, "
        summary += f"with {high_impact_count} high-impact findings. "

        if quality_score > 0.8:
            summary += "Analysis quality is excellent with high confidence in recommendations. "
        elif quality_score > 0.6:
            summary += "Analysis quality is good with reliable recommendations. "
        else:
            summary += "Analysis quality requires attention; recommendations should be validated. "

        # Add key insight types
        if "recommendation" in insight_counts:
            summary += f"Generated {insight_counts['recommendation']} strategic recommendations. "

        if "opportunity" in insight_counts:
            summary += f"Identified {insight_counts['opportunity']} business opportunities. "

        if "risk" in insight_counts:
            summary += f"Assessed {insight_counts['risk']} risk factors. "

        summary += "Implementation should proceed based on priority and impact assessment."

        return summary

    async def _extract_key_findings(self, insights: List[StrategicInsight],
                                  results: Dict[str, Any]) -> List[str]:
        """Extract key findings from strategic insights."""
        findings = []

        # Get high-impact insights
        high_impact_insights = [i for i in insights
                              if i.business_impact in [BusinessImpact.CRITICAL, BusinessImpact.HIGH]]

        for insight in high_impact_insights[:5]:  # Top 5
            if insight.insight_type == InsightType.OPPORTUNITY:
                findings.append(f"Opportunity: {insight.description}")
            elif insight.insight_type == InsightType.RISK:
                findings.append(f"Risk: {insight.description}")
            elif insight.insight_type == InsightType.RECOMMENDATION:
                findings.append(f"Recommendation: {insight.description}")
            else:
                findings.append(insight.description)

        # Add causal relationships as findings
        causal_insights = [i for i in insights if i.insight_type == InsightType.CAUSAL_RELATIONSHIP]
        for causal in causal_insights[:2]:  # Top 2
            findings.append(f"Key Relationship: {causal.description}")

        return findings

    async def _generate_next_actions(self, insights: List[StrategicInsight]) -> List[str]:
        """Generate actionable next steps."""
        actions = []

        # Extract actionable recommendations
        for insight in insights:
            if insight.actionable_recommendations:
                for rec in insight.actionable_recommendations[:2]:  # Max 2 per insight
                    if rec not in actions:
                        actions.append(rec)

        # Add generic next actions if none found
        if not actions:
            actions = [
                "Review and validate analysis findings with stakeholders",
                "Develop implementation timeline for recommended strategies",
                "Assess resource requirements for proposed changes",
                "Monitor key performance indicators"
            ]

        return actions[:6]  # Limit to 6 actions

    async def _assess_risks(self, insights: List[StrategicInsight]) -> str:
        """Assess overall risk landscape."""
        risk_insights = [i for i in insights if i.insight_type == InsightType.RISK]

        if not risk_insights:
            return "Risk assessment indicates low to moderate risk levels with standard estate planning considerations."

        # Categorize risks by impact
        critical_risks = [i for i in risk_insights if i.business_impact == BusinessImpact.CRITICAL]
        high_risks = [i for i in risk_insights if i.business_impact == BusinessImpact.HIGH]

        assessment = f"Identified {len(risk_insights)} risk factors. "

        if critical_risks:
            assessment += f"{len(critical_risks)} critical risks require immediate attention. "

        if high_risks:
            assessment += f"{len(high_risks)} high-impact risks need mitigation planning. "

        assessment += "Recommended risk management strategies should be implemented as part of overall planning."

        return assessment

    async def _analyze_opportunities(self, insights: List[StrategicInsight]) -> str:
        """Analyze opportunity landscape."""
        opportunity_insights = [i for i in insights if i.insight_type == InsightType.OPPORTUNITY]

        if not opportunity_insights:
            return "Opportunity analysis reveals standard estate planning benefits with potential for optimization."

        # Assess opportunity value
        high_value_opps = [i for i in opportunity_insights
                          if i.business_impact in [BusinessImpact.CRITICAL, BusinessImpact.HIGH]]

        analysis = f"Identified {len(opportunity_insights)} opportunities for value creation. "

        if high_value_opps:
            analysis += f"{len(high_value_opps)} high-value opportunities present significant potential. "

        # Look for value estimates
        valued_opps = [i for i in opportunity_insights if i.estimated_value]
        if valued_opps:
            analysis += f"Quantified value potential identified in {len(valued_opps)} opportunities. "

        analysis += "Prioritize opportunities based on implementation complexity and expected return."

        return analysis

    def _calculate_overall_confidence(self, insights: List[StrategicInsight],
                                    audit_result: AuditResult) -> float:
        """Calculate overall confidence in the strategic synthesis."""
        if not insights:
            return 0.5

        # Weight confidence by business impact
        weighted_confidences = []
        for insight in insights:
            impact_weight = {
                BusinessImpact.CRITICAL: 1.0,
                BusinessImpact.HIGH: 0.8,
                BusinessImpact.MEDIUM: 0.6,
                BusinessImpact.LOW: 0.4,
                BusinessImpact.MINIMAL: 0.2
            }.get(insight.business_impact, 0.6)

            confidence_value = {
                ConfidenceLevel.VERY_HIGH: 0.95,
                ConfidenceLevel.HIGH: 0.8,
                ConfidenceLevel.MEDIUM: 0.6,
                ConfidenceLevel.LOW: 0.4,
                ConfidenceLevel.VERY_LOW: 0.2
            }.get(insight.confidence, 0.5)

            weighted_confidences.append(confidence_value * impact_weight)

        insight_confidence = statistics.mean(weighted_confidences)

        # Factor in audit quality
        audit_confidence = audit_result.quality_metrics.confidence_score

        # Combined confidence (70% insights, 30% audit)
        return (insight_confidence * 0.7) + (audit_confidence * 0.3)

    def _calculate_business_value(self, insights: List[StrategicInsight]) -> float:
        """Calculate business value score."""
        if not insights:
            return 0.5

        # Score based on number and impact of opportunities
        opportunity_score = 0
        for insight in insights:
            if insight.insight_type == InsightType.OPPORTUNITY:
                impact_score = {
                    BusinessImpact.CRITICAL: 1.0,
                    BusinessImpact.HIGH: 0.8,
                    BusinessImpact.MEDIUM: 0.6,
                    BusinessImpact.LOW: 0.4,
                    BusinessImpact.MINIMAL: 0.2
                }.get(insight.business_impact, 0.5)

                opportunity_score += impact_score

        # Normalize and combine with recommendation quality
        normalized_opp_score = min(1.0, opportunity_score / 3)  # Cap at 3 opportunities

        recommendation_count = len([i for i in insights if i.insight_type == InsightType.RECOMMENDATION])
        rec_score = min(1.0, recommendation_count / 5)  # Cap at 5 recommendations

        return (normalized_opp_score * 0.6) + (rec_score * 0.4)

    def _assess_implementation_complexity(self, insights: List[StrategicInsight]) -> str:
        """Assess implementation complexity of strategic recommendations."""
        recommendation_count = len([i for i in insights if i.insight_type == InsightType.RECOMMENDATION])
        high_impact_count = len([i for i in insights if i.business_impact in [BusinessImpact.CRITICAL, BusinessImpact.HIGH]])

        if recommendation_count > 5 or high_impact_count > 3:
            return "High - Multiple complex recommendations requiring coordinated implementation"
        elif recommendation_count > 2 or high_impact_count > 1:
            return "Medium - Several recommendations with moderate coordination requirements"
        else:
            return "Low - Straightforward implementation with minimal coordination"

    def _calculate_success_probability(self, insights: List[StrategicInsight],
                                     audit_result: AuditResult) -> float:
        """Calculate probability of successful implementation."""
        # Base success probability on confidence and quality
        base_probability = (self._calculate_overall_confidence(insights, audit_result) +
                          audit_result.quality_metrics.overall_quality_score) / 2

        # Adjust for complexity
        complexity = self._assess_implementation_complexity(insights)
        if "High" in complexity:
            complexity_modifier = -0.2
        elif "Medium" in complexity:
            complexity_modifier = -0.1
        else:
            complexity_modifier = 0.0

        # Adjust for risk factors
        risk_count = len([i for i in insights if i.insight_type == InsightType.RISK])
        risk_modifier = -min(0.3, risk_count * 0.05)

        return max(0.1, min(1.0, base_probability + complexity_modifier + risk_modifier))

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the strategist."""
        return {
            "status": "healthy",
            "component": "estate_strategist",
            "causal_indicators": sum(len(patterns) for patterns in self.causal_indicators.values()),
            "impact_keywords": sum(len(keywords) for keywords in self.impact_keywords.values()),
            "value_patterns": len(self.value_patterns),
            "last_check": datetime.now().isoformat()
        }


# Example usage and testing
async def main():
    """Example usage of Estate Strategist."""
    from .auditor import AuditResult, QualityMetrics

    strategist = EstateStrategist()

    # Mock audit result
    quality_metrics = QualityMetrics(
        accuracy_score=0.85,
        completeness_score=0.90,
        consistency_score=0.80,
        confidence_score=0.82,
        reliability_score=0.88,
        overall_quality_score=0.85,
        total_findings=3,
        critical_findings=0,
        execution_time=450.0,
        success_rate=0.90
    )

    audit_result = AuditResult(
        audit_id="test_audit_001",
        passes_quality_check=True,
        quality_metrics=quality_metrics,
        findings=[],
        corrections_applied=[],
        confidence_score=0.82,
        recommendation="Good quality analysis"
    )

    # Mock analysis results
    results = {
        "task_001": {
            "recommendations": [
                "Implement a revocable living trust structure",
                "Utilize annual gift tax exclusions for tax efficiency"
            ],
            "insights": [
                "Trust structure will provide estate tax savings of approximately $500,000"
            ],
            "opportunities": [
                "Charitable giving strategy could provide additional tax benefits"
            ],
            "confidence_score": 0.85
        },
        "task_002": {
            "recommendations": [
                "Consider business succession planning for family company"
            ],
            "risks": [
                "Potential liquidity constraints during estate settlement",
                "Market volatility may affect asset valuations"
            ],
            "confidence_score": 0.78
        }
    }

    print("Performing strategic synthesis...")
    synthesis = await strategist.synthesize(audit_result, results)

    print(f"\nSynthesis ID: {synthesis.synthesis_id}")
    print(f"Overall Confidence: {synthesis.overall_confidence:.2f}")
    print(f"Business Value Score: {synthesis.business_value_score:.2f}")
    print(f"Success Probability: {synthesis.success_probability:.2f}")

    print(f"\nPrimary Recommendation:")
    print(f"  {synthesis.primary_recommendation}")

    print(f"\nExecutive Summary:")
    print(f"  {synthesis.executive_summary}")

    print(f"\nStrategic Insights ({len(synthesis.strategic_insights)}):")
    for insight in synthesis.strategic_insights:
        print(f"  - {insight.insight_type.value.title()}: {insight.title}")
        print(f"    Impact: {insight.business_impact.value}, Confidence: {insight.confidence.value}")

    if synthesis.causal_relationships:
        print(f"\nCausal Relationships ({len(synthesis.causal_relationships)}):")
        for rel in synthesis.causal_relationships:
            print(f"  - {rel.cause} → {rel.effect} (strength: {rel.strength:.0%})")

    print(f"\nKey Findings:")
    for finding in synthesis.key_findings:
        print(f"  - {finding}")

    print(f"\nNext Actions:")
    for action in synthesis.next_actions:
        print(f"  - {action}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())