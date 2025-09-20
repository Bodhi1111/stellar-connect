"""
Result synthesis module with confidence scoring.
Combines multiple analysis results into coherent responses with confidence metrics.
"""

import math
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re


class ConfidenceLevel(Enum):
    VERY_LOW = "very_low"    # 0-20%
    LOW = "low"              # 20-40%
    MODERATE = "moderate"    # 40-60%
    HIGH = "high"            # 60-80%
    VERY_HIGH = "very_high"  # 80-100%


@dataclass
class SourceInfo:
    id: str
    type: str
    reliability: float
    recency: float
    relevance: float
    weight: float = 1.0

    @property
    def quality_score(self) -> float:
        """Calculate overall quality score for this source."""
        return (self.reliability * 0.4 +
                self.recency * 0.3 +
                self.relevance * 0.3) * self.weight


@dataclass
class SynthesisResult:
    content: str
    confidence_score: float
    confidence_level: ConfidenceLevel
    sources: List[SourceInfo]
    supporting_evidence: List[str]
    conflicting_information: List[str]
    uncertainty_factors: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def evidence_strength(self) -> float:
        """Calculate strength of supporting evidence."""
        if not self.sources:
            return 0.0
        return statistics.mean([source.quality_score for source in self.sources])

    @property
    def consistency_score(self) -> float:
        """Calculate consistency score based on conflicts."""
        total_statements = len(self.supporting_evidence) + len(self.conflicting_information)
        if total_statements == 0:
            return 1.0
        return len(self.supporting_evidence) / total_statements


class ResultSynthesizer:
    """Synthesizes multiple analysis results into coherent responses with confidence scoring."""

    def __init__(self):
        self.confidence_thresholds = {
            ConfidenceLevel.VERY_LOW: (0.0, 0.2),
            ConfidenceLevel.LOW: (0.2, 0.4),
            ConfidenceLevel.MODERATE: (0.4, 0.6),
            ConfidenceLevel.HIGH: (0.6, 0.8),
            ConfidenceLevel.VERY_HIGH: (0.8, 1.0)
        }

    def synthesize_results(self,
                          step_results: Dict[str, Any],
                          query: str,
                          query_type: str) -> SynthesisResult:
        """
        Synthesize multiple step results into a coherent response.

        Args:
            step_results: Results from each execution step
            query: Original user query
            query_type: Type of query being processed

        Returns:
            SynthesisResult with synthesized content and confidence metrics
        """
        # Extract and organize information from step results
        extracted_info = self._extract_information(step_results)

        # Analyze source reliability and relevance
        sources = self._analyze_sources(extracted_info, query)

        # Identify supporting evidence and conflicts
        evidence, conflicts = self._analyze_evidence(extracted_info)

        # Calculate base confidence score
        confidence_score = self._calculate_confidence(sources, evidence, conflicts, query_type)

        # Identify uncertainty factors
        uncertainty_factors = self._identify_uncertainty_factors(step_results, evidence, conflicts)

        # Adjust confidence based on uncertainty
        adjusted_confidence = self._adjust_confidence_for_uncertainty(confidence_score, uncertainty_factors)

        # Determine confidence level
        confidence_level = self._determine_confidence_level(adjusted_confidence)

        # Generate synthesized content
        content = self._generate_synthesized_content(
            extracted_info, evidence, conflicts, query_type, confidence_level
        )

        return SynthesisResult(
            content=content,
            confidence_score=adjusted_confidence,
            confidence_level=confidence_level,
            sources=sources,
            supporting_evidence=evidence,
            conflicting_information=conflicts,
            uncertainty_factors=uncertainty_factors,
            metadata={
                "query": query,
                "query_type": query_type,
                "synthesis_timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
                "step_count": len(step_results)
            }
        )

    def _extract_information(self, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and organize information from step results."""
        extracted = {
            "facts": [],
            "analyses": [],
            "insights": [],
            "data_points": [],
            "sources": [],
            "errors": []
        }

        for step_id, result in step_results.items():
            if isinstance(result, dict):
                # Extract facts
                if "facts" in result:
                    extracted["facts"].extend(result["facts"])

                # Extract analyses
                if "analysis" in result:
                    extracted["analyses"].append(result["analysis"])

                # Extract insights
                if "insights" in result:
                    extracted["insights"].extend(result["insights"])

                # Extract data points
                if "data" in result:
                    extracted["data_points"].append(result["data"])

                # Extract sources
                if "sources" in result:
                    extracted["sources"].extend(result["sources"])

                # Extract errors
                if "error" in result:
                    extracted["errors"].append(result["error"])

        return extracted

    def _analyze_sources(self, extracted_info: Dict[str, Any], query: str) -> List[SourceInfo]:
        """Analyze and score information sources."""
        sources = []

        for i, source_data in enumerate(extracted_info.get("sources", [])):
            if isinstance(source_data, dict):
                source = SourceInfo(
                    id=source_data.get("id", f"source_{i}"),
                    type=source_data.get("type", "unknown"),
                    reliability=self._assess_reliability(source_data),
                    recency=self._assess_recency(source_data),
                    relevance=self._assess_relevance(source_data, query)
                )
                sources.append(source)

        # If no explicit sources, create implicit sources from step results
        if not sources:
            sources.append(SourceInfo(
                id="analysis_engine",
                type="computational",
                reliability=0.8,
                recency=1.0,
                relevance=0.9
            ))

        return sources

    def _assess_reliability(self, source_data: Dict[str, Any]) -> float:
        """Assess the reliability of a source."""
        source_type = source_data.get("type", "unknown")

        reliability_scores = {
            "academic": 0.9,
            "official": 0.85,
            "news": 0.7,
            "encyclopedia": 0.8,
            "computational": 0.8,
            "user_generated": 0.4,
            "unknown": 0.5
        }

        base_score = reliability_scores.get(source_type, 0.5)

        # Adjust based on additional factors
        if source_data.get("peer_reviewed", False):
            base_score += 0.1

        if source_data.get("citation_count", 0) > 100:
            base_score += 0.05

        return min(base_score, 1.0)

    def _assess_recency(self, source_data: Dict[str, Any]) -> float:
        """Assess the recency/freshness of a source."""
        # For now, return a default score
        # In practice, would calculate based on publication date
        publication_year = source_data.get("year", 2024)
        current_year = 2024

        years_old = current_year - publication_year

        if years_old <= 1:
            return 1.0
        elif years_old <= 3:
            return 0.8
        elif years_old <= 5:
            return 0.6
        elif years_old <= 10:
            return 0.4
        else:
            return 0.2

    def _assess_relevance(self, source_data: Dict[str, Any], query: str) -> float:
        """Assess how relevant a source is to the query."""
        # Simple keyword matching approach
        # In practice, would use more sophisticated NLP techniques

        source_text = str(source_data).lower()
        query_words = set(re.findall(r'\b\w+\b', query.lower()))

        if not query_words:
            return 0.5

        matches = sum(1 for word in query_words if word in source_text)
        return min(matches / len(query_words), 1.0)

    def _analyze_evidence(self, extracted_info: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Analyze evidence and identify supporting vs conflicting information."""
        evidence = []
        conflicts = []

        # Collect all statements
        all_statements = []
        all_statements.extend(extracted_info.get("facts", []))
        all_statements.extend(extracted_info.get("insights", []))

        # Simple conflict detection (would be more sophisticated in practice)
        for statement in all_statements:
            if isinstance(statement, str):
                # Look for contradictory language
                if any(word in statement.lower() for word in ["however", "but", "although", "despite"]):
                    conflicts.append(statement)
                else:
                    evidence.append(statement)

        return evidence, conflicts

    def _calculate_confidence(self,
                            sources: List[SourceInfo],
                            evidence: List[str],
                            conflicts: List[str],
                            query_type: str) -> float:
        """Calculate base confidence score."""
        # Source quality component
        if sources:
            source_score = statistics.mean([source.quality_score for source in sources])
        else:
            source_score = 0.5

        # Evidence consistency component
        total_statements = len(evidence) + len(conflicts)
        if total_statements > 0:
            consistency_score = len(evidence) / total_statements
        else:
            consistency_score = 0.5

        # Evidence quantity component
        evidence_quantity_score = min(len(evidence) / 5.0, 1.0)  # Normalize to 5 pieces of evidence

        # Query type adjustment
        query_type_weights = {
            "simple_lookup": {"source": 0.6, "consistency": 0.3, "quantity": 0.1},
            "complex_analysis": {"source": 0.4, "consistency": 0.4, "quantity": 0.2},
            "comparison": {"source": 0.3, "consistency": 0.5, "quantity": 0.2},
            "trend_analysis": {"source": 0.5, "consistency": 0.3, "quantity": 0.2},
            "aggregation": {"source": 0.4, "consistency": 0.2, "quantity": 0.4}
        }

        weights = query_type_weights.get(query_type, {"source": 0.4, "consistency": 0.4, "quantity": 0.2})

        confidence = (source_score * weights["source"] +
                     consistency_score * weights["consistency"] +
                     evidence_quantity_score * weights["quantity"])

        return min(confidence, 1.0)

    def _identify_uncertainty_factors(self,
                                    step_results: Dict[str, Any],
                                    evidence: List[str],
                                    conflicts: List[str]) -> List[str]:
        """Identify factors that increase uncertainty."""
        uncertainty_factors = []

        # Check for errors in step execution
        errors = []
        for result in step_results.values():
            if isinstance(result, dict) and "error" in result:
                errors.append(result["error"])

        if errors:
            uncertainty_factors.append(f"Execution errors in {len(errors)} steps")

        # Check for conflicting information
        if conflicts:
            uncertainty_factors.append(f"Found {len(conflicts)} conflicting statements")

        # Check for insufficient evidence
        if len(evidence) < 2:
            uncertainty_factors.append("Limited supporting evidence available")

        # Check for missing data
        missing_data_indicators = ["no data available", "information not found", "unable to determine"]
        for result in step_results.values():
            result_str = str(result).lower()
            if any(indicator in result_str for indicator in missing_data_indicators):
                uncertainty_factors.append("Missing or unavailable data sources")
                break

        return uncertainty_factors

    def _adjust_confidence_for_uncertainty(self,
                                         base_confidence: float,
                                         uncertainty_factors: List[str]) -> float:
        """Adjust confidence score based on uncertainty factors."""
        if not uncertainty_factors:
            return base_confidence

        # Apply penalty for each uncertainty factor
        penalty_per_factor = 0.1
        total_penalty = min(len(uncertainty_factors) * penalty_per_factor, 0.5)

        adjusted_confidence = base_confidence * (1 - total_penalty)
        return max(adjusted_confidence, 0.1)  # Minimum confidence of 10%

    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level from numeric score."""
        for level, (min_thresh, max_thresh) in self.confidence_thresholds.items():
            if min_thresh <= confidence_score < max_thresh:
                return level

        # Handle edge case for exactly 1.0
        if confidence_score >= 1.0:
            return ConfidenceLevel.VERY_HIGH

        return ConfidenceLevel.VERY_LOW

    def _generate_synthesized_content(self,
                                    extracted_info: Dict[str, Any],
                                    evidence: List[str],
                                    conflicts: List[str],
                                    query_type: str,
                                    confidence_level: ConfidenceLevel) -> str:
        """Generate synthesized content from extracted information."""
        content_parts = []

        # Start with main findings
        if evidence:
            content_parts.append("Based on the analysis, key findings include:")
            for i, fact in enumerate(evidence[:5], 1):  # Limit to top 5
                content_parts.append(f"{i}. {fact}")

        # Add insights if available
        insights = extracted_info.get("insights", [])
        if insights:
            content_parts.append("\nKey insights:")
            for insight in insights[:3]:  # Limit to top 3
                content_parts.append(f"• {insight}")

        # Address conflicts if any
        if conflicts:
            content_parts.append(f"\nNote: Some conflicting information was found:")
            for conflict in conflicts[:2]:  # Limit to top 2
                content_parts.append(f"• {conflict}")

        # Add confidence qualifier
        confidence_qualifiers = {
            ConfidenceLevel.VERY_LOW: "This analysis has very low confidence due to limited or uncertain information.",
            ConfidenceLevel.LOW: "This analysis has low confidence; additional verification is recommended.",
            ConfidenceLevel.MODERATE: "This analysis has moderate confidence based on available information.",
            ConfidenceLevel.HIGH: "This analysis has high confidence based on reliable sources and evidence.",
            ConfidenceLevel.VERY_HIGH: "This analysis has very high confidence based on strong, consistent evidence."
        }

        content_parts.append(f"\nConfidence Assessment: {confidence_qualifiers[confidence_level]}")

        return "\n".join(content_parts)

    def generate_confidence_explanation(self, result: SynthesisResult) -> str:
        """Generate a detailed explanation of the confidence score."""
        explanation_parts = [
            f"Confidence Score: {result.confidence_score:.2f} ({result.confidence_level.value.replace('_', ' ').title()})",
            f"Evidence Strength: {result.evidence_strength:.2f}",
            f"Consistency Score: {result.consistency_score:.2f}"
        ]

        if result.sources:
            avg_source_quality = statistics.mean([s.quality_score for s in result.sources])
            explanation_parts.append(f"Average Source Quality: {avg_source_quality:.2f}")

        if result.uncertainty_factors:
            explanation_parts.append("Uncertainty Factors:")
            for factor in result.uncertainty_factors:
                explanation_parts.append(f"• {factor}")

        return "\n".join(explanation_parts)

    def compare_synthesis_results(self, results: List[SynthesisResult]) -> Dict[str, Any]:
        """Compare multiple synthesis results and provide recommendations."""
        if not results:
            return {"error": "No results to compare"}

        if len(results) == 1:
            return {"message": "Only one result available", "result": results[0]}

        # Find highest confidence result
        best_result = max(results, key=lambda r: r.confidence_score)

        # Calculate average confidence
        avg_confidence = statistics.mean([r.confidence_score for r in results])

        # Identify consensus and disagreements
        all_evidence = []
        for result in results:
            all_evidence.extend(result.supporting_evidence)

        # Simple consensus detection (count common statements)
        evidence_counts = {}
        for evidence in all_evidence:
            evidence_counts[evidence] = evidence_counts.get(evidence, 0) + 1

        consensus_evidence = [evidence for evidence, count in evidence_counts.items()
                            if count > len(results) / 2]

        return {
            "best_result": best_result,
            "average_confidence": avg_confidence,
            "consensus_evidence": consensus_evidence,
            "total_results": len(results),
            "confidence_range": {
                "min": min(r.confidence_score for r in results),
                "max": max(r.confidence_score for r in results)
            }
        }