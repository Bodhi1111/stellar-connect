"""
Estate Librarian Agent for Stellar Connect
Implements Story 5.2: Sales Specialist Agent Team - Estate Librarian

The Estate Librarian specializes in document retrieval, similar case finding,
and building a comprehensive knowledge base of successful sales patterns
and effective rebuttals for estate planning sales.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import re

from .base_specialist import (
    BaseSpecialist, SpecialistTask, SpecialistExpertise,
    SpecialistCapability, TaskStatus
)
from .similar_case_finder import SimilarCaseFinder, SimilarCase as FinderSimilarCase
from .rebuttal_library import RebuttalLibrary, ObjectionAnalysis, RebuttalRecommendation


@dataclass
class SimilarCase:
    """Represents a similar case found by the Estate Librarian."""
    case_id: str
    client_name: str
    estate_value: float
    family_structure: str
    primary_concerns: List[str]
    solutions_implemented: List[str]
    outcome: str
    success_factors: List[str]
    lessons_learned: List[str]
    similarity_score: float
    case_date: datetime
    advisor: str


@dataclass
class RebuttalPattern:
    """Represents an effective rebuttal pattern."""
    rebuttal_id: str
    objection_type: str
    objection_text: str
    rebuttal_text: str
    effectiveness_score: float
    usage_count: int
    success_rate: float
    context_tags: List[str]
    advisor: str
    date_created: datetime
    last_used: Optional[datetime] = None


@dataclass
class ContentSearchResult:
    """Represents a content search result."""
    content_id: str
    content_type: str  # transcript, rebuttal, case_study, template
    title: str
    excerpt: str
    full_content: str
    relevance_score: float
    metadata: Dict[str, Any]
    source_file: str
    created_date: datetime


@dataclass
class KnowledgePattern:
    """Represents a pattern extracted from the knowledge base."""
    pattern_id: str
    pattern_type: str  # success_factor, warning_sign, timing_indicator
    description: str
    indicators: List[str]
    frequency: int
    confidence_score: float
    supporting_cases: List[str]
    recommendations: List[str]


class EstateLibrarianAgent(BaseSpecialist):
    """
    Estate Librarian Agent - Specialist in document retrieval and knowledge management.

    Capabilities:
    - Find similar cases based on estate characteristics
    - Retrieve effective rebuttals for common objections
    - Search knowledge base for relevant content
    - Identify successful patterns across cases
    - Generate case studies and best practices
    """

    def __init__(self, knowledge_base_path: str = "data/knowledge_base"):
        # Define capabilities
        capabilities = [
            SpecialistCapability(
                name="similar_case_search",
                description="Find similar cases based on estate characteristics",
                input_types=["estate_profile", "search_criteria"],
                output_types=["similar_cases", "case_analysis"]
            ),
            SpecialistCapability(
                name="rebuttal_retrieval",
                description="Retrieve effective rebuttals for objections",
                input_types=["objection_text", "objection_category"],
                output_types=["rebuttal_suggestions", "effectiveness_scores"]
            ),
            SpecialistCapability(
                name="content_search",
                description="Search knowledge base for relevant content",
                input_types=["search_query", "content_filters"],
                output_types=["search_results", "content_summaries"]
            ),
            SpecialistCapability(
                name="pattern_analysis",
                description="Identify patterns across successful cases",
                input_types=["case_data", "analysis_criteria"],
                output_types=["success_patterns", "recommendations"]
            ),
            SpecialistCapability(
                name="case_study_generation",
                description="Generate case studies from successful interactions",
                input_types=["transcript_data", "outcome_data"],
                output_types=["case_study", "key_insights"]
            )
        ]

        super().__init__(
            name="Estate Librarian",
            expertise=SpecialistExpertise.DOCUMENT_RETRIEVAL,
            description="Specialist in document retrieval, case analysis, and knowledge management for estate planning sales",
            capabilities=capabilities,
            max_concurrent_tasks=5
        )

        # Initialize knowledge base
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)

        # Initialize internal data structures
        self.similar_cases: List[SimilarCase] = []
        self.rebuttal_patterns: List[RebuttalPattern] = []
        self.knowledge_patterns: List[KnowledgePattern] = []
        self.content_index: Dict[str, ContentSearchResult] = {}

        # Initialize utility classes
        self.similar_case_finder = SimilarCaseFinder(knowledge_base_path)
        self.rebuttal_library = RebuttalLibrary(knowledge_base_path)

        # Load existing knowledge base
        asyncio.create_task(self._load_knowledge_base())

        self.logger = logging.getLogger(f"{__name__}.EstateLibrarianAgent")

    def get_task_types(self) -> List[str]:
        """Return list of task types this specialist can handle."""
        return [
            "find_similar_cases",
            "retrieve_rebuttals",
            "search_content",
            "analyze_patterns",
            "generate_case_study",
            "build_knowledge_index",
            "extract_best_practices"
        ]

    async def validate_input(self, task: SpecialistTask) -> Tuple[bool, Optional[str]]:
        """Validate task input data."""
        task_type = task.task_type
        input_data = task.input_data

        try:
            if task_type == "find_similar_cases":
                required_fields = ["estate_value", "family_structure"]
                for field in required_fields:
                    if field not in input_data:
                        return False, f"Missing required field: {field}"

            elif task_type == "retrieve_rebuttals":
                if "objection_text" not in input_data and "objection_category" not in input_data:
                    return False, "Must provide either objection_text or objection_category"

            elif task_type == "search_content":
                if "search_query" not in input_data:
                    return False, "Missing required field: search_query"

            elif task_type == "analyze_patterns":
                if "analysis_type" not in input_data:
                    return False, "Missing required field: analysis_type"

            elif task_type == "generate_case_study":
                required_fields = ["transcript_content", "outcome"]
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
            if task_type == "find_similar_cases":
                return await self._find_similar_cases(input_data)

            elif task_type == "retrieve_rebuttals":
                return await self._retrieve_rebuttals(input_data)

            elif task_type == "search_content":
                return await self._search_content(input_data)

            elif task_type == "analyze_patterns":
                return await self._analyze_patterns(input_data)

            elif task_type == "generate_case_study":
                return await self._generate_case_study(input_data)

            elif task_type == "build_knowledge_index":
                return await self._build_knowledge_index(input_data)

            elif task_type == "extract_best_practices":
                return await self._extract_best_practices(input_data)

            else:
                raise ValueError(f"Unsupported task type: {task_type}")

        except Exception as e:
            self.logger.error(f"Error processing task {task.task_id}: {str(e)}")
            raise

    async def _find_similar_cases(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find similar cases based on estate characteristics."""
        estate_value = input_data.get("estate_value", 0)
        family_structure = input_data.get("family_structure", "")
        primary_concerns = input_data.get("primary_concerns", [])
        objection_types = input_data.get("objection_types", [])
        limit = input_data.get("limit", 5)

        # Use the SimilarCaseFinder utility
        similar_cases = await self.similar_case_finder.find_matches(
            estate_value=estate_value,
            family_structure=family_structure,
            objection_types=objection_types,
            primary_concerns=primary_concerns
        )

        # Limit results
        similar_cases = similar_cases[:limit]

        # Convert to local SimilarCase format for compatibility
        local_similar_cases = []
        for finder_case in similar_cases:
            local_case = SimilarCase(
                case_id=finder_case.case_id,
                client_name=finder_case.client_name,
                estate_value=finder_case.estate_value,
                family_structure=finder_case.family_structure,
                primary_concerns=finder_case.primary_concerns,
                solutions_implemented=finder_case.solutions_implemented,
                outcome=finder_case.outcome,
                success_factors=finder_case.success_factors,
                lessons_learned=finder_case.lessons_learned,
                similarity_score=finder_case.similarity_score,
                case_date=finder_case.case_date,
                advisor=finder_case.advisor
            )
            local_similar_cases.append(local_case)

        # Extract insights from similar cases
        insights = self._extract_case_insights(local_similar_cases)

        return {
            "similar_cases": [self._case_to_dict(case) for case in local_similar_cases],
            "total_found": len(local_similar_cases),
            "insights": insights,
            "search_criteria": input_data,
            "matching_factors": [case.matching_factors for case in similar_cases],
            "similarity_breakdown": [case.similarity_breakdown for case in similar_cases],
            "timestamp": datetime.now().isoformat()
        }

    async def _retrieve_rebuttals(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve effective rebuttals for objections."""
        objection_text = input_data.get("objection_text", "")
        client_context = input_data.get("client_context", {})
        limit = input_data.get("limit", 3)

        # Analyze the objection using RebuttalLibrary
        objection_analysis = await self.rebuttal_library.analyze_objection(
            objection_text, client_context
        )

        # Get rebuttal recommendations
        recommendations = self.rebuttal_library.get_effective_rebuttals(
            objection_analysis, max_results=limit
        )

        # Convert recommendations to response format
        rebuttals = []
        for rec in recommendations:
            rebuttal_data = {
                "rebuttal_id": rec.rebuttal_pattern.rebuttal_id,
                "rebuttal_text": rec.rebuttal_pattern.rebuttal_text,
                "rebuttal_type": rec.rebuttal_pattern.rebuttal_type.value,
                "effectiveness_score": rec.rebuttal_pattern.effectiveness_score,
                "success_rate": rec.rebuttal_pattern.success_rate,
                "relevance_score": rec.relevance_score,
                "usage_count": rec.rebuttal_pattern.usage_count,
                "context_tags": rec.rebuttal_pattern.context_tags,
                "customization_suggestions": rec.customization_suggestions,
                "context_match_factors": rec.context_match_factors,
                "risk_factors": rec.risk_factors,
                "follow_up_strategy": rec.follow_up_strategy
            }
            rebuttals.append(rebuttal_data)

        return {
            "objection_analysis": {
                "category": objection_analysis.category.value,
                "emotional_intensity": objection_analysis.emotional_intensity,
                "specific_concerns": objection_analysis.specific_concerns,
                "urgency_level": objection_analysis.urgency_level,
                "confidence_score": objection_analysis.confidence_score
            },
            "rebuttals": rebuttals,
            "total_found": len(rebuttals),
            "search_criteria": input_data,
            "timestamp": datetime.now().isoformat()
        }

    async def _search_content(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Search knowledge base for relevant content."""
        search_query = input_data.get("search_query", "")
        content_types = input_data.get("content_types", [])
        limit = input_data.get("limit", 10)

        search_results = []

        for content_id, content in self.content_index.items():
            # Filter by content type if specified
            if content_types and content.content_type not in content_types:
                continue

            # Calculate relevance score
            relevance_score = self._calculate_content_relevance(search_query, content)

            if relevance_score > 0.1:  # Minimum relevance threshold
                search_results.append((content, relevance_score))

        # Sort by relevance
        search_results.sort(key=lambda x: x[1], reverse=True)
        search_results = search_results[:limit]

        return {
            "search_results": [
                {
                    "title": content.title,
                    "excerpt": content.excerpt,
                    "content_type": content.content_type,
                    "relevance_score": relevance_score,
                    "source_file": content.source_file,
                    "metadata": content.metadata
                }
                for content, relevance_score in search_results
            ],
            "total_found": len(search_results),
            "search_query": search_query,
            "timestamp": datetime.now().isoformat()
        }

    async def _analyze_patterns(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns across successful cases."""
        analysis_type = input_data.get("analysis_type", "success_factors")
        time_period = input_data.get("time_period_days", 90)

        cutoff_date = datetime.now() - timedelta(days=time_period)

        if analysis_type == "success_factors":
            patterns = self._analyze_success_factors(cutoff_date)
        elif analysis_type == "warning_signs":
            patterns = self._analyze_warning_signs(cutoff_date)
        elif analysis_type == "timing_patterns":
            patterns = self._analyze_timing_patterns(cutoff_date)
        else:
            patterns = []

        return {
            "patterns": [self._pattern_to_dict(pattern) for pattern in patterns],
            "analysis_type": analysis_type,
            "time_period_days": time_period,
            "patterns_found": len(patterns),
            "timestamp": datetime.now().isoformat()
        }

    async def _generate_case_study(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a case study from successful interaction."""
        transcript_content = input_data.get("transcript_content", "")
        outcome = input_data.get("outcome", "")
        client_profile = input_data.get("client_profile", {})

        # Extract key elements from transcript
        key_moments = self._extract_key_moments(transcript_content)
        success_factors = self._identify_success_factors(transcript_content, outcome)
        lessons_learned = self._extract_lessons_learned(transcript_content, outcome)

        # Generate case study
        case_study = {
            "case_id": f"CS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "title": self._generate_case_title(client_profile, outcome),
            "summary": self._generate_case_summary(client_profile, key_moments, outcome),
            "client_profile": client_profile,
            "key_moments": key_moments,
            "success_factors": success_factors,
            "lessons_learned": lessons_learned,
            "outcome": outcome,
            "created_date": datetime.now().isoformat(),
            "tags": self._generate_case_tags(transcript_content, client_profile)
        }

        # Store case study in knowledge base
        await self._store_case_study(case_study)

        return {
            "case_study": case_study,
            "insights_extracted": len(success_factors) + len(lessons_learned),
            "timestamp": datetime.now().isoformat()
        }

    async def _build_knowledge_index(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build or rebuild the knowledge base index."""
        source_directory = input_data.get("source_directory", "data/transcripts")
        content_types = input_data.get("content_types", ["transcript", "case_study"])

        indexed_count = 0
        error_count = 0

        for content_type in content_types:
            if content_type == "transcript":
                count, errors = await self._index_transcripts(source_directory)
            elif content_type == "case_study":
                count, errors = await self._index_case_studies()
            else:
                continue

            indexed_count += count
            error_count += errors

        return {
            "indexed_items": indexed_count,
            "errors": error_count,
            "content_types_processed": content_types,
            "index_size": len(self.content_index),
            "timestamp": datetime.now().isoformat()
        }

    async def _extract_best_practices(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract best practices from successful cases."""
        category = input_data.get("category", "all")
        min_success_rate = input_data.get("min_success_rate", 0.8)

        best_practices = []

        # Analyze rebuttal patterns
        effective_rebuttals = [
            r for r in self.rebuttal_patterns
            if r.success_rate >= min_success_rate and r.usage_count >= 3
        ]

        for rebuttal in effective_rebuttals:
            best_practices.append({
                "type": "rebuttal_technique",
                "category": rebuttal.objection_type,
                "description": f"Effective rebuttal for {rebuttal.objection_type}",
                "technique": rebuttal.rebuttal_text,
                "success_rate": rebuttal.success_rate,
                "usage_count": rebuttal.usage_count,
                "context_tags": rebuttal.context_tags
            })

        # Analyze successful case patterns
        successful_cases = [
            c for c in self.similar_cases
            if c.outcome == "closed_won" and c.case_date >= datetime.now() - timedelta(days=180)
        ]

        pattern_frequency = {}
        for case in successful_cases:
            for factor in case.success_factors:
                pattern_frequency[factor] = pattern_frequency.get(factor, 0) + 1

        for pattern, frequency in pattern_frequency.items():
            if frequency >= 3:  # Minimum frequency threshold
                best_practices.append({
                    "type": "success_pattern",
                    "category": "general",
                    "description": pattern,
                    "frequency": frequency,
                    "success_rate": frequency / len(successful_cases)
                })

        return {
            "best_practices": best_practices,
            "total_practices": len(best_practices),
            "analysis_criteria": input_data,
            "timestamp": datetime.now().isoformat()
        }

    # Helper methods
    def _calculate_case_similarity(self, estate_value: float, family_structure: str,
                                 concerns: List[str], case: SimilarCase) -> float:
        """Calculate similarity score between search criteria and a case."""
        score = 0.0

        # Estate value similarity (30% weight)
        if estate_value > 0 and case.estate_value > 0:
            value_ratio = min(estate_value, case.estate_value) / max(estate_value, case.estate_value)
            score += value_ratio * 0.3

        # Family structure similarity (25% weight)
        if family_structure and case.family_structure:
            structure_similarity = self._calculate_text_similarity(family_structure, case.family_structure)
            score += structure_similarity * 0.25

        # Concerns overlap (45% weight)
        if concerns and case.primary_concerns:
            concern_overlap = len(set(concerns) & set(case.primary_concerns)) / len(set(concerns) | set(case.primary_concerns))
            score += concern_overlap * 0.45

        return score

    def _calculate_rebuttal_relevance(self, objection_text: str, objection_category: str,
                                    rebuttal: RebuttalPattern) -> float:
        """Calculate relevance score for a rebuttal pattern."""
        score = 0.0

        # Category match (60% weight)
        if objection_category and objection_category == rebuttal.objection_type:
            score += 0.6

        # Text similarity (40% weight)
        if objection_text and rebuttal.objection_text:
            text_similarity = self._calculate_text_similarity(objection_text, rebuttal.objection_text)
            score += text_similarity * 0.4

        return score

    def _calculate_content_relevance(self, search_query: str, content: ContentSearchResult) -> float:
        """Calculate relevance score for content search."""
        query_words = set(search_query.lower().split())

        # Title relevance (40% weight)
        title_words = set(content.title.lower().split())
        title_overlap = len(query_words & title_words) / len(query_words | title_words) if query_words | title_words else 0

        # Content relevance (60% weight)
        content_words = set(content.excerpt.lower().split())
        content_overlap = len(query_words & content_words) / len(query_words | content_words) if query_words | content_words else 0

        return (title_overlap * 0.4) + (content_overlap * 0.6)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _extract_case_insights(self, similar_cases: List[SimilarCase]) -> List[str]:
        """Extract insights from similar cases."""
        insights = []

        if not similar_cases:
            return insights

        # Common success factors
        all_success_factors = []
        for case in similar_cases:
            all_success_factors.extend(case.success_factors)

        factor_counts = {}
        for factor in all_success_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1

        common_factors = [factor for factor, count in factor_counts.items() if count >= len(similar_cases) * 0.5]

        if common_factors:
            insights.append(f"Common success factors: {', '.join(common_factors)}")

        # Average estate value
        avg_estate_value = sum(case.estate_value for case in similar_cases) / len(similar_cases)
        insights.append(f"Average estate value in similar cases: ${avg_estate_value:,.0f}")

        return insights

    def _case_to_dict(self, case: SimilarCase) -> Dict[str, Any]:
        """Convert SimilarCase to dictionary."""
        return {
            "case_id": case.case_id,
            "client_name": case.client_name,
            "estate_value": case.estate_value,
            "family_structure": case.family_structure,
            "primary_concerns": case.primary_concerns,
            "solutions_implemented": case.solutions_implemented,
            "outcome": case.outcome,
            "success_factors": case.success_factors,
            "lessons_learned": case.lessons_learned,
            "similarity_score": case.similarity_score,
            "case_date": case.case_date.isoformat(),
            "advisor": case.advisor
        }

    def _pattern_to_dict(self, pattern: KnowledgePattern) -> Dict[str, Any]:
        """Convert KnowledgePattern to dictionary."""
        return {
            "pattern_id": pattern.pattern_id,
            "pattern_type": pattern.pattern_type,
            "description": pattern.description,
            "indicators": pattern.indicators,
            "frequency": pattern.frequency,
            "confidence_score": pattern.confidence_score,
            "supporting_cases": pattern.supporting_cases,
            "recommendations": pattern.recommendations
        }

    async def _load_knowledge_base(self):
        """Load existing knowledge base data."""
        try:
            # Load similar cases
            cases_file = self.knowledge_base_path / "similar_cases.json"
            if cases_file.exists():
                with open(cases_file, 'r') as f:
                    cases_data = json.load(f)
                    self.similar_cases = [
                        SimilarCase(**case_data) for case_data in cases_data
                    ]

            # Load rebuttal patterns
            rebuttals_file = self.knowledge_base_path / "rebuttal_patterns.json"
            if rebuttals_file.exists():
                with open(rebuttals_file, 'r') as f:
                    rebuttals_data = json.load(f)
                    self.rebuttal_patterns = [
                        RebuttalPattern(**rebuttal_data) for rebuttal_data in rebuttals_data
                    ]

            self.logger.info(f"Loaded {len(self.similar_cases)} cases and {len(self.rebuttal_patterns)} rebuttals")

        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {str(e)}")

    # Additional helper methods implementation

    def _analyze_success_factors(self, cutoff_date: datetime) -> List[KnowledgePattern]:
        """Analyze success factors from recent cases."""
        patterns = []
        recent_cases = [
            case for case in self.similar_cases
            if case.case_date >= cutoff_date and case.outcome == "closed_won"
        ]

        if not recent_cases:
            return patterns

        # Analyze common success factors
        factor_frequency = {}
        for case in recent_cases:
            for factor in case.success_factors:
                factor_frequency[factor] = factor_frequency.get(factor, 0) + 1

        # Create patterns for factors that appear in at least 30% of cases
        threshold = max(1, len(recent_cases) * 0.3)

        for factor, frequency in factor_frequency.items():
            if frequency >= threshold:
                confidence_score = frequency / len(recent_cases)
                supporting_cases = [
                    case.case_id for case in recent_cases
                    if factor in case.success_factors
                ]

                pattern = KnowledgePattern(
                    pattern_id=f"SF_{hash(factor) % 10000}",
                    pattern_type="success_factor",
                    description=f"Success factor: {factor}",
                    indicators=[factor],
                    frequency=frequency,
                    confidence_score=confidence_score,
                    supporting_cases=supporting_cases[:5],  # Limit to top 5
                    recommendations=[f"Focus on {factor} in future cases"]
                )
                patterns.append(pattern)

        return patterns

    def _analyze_warning_signs(self, cutoff_date: datetime) -> List[KnowledgePattern]:
        """Analyze warning signs from failed cases."""
        patterns = []
        failed_cases = [
            case for case in self.similar_cases
            if case.case_date >= cutoff_date and case.outcome in ["closed_lost", "stalled"]
        ]

        if not failed_cases:
            return patterns

        # Extract common warning indicators from failed cases
        warning_indicators = []
        for case in failed_cases:
            # Look for patterns in primary concerns that led to failure
            for concern in case.primary_concerns:
                if any(word in concern.lower() for word in ['cost', 'expensive', 'complicated', 'time']):
                    warning_indicators.append(concern)

        # Analyze frequency of warning signs
        indicator_frequency = {}
        for indicator in warning_indicators:
            indicator_frequency[indicator] = indicator_frequency.get(indicator, 0) + 1

        threshold = max(1, len(failed_cases) * 0.2)

        for indicator, frequency in indicator_frequency.items():
            if frequency >= threshold:
                confidence_score = frequency / len(failed_cases)
                supporting_cases = [
                    case.case_id for case in failed_cases
                    if indicator in case.primary_concerns
                ]

                pattern = KnowledgePattern(
                    pattern_id=f"WS_{hash(indicator) % 10000}",
                    pattern_type="warning_sign",
                    description=f"Warning sign: {indicator}",
                    indicators=[indicator],
                    frequency=frequency,
                    confidence_score=confidence_score,
                    supporting_cases=supporting_cases[:5],
                    recommendations=[f"Address {indicator} concerns early in the process"]
                )
                patterns.append(pattern)

        return patterns

    def _analyze_timing_patterns(self, cutoff_date: datetime) -> List[KnowledgePattern]:
        """Analyze timing patterns in successful cases."""
        patterns = []
        recent_cases = [
            case for case in self.similar_cases
            if case.case_date >= cutoff_date
        ]

        if not recent_cases:
            return patterns

        # Analyze seasonal patterns
        monthly_success = {}
        monthly_total = {}

        for case in recent_cases:
            month = case.case_date.month
            monthly_total[month] = monthly_total.get(month, 0) + 1
            if case.outcome == "closed_won":
                monthly_success[month] = monthly_success.get(month, 0) + 1

        # Find months with above-average success rates
        overall_success_rate = len([c for c in recent_cases if c.outcome == "closed_won"]) / len(recent_cases)

        for month, total in monthly_total.items():
            if total >= 3:  # Minimum sample size
                success_count = monthly_success.get(month, 0)
                success_rate = success_count / total

                if success_rate > overall_success_rate * 1.2:  # 20% above average
                    pattern = KnowledgePattern(
                        pattern_id=f"TP_MONTH_{month}",
                        pattern_type="timing_indicator",
                        description=f"High success rate in month {month}",
                        indicators=[f"month_{month}"],
                        frequency=success_count,
                        confidence_score=success_rate,
                        supporting_cases=[
                            case.case_id for case in recent_cases
                            if case.case_date.month == month and case.outcome == "closed_won"
                        ][:5],
                        recommendations=[f"Increase marketing efforts in month {month}"]
                    )
                    patterns.append(pattern)

        return patterns

    def _extract_key_moments(self, transcript_content: str) -> List[str]:
        """Extract key moments from transcript content."""
        key_moments = []
        sentences = transcript_content.split('.')

        # Keywords that indicate key moments
        moment_indicators = [
            'breakthrough', 'aha', 'understand', 'makes sense', 'convinced',
            'decided', 'ready to move forward', 'let\'s do it', 'sounds good',
            'that addresses my concern', 'perfect solution', 'exactly what I need'
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Skip very short sentences
                sentence_lower = sentence.lower()
                for indicator in moment_indicators:
                    if indicator in sentence_lower:
                        key_moments.append(sentence)
                        break

        return key_moments[:5]  # Limit to top 5 key moments

    def _identify_success_factors(self, transcript_content: str, outcome: str) -> List[str]:
        """Identify success factors from transcript and outcome."""
        success_factors = []

        if outcome != "closed_won":
            return success_factors

        content_lower = transcript_content.lower()

        # Analyze transcript for success indicators
        success_patterns = {
            'clear_value_proposition': ['value', 'benefit', 'saves money', 'protects'],
            'addressed_concerns': ['understand your concern', 'that makes sense', 'let me address'],
            'family_focus': ['family', 'children', 'spouse', 'legacy'],
            'urgency_creation': ['important to act', 'time sensitive', 'sooner rather than later'],
            'trust_building': ['trust me', 'experience', 'helped many families', 'track record'],
            'clear_next_steps': ['next step', 'here\'s what we do', 'process is simple']
        }

        for factor, patterns in success_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                success_factors.append(factor)

        return success_factors

    def _extract_lessons_learned(self, transcript_content: str, outcome: str) -> List[str]:
        """Extract lessons learned from the interaction."""
        lessons = []
        content_lower = transcript_content.lower()

        if outcome == "closed_won":
            # Lessons from successful cases
            if 'tax' in content_lower and 'savings' in content_lower:
                lessons.append("Tax savings messaging resonates well with clients")
            if 'simple' in content_lower and 'process' in content_lower:
                lessons.append("Emphasizing simplicity of the process helps close deals")
            if 'family' in content_lower and 'protect' in content_lower:
                lessons.append("Family protection angle is compelling for estate planning clients")

        elif outcome in ["closed_lost", "stalled"]:
            # Lessons from failed cases
            if 'expensive' in content_lower or 'cost' in content_lower:
                lessons.append("Cost objections need to be addressed earlier in the process")
            if 'complicated' in content_lower or 'complex' in content_lower:
                lessons.append("Clients need more education on trust structures")
            if 'think about it' in content_lower:
                lessons.append("Create more urgency and clear next steps")

        return lessons

    def _generate_case_title(self, client_profile: Dict[str, Any], outcome: str) -> str:
        """Generate a descriptive title for the case study."""
        estate_value = client_profile.get('estate_value', 0)
        family_structure = client_profile.get('family_structure', 'family')

        if estate_value > 5000000:
            estate_desc = "High Net Worth"
        elif estate_value > 1000000:
            estate_desc = "Affluent"
        else:
            estate_desc = "Middle Market"

        outcome_desc = "Successful" if outcome == "closed_won" else "Challenging"

        return f"{outcome_desc} {estate_desc} Estate Planning Case - {family_structure}"

    def _generate_case_summary(self, client_profile: Dict[str, Any],
                             key_moments: List[str], outcome: str) -> str:
        """Generate a summary of the case study."""
        estate_value = client_profile.get('estate_value', 0)
        family_structure = client_profile.get('family_structure', 'family')

        summary = f"Estate planning consultation for {family_structure} with ${estate_value:,.0f} estate. "

        if key_moments:
            summary += f"Key breakthrough: {key_moments[0][:100]}... "

        if outcome == "closed_won":
            summary += "Successfully closed with comprehensive trust solution."
        else:
            summary += "Case stalled due to client concerns requiring follow-up."

        return summary

    def _generate_case_tags(self, transcript_content: str,
                          client_profile: Dict[str, Any]) -> List[str]:
        """Generate tags for the case study."""
        tags = []
        content_lower = transcript_content.lower()

        # Estate value tags
        estate_value = client_profile.get('estate_value', 0)
        if estate_value > 5000000:
            tags.append("high_net_worth")
        elif estate_value > 1000000:
            tags.append("affluent")
        else:
            tags.append("middle_market")

        # Content-based tags
        tag_patterns = {
            'trust_focus': ['trust', 'revocable', 'irrevocable'],
            'tax_planning': ['tax', 'estate tax', 'gift tax'],
            'business_succession': ['business', 'succession', 'company'],
            'charitable_giving': ['charity', 'charitable', 'foundation'],
            'generation_skipping': ['grandchildren', 'generation skipping', 'gst'],
            'asset_protection': ['creditor', 'protection', 'shield']
        }

        for tag, patterns in tag_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                tags.append(tag)

        # Family structure tags
        family_structure = client_profile.get('family_structure', '')
        if 'children' in family_structure:
            tags.append("with_children")
        if 'spouse' in family_structure:
            tags.append("married")

        return tags

    async def _store_case_study(self, case_study: Dict[str, Any]):
        """Store case study in the knowledge base."""
        case_studies_file = self.knowledge_base_path / "case_studies.json"

        # Load existing case studies
        case_studies = []
        if case_studies_file.exists():
            try:
                with open(case_studies_file, 'r') as f:
                    case_studies = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading case studies: {e}")

        # Add new case study
        case_studies.append(case_study)

        # Save back to file
        try:
            with open(case_studies_file, 'w') as f:
                json.dump(case_studies, f, indent=2, default=str)
            self.logger.info(f"Stored case study: {case_study['case_id']}")
        except Exception as e:
            self.logger.error(f"Error storing case study: {e}")

    async def _index_transcripts(self, source_directory: str) -> Tuple[int, int]:
        """Index transcript files for content search."""
        indexed_count = 0
        error_count = 0

        transcript_dir = Path(source_directory)
        if not transcript_dir.exists():
            self.logger.warning(f"Transcript directory does not exist: {source_directory}")
            return 0, 1

        # Process transcript files
        for file_path in transcript_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract title from filename or first line
                title = file_path.stem.replace('_', ' ').title()
                excerpt = content[:200] + "..." if len(content) > 200 else content

                # Create content search result
                content_result = ContentSearchResult(
                    content_id=f"transcript_{file_path.stem}",
                    content_type="transcript",
                    title=title,
                    excerpt=excerpt,
                    full_content=content,
                    relevance_score=1.0,  # Base relevance
                    metadata={
                        "file_path": str(file_path),
                        "file_size": len(content),
                        "word_count": len(content.split())
                    },
                    source_file=str(file_path),
                    created_date=datetime.fromtimestamp(file_path.stat().st_mtime)
                )

                self.content_index[content_result.content_id] = content_result
                indexed_count += 1

            except Exception as e:
                self.logger.error(f"Error indexing transcript {file_path}: {e}")
                error_count += 1

        return indexed_count, error_count

    async def _index_case_studies(self) -> Tuple[int, int]:
        """Index case studies for content search."""
        indexed_count = 0
        error_count = 0

        case_studies_file = self.knowledge_base_path / "case_studies.json"
        if not case_studies_file.exists():
            return 0, 0

        try:
            with open(case_studies_file, 'r') as f:
                case_studies = json.load(f)

            for case_study in case_studies:
                try:
                    # Create content search result for case study
                    content_result = ContentSearchResult(
                        content_id=case_study['case_id'],
                        content_type="case_study",
                        title=case_study.get('title', ''),
                        excerpt=case_study.get('summary', '')[:200],
                        full_content=json.dumps(case_study, indent=2),
                        relevance_score=1.0,
                        metadata={
                            "outcome": case_study.get('outcome', ''),
                            "tags": case_study.get('tags', []),
                            "client_profile": case_study.get('client_profile', {})
                        },
                        source_file=str(case_studies_file),
                        created_date=datetime.fromisoformat(case_study.get('created_date', datetime.now().isoformat()))
                    )

                    self.content_index[content_result.content_id] = content_result
                    indexed_count += 1

                except Exception as e:
                    self.logger.error(f"Error indexing case study {case_study.get('case_id', 'unknown')}: {e}")
                    error_count += 1

        except Exception as e:
            self.logger.error(f"Error loading case studies file: {e}")
            error_count += 1

        return indexed_count, error_count


# Example usage
async def main():
    """Example usage of Estate Librarian Agent."""
    librarian = EstateLibrarianAgent()

    # Test similar case search
    search_task = SpecialistTask(
        task_type="find_similar_cases",
        description="Find similar cases for estate planning",
        input_data={
            "estate_value": 2500000,
            "family_structure": "married with 2 adult children",
            "primary_concerns": ["tax minimization", "family harmony"],
            "limit": 3
        }
    )

    await librarian.assign_task(search_task)
    result = await librarian.execute_task(search_task.task_id)

    print(f"Estate Librarian search result: {result}")
    print(f"Agent status: {librarian.get_status()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())