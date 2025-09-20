"""
Intelligent query validation module with clarifying questions.
Validates user queries and generates clarifying questions when needed.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    SIMPLE_LOOKUP = "simple_lookup"
    COMPLEX_ANALYSIS = "complex_analysis"
    COMPARISON = "comparison"
    TREND_ANALYSIS = "trend_analysis"
    AGGREGATION = "aggregation"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    is_valid: bool
    confidence: float
    query_type: QueryType
    clarifying_questions: List[str]
    suggested_refinements: List[str]
    missing_context: List[str]


class QueryValidator:
    """Validates queries and generates clarifying questions for better analysis."""

    def __init__(self):
        self.query_patterns = {
            QueryType.SIMPLE_LOOKUP: [
                r'\bwhat is\b',
                r'\bwho is\b',
                r'\bwhen was\b',
                r'\bwhere is\b',
                r'\bdefine\b',
                r'\bexplain\b'
            ],
            QueryType.COMPLEX_ANALYSIS: [
                r'\banalyze\b',
                r'\bevaluate\b',
                r'\bassess\b',
                r'\bcompare and contrast\b',
                r'\bwhat are the implications\b'
            ],
            QueryType.COMPARISON: [
                r'\bcompare\b',
                r'\bvs\b',
                r'\bversus\b',
                r'\bdifference between\b',
                r'\bsimilarities\b'
            ],
            QueryType.TREND_ANALYSIS: [
                r'\btrend\b',
                r'\bover time\b',
                r'\bhistorical\b',
                r'\bevolution\b',
                r'\bchanges\b'
            ],
            QueryType.AGGREGATION: [
                r'\btotal\b',
                r'\bsum\b',
                r'\baverage\b',
                r'\bcount\b',
                r'\bstatistics\b'
            ]
        }

        self.domain_keywords = {
            'financial': ['revenue', 'profit', 'loss', 'investment', 'market', 'stock'],
            'technical': ['algorithm', 'code', 'system', 'architecture', 'performance'],
            'business': ['strategy', 'competition', 'customer', 'growth', 'market share'],
            'research': ['study', 'hypothesis', 'methodology', 'results', 'conclusion']
        }

    def validate_query(self, query: str) -> ValidationResult:
        """
        Validate a user query and generate clarifying questions if needed.

        Args:
            query: The user's input query

        Returns:
            ValidationResult with validation status and recommendations
        """
        if not query or len(query.strip()) < 3:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                query_type=QueryType.UNKNOWN,
                clarifying_questions=["Could you please provide a more detailed question?"],
                suggested_refinements=[],
                missing_context=["Query is too short or empty"]
            )

        query_lower = query.lower()
        query_type = self._classify_query(query_lower)
        confidence = self._calculate_confidence(query_lower, query_type)

        clarifying_questions = self._generate_clarifying_questions(query_lower, query_type)
        suggested_refinements = self._generate_refinements(query_lower, query_type)
        missing_context = self._identify_missing_context(query_lower, query_type)

        is_valid = confidence > 0.6 and len(missing_context) <= 2

        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            query_type=query_type,
            clarifying_questions=clarifying_questions,
            suggested_refinements=suggested_refinements,
            missing_context=missing_context
        )

    def _classify_query(self, query: str) -> QueryType:
        """Classify the query type based on patterns."""
        max_matches = 0
        best_type = QueryType.UNKNOWN

        for query_type, patterns in self.query_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, query))
            if matches > max_matches:
                max_matches = matches
                best_type = query_type

        return best_type

    def _calculate_confidence(self, query: str, query_type: QueryType) -> float:
        """Calculate confidence score for the query classification."""
        if query_type == QueryType.UNKNOWN:
            return 0.3

        pattern_matches = 0
        total_patterns = len(self.query_patterns[query_type])

        for pattern in self.query_patterns[query_type]:
            if re.search(pattern, query):
                pattern_matches += 1

        pattern_score = pattern_matches / total_patterns if total_patterns > 0 else 0

        # Check for domain-specific keywords
        domain_score = 0
        for domain, keywords in self.domain_keywords.items():
            domain_matches = sum(1 for keyword in keywords if keyword in query)
            domain_score = max(domain_score, domain_matches / len(keywords))

        # Consider query length and complexity
        length_score = min(len(query.split()) / 10, 1.0)

        # Weighted combination
        confidence = 0.5 * pattern_score + 0.3 * domain_score + 0.2 * length_score
        return min(confidence, 1.0)

    def _generate_clarifying_questions(self, query: str, query_type: QueryType) -> List[str]:
        """Generate clarifying questions based on query type and content."""
        questions = []

        if query_type == QueryType.SIMPLE_LOOKUP:
            if not any(word in query for word in ['specific', 'detailed', 'comprehensive']):
                questions.append("Would you like a detailed explanation or a brief overview?")

        elif query_type == QueryType.COMPLEX_ANALYSIS:
            questions.extend([
                "What specific aspects would you like me to focus on?",
                "Are there particular metrics or criteria you'd like me to consider?"
            ])

        elif query_type == QueryType.COMPARISON:
            if query.count('and') < 1 and query.count('vs') < 1:
                questions.append("What specific items would you like me to compare?")
            questions.append("What criteria should I use for the comparison?")

        elif query_type == QueryType.TREND_ANALYSIS:
            questions.extend([
                "What time period would you like me to analyze?",
                "Are you interested in specific trends or patterns?"
            ])

        elif query_type == QueryType.AGGREGATION:
            questions.append("What data sources should I consider for this analysis?")

        # Add general questions for ambiguous queries
        if len(questions) == 0 or query_type == QueryType.UNKNOWN:
            questions.extend([
                "Could you provide more context about what you're looking for?",
                "What would be most helpful for your current task?"
            ])

        return questions[:3]  # Limit to 3 questions to avoid overwhelming

    def _generate_refinements(self, query: str, query_type: QueryType) -> List[str]:
        """Generate suggested query refinements."""
        refinements = []

        if query_type == QueryType.SIMPLE_LOOKUP:
            refinements.append(f"For a comprehensive analysis: 'Provide a detailed analysis of {query.replace(\"what is\", \"\").strip()}'")

        elif query_type == QueryType.COMPARISON:
            refinements.append("Consider specifying the comparison criteria or context")

        elif query_type == QueryType.TREND_ANALYSIS:
            refinements.append("Include specific time periods or data points of interest")

        if len(query.split()) < 5:
            refinements.append("Consider adding more specific details or context")

        return refinements

    def _identify_missing_context(self, query: str, query_type: QueryType) -> List[str]:
        """Identify missing context that could improve the query."""
        missing = []

        # Check for time context
        time_indicators = ['when', 'time', 'period', 'date', 'year', 'month']
        if not any(indicator in query for indicator in time_indicators):
            if query_type in [QueryType.TREND_ANALYSIS, QueryType.AGGREGATION]:
                missing.append("Time period or timeframe")

        # Check for scope context
        scope_indicators = ['all', 'some', 'specific', 'particular', 'which']
        if not any(indicator in query for indicator in scope_indicators):
            if query_type == QueryType.COMPARISON:
                missing.append("Scope or specific items to compare")

        # Check for domain context
        domain_found = False
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query for keyword in keywords):
                domain_found = True
                break

        if not domain_found:
            missing.append("Domain or subject area context")

        # Check for objective context
        objective_indicators = ['why', 'how', 'purpose', 'goal', 'objective']
        if not any(indicator in query for indicator in objective_indicators):
            if query_type == QueryType.COMPLEX_ANALYSIS:
                missing.append("Analysis objective or purpose")

        return missing

    def suggest_follow_up_queries(self, original_query: str, validation_result: ValidationResult) -> List[str]:
        """Suggest follow-up queries based on the validation result."""
        follow_ups = []

        if validation_result.query_type == QueryType.SIMPLE_LOOKUP:
            follow_ups.extend([
                f"What are the implications of {original_query.lower()}?",
                f"How does {original_query.lower()} compare to alternatives?",
                f"What are the latest developments regarding {original_query.lower()}?"
            ])

        elif validation_result.query_type == QueryType.COMPLEX_ANALYSIS:
            follow_ups.extend([
                f"What are the key metrics for {original_query.lower()}?",
                f"What are the potential risks in {original_query.lower()}?",
                f"How can we improve {original_query.lower()}?"
            ])

        return follow_ups[:3]