"""
Enhanced Transcript Processor for Stellar Connect
Implements Story 5.1: Enhanced Sales Knowledge Base with Multi-Layered Understanding

This module provides structure-aware transcript parsing that identifies sales methodology
sections and generates domain-specific metadata for estate planning conversations.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import asyncio
import logging
from datetime import datetime


@dataclass
class SalesSection:
    """Represents a section of a sales conversation with methodology awareness."""
    section_type: str  # discovery, needs_analysis, presentation, objection_handling, closing, next_steps
    content: str
    start_position: int
    end_position: int
    confidence_score: float
    key_moments: List[str]
    emotional_indicators: List[str]


@dataclass
class EstateMetadata:
    """Domain-specific metadata for estate planning conversations."""
    estate_value_indicators: List[str]
    family_structure_mentions: List[str]
    objection_categories: List[str]
    closing_probability: float
    urgency_indicators: List[str]
    trust_vs_will_discussion: bool
    tax_planning_focus: bool
    generation_skipping_mentioned: bool


@dataclass
class EnhancedTranscript:
    """Enhanced transcript with multi-layered understanding."""
    original_content: str
    sales_sections: List[SalesSection]
    estate_metadata: EstateMetadata
    knowledge_layers: Dict[str, any]
    processing_timestamp: datetime
    confidence_score: float


class EstateSectionParser:
    """Parses transcripts to identify sales methodology sections."""

    def __init__(self):
        self.section_patterns = {
            'discovery': [
                'tell me about', 'help me understand', 'what brings you',
                'family situation', 'current estate', 'assets include'
            ],
            'needs_analysis': [
                'concerns about', 'worried about', 'priority is',
                'important to', 'goals for', 'planning objectives'
            ],
            'presentation': [
                'recommend', 'solution would be', 'trust structure',
                'benefit of this', 'this approach', 'strategy involves'
            ],
            'objection_handling': [
                'but what about', 'concerned that', 'not sure if',
                'expensive', 'complicated', 'really necessary'
            ],
            'closing': [
                'ready to move forward', 'next step would be',
                'start the process', 'schedule', 'paperwork'
            ],
            'next_steps': [
                'follow up', 'send you', 'next meeting',
                'documents needed', 'timeline', 'schedule'
            ]
        }

    async def parse(self, transcript_content: str) -> List[SalesSection]:
        """Parse transcript into sales methodology sections."""
        sections = []

        # Split transcript into paragraphs for analysis
        paragraphs = [p.strip() for p in transcript_content.split('\n\n') if p.strip()]

        for i, paragraph in enumerate(paragraphs):
            section_type = self._identify_section_type(paragraph)
            if section_type:
                section = SalesSection(
                    section_type=section_type,
                    content=paragraph,
                    start_position=i,
                    end_position=i,
                    confidence_score=self._calculate_confidence(paragraph, section_type),
                    key_moments=self._extract_key_moments(paragraph),
                    emotional_indicators=self._extract_emotional_indicators(paragraph)
                )
                sections.append(section)

        return sections

    def _identify_section_type(self, content: str) -> Optional[str]:
        """Identify the sales methodology section type."""
        content_lower = content.lower()

        # Score each section type based on pattern matches
        scores = {}
        for section_type, patterns in self.section_patterns.items():
            score = sum(1 for pattern in patterns if pattern in content_lower)
            if score > 0:
                scores[section_type] = score

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return None

    def _calculate_confidence(self, content: str, section_type: str) -> float:
        """Calculate confidence score for section classification."""
        content_lower = content.lower()
        patterns = self.section_patterns.get(section_type, [])
        matches = sum(1 for pattern in patterns if pattern in content_lower)

        # Confidence based on pattern matches and content length
        base_confidence = min(matches / len(patterns), 1.0) if patterns else 0.0
        length_factor = min(len(content) / 200, 1.0)  # Longer content generally more reliable

        return (base_confidence * 0.7) + (length_factor * 0.3)

    def _extract_key_moments(self, content: str) -> List[str]:
        """Extract key moments from the content."""
        key_moment_indicators = [
            'breakthrough moment', 'aha moment', 'suddenly understood',
            'changed their mind', 'convinced', 'decided to'
        ]

        content_lower = content.lower()
        moments = []

        for indicator in key_moment_indicators:
            if indicator in content_lower:
                # Extract sentence containing the indicator
                sentences = content.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        moments.append(sentence.strip())

        return moments

    def _extract_emotional_indicators(self, content: str) -> List[str]:
        """Extract emotional indicators from the content."""
        emotional_words = [
            'excited', 'worried', 'concerned', 'relieved', 'frustrated',
            'confident', 'uncertain', 'anxious', 'pleased', 'upset'
        ]

        content_lower = content.lower()
        indicators = []

        for emotion in emotional_words:
            if emotion in content_lower:
                indicators.append(emotion)

        return list(set(indicators))  # Remove duplicates


class EstateMetadataGenerator:
    """Generates domain-specific metadata for estate planning conversations."""

    def __init__(self):
        self.estate_value_patterns = [
            r'\$[\d,]+', r'million', r'thousand', r'estate worth',
            r'assets total', r'net worth', r'property value'
        ]
        self.family_patterns = [
            'children', 'spouse', 'grandchildren', 'siblings',
            'family members', 'heirs', 'beneficiaries'
        ]
        self.objection_patterns = {
            'cost': ['expensive', 'cost', 'fees', 'afford'],
            'complexity': ['complicated', 'complex', 'difficult', 'confusing'],
            'necessity': ['necessary', 'need', 'required', 'must have'],
            'timing': ['rush', 'hurry', 'time', 'urgent', 'delay']
        }

    async def generate(self, sections: List[SalesSection]) -> EstateMetadata:
        """Generate estate-specific metadata from parsed sections."""
        all_content = ' '.join([section.content for section in sections])

        return EstateMetadata(
            estate_value_indicators=self._extract_estate_values(all_content),
            family_structure_mentions=self._extract_family_structure(all_content),
            objection_categories=self._categorize_objections(sections),
            closing_probability=self._calculate_closing_probability(sections),
            urgency_indicators=self._extract_urgency_indicators(all_content),
            trust_vs_will_discussion=self._detect_trust_vs_will_discussion(all_content),
            tax_planning_focus=self._detect_tax_planning_focus(all_content),
            generation_skipping_mentioned=self._detect_generation_skipping(all_content)
        )

    def _extract_estate_values(self, content: str) -> List[str]:
        """Extract estate value indicators from content."""
        import re
        values = []

        # Find monetary values
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        monetary_values = re.findall(money_pattern, content)
        values.extend(monetary_values)

        # Find value descriptors
        value_descriptors = ['million', 'thousand', 'substantial estate', 'significant assets']
        content_lower = content.lower()

        for descriptor in value_descriptors:
            if descriptor in content_lower:
                values.append(descriptor)

        return values

    def _extract_family_structure(self, content: str) -> List[str]:
        """Extract family structure mentions."""
        content_lower = content.lower()
        mentions = []

        for pattern in self.family_patterns:
            if pattern in content_lower:
                mentions.append(pattern)

        return mentions

    def _categorize_objections(self, sections: List[SalesSection]) -> List[str]:
        """Categorize objections found in the conversation."""
        objection_sections = [s for s in sections if s.section_type == 'objection_handling']
        categories = []

        for section in objection_sections:
            content_lower = section.content.lower()
            for category, patterns in self.objection_patterns.items():
                if any(pattern in content_lower for pattern in patterns):
                    categories.append(category)

        return list(set(categories))

    def _calculate_closing_probability(self, sections: List[SalesSection]) -> float:
        """Calculate probability of closing based on conversation analysis."""
        # Scoring factors
        discovery_score = len([s for s in sections if s.section_type == 'discovery']) * 0.1
        presentation_score = len([s for s in sections if s.section_type == 'presentation']) * 0.2
        objection_handling_score = len([s for s in sections if s.section_type == 'objection_handling']) * 0.15
        closing_score = len([s for s in sections if s.section_type == 'closing']) * 0.3
        next_steps_score = len([s for s in sections if s.section_type == 'next_steps']) * 0.25

        total_score = discovery_score + presentation_score + objection_handling_score + closing_score + next_steps_score

        # Normalize to 0-1 range
        return min(total_score, 1.0)

    def _extract_urgency_indicators(self, content: str) -> List[str]:
        """Extract urgency indicators from content."""
        urgency_patterns = [
            'urgent', 'soon', 'quickly', 'immediate', 'asap',
            'time sensitive', 'deadline', 'before', 'health issues'
        ]

        content_lower = content.lower()
        indicators = []

        for pattern in urgency_patterns:
            if pattern in content_lower:
                indicators.append(pattern)

        return indicators

    def _detect_trust_vs_will_discussion(self, content: str) -> bool:
        """Detect if trust vs will discussion occurred."""
        trust_indicators = ['trust', 'revocable', 'irrevocable', 'living trust']
        will_indicators = ['will', 'testament', 'last will']

        content_lower = content.lower()
        has_trust = any(indicator in content_lower for indicator in trust_indicators)
        has_will = any(indicator in content_lower for indicator in will_indicators)

        return has_trust and has_will

    def _detect_tax_planning_focus(self, content: str) -> bool:
        """Detect if tax planning was a focus."""
        tax_patterns = ['tax', 'estate tax', 'gift tax', 'irs', 'deduction', 'exemption']
        content_lower = content.lower()

        return any(pattern in content_lower for pattern in tax_patterns)

    def _detect_generation_skipping(self, content: str) -> bool:
        """Detect if generation skipping was mentioned."""
        gs_patterns = ['generation skipping', 'gst', 'skip generation', 'grandchildren trust']
        content_lower = content.lower()

        return any(pattern in content_lower for pattern in gs_patterns)


class MultiLayerKnowledgeBuilder:
    """Builds multi-layered knowledge representation from enhanced transcript data."""

    async def build_layers(self, sections: List[SalesSection], metadata: EstateMetadata) -> Dict[str, any]:
        """Build knowledge layers for enhanced understanding."""
        return {
            'structural_layer': self._build_structural_layer(sections),
            'semantic_layer': await self._build_semantic_layer(sections),
            'relational_layer': self._build_relational_layer(sections, metadata),
            'predictive_layer': self._build_predictive_layer(sections, metadata)
        }

    def _build_structural_layer(self, sections: List[SalesSection]) -> Dict[str, any]:
        """Build structural representation of the conversation."""
        return {
            'conversation_flow': [s.section_type for s in sections],
            'section_confidence': {s.section_type: s.confidence_score for s in sections},
            'key_moments_count': sum(len(s.key_moments) for s in sections),
            'emotional_indicators_count': sum(len(s.emotional_indicators) for s in sections)
        }

    async def _build_semantic_layer(self, sections: List[SalesSection]) -> Dict[str, any]:
        """Build semantic representation for vector search."""
        # This would integrate with embedding models
        # For now, return structured semantic information
        return {
            'main_topics': self._extract_main_topics(sections),
            'conversation_themes': self._identify_themes(sections),
            'sentiment_analysis': self._analyze_sentiment(sections)
        }

    def _build_relational_layer(self, sections: List[SalesSection], metadata: EstateMetadata) -> Dict[str, any]:
        """Build relational connections between conversation elements."""
        return {
            'objection_to_resolution': self._map_objections_to_resolutions(sections),
            'need_to_solution': self._map_needs_to_solutions(sections),
            'family_to_strategy': self._map_family_to_strategy(sections, metadata)
        }

    def _build_predictive_layer(self, sections: List[SalesSection], metadata: EstateMetadata) -> Dict[str, any]:
        """Build predictive insights for future conversations."""
        return {
            'success_indicators': self._identify_success_indicators(sections, metadata),
            'risk_factors': self._identify_risk_factors(sections, metadata),
            'next_best_actions': self._suggest_next_actions(sections, metadata)
        }

    # Helper methods for layer building
    def _extract_main_topics(self, sections: List[SalesSection]) -> List[str]:
        """Extract main topics from conversation sections."""
        topics = []
        for section in sections:
            # Simple keyword extraction - would be enhanced with NLP
            words = section.content.lower().split()
            topic_words = [w for w in words if len(w) > 5 and w.isalpha()]
            topics.extend(topic_words[:3])  # Top 3 words per section

        return list(set(topics))

    def _identify_themes(self, sections: List[SalesSection]) -> List[str]:
        """Identify conversation themes."""
        themes = []
        all_content = ' '.join([s.content for s in sections]).lower()

        theme_patterns = {
            'family_protection': ['protect family', 'family security', 'children future'],
            'tax_optimization': ['reduce taxes', 'tax savings', 'minimize tax'],
            'legacy_planning': ['legacy', 'generational', 'future generations'],
            'asset_protection': ['protect assets', 'creditor protection', 'shield assets']
        }

        for theme, patterns in theme_patterns.items():
            if any(pattern in all_content for pattern in patterns):
                themes.append(theme)

        return themes

    def _analyze_sentiment(self, sections: List[SalesSection]) -> Dict[str, float]:
        """Analyze sentiment for each section type."""
        # Simplified sentiment analysis - would use proper NLP models
        sentiment_scores = {}

        for section in sections:
            positive_words = ['great', 'excellent', 'perfect', 'amazing', 'wonderful']
            negative_words = ['concerned', 'worried', 'problem', 'difficult', 'expensive']

            content_lower = section.content.lower()
            positive_count = sum(1 for word in positive_words if word in content_lower)
            negative_count = sum(1 for word in negative_words if word in content_lower)

            if positive_count + negative_count > 0:
                sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                sentiment = 0.0

            sentiment_scores[section.section_type] = sentiment

        return sentiment_scores

    def _map_objections_to_resolutions(self, sections: List[SalesSection]) -> Dict[str, str]:
        """Map objections to their resolutions."""
        objection_sections = [s for s in sections if s.section_type == 'objection_handling']
        mappings = {}

        for section in objection_sections:
            # Simple pattern matching - would be enhanced with NLP
            content = section.content
            if 'expensive' in content.lower() and 'value' in content.lower():
                mappings['cost_objection'] = 'value_demonstration'
            elif 'complicated' in content.lower() and 'simple' in content.lower():
                mappings['complexity_objection'] = 'simplification_approach'

        return mappings

    def _map_needs_to_solutions(self, sections: List[SalesSection]) -> Dict[str, str]:
        """Map identified needs to proposed solutions."""
        needs_sections = [s for s in sections if s.section_type == 'needs_analysis']
        solution_sections = [s for s in sections if s.section_type == 'presentation']

        mappings = {}
        # This would be enhanced with proper NLP to match needs to solutions
        for i, need_section in enumerate(needs_sections):
            if i < len(solution_sections):
                mappings[f'need_{i}'] = f'solution_{i}'

        return mappings

    def _map_family_to_strategy(self, sections: List[SalesSection], metadata: EstateMetadata) -> Dict[str, str]:
        """Map family structure to recommended strategy."""
        mappings = {}

        if 'children' in metadata.family_structure_mentions:
            mappings['has_children'] = 'generation_skipping_trust'
        if 'spouse' in metadata.family_structure_mentions:
            mappings['married'] = 'marital_deduction_strategy'

        return mappings

    def _identify_success_indicators(self, sections: List[SalesSection], metadata: EstateMetadata) -> List[str]:
        """Identify indicators of likely success."""
        indicators = []

        if metadata.closing_probability > 0.7:
            indicators.append('high_closing_probability')
        if len([s for s in sections if s.section_type == 'next_steps']) > 0:
            indicators.append('next_steps_discussed')
        if len(metadata.urgency_indicators) > 0:
            indicators.append('urgency_present')

        return indicators

    def _identify_risk_factors(self, sections: List[SalesSection], metadata: EstateMetadata) -> List[str]:
        """Identify risk factors that might prevent closing."""
        risks = []

        if 'cost' in metadata.objection_categories:
            risks.append('price_sensitivity')
        if 'complexity' in metadata.objection_categories:
            risks.append('overwhelmed_by_complexity')
        if len([s for s in sections if s.section_type == 'objection_handling']) > 2:
            risks.append('multiple_objections')

        return risks

    def _suggest_next_actions(self, sections: List[SalesSection], metadata: EstateMetadata) -> List[str]:
        """Suggest next best actions based on conversation analysis."""
        actions = []

        if metadata.closing_probability > 0.8:
            actions.append('send_engagement_letter')
        elif metadata.closing_probability > 0.5:
            actions.append('schedule_follow_up_meeting')
        else:
            actions.append('address_remaining_concerns')

        if len(metadata.urgency_indicators) > 0:
            actions.append('expedite_process')

        return actions


class EnhancedTranscriptProcessor:
    """Main processor for enhanced transcript processing with multi-layered understanding."""

    def __init__(self):
        self.section_parser = EstateSectionParser()
        self.metadata_generator = EstateMetadataGenerator()
        self.knowledge_builder = MultiLayerKnowledgeBuilder()
        self.logger = logging.getLogger(__name__)

    async def process_transcript(self, transcript_path: Path) -> EnhancedTranscript:
        """Process a transcript file with enhanced understanding."""
        try:
            # Read transcript content
            with open(transcript_path, 'r', encoding='utf-8') as file:
                content = file.read()

            self.logger.info(f"Processing transcript: {transcript_path}")

            # Parse into sales methodology sections
            sections = await self.section_parser.parse(content)
            self.logger.debug(f"Identified {len(sections)} sales sections")

            # Generate estate-specific metadata
            metadata = await self.metadata_generator.generate(sections)
            self.logger.debug(f"Generated metadata with {len(metadata.objection_categories)} objection categories")

            # Build multi-layered knowledge representation
            knowledge_layers = await self.knowledge_builder.build_layers(sections, metadata)
            self.logger.debug(f"Built {len(knowledge_layers)} knowledge layers")

            # Calculate overall confidence score
            section_confidences = [s.confidence_score for s in sections]
            overall_confidence = sum(section_confidences) / len(section_confidences) if section_confidences else 0.0

            enhanced_transcript = EnhancedTranscript(
                original_content=content,
                sales_sections=sections,
                estate_metadata=metadata,
                knowledge_layers=knowledge_layers,
                processing_timestamp=datetime.now(),
                confidence_score=overall_confidence
            )

            self.logger.info(f"Successfully processed transcript with confidence score: {overall_confidence:.2f}")
            return enhanced_transcript

        except Exception as e:
            self.logger.error(f"Error processing transcript {transcript_path}: {str(e)}")
            raise


# Example usage
async def main():
    """Example usage of enhanced transcript processor."""
    processor = EnhancedTranscriptProcessor()

    # This would process a real transcript file
    # enhanced_transcript = await processor.process_transcript(Path("sample_transcript.txt"))
    # print(f"Processed transcript with {len(enhanced_transcript.sales_sections)} sections")

    print("Enhanced Transcript Processor initialized successfully!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run example
    asyncio.run(main())