"""
Demo script for enhanced query processing.
Demonstrates the capabilities of the advanced query processing system.
"""

import asyncio
import json
from typing import Dict, Any

from query_validator import QueryValidator
from multi_step_executor import MultiStepExecutor
from result_synthesizer import ResultSynthesizer


async def demo_simple_query():
    """Demonstrate processing a simple query."""
    print("=" * 60)
    print("DEMO: Simple Query Processing")
    print("=" * 60)

    query = "What is estate planning?"

    # Initialize components
    validator = QueryValidator()
    executor = MultiStepExecutor()
    synthesizer = ResultSynthesizer()

    # Step 1: Validate query
    print(f"\n1. Validating query: '{query}'")
    validation_result = validator.validate_query(query)

    print(f"   ✓ Valid: {validation_result.is_valid}")
    print(f"   ✓ Confidence: {validation_result.confidence:.1%}")
    print(f"   ✓ Query Type: {validation_result.query_type.value}")

    if validation_result.clarifying_questions:
        print("   ✓ Clarifying Questions:")
        for q in validation_result.clarifying_questions:
            print(f"     - {q}")

    # Step 2: Create execution plan
    print(f"\n2. Creating execution plan...")
    execution_plan = executor.create_execution_plan(query, validation_result.query_type.value)

    print(f"   ✓ Created plan with {len(execution_plan.steps)} steps:")
    for i, step in enumerate(execution_plan.steps, 1):
        print(f"     {i}. {step.name}: {step.description}")

    # Step 3: Execute plan
    print(f"\n3. Executing plan...")
    context = {
        'query': query,
        'validation_result': validation_result,
        'plan': execution_plan
    }

    execution_results = await executor.execute_plan(execution_plan, context)
    print(f"   ✓ Execution completed with status: {execution_results['status']}")
    print(f"   ✓ Duration: {execution_results.get('duration', 0):.2f} seconds")
    print(f"   ✓ Progress: {execution_results['total_progress']:.1f}%")

    # Step 4: Synthesize results
    print(f"\n4. Synthesizing results...")
    synthesis_result = synthesizer.synthesize_results(
        execution_results.get('results', {}),
        query,
        validation_result.query_type.value
    )

    print(f"   ✓ Synthesis completed")
    print(f"   ✓ Confidence: {synthesis_result.confidence_score:.1%}")
    print(f"   ✓ Confidence Level: {synthesis_result.confidence_level.value}")

    # Step 5: Display results
    print(f"\n5. Final Results:")
    print("-" * 40)
    print(synthesis_result.content)
    print("-" * 40)

    return synthesis_result


async def demo_complex_analysis():
    """Demonstrate processing a complex analysis query."""
    print("\n\n" + "=" * 60)
    print("DEMO: Complex Analysis Query Processing")
    print("=" * 60)

    query = "Analyze the effectiveness of different trust structures for high-net-worth families with multiple generations and international assets, considering tax implications and succession planning"

    # Initialize components
    validator = QueryValidator()
    executor = MultiStepExecutor()
    synthesizer = ResultSynthesizer()

    # Process the complex query
    print(f"\nProcessing complex query: '{query[:100]}...'")

    validation_result = validator.validate_query(query)
    print(f"✓ Query Type: {validation_result.query_type.value}")
    print(f"✓ Confidence: {validation_result.confidence:.1%}")

    execution_plan = executor.create_execution_plan(query, validation_result.query_type.value)
    print(f"✓ Execution plan created with {len(execution_plan.steps)} steps")

    # Add progress tracking
    progress_updates = []

    def progress_callback(event):
        progress_updates.append(event)
        print(f"   Progress: {event['type']} - {event.get('message', '')}")

    executor.progress_tracker.add_callback(progress_callback)

    context = {
        'query': query,
        'validation_result': validation_result,
        'plan': execution_plan
    }

    execution_results = await executor.execute_plan(execution_plan, context)
    synthesis_result = synthesizer.synthesize_results(
        execution_results.get('results', {}),
        query,
        validation_result.query_type.value
    )

    print(f"\n✓ Analysis completed!")
    print(f"✓ Final confidence: {synthesis_result.confidence_score:.1%}")

    if synthesis_result.uncertainty_factors:
        print("✓ Uncertainty factors identified:")
        for factor in synthesis_result.uncertainty_factors:
            print(f"  - {factor}")

    return synthesis_result


async def demo_comparison_query():
    """Demonstrate processing a comparison query."""
    print("\n\n" + "=" * 60)
    print("DEMO: Comparison Query Processing")
    print("=" * 60)

    query = "Compare revocable living trusts vs irrevocable trusts for estate tax planning"

    validator = QueryValidator()
    executor = MultiStepExecutor()
    synthesizer = ResultSynthesizer()

    validation_result = validator.validate_query(query)
    print(f"✓ Detected as {validation_result.query_type.value} query")

    if validation_result.clarifying_questions:
        print("✓ Clarifying questions generated:")
        for q in validation_result.clarifying_questions:
            print(f"  - {q}")

    execution_plan = executor.create_execution_plan(query, validation_result.query_type.value)
    print(f"✓ Comparison-specific execution plan with {len(execution_plan.steps)} steps:")
    for step in execution_plan.steps:
        print(f"  - {step.name}")

    context = {'query': query, 'validation_result': validation_result, 'plan': execution_plan}
    execution_results = await executor.execute_plan(execution_plan, context)
    synthesis_result = synthesizer.synthesize_results(
        execution_results.get('results', {}),
        query,
        validation_result.query_type.value
    )

    print(f"\n✓ Comparison analysis completed with {synthesis_result.confidence_score:.1%} confidence")

    return synthesis_result


def demo_confidence_explanation():
    """Demonstrate confidence scoring explanation."""
    print("\n\n" + "=" * 60)
    print("DEMO: Confidence Scoring System")
    print("=" * 60)

    synthesizer = ResultSynthesizer()

    # Create mock synthesis result for demonstration
    from result_synthesizer import SynthesisResult, SourceInfo, ConfidenceLevel

    mock_sources = [
        SourceInfo("academic_paper", "academic", 0.9, 0.8, 0.9),
        SourceInfo("legal_database", "official", 0.85, 0.9, 0.95),
        SourceInfo("industry_report", "news", 0.7, 0.6, 0.8)
    ]

    mock_result = SynthesisResult(
        content="Mock analysis result",
        confidence_score=0.78,
        confidence_level=ConfidenceLevel.HIGH,
        sources=mock_sources,
        supporting_evidence=["Evidence 1", "Evidence 2", "Evidence 3"],
        conflicting_information=["Minor conflict found"],
        uncertainty_factors=["Limited recent data"],
        metadata={}
    )

    explanation = synthesizer.generate_confidence_explanation(mock_result)
    print("Confidence Analysis:")
    print("-" * 40)
    print(explanation)
    print("-" * 40)

    print(f"\nSource Quality Breakdown:")
    for source in mock_sources:
        print(f"  {source.id}: {source.quality_score:.1%} quality")
        print(f"    - Reliability: {source.reliability:.1%}")
        print(f"    - Recency: {source.recency:.1%}")
        print(f"    - Relevance: {source.relevance:.1%}")


async def main():
    """Run all demos."""
    print("Enhanced Query Processing System Demo")
    print("====================================")

    try:
        # Run all demonstrations
        await demo_simple_query()
        await demo_complex_analysis()
        await demo_comparison_query()
        demo_confidence_explanation()

        print("\n\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Intelligent query validation with type classification")
        print("✓ Multi-step execution with progress tracking")
        print("✓ Result synthesis with confidence scoring")
        print("✓ Clarifying questions for ambiguous queries")
        print("✓ Detailed confidence explanations")
        print("✓ Support for various query types (simple, complex, comparison)")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())