"""
Gradio interface for enhanced query processing.
Provides a web interface for interactive query processing with progress tracking.
"""

import gradio as gr
import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Tuple
import threading
from dataclasses import asdict

from .query_validator import QueryValidator, ValidationResult
from .multi_step_executor import MultiStepExecutor, ExecutionPlan, ProgressTracker
from .result_synthesizer import ResultSynthesizer, SynthesisResult


class EnhancedQueryInterface:
    """Gradio interface for enhanced query processing with real-time progress tracking."""

    def __init__(self):
        self.query_validator = QueryValidator()
        self.executor = MultiStepExecutor()
        self.synthesizer = ResultSynthesizer()
        self.current_execution = None
        self.progress_updates = []

    def create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface."""

        with gr.Blocks(
            title="Stellar Connect - Enhanced Query Processing",
            theme=gr.themes.Soft(),
            css=self._get_custom_css()
        ) as interface:

            gr.Markdown("# ⭐ Stellar Connect - Enhanced Query Processing")
            gr.Markdown("Advanced query analysis with intelligent validation, multi-step execution, and confidence scoring.")

            with gr.Row():
                with gr.Column(scale=2):
                    # Query input section
                    with gr.Group():
                        gr.Markdown("## Query Input")
                        query_input = gr.Textbox(
                            label="Enter your query",
                            placeholder="Ask a complex question about estate planning, sales analysis, or client insights...",
                            lines=3
                        )

                        with gr.Row():
                            submit_btn = gr.Button("Process Query", variant="primary")
                            clear_btn = gr.Button("Clear", variant="secondary")

                    # Validation results section
                    with gr.Group():
                        gr.Markdown("## Query Validation")
                        validation_status = gr.HTML()
                        clarifying_questions = gr.Markdown(visible=False)
                        suggested_refinements = gr.Markdown(visible=False)

                with gr.Column(scale=1):
                    # Progress tracking section
                    with gr.Group():
                        gr.Markdown("## Execution Progress")
                        progress_bar = gr.Progress()
                        current_step = gr.Textbox(
                            label="Current Step",
                            interactive=False,
                            value="Ready"
                        )
                        execution_log = gr.Textbox(
                            label="Execution Log",
                            lines=8,
                            interactive=False,
                            value=""
                        )

            # Results section
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        gr.Markdown("## Analysis Results")
                        results_tabs = gr.Tabs()

                        with results_tabs:
                            with gr.TabItem("Synthesized Response"):
                                main_response = gr.Markdown()
                                confidence_display = gr.HTML()

                            with gr.TabItem("Detailed Analysis"):
                                step_results = gr.JSON(label="Step-by-Step Results")
                                execution_metadata = gr.JSON(label="Execution Metadata")

                            with gr.TabItem("Sources & Evidence"):
                                sources_info = gr.Markdown()
                                supporting_evidence = gr.Markdown()
                                conflicts_info = gr.Markdown()

                            with gr.TabItem("Confidence Analysis"):
                                confidence_explanation = gr.Markdown()
                                uncertainty_factors = gr.Markdown()

            # Event handlers
            submit_btn.click(
                fn=self.process_query_wrapper,
                inputs=[query_input],
                outputs=[
                    validation_status,
                    clarifying_questions,
                    suggested_refinements,
                    current_step,
                    execution_log,
                    main_response,
                    confidence_display,
                    step_results,
                    execution_metadata,
                    sources_info,
                    supporting_evidence,
                    conflicts_info,
                    confidence_explanation,
                    uncertainty_factors
                ]
            )

            clear_btn.click(
                fn=self.clear_interface,
                outputs=[
                    query_input,
                    validation_status,
                    clarifying_questions,
                    suggested_refinements,
                    current_step,
                    execution_log,
                    main_response,
                    confidence_display,
                    step_results,
                    execution_metadata,
                    sources_info,
                    supporting_evidence,
                    conflicts_info,
                    confidence_explanation,
                    uncertainty_factors
                ]
            )

        return interface

    def process_query_wrapper(self, query: str) -> Tuple:
        """Wrapper for processing queries with error handling."""
        try:
            return asyncio.run(self.process_query(query))
        except Exception as e:
            error_html = f'<div class="error">Error: {str(e)}</div>'
            return (
                error_html,  # validation_status
                gr.Markdown(visible=False),  # clarifying_questions
                gr.Markdown(visible=False),  # suggested_refinements
                "Error",  # current_step
                f"Error occurred: {str(e)}",  # execution_log
                f"**Error:** {str(e)}",  # main_response
                "",  # confidence_display
                {},  # step_results
                {},  # execution_metadata
                "",  # sources_info
                "",  # supporting_evidence
                "",  # conflicts_info
                "",  # confidence_explanation
                ""   # uncertainty_factors
            )

    async def process_query(self, query: str) -> Tuple:
        """Process a query through the enhanced pipeline."""
        if not query.strip():
            return self._empty_results()

        self.progress_updates = []

        # Step 1: Validate query
        validation_result = self.query_validator.validate_query(query)
        validation_html = self._format_validation_result(validation_result)

        # Prepare clarifying questions and suggestions
        clarifying_md = ""
        suggestions_md = ""

        if validation_result.clarifying_questions:
            clarifying_md = "### Clarifying Questions:\n" + "\n".join(
                f"- {q}" for q in validation_result.clarifying_questions
            )

        if validation_result.suggested_refinements:
            suggestions_md = "### Suggested Refinements:\n" + "\n".join(
                f"- {r}" for r in validation_result.suggested_refinements
            )

        if not validation_result.is_valid:
            return (
                validation_html,
                gr.Markdown(clarifying_md, visible=bool(clarifying_md)),
                gr.Markdown(suggestions_md, visible=bool(suggestions_md)),
                "Validation Failed",
                "Query validation failed. Please refine your query.",
                "Please refine your query based on the validation feedback above.",
                "",
                {},
                {"validation": asdict(validation_result)},
                "",
                "",
                "",
                "",
                ""
            )

        # Step 2: Create execution plan
        execution_plan = self.executor.create_execution_plan(
            query, validation_result.query_type.value
        )

        # Step 3: Set up progress tracking
        self.executor.progress_tracker.add_callback(self._progress_callback)

        # Step 4: Execute plan
        execution_context = {
            'query': query,
            'validation_result': validation_result,
            'plan': execution_plan
        }

        execution_results = await self.executor.execute_plan(execution_plan, execution_context)

        # Step 5: Synthesize results
        synthesis_result = self.synthesizer.synthesize_results(
            execution_results.get('results', {}),
            query,
            validation_result.query_type.value
        )

        # Format outputs
        return self._format_complete_results(
            validation_result,
            execution_results,
            synthesis_result,
            clarifying_md,
            suggestions_md
        )

    def _progress_callback(self, event: Dict[str, Any]):
        """Handle progress updates."""
        self.progress_updates.append(event)

    def _format_validation_result(self, validation: ValidationResult) -> str:
        """Format validation result as HTML."""
        if validation.is_valid:
            status_class = "success"
            status_text = "✅ Valid"
        else:
            status_class = "warning"
            status_text = "⚠️ Needs Refinement"

        confidence_color = self._get_confidence_color(validation.confidence)

        return f'''
        <div class="validation-result {status_class}">
            <div class="status">{status_text}</div>
            <div class="confidence" style="color: {confidence_color}">
                Confidence: {validation.confidence:.1%}
            </div>
            <div class="query-type">
                Query Type: {validation.query_type.value.replace('_', ' ').title()}
            </div>
        </div>
        '''

    def _format_complete_results(self,
                                validation: ValidationResult,
                                execution: Dict[str, Any],
                                synthesis: SynthesisResult,
                                clarifying_md: str,
                                suggestions_md: str) -> Tuple:
        """Format complete results for display."""

        # Main response
        main_response = synthesis.content

        # Confidence display
        confidence_color = self._get_confidence_color(synthesis.confidence_score)
        confidence_html = f'''
        <div class="confidence-score">
            <div class="score" style="color: {confidence_color}">
                Confidence: {synthesis.confidence_score:.1%}
            </div>
            <div class="level">
                Level: {synthesis.confidence_level.value.replace('_', ' ').title()}
            </div>
        </div>
        '''

        # Sources information
        sources_md = "### Information Sources:\n"
        for source in synthesis.sources:
            sources_md += f"- **{source.id}** ({source.type}) - Quality: {source.quality_score:.1%}\n"

        # Supporting evidence
        evidence_md = "### Supporting Evidence:\n"
        for evidence in synthesis.supporting_evidence:
            evidence_md += f"- {evidence}\n"

        # Conflicts
        conflicts_md = ""
        if synthesis.conflicting_information:
            conflicts_md = "### Conflicting Information:\n"
            for conflict in synthesis.conflicting_information:
                conflicts_md += f"- {conflict}\n"

        # Confidence explanation
        confidence_explanation = self.synthesizer.generate_confidence_explanation(synthesis)

        # Uncertainty factors
        uncertainty_md = ""
        if synthesis.uncertainty_factors:
            uncertainty_md = "### Uncertainty Factors:\n"
            for factor in synthesis.uncertainty_factors:
                uncertainty_md += f"- {factor}\n"

        # Execution log
        execution_log = "\n".join([
            f"[{event.get('timestamp', 'N/A')}] {event.get('type', 'unknown')}: {event.get('message', '')}"
            for event in self.progress_updates
        ])

        return (
            self._format_validation_result(validation),
            gr.Markdown(clarifying_md, visible=bool(clarifying_md)),
            gr.Markdown(suggestions_md, visible=bool(suggestions_md)),
            execution.get('status', 'Completed'),
            execution_log,
            main_response,
            confidence_html,
            execution.get('results', {}),
            {
                'validation': asdict(validation),
                'execution': execution,
                'synthesis_metadata': synthesis.metadata
            },
            sources_md,
            evidence_md,
            conflicts_md,
            confidence_explanation,
            uncertainty_md
        )

    def _get_confidence_color(self, confidence: float) -> str:
        """Get color code for confidence level."""
        if confidence >= 0.8:
            return "#22c55e"  # Green
        elif confidence >= 0.6:
            return "#84cc16"  # Light green
        elif confidence >= 0.4:
            return "#eab308"  # Yellow
        elif confidence >= 0.2:
            return "#f97316"  # Orange
        else:
            return "#ef4444"  # Red

    def clear_interface(self) -> Tuple:
        """Clear all interface elements."""
        return self._empty_results() + ("",)  # Add empty query input

    def _empty_results(self) -> Tuple:
        """Return empty/default values for all output components."""
        return (
            "",  # validation_status
            gr.Markdown(visible=False),  # clarifying_questions
            gr.Markdown(visible=False),  # suggested_refinements
            "Ready",  # current_step
            "",  # execution_log
            "",  # main_response
            "",  # confidence_display
            {},  # step_results
            {},  # execution_metadata
            "",  # sources_info
            "",  # supporting_evidence
            "",  # conflicts_info
            "",  # confidence_explanation
            ""   # uncertainty_factors
        )

    def _get_custom_css(self) -> str:
        """Get custom CSS for the interface."""
        return """
        .validation-result {
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }

        .validation-result.success {
            background-color: #dcfce7;
            border: 1px solid #22c55e;
        }

        .validation-result.warning {
            background-color: #fef3c7;
            border: 1px solid #f59e0b;
        }

        .validation-result.error {
            background-color: #fee2e2;
            border: 1px solid #ef4444;
        }

        .status {
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 5px;
        }

        .confidence {
            font-weight: 600;
        }

        .query-type {
            font-style: italic;
            color: #6b7280;
        }

        .confidence-score {
            padding: 10px;
            border-radius: 6px;
            background-color: #f9fafb;
            border: 1px solid #e5e7eb;
        }

        .confidence-score .score {
            font-size: 1.2em;
            font-weight: bold;
        }

        .confidence-score .level {
            font-size: 0.9em;
            color: #6b7280;
        }

        .error {
            color: #ef4444;
            font-weight: bold;
        }
        """


def create_gradio_app() -> gr.Blocks:
    """Create and return the Gradio application."""
    interface = EnhancedQueryInterface()
    return interface.create_interface()


def launch_gradio_app(
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int = 7860
) -> None:
    """Launch the Gradio application."""
    app = create_gradio_app()
    app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True
    )


if __name__ == "__main__":
    launch_gradio_app()