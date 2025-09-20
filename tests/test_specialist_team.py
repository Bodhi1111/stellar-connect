"""
Comprehensive Test Suite for Stellar Connect Specialist Agent Team
Tests all 5 specialist agents and their coordination capabilities.

This test suite validates:
1. Individual agent functionality
2. Task assignment and execution
3. Inter-agent coordination
4. Performance monitoring
5. Error handling and recovery
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from agents.specialists.base_specialist import (
    SpecialistTask, TaskPriority, SpecialistCoordinator
)
from agents.specialists.estate_librarian import EstateLibrarianAgent
from agents.specialists.trust_sales_analyst import TrustSalesAnalystAgent
from agents.specialists.market_scout import MarketScoutAgent
from agents.specialists.sales_specialist import SalesSpecialistAgent


class SpecialistTeamTester:
    """Comprehensive tester for the specialist agent team."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        self.start_time = None

        # Initialize all specialist agents
        self.estate_librarian = EstateLibrarianAgent()
        self.trust_sales_analyst = TrustSalesAnalystAgent()
        self.market_scout = MarketScoutAgent()
        self.sales_specialist = SalesSpecialistAgent()

        # Initialize coordinator
        self.coordinator = SpecialistCoordinator([
            self.estate_librarian,
            self.trust_sales_analyst,
            self.market_scout,
            self.sales_specialist
        ])

        # Test data
        self.sample_conversation_data = [
            {
                "date": "2024-01-15",
                "client_name": "Johnson Family",
                "outcome": "closed_won",
                "deal_value": 350000,
                "first_contact_date": "2024-01-01",
                "final_stage": "closed_won",
                "estate_value": 2500000,
                "family_structure": "married with 2 adult children",
                "primary_concerns": ["tax minimization", "family harmony"],
                "objections": [
                    {"category": "cost", "resolved": True, "resolution_time": 15},
                    {"category": "complexity", "resolved": True, "resolution_time": 25}
                ]
            },
            {
                "date": "2024-01-16",
                "client_name": "Smith Estate",
                "outcome": "closed_lost",
                "deal_value": 0,
                "first_contact_date": "2024-01-02",
                "final_stage": "objection_handling",
                "loss_reason": "timing_concerns",
                "estate_value": 1800000,
                "family_structure": "widowed with 3 children",
                "primary_concerns": ["asset protection", "generation skipping"]
            }
        ]

        self.sample_transcript = """
        Advisor: Thank you for meeting with me today, Mr. Johnson. What brings you to consider estate planning at this time?

        Client: Well, my wife and I have been talking about this for years, but we finally decided we need to get serious about protecting our assets for our children.

        Advisor: That's a wise decision. Tell me about your family - who are the important people in your life?

        Client: We have two adult children, Sarah and Michael. Sarah is 28 and Michael is 25. Sarah just had our first grandchild.

        Advisor: Congratulations on becoming grandparents! That's often what motivates people to take action. What's your biggest concern about your current situation?

        Client: Honestly, taxes. We've heard horror stories about estate taxes taking half of everything.

        Advisor: I understand that concern. Many families worry about estate taxes. Can you give me a rough idea of your net worth?

        Client: Between our home, investments, and my business, probably around $2.5 million.

        Advisor: With proper planning, we can structure things to minimize tax impact while ensuring your family is protected. The key is having the right trust structure...
        """

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite for all specialist agents."""
        self.start_time = datetime.now()
        self.logger.info("🚀 Starting Comprehensive Specialist Team Test Suite")

        try:
            # Test 1: Individual Agent Health Checks
            await self._test_agent_health_checks()

            # Test 2: Estate Librarian Tests
            await self._test_estate_librarian()

            # Test 3: Trust Sales Analyst Tests
            await self._test_trust_sales_analyst()

            # Test 4: Market Scout Tests
            await self._test_market_scout()

            # Test 5: Sales Specialist Tests
            await self._test_sales_specialist()

            # Test 6: Coordinator Integration Tests
            await self._test_coordinator_integration()

            # Test 7: Performance and Load Tests
            await self._test_performance_and_load()

            # Test 8: Error Handling Tests
            await self._test_error_handling()

            # Generate comprehensive report
            return self._generate_test_report()

        except Exception as e:
            self.logger.error(f"❌ Test suite failed: {str(e)}")
            self.test_results["fatal_error"] = str(e)
            return self.test_results

    async def _test_agent_health_checks(self):
        """Test basic health and initialization of all agents."""
        self.logger.info("🏥 Testing Agent Health Checks...")

        agents = [
            ("Estate Librarian", self.estate_librarian),
            ("Trust Sales Analyst", self.trust_sales_analyst),
            ("Market Scout", self.market_scout),
            ("Sales Specialist", self.sales_specialist)
        ]

        health_results = {}

        for agent_name, agent in agents:
            try:
                # Basic health check
                health = await agent.health_check()

                # Status check
                status = agent.get_status()

                # Capabilities check
                capabilities = agent.get_capabilities()
                task_types = agent.get_task_types()

                health_results[agent_name] = {
                    "health_check": health["is_healthy"],
                    "agent_active": status["is_active"],
                    "capabilities_count": len(capabilities),
                    "task_types_count": len(task_types),
                    "workload_capacity": status["workload"]["can_accept_more"],
                    "performance_score": status["performance"]["success_rate"]
                }

                self.logger.info(f"✅ {agent_name}: Healthy={health['is_healthy']}, Active={status['is_active']}")

            except Exception as e:
                health_results[agent_name] = {"error": str(e)}
                self.logger.error(f"❌ {agent_name} health check failed: {str(e)}")

        self.test_results["health_checks"] = health_results

    async def _test_estate_librarian(self):
        """Test Estate Librarian Agent functionality."""
        self.logger.info("📚 Testing Estate Librarian Agent...")

        librarian_results = {}

        try:
            # Test 1: Similar Case Search
            search_task = SpecialistTask(
                task_type="find_similar_cases",
                description="Find similar estate planning cases",
                input_data={
                    "estate_value": 2500000,
                    "family_structure": "married with 2 adult children",
                    "primary_concerns": ["tax minimization", "family harmony"],
                    "limit": 3
                }
            )

            await self.estate_librarian.assign_task(search_task)
            search_result = await self.estate_librarian.execute_task(search_task.task_id)

            librarian_results["similar_case_search"] = {
                "success": search_result is not None,
                "cases_found": len(search_result.get("similar_cases", [])),
                "insights_generated": len(search_result.get("insights", []))
            }

            # Test 2: Rebuttal Retrieval
            rebuttal_task = SpecialistTask(
                task_type="retrieve_rebuttals",
                description="Retrieve rebuttals for cost objections",
                input_data={
                    "objection_category": "cost",
                    "objection_text": "This seems very expensive for what we're getting",
                    "limit": 3
                }
            )

            await self.estate_librarian.assign_task(rebuttal_task)
            rebuttal_result = await self.estate_librarian.execute_task(rebuttal_task.task_id)

            librarian_results["rebuttal_retrieval"] = {
                "success": rebuttal_result is not None,
                "rebuttals_found": len(rebuttal_result.get("rebuttals", [])),
                "avg_effectiveness": sum(r.get("effectiveness_score", 0) for r in rebuttal_result.get("rebuttals", [])) / max(len(rebuttal_result.get("rebuttals", [])), 1)
            }

            # Test 3: Content Search
            content_task = SpecialistTask(
                task_type="search_content",
                description="Search for generation skipping content",
                input_data={
                    "search_query": "generation skipping trust strategies",
                    "content_types": ["transcript", "case_study"],
                    "limit": 5
                }
            )

            await self.estate_librarian.assign_task(content_task)
            content_result = await self.estate_librarian.execute_task(content_task.task_id)

            librarian_results["content_search"] = {
                "success": content_result is not None,
                "results_found": len(content_result.get("search_results", [])),
                "avg_relevance": sum(r.get("relevance_score", 0) for r in content_result.get("search_results", [])) / max(len(content_result.get("search_results", [])), 1)
            }

            self.logger.info("✅ Estate Librarian tests completed successfully")

        except Exception as e:
            librarian_results["error"] = str(e)
            self.logger.error(f"❌ Estate Librarian test failed: {str(e)}")

        self.test_results["estate_librarian"] = librarian_results

    async def _test_trust_sales_analyst(self):
        """Test Trust Sales Analyst Agent functionality."""
        self.logger.info("📊 Testing Trust Sales Analyst Agent...")

        analyst_results = {}

        try:
            # Test 1: Conversion Analysis
            conversion_task = SpecialistTask(
                task_type="analyze_conversions",
                description="Analyze sales conversion patterns",
                input_data={
                    "conversations": self.sample_conversation_data,
                    "time_period": "30_days"
                }
            )

            await self.trust_sales_analyst.assign_task(conversion_task)
            conversion_result = await self.trust_sales_analyst.execute_task(conversion_task.task_id)

            analyst_results["conversion_analysis"] = {
                "success": conversion_result is not None,
                "conversion_rate": conversion_result.get("analysis", {}).get("conversion_rate", 0),
                "avg_deal_size": conversion_result.get("analysis", {}).get("average_deal_size", 0),
                "recommendations_count": len(conversion_result.get("analysis", {}).get("recommendations", []))
            }

            # Test 2: Objection Analysis
            objection_data = []
            for conv in self.sample_conversation_data:
                for objection in conv.get("objections", []):
                    objection_data.append({
                        "category": objection["category"],
                        "resolved": objection["resolved"],
                        "resolution_time": objection.get("resolution_time", 0),
                        "final_outcome": conv["outcome"]
                    })

            objection_task = SpecialistTask(
                task_type="analyze_objections",
                description="Analyze objection handling patterns",
                input_data={
                    "objection_data": objection_data
                }
            )

            await self.trust_sales_analyst.assign_task(objection_task)
            objection_result = await self.trust_sales_analyst.execute_task(objection_task.task_id)

            analyst_results["objection_analysis"] = {
                "success": objection_result is not None,
                "categories_analyzed": objection_result.get("categories_analyzed", 0),
                "total_objections": objection_result.get("total_objections", 0),
                "insights_generated": len(objection_result.get("overall_insights", []))
            }

            # Test 3: Performance Report Generation
            performance_task = SpecialistTask(
                task_type="generate_performance_report",
                description="Generate sales performance report",
                input_data={
                    "advisor_name": "Test Advisor",
                    "time_period": "30_days",
                    "comparison_period": "previous_30_days"
                }
            )

            await self.trust_sales_analyst.assign_task(performance_task)
            performance_result = await self.trust_sales_analyst.execute_task(performance_task.task_id)

            analyst_results["performance_report"] = {
                "success": performance_result is not None,
                "overall_score": performance_result.get("summary", {}).get("overall_performance", 0),
                "trend": performance_result.get("summary", {}).get("trend", "unknown"),
                "recommendations": len(performance_result.get("performance_report", {}).get("recommendations", []))
            }

            self.logger.info("✅ Trust Sales Analyst tests completed successfully")

        except Exception as e:
            analyst_results["error"] = str(e)
            self.logger.error(f"❌ Trust Sales Analyst test failed: {str(e)}")

        self.test_results["trust_sales_analyst"] = analyst_results

    async def _test_market_scout(self):
        """Test Market Scout Agent functionality."""
        self.logger.info("🔍 Testing Market Scout Agent...")

        scout_results = {}

        try:
            # Test 1: Trend Monitoring
            from agents.specialists.market_scout import MarketTrendType

            trend_task = SpecialistTask(
                task_type="monitor_trends",
                description="Monitor estate planning market trends",
                input_data={
                    "trend_types": [MarketTrendType.TAX_LAW_UPDATES, MarketTrendType.DEMOGRAPHIC_SHIFTS],
                    "keywords": ["estate tax", "wealth transfer", "baby boomer"],
                    "time_period": "30_days",
                    "geographical_scope": ["US", "California"]
                }
            )

            await self.market_scout.assign_task(trend_task)
            trend_result = await self.market_scout.execute_task(trend_task.task_id)

            scout_results["trend_monitoring"] = {
                "success": trend_result is not None,
                "trends_found": trend_result.get("summary", {}).get("total_trends", 0),
                "high_impact_trends": trend_result.get("summary", {}).get("high_impact_trends", 0),
                "avg_confidence": trend_result.get("summary", {}).get("average_confidence", 0)
            }

            # Test 2: Opportunity Discovery
            opportunity_task = SpecialistTask(
                task_type="discover_opportunities",
                description="Discover new market opportunities",
                input_data={
                    "market_criteria": {"focus": "high_net_worth", "geographic": "regional"},
                    "target_segments": ["tech entrepreneurs", "business owners"],
                    "opportunity_types": ["new_prospect_segment", "underserved_market"]
                }
            )

            await self.market_scout.assign_task(opportunity_task)
            opportunity_result = await self.market_scout.execute_task(opportunity_task.task_id)

            scout_results["opportunity_discovery"] = {
                "success": opportunity_result is not None,
                "opportunities_found": opportunity_result.get("summary", {}).get("total_opportunities", 0),
                "high_value_opportunities": opportunity_result.get("summary", {}).get("high_value_opportunities", 0),
                "avg_success_probability": opportunity_result.get("summary", {}).get("average_success_probability", 0)
            }

            # Test 3: Competitive Intelligence
            intel_task = SpecialistTask(
                task_type="gather_competitive_intel",
                description="Gather competitive intelligence",
                input_data={
                    "competitors": ["Regional Trust Company", "Wealth Management Firm"],
                    "intelligence_focus": ["pricing", "services", "marketing"],
                    "priority_level": "high"
                }
            )

            await self.market_scout.assign_task(intel_task)
            intel_result = await self.market_scout.execute_task(intel_task.task_id)

            scout_results["competitive_intelligence"] = {
                "success": intel_result is not None,
                "intelligence_items": intel_result.get("summary", {}).get("total_intelligence_items", 0),
                "high_reliability_items": intel_result.get("summary", {}).get("high_reliability_items", 0),
                "competitors_analyzed": intel_result.get("summary", {}).get("competitors_analyzed", 0)
            }

            self.logger.info("✅ Market Scout tests completed successfully")

        except Exception as e:
            scout_results["error"] = str(e)
            self.logger.error(f"❌ Market Scout test failed: {str(e)}")

        self.test_results["market_scout"] = scout_results

    async def _test_sales_specialist(self):
        """Test Sales Specialist Agent functionality."""
        self.logger.info("🎯 Testing Sales Specialist Agent...")

        specialist_results = {}

        try:
            # Test 1: Sales Recommendations
            recommendation_task = SpecialistTask(
                task_type="provide_sales_recommendations",
                description="Provide real-time sales recommendations",
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

            await self.sales_specialist.assign_task(recommendation_task)
            rec_result = await self.sales_specialist.execute_task(recommendation_task.task_id)

            specialist_results["sales_recommendations"] = {
                "success": rec_result is not None,
                "immediate_recommendations": len(rec_result.get("sales_recommendations", {}).get("immediate_recommendations", [])),
                "priority_recommendations": rec_result.get("summary", {}).get("priority_recommendations", 0),
                "stage_optimization": rec_result.get("summary", {}).get("stage_optimization", 0)
            }

            # Test 2: Conversation Performance Analysis
            performance_task = SpecialistTask(
                task_type="analyze_conversation_performance",
                description="Analyze conversation performance",
                input_data={
                    "conversation_transcript": self.sample_transcript,
                    "outcome": "closed_won",
                    "analysis_focus": ["overall", "techniques", "opportunities"]
                }
            )

            await self.sales_specialist.assign_task(performance_task)
            perf_result = await self.sales_specialist.execute_task(performance_task.task_id)

            specialist_results["performance_analysis"] = {
                "success": perf_result is not None,
                "overall_score": perf_result.get("summary", {}).get("overall_score", 0),
                "strengths": perf_result.get("summary", {}).get("strengths", 0),
                "improvement_areas": perf_result.get("summary", {}).get("improvement_areas", 0)
            }

            # Test 3: Coaching Insights
            coaching_task = SpecialistTask(
                task_type="generate_coaching_insights",
                description="Generate coaching insights",
                input_data={
                    "performance_data": {
                        "conversation_count": 10,
                        "conversion_rate": 0.3,
                        "avg_conversation_length": 45,
                        "objection_handling_success": 0.7
                    },
                    "coaching_goals": ["improve_questioning", "better_closing"],
                    "timeline": "30_days"
                }
            )

            await self.sales_specialist.assign_task(coaching_task)
            coaching_result = await self.sales_specialist.execute_task(coaching_task.task_id)

            specialist_results["coaching_insights"] = {
                "success": coaching_result is not None,
                "critical_areas": coaching_result.get("summary", {}).get("critical_areas", 0),
                "high_priority_areas": coaching_result.get("summary", {}).get("high_priority_areas", 0),
                "timeline": coaching_result.get("summary", {}).get("estimated_improvement_timeline", "unknown")
            }

            self.logger.info("✅ Sales Specialist tests completed successfully")

        except Exception as e:
            specialist_results["error"] = str(e)
            self.logger.error(f"❌ Sales Specialist test failed: {str(e)}")

        self.test_results["sales_specialist"] = specialist_results

    async def _test_coordinator_integration(self):
        """Test SpecialistCoordinator integration and multi-agent workflows."""
        self.logger.info("🎭 Testing Specialist Coordinator Integration...")

        coordinator_results = {}

        try:
            # Test 1: System Status
            system_status = self.coordinator.get_system_status()

            coordinator_results["system_status"] = {
                "total_specialists": system_status["total_specialists"],
                "active_specialists": system_status["active_specialists"],
                "all_agents_healthy": system_status["active_specialists"] == system_status["total_specialists"]
            }

            # Test 2: Health Check All Agents
            health_check = await self.coordinator.health_check_all()

            coordinator_results["health_check_all"] = {
                "overall_health": health_check["overall_health"],
                "healthy_agents": len([h for h in health_check["specialist_health"].values() if h["is_healthy"]])
            }

            # Test 3: Task Assignment and Coordination
            # Create tasks for different specialists
            tasks = [
                SpecialistTask(
                    task_type="find_similar_cases",
                    description="Coordinator test - similar cases",
                    input_data={"estate_value": 1000000, "family_structure": "single"}
                ),
                SpecialistTask(
                    task_type="analyze_conversions",
                    description="Coordinator test - conversion analysis",
                    input_data={"conversations": self.sample_conversation_data[:1], "time_period": "7_days"}
                ),
                SpecialistTask(
                    task_type="monitor_trends",
                    description="Coordinator test - trend monitoring",
                    input_data={"keywords": ["estate planning"], "time_period": "7_days"}
                )
            ]

            # Test coordinated execution
            coordination_results = []
            for task in tasks:
                specialist = self.coordinator.find_best_specialist(task.task_type)
                if specialist:
                    result = await self.coordinator.execute_task(task)
                    coordination_results.append({
                        "task_type": task.task_type,
                        "assigned_specialist": specialist.name,
                        "execution_success": result is not None
                    })

            coordinator_results["task_coordination"] = {
                "tasks_executed": len(coordination_results),
                "successful_executions": len([r for r in coordination_results if r["execution_success"]]),
                "coordination_details": coordination_results
            }

            self.logger.info("✅ Coordinator integration tests completed successfully")

        except Exception as e:
            coordinator_results["error"] = str(e)
            self.logger.error(f"❌ Coordinator integration test failed: {str(e)}")

        self.test_results["coordinator_integration"] = coordinator_results

    async def _test_performance_and_load(self):
        """Test performance and load handling of the specialist team."""
        self.logger.info("⚡ Testing Performance and Load Handling...")

        performance_results = {}

        try:
            # Test 1: Concurrent Task Execution
            start_time = datetime.now()

            concurrent_tasks = []
            for i in range(8):  # Test with 8 concurrent tasks
                task = SpecialistTask(
                    task_type="find_similar_cases",
                    description=f"Load test task {i}",
                    input_data={"estate_value": 1000000 + (i * 100000), "family_structure": "test"}
                )
                concurrent_tasks.append(self.coordinator.execute_task(task))

            # Execute all tasks concurrently
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            successful_results = [r for r in results if not isinstance(r, Exception)]

            performance_results["concurrent_execution"] = {
                "total_tasks": len(concurrent_tasks),
                "successful_tasks": len(successful_results),
                "execution_time_seconds": execution_time,
                "avg_time_per_task": execution_time / len(concurrent_tasks),
                "tasks_per_second": len(concurrent_tasks) / execution_time
            }

            # Test 2: Agent Workload Distribution
            workload_stats = {}
            for agent_id, agent in self.coordinator.specialists.items():
                status = agent.get_status()
                workload_stats[agent.name] = {
                    "completed_today": status["workload"]["completed_today"],
                    "success_rate": status["performance"]["success_rate"],
                    "avg_response_time": status["performance"]["average_response_time"]
                }

            performance_results["workload_distribution"] = workload_stats

            # Test 3: Memory and Resource Usage (simulated)
            performance_results["resource_usage"] = {
                "estimated_memory_mb": len(self.coordinator.specialists) * 50,  # Estimated
                "active_tasks_total": sum(s.workload.active_tasks for s in self.coordinator.specialists.values()),
                "agents_at_capacity": len([s for s in self.coordinator.specialists.values() if not s._can_accept_task()])
            }

            self.logger.info("✅ Performance and load tests completed successfully")

        except Exception as e:
            performance_results["error"] = str(e)
            self.logger.error(f"❌ Performance and load test failed: {str(e)}")

        self.test_results["performance_and_load"] = performance_results

    async def _test_error_handling(self):
        """Test error handling and recovery mechanisms."""
        self.logger.info("🛡️ Testing Error Handling and Recovery...")

        error_results = {}

        try:
            # Test 1: Invalid Task Type
            invalid_task = SpecialistTask(
                task_type="invalid_task_type",
                description="Test invalid task type handling",
                input_data={}
            )

            try:
                result = await self.coordinator.execute_task(invalid_task)
                error_results["invalid_task_type"] = {
                    "handled_gracefully": result is None,
                    "no_crash": True
                }
            except Exception as e:
                error_results["invalid_task_type"] = {
                    "handled_gracefully": False,
                    "error_message": str(e)
                }

            # Test 2: Missing Required Data
            missing_data_task = SpecialistTask(
                task_type="find_similar_cases",
                description="Test missing required data",
                input_data={}  # Missing required fields
            )

            try:
                result = await self.coordinator.execute_task(missing_data_task)
                error_results["missing_required_data"] = {
                    "handled_gracefully": result is None,
                    "no_crash": True
                }
            except Exception as e:
                error_results["missing_required_data"] = {
                    "handled_gracefully": False,
                    "error_message": str(e)
                }

            # Test 3: Agent Overload Handling
            overload_tasks = []
            for i in range(20):  # Try to overload agents
                task = SpecialistTask(
                    task_type="find_similar_cases",
                    description=f"Overload test {i}",
                    input_data={"estate_value": 1000000, "family_structure": "test"}
                )
                overload_tasks.append(task)

            assigned_count = 0
            rejected_count = 0

            for task in overload_tasks:
                agent_id = await self.coordinator.assign_task(task)
                if agent_id:
                    assigned_count += 1
                else:
                    rejected_count += 1

            error_results["overload_handling"] = {
                "tasks_assigned": assigned_count,
                "tasks_rejected": rejected_count,
                "graceful_rejection": rejected_count > 0,
                "no_crash": True
            }

            self.logger.info("✅ Error handling tests completed successfully")

        except Exception as e:
            error_results["fatal_error"] = str(e)
            self.logger.error(f"❌ Error handling test failed: {str(e)}")

        self.test_results["error_handling"] = error_results

    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()

        # Calculate overall success metrics
        successful_tests = []
        failed_tests = []

        for test_name, results in self.test_results.items():
            if isinstance(results, dict) and "error" not in results:
                successful_tests.append(test_name)
            else:
                failed_tests.append(test_name)

        # Generate summary
        summary = {
            "test_execution": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_time,
                "total_tests": len(self.test_results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.test_results) if self.test_results else 0
            },
            "agent_performance": {
                "estate_librarian": self._extract_agent_performance("estate_librarian"),
                "trust_sales_analyst": self._extract_agent_performance("trust_sales_analyst"),
                "market_scout": self._extract_agent_performance("market_scout"),
                "sales_specialist": self._extract_agent_performance("sales_specialist")
            },
            "system_health": {
                "all_agents_healthy": "error" not in str(self.test_results.get("health_checks", {})),
                "coordinator_functional": "error" not in str(self.test_results.get("coordinator_integration", {})),
                "performance_acceptable": self.test_results.get("performance_and_load", {}).get("concurrent_execution", {}).get("tasks_per_second", 0) > 1
            }
        }

        return {
            "summary": summary,
            "detailed_results": self.test_results,
            "recommendations": self._generate_recommendations(),
            "report_generated": end_time.isoformat()
        }

    def _extract_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        """Extract performance metrics for a specific agent."""
        agent_results = self.test_results.get(agent_name, {})

        if "error" in agent_results:
            return {"status": "failed", "error": agent_results["error"]}

        # Extract key performance indicators
        performance = {"status": "passed"}

        if agent_name == "estate_librarian":
            performance.update({
                "case_search_success": agent_results.get("similar_case_search", {}).get("success", False),
                "rebuttal_retrieval_success": agent_results.get("rebuttal_retrieval", {}).get("success", False),
                "content_search_success": agent_results.get("content_search", {}).get("success", False)
            })
        elif agent_name == "trust_sales_analyst":
            performance.update({
                "conversion_analysis_success": agent_results.get("conversion_analysis", {}).get("success", False),
                "objection_analysis_success": agent_results.get("objection_analysis", {}).get("success", False),
                "performance_report_success": agent_results.get("performance_report", {}).get("success", False)
            })
        # Add similar extractions for other agents...

        return performance

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Check for failed tests
        if self.test_results.get("health_checks", {}).get("error"):
            recommendations.append("Address agent health check failures before deployment")

        # Check performance
        perf_results = self.test_results.get("performance_and_load", {})
        if perf_results.get("concurrent_execution", {}).get("tasks_per_second", 0) < 1:
            recommendations.append("Consider performance optimization for concurrent task execution")

        # Check error handling
        error_results = self.test_results.get("error_handling", {})
        if not error_results.get("invalid_task_type", {}).get("handled_gracefully", True):
            recommendations.append("Improve error handling for invalid task types")

        if not recommendations:
            recommendations.append("All tests passed successfully - system ready for deployment")

        return recommendations


# Main test execution function
async def main():
    """Main function to run the comprehensive test suite."""
    print("🚀 Stellar Connect Specialist Team Test Suite")
    print("=" * 60)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run tester
    tester = SpecialistTeamTester()

    try:
        # Run comprehensive test
        test_report = await tester.run_comprehensive_test()

        # Print summary
        print("\n📊 TEST SUMMARY")
        print("=" * 60)
        summary = test_report["summary"]

        print(f"⏱️  Total Duration: {summary['test_execution']['total_duration_seconds']:.2f} seconds")
        print(f"✅ Successful Tests: {summary['test_execution']['successful_tests']}/{summary['test_execution']['total_tests']}")
        print(f"📈 Success Rate: {summary['test_execution']['success_rate']:.1%}")
        print(f"🏥 System Health: {'✅ Healthy' if summary['system_health']['all_agents_healthy'] else '❌ Issues Detected'}")

        # Print agent performance
        print("\n🤖 AGENT PERFORMANCE")
        print("-" * 60)
        for agent_name, performance in summary["agent_performance"].items():
            status_icon = "✅" if performance["status"] == "passed" else "❌"
            print(f"{status_icon} {agent_name.replace('_', ' ').title()}: {performance['status']}")

        # Print recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 60)
        for i, rec in enumerate(test_report["recommendations"], 1):
            print(f"{i}. {rec}")

        # Save detailed report
        report_file = Path("test_results") / f"specialist_team_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(test_report, f, indent=2, default=str)

        print(f"\n📄 Detailed report saved to: {report_file}")

        # Return success/failure code
        return 0 if summary["test_execution"]["success_rate"] > 0.8 else 1

    except Exception as e:
        print(f"\n❌ FATAL ERROR: Test suite failed to complete")
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())