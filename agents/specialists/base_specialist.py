"""
Base Specialist Agent for Stellar Connect
Implements Story 5.2: Sales Specialist Agent Team Implementation

This module provides the foundation for domain-expert specialist agents that work
together to provide comprehensive sales intelligence and analysis.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import logging
import uuid


class SpecialistExpertise(Enum):
    """Enumeration of specialist agent expertise areas."""
    DOCUMENT_RETRIEVAL = "document_retrieval"
    REVENUE_OPERATIONS = "revenue_operations"
    SALES_PERFORMANCE = "sales_performance"
    COMPETITIVE_INTELLIGENCE = "competitive_intelligence"
    SALES_ENABLEMENT = "sales_enablement"
    TRUST_SALES_ANALYSIS = "trust_sales_analysis"
    MARKET_INTELLIGENCE = "market_intelligence"


class TaskPriority(Enum):
    """Task priority levels for specialist agents."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SpecialistTask:
    """Represents a task assigned to a specialist agent."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    description: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpecialistCapability:
    """Represents a capability of a specialist agent."""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    reliability_score: float = 1.0


@dataclass
class SpecialistWorkload:
    """Tracks the workload of a specialist agent."""
    active_tasks: int = 0
    pending_tasks: int = 0
    completed_tasks_today: int = 0
    average_task_duration: float = 0.0
    success_rate: float = 1.0
    last_activity: Optional[datetime] = None


@dataclass
class SpecialistPerformance:
    """Tracks performance metrics for a specialist agent."""
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    average_response_time: float = 0.0
    accuracy_score: float = 1.0
    user_satisfaction_score: float = 1.0
    uptime_percentage: float = 100.0
    success_rate: float = 1.0
    last_performance_update: datetime = field(default_factory=datetime.now)


class BaseSpecialist(ABC):
    """Base class for all specialist agents in the Stellar Connect system."""

    def __init__(
        self,
        name: str,
        expertise: SpecialistExpertise,
        description: str,
        capabilities: List[SpecialistCapability],
        max_concurrent_tasks: int = 3
    ):
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.expertise = expertise
        self.description = description
        self.capabilities = {cap.name: cap for cap in capabilities}
        self.max_concurrent_tasks = max_concurrent_tasks

        # State management
        self.workload = SpecialistWorkload()
        self.performance = SpecialistPerformance()
        self.active_tasks: Dict[str, SpecialistTask] = {}
        self.task_history: List[SpecialistTask] = []

        # Configuration
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.is_active = True
        self.created_at = datetime.now()

    @abstractmethod
    async def process_task(self, task: SpecialistTask) -> Dict[str, Any]:
        """Process a specific task. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_task_types(self) -> List[str]:
        """Return list of task types this specialist can handle."""
        pass

    @abstractmethod
    async def validate_input(self, task: SpecialistTask) -> Tuple[bool, Optional[str]]:
        """Validate task input data. Returns (is_valid, error_message)."""
        pass

    async def assign_task(self, task: SpecialistTask) -> bool:
        """Assign a task to this specialist agent."""
        try:
            # Check if agent can handle more tasks
            if not self._can_accept_task():
                self.logger.warning(f"Cannot accept task {task.task_id}: at capacity")
                return False

            # Validate task type
            if task.task_type not in self.get_task_types():
                self.logger.error(f"Cannot handle task type: {task.task_type}")
                return False

            # Validate input data
            is_valid, error_msg = await self.validate_input(task)
            if not is_valid:
                self.logger.error(f"Invalid input for task {task.task_id}: {error_msg}")
                task.status = TaskStatus.FAILED
                task.error_message = error_msg
                return False

            # Assign task
            task.assigned_agent = self.agent_id
            task.status = TaskStatus.PENDING
            self.active_tasks[task.task_id] = task
            self.workload.pending_tasks += 1

            self.logger.info(f"Assigned task {task.task_id} to {self.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error assigning task {task.task_id}: {str(e)}")
            return False

    async def execute_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Execute a specific task by ID."""
        if task_id not in self.active_tasks:
            self.logger.error(f"Task {task_id} not found in active tasks")
            return None

        task = self.active_tasks[task_id]

        try:
            # Update task status
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now()
            self.workload.pending_tasks -= 1
            self.workload.active_tasks += 1

            self.logger.info(f"Starting execution of task {task_id}")

            # Process the task
            result = await self.process_task(task)

            # Update task completion
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result

            # Update workload
            self.workload.active_tasks -= 1
            self.workload.completed_tasks_today += 1

            # Move to history
            self.task_history.append(task)
            del self.active_tasks[task_id]

            # Update performance metrics
            self._update_performance_metrics(task, success=True)

            self.logger.info(f"Successfully completed task {task_id}")
            return result

        except Exception as e:
            # Handle task failure
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            task.error_message = str(e)

            self.workload.active_tasks -= 1
            self.task_history.append(task)
            del self.active_tasks[task_id]

            self._update_performance_metrics(task, success=False)

            self.logger.error(f"Task {task_id} failed: {str(e)}")
            return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or in-progress task."""
        if task_id not in self.active_tasks:
            return False

        task = self.active_tasks[task_id]
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()

        if task.status == TaskStatus.PENDING:
            self.workload.pending_tasks -= 1
        elif task.status == TaskStatus.IN_PROGRESS:
            self.workload.active_tasks -= 1

        self.task_history.append(task)
        del self.active_tasks[task_id]

        self.logger.info(f"Cancelled task {task_id}")
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the specialist agent."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "expertise": self.expertise.value,
            "is_active": self.is_active,
            "workload": {
                "active_tasks": self.workload.active_tasks,
                "pending_tasks": self.workload.pending_tasks,
                "completed_today": self.workload.completed_tasks_today,
                "can_accept_more": self._can_accept_task()
            },
            "performance": {
                "total_completed": self.performance.total_tasks_completed,
                "total_failed": self.performance.total_tasks_failed,
                "success_rate": self.performance.success_rate,
                "average_response_time": self.performance.average_response_time,
                "accuracy_score": self.performance.accuracy_score
            },
            "capabilities": list(self.capabilities.keys()),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.workload.last_activity.isoformat() if self.workload.last_activity else None
        }

    def get_capabilities(self) -> List[SpecialistCapability]:
        """Get list of agent capabilities."""
        return list(self.capabilities.values())

    def get_capability(self, capability_name: str) -> Optional[SpecialistCapability]:
        """Get a specific capability by name."""
        return self.capabilities.get(capability_name)

    def update_capability_performance(self, capability_name: str, metrics: Dict[str, float]):
        """Update performance metrics for a specific capability."""
        if capability_name in self.capabilities:
            self.capabilities[capability_name].performance_metrics.update(metrics)

    def get_workload_score(self) -> float:
        """Calculate current workload score (0.0 = no load, 1.0 = at capacity)."""
        total_tasks = self.workload.active_tasks + self.workload.pending_tasks
        return min(total_tasks / self.max_concurrent_tasks, 1.0)

    def get_recommendation_score(self, task_type: str) -> float:
        """Get recommendation score for assigning a specific task type."""
        # Base score on capability match
        if task_type not in self.get_task_types():
            return 0.0

        # Factor in current workload (prefer less loaded agents)
        workload_factor = 1.0 - self.get_workload_score()

        # Factor in performance history
        performance_factor = self.performance.success_rate * self.performance.accuracy_score

        # Factor in agent availability
        availability_factor = 1.0 if self.is_active else 0.0

        return (workload_factor * 0.4 + performance_factor * 0.5 + availability_factor * 0.1)

    def _can_accept_task(self) -> bool:
        """Check if agent can accept additional tasks."""
        total_tasks = self.workload.active_tasks + self.workload.pending_tasks
        return self.is_active and total_tasks < self.max_concurrent_tasks

    def _update_performance_metrics(self, task: SpecialistTask, success: bool):
        """Update performance metrics based on task completion."""
        if success:
            self.performance.total_tasks_completed += 1
        else:
            self.performance.total_tasks_failed += 1

        # Update success rate
        total_tasks = self.performance.total_tasks_completed + self.performance.total_tasks_failed
        self.performance.success_rate = self.performance.total_tasks_completed / total_tasks

        # Update response time
        if task.started_at and task.completed_at:
            response_time = (task.completed_at - task.started_at).total_seconds()

            # Running average
            current_count = self.performance.total_tasks_completed + self.performance.total_tasks_failed
            if current_count == 1:
                self.performance.average_response_time = response_time
            else:
                self.performance.average_response_time = (
                    (self.performance.average_response_time * (current_count - 1) + response_time) / current_count
                )

        # Update last activity
        self.workload.last_activity = datetime.now()
        self.performance.last_performance_update = datetime.now()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the specialist agent."""
        try:
            # Test basic functionality
            test_task = SpecialistTask(
                task_type="health_check",
                description="Health check test",
                input_data={"test": True}
            )

            # Quick validation test
            is_valid, _ = await self.validate_input(test_task)

            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "is_healthy": True,
                "is_active": self.is_active,
                "validation_working": is_valid,
                "workload_score": self.get_workload_score(),
                "success_rate": self.performance.success_rate,
                "last_activity": self.workload.last_activity.isoformat() if self.workload.last_activity else None,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                "agent_id": self.agent_id,
                "name": self.name,
                "is_healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def shutdown(self):
        """Gracefully shutdown the specialist agent."""
        self.logger.info(f"Shutting down specialist agent: {self.name}")

        # Cancel all pending tasks
        pending_task_ids = [
            task_id for task_id, task in self.active_tasks.items()
            if task.status == TaskStatus.PENDING
        ]

        for task_id in pending_task_ids:
            asyncio.create_task(self.cancel_task(task_id))

        self.is_active = False
        self.logger.info(f"Specialist agent {self.name} shutdown complete")


class SpecialistCoordinator:
    """Coordinates tasks across multiple specialist agents."""

    def __init__(self, specialists: List[BaseSpecialist]):
        self.specialists = {spec.agent_id: spec for spec in specialists}
        self.task_queue: List[SpecialistTask] = []
        self.logger = logging.getLogger(f"{__name__}.SpecialistCoordinator")

    def register_specialist(self, specialist: BaseSpecialist):
        """Register a new specialist agent."""
        self.specialists[specialist.agent_id] = specialist
        self.logger.info(f"Registered specialist: {specialist.name}")

    def unregister_specialist(self, agent_id: str):
        """Unregister a specialist agent."""
        if agent_id in self.specialists:
            specialist = self.specialists[agent_id]
            specialist.shutdown()
            del self.specialists[agent_id]
            self.logger.info(f"Unregistered specialist: {specialist.name}")

    def find_best_specialist(self, task_type: str) -> Optional[BaseSpecialist]:
        """Find the best specialist for a given task type."""
        candidates = []

        for specialist in self.specialists.values():
            if task_type in specialist.get_task_types():
                score = specialist.get_recommendation_score(task_type)
                candidates.append((specialist, score))

        if not candidates:
            return None

        # Return specialist with highest recommendation score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    async def assign_task(self, task: SpecialistTask) -> Optional[str]:
        """Assign a task to the best available specialist."""
        specialist = self.find_best_specialist(task.task_type)

        if not specialist:
            self.logger.error(f"No suitable specialist found for task type: {task.task_type}")
            return None

        success = await specialist.assign_task(task)
        if success:
            self.logger.info(f"Assigned task {task.task_id} to {specialist.name}")
            return specialist.agent_id
        else:
            self.logger.error(f"Failed to assign task {task.task_id} to {specialist.name}")
            return None

    async def execute_task(self, task: SpecialistTask) -> Optional[Dict[str, Any]]:
        """Assign and execute a task."""
        agent_id = await self.assign_task(task)
        if not agent_id:
            return None

        specialist = self.specialists[agent_id]
        return await specialist.execute_task(task.task_id)

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all specialist agents."""
        return {
            "total_specialists": len(self.specialists),
            "active_specialists": len([s for s in self.specialists.values() if s.is_active]),
            "specialists": [spec.get_status() for spec in self.specialists.values()],
            "task_queue_length": len(self.task_queue),
            "timestamp": datetime.now().isoformat()
        }

    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all specialists."""
        health_results = {}

        for agent_id, specialist in self.specialists.items():
            health_results[agent_id] = await specialist.health_check()

        return {
            "overall_health": all(result["is_healthy"] for result in health_results.values()),
            "specialist_health": health_results,
            "timestamp": datetime.now().isoformat()
        }


# Example usage and testing
async def main():
    """Example usage of the base specialist framework."""

    # This would be implemented by actual specialist agents
    class DummySpecialist(BaseSpecialist):
        def __init__(self):
            capabilities = [
                SpecialistCapability(
                    name="dummy_processing",
                    description="Dummy task processing",
                    input_types=["text"],
                    output_types=["result"]
                )
            ]
            super().__init__(
                name="Dummy Specialist",
                expertise=SpecialistExpertise.DOCUMENT_RETRIEVAL,
                description="A dummy specialist for testing",
                capabilities=capabilities
            )

        async def process_task(self, task: SpecialistTask) -> Dict[str, Any]:
            await asyncio.sleep(0.1)  # Simulate processing
            return {"result": f"Processed task: {task.description}"}

        def get_task_types(self) -> List[str]:
            return ["dummy_task"]

        async def validate_input(self, task: SpecialistTask) -> Tuple[bool, Optional[str]]:
            return True, None

    # Create and test dummy specialist
    specialist = DummySpecialist()
    print(f"Created specialist: {specialist.name}")
    print(f"Status: {specialist.get_status()}")

    # Test task assignment and execution
    task = SpecialistTask(
        task_type="dummy_task",
        description="Test task",
        input_data={"test": "data"}
    )

    success = await specialist.assign_task(task)
    print(f"Task assignment success: {success}")

    if success:
        result = await specialist.execute_task(task.task_id)
        print(f"Task result: {result}")

    print("Base Specialist framework test completed!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run example
    asyncio.run(main())