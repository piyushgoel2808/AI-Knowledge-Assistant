"""Integration tests for Ticket 01: Asynchronous Ingestion & Task Progress Tracking.

Highest seam integration test hitting the FastAPI HTTP endpoints directly via TestClient.
"""

from __future__ import annotations

import time
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_upload_returns_202_accepted_with_job_id():
    """POST /documents/upload should return 202 Accepted with a job_id immediately."""
    files = [
        (
            "files",
            (
                "employee_leave_policy.txt",
                b"Employees receive 18 annual leave days per calendar year upon completion of probation.",
                "text/plain",
            ),
        )
    ]
    response = client.post("/documents/upload", files=files)

    assert response.status_code == 202, f"Expected 202 Accepted, got {response.status_code}: {response.text}"
    data = response.json()
    assert "job_id" in data, "Response must include a job_id"
    assert data["job_id"].startswith("job_"), f"job_id must start with 'job_', got {data['job_id']}"
    assert data["status"] in ("queued", "processing"), f"Initial status must be queued or processing, got {data['status']}"
    assert "files" in data
    assert "employee_leave_policy.txt" in data["files"]


def test_get_task_status_lifecycle():
    """GET /documents/tasks/{job_id} tracks task progression and completes successfully."""
    files = [
        (
            "files",
            (
                "onboarding_guide.txt",
                b"The onboarding process consists of equipment setup, HR orientation, and team pairing.",
                "text/plain",
            ),
        )
    ]
    upload_res = client.post("/documents/upload", files=files)
    assert upload_res.status_code == 202
    job_id = upload_res.json()["job_id"]

    # Check task polling endpoint exists and returns valid schema
    task_res = client.get(f"/documents/tasks/{job_id}")
    assert task_res.status_code == 200, f"Expected 200, got {task_res.status_code}: {task_res.text}"
    task_data = task_res.json()
    assert task_data["job_id"] == job_id
    assert "progress_percent" in task_data
    assert 0 <= task_data["progress_percent"] <= 100
    assert task_data["stage"] in ("queued", "parsing", "chunking", "embedding", "completed")

    # Poll until background task finishes (timeout 10s)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        res = client.get(f"/documents/tasks/{job_id}")
        assert res.status_code == 200
        poll_data = res.json()
        if poll_data["status"] == "completed":
            assert poll_data["progress_percent"] == 100
            assert poll_data["stage"] == "completed"
            assert poll_data["stored_chunks"] >= 1
            return
        time.sleep(0.1)

    assert False, f"Job {job_id} did not complete within 10s. Last state: {task_res.json()}"


def test_get_nonexistent_task_returns_404():
    """GET /documents/tasks/job_nonexistent should return 404 Not Found."""
    response = client.get("/documents/tasks/job_nonexistent_99999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
