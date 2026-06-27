"""
Locust load test for the Search Engine API.

Run headless (CLI):
    locust -f tests/locustfile.py \
           --host http://localhost:8000 \
           --users 50 --spawn-rate 5 \
           --run-time 60s --headless

Run with web UI:
    locust -f tests/locustfile.py --host http://localhost:8000
    # then open http://localhost:8089
"""

import random
from locust import HttpUser, task, between

# ── Realistic sample queries drawn from MS MARCO topics ──────────────────────
SAMPLE_QUERIES = [
    "what is the capital of france",
    "how does photosynthesis work",
    "symptoms of diabetes type 2",
    "best programming languages 2024",
    "how to learn machine learning",
    "what causes earthquakes",
    "history of the roman empire",
    "how to make sourdough bread",
    "what is quantum computing",
    "treatments for high blood pressure",
    "how do vaccines work",
    "what is the speed of light",
    "how to improve sleep quality",
    "what is blockchain technology",
    "causes of climate change",
    "how to write a resume",
    "what is artificial intelligence",
    "symptoms of vitamin D deficiency",
    "how does the stock market work",
    "what is machine learning",
]


class SearchUser(HttpUser):
    """Simulates a user hitting the search API."""

    # Each simulated user waits 0.5–2s between requests (realistic think time)
    wait_time = between(0.5, 2.0)

    # ── Tasks ─────────────────────────────────────────────────────────────────

    @task(10)
    def search(self):
        """POST /search — the hot path, weighted 10x."""
        query = random.choice(SAMPLE_QUERIES)
        top_k = random.choice([5, 10, 20])
        with self.client.post(
            "/search",
            json={"query": query, "top_k": top_k},
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("num_results", 0) == 0:
                    resp.failure("Search returned 0 results")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    @task(1)
    def health_check(self):
        """GET /health — lightweight, weighted 1x."""
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 200 and resp.json().get("status") == "healthy":
                resp.success()
            else:
                resp.failure(f"Health check failed: {resp.text[:200]}")

    @task(1)
    def root(self):
        """GET / — sanity check."""
        self.client.get("/")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_start(self):
        """Called once per simulated user when they start.
        Verifies the service is reachable before hammering it."""
        resp = self.client.get("/health")
        if resp.status_code != 200:
            self.environment.runner.quit()