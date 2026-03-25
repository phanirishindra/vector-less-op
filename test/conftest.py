"""Pytest configuration and fixtures."""
import pytest
import os

os.environ["LLM_BASE_URL"] = "http://127.0.0.1:8000/v1"
os.environ["LLM_API_KEY"] = "sk-local"
os.environ["LOG_LEVEL"] = "WARNING"


@pytest.fixture
def sample_html():
    return """
    <html>
    <head><title>Test</title></head>
    <body>
        <nav>Navigation</nav>
        <main>
            <h1>Main Title</h1>
            <p>This is the main content paragraph.</p>
            <h2>Section</h2>
            <p>Section content here.</p>
        </main>
        <footer>Footer</footer>
        <script>console.log('x')</script>
    </body>
    </html>
    """


@pytest.fixture
def sample_markdown():
    return """
# Main Title

This is the introduction paragraph with important information.

## Section One

First section content. It contains details about the topic.
Multiple sentences here. Last sentence of section one.

## Section Two

Second section with different content. More details follow.
Final sentence of section two.

### Subsection

Nested content under section two.
"""
