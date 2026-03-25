"""Tests for DOM pruner."""
import pytest
from vnull.parser.dom_pruner import DOMPruner


def test_prune_removes_scripts():
    html = "<html><body><script>alert('x')</script><p>Content</p></body></html>"
    pruner = DOMPruner()
    result = pruner.prune(html)
    assert "<script>" not in result.pruned_html
    assert "Content" in result.pruned_html


def test_prune_removes_nav():
    html = "<html><body><nav>Menu</nav><main>Content</main></body></html>"
    pruner = DOMPruner(remove_nav=True)
    result = pruner.prune(html)
    assert "Menu" not in result.pruned_html
    assert "Content" in result.pruned_html


def test_prune_reduction():
    html = "<html><body><script>x</script><style>.a{}</style><nav>nav</nav><p>Content here</p></body></html>"
    pruner = DOMPruner()
    result = pruner.prune(html)
    assert result.reduction_percent > 0
