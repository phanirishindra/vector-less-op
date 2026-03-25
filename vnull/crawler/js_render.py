"""JavaScript rendering using Playwright with stealth configuration.

For JS-heavy sites that require full browser rendering.
Uses stealth techniques to avoid bot detection.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from vnull.core.config import settings
from vnull.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RenderResult:
    """Result of rendering a page with JS."""
    url: str
    final_url: str
    content: str
    title: str
    rendered_at: datetime
    render_time_ms: float
    error: str | None = None
    
    @property
    def is_success(self) -> bool:
        return self.error is None


class JSRenderer:
    """Playwright-based JavaScript renderer with stealth config.
    
    Features:
    - Headless Chromium with stealth patches
    - Configurable wait strategies
    - Screenshot capability for debugging
    - Resource blocking for faster loads
    
    Example:
        >>> async with JSRenderer() as renderer:
        ...     result = await renderer.render("https://spa-example.com")
        ...     print(result.content[:100])
    """
    
    # Stealth JavaScript to inject
    STEALTH_SCRIPT = """
    // Overwrite navigator properties
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    
    // Overwrite chrome property
    window.chrome = { runtime: {} };
    
    // Overwrite permissions
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
        Promise.resolve({ state: Notification.permission }) :
        originalQuery(parameters)
    );
    """
    
    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 30000,
        wait_until: str = "networkidle",
        block_resources: bool = True,
    ) -> None:
        """Initialize JS renderer.
        
        Args:
            headless: Run browser in headless mode.
            timeout_ms: Page load timeout in milliseconds.
            wait_until: Wait strategy (load, domcontentloaded, networkidle).
            block_resources: Block images/fonts/media for faster loads.
        """
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.wait_until = wait_until
        self.block_resources = block_resources
        
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
    
    async def _ensure_browser(self) -> None:
        """Ensure browser is launched."""
        if self._browser is not None:
            return
        
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )
        
        self._playwright = await async_playwright().start()
        
        # Launch with stealth args
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        
        # Create context with realistic viewport and user agent
        self._context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=settings.user_agent,
            locale="en-US",
            timezone_id="America/New_York",
            geolocation={"latitude": 40.7128, "longitude": -74.0060},
            permissions=["geolocation"],
        )
        
        # Inject stealth script on every page
        await self._context.add_init_script(self.STEALTH_SCRIPT)
        
        logger.info("Browser launched", headless=self.headless)
    
    async def render(
        self,
        url: str,
        wait_selector: str | None = None,
        extra_wait_ms: int = 0,
    ) -> RenderResult:
        """Render a page with JavaScript execution.
        
        Args:
            url: URL to render.
            wait_selector: Optional CSS selector to wait for.
            extra_wait_ms: Additional wait time after load.
            
        Returns:
            RenderResult with rendered HTML content.
        """
        await self._ensure_browser()
        
        page = await self._context.new_page()
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Block unnecessary resources if enabled
            if self.block_resources:
                await page.route(
                    "**/*",
                    lambda route: (
                        route.abort()
                        if route.request.resource_type in ["image", "font", "media", "stylesheet"]
                        else route.continue_()
                    ),
                )
            
            # Navigate to page
            response = await page.goto(
                url,
                wait_until=self.wait_until,
                timeout=self.timeout_ms,
            )
            
            # Wait for specific selector if provided
            if wait_selector:
                await page.wait_for_selector(wait_selector, timeout=self.timeout_ms)
            
            # Extra wait for dynamic content
            if extra_wait_ms > 0:
                await asyncio.sleep(extra_wait_ms / 1000)
            
            # Get rendered content
            content = await page.content()
            title = await page.title()
            final_url = page.url
            
            render_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            logger.debug(
                "Page rendered",
                url=url,
                final_url=final_url,
                render_time_ms=round(render_time, 2),
                content_length=len(content),
            )
            
            return RenderResult(
                url=url,
                final_url=final_url,
                content=content,
                title=title,
                rendered_at=datetime.now(timezone.utc),
                render_time_ms=render_time,
            )
            
        except Exception as e:
            render_time = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.error("Render failed", url=url, error=str(e))
            
            return RenderResult(
                url=url,
                final_url=url,
                content="",
                title="",
                rendered_at=datetime.now(timezone.utc),
                render_time_ms=render_time,
                error=str(e),
            )
        finally:
            await page.close()
    
    async def render_many(
        self,
        urls: list[str],
        max_concurrent: int = 3,
    ) -> list[RenderResult]:
        """Render multiple pages concurrently.
        
        Args:
            urls: List of URLs to render.
            max_concurrent: Max concurrent renders.
            
        Returns:
            List of RenderResults.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def render_with_semaphore(url: str) -> RenderResult:
            async with semaphore:
                return await self.render(url)
        
        results = await asyncio.gather(
            *[render_with_semaphore(url) for url in urls],
            return_exceptions=True,
        )
        
        # Convert exceptions to error results
        processed: list[RenderResult] = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                processed.append(RenderResult(
                    url=url,
                    final_url=url,
                    content="",
                    title="",
                    rendered_at=datetime.now(timezone.utc),
                    render_time_ms=0,
                    error=str(result),
                ))
            else:
                processed.append(result)
        
        return processed
    
    async def close(self) -> None:
        """Close browser and cleanup."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        
        self._context = None
        self._browser = None
        self._playwright = None
        
        logger.info("Browser closed")
    
    async def __aenter__(self) -> "JSRenderer":
        await self._ensure_browser()
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        await self.close()
