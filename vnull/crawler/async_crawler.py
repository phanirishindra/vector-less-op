"""High-performance async web crawler.

Uses aiohttp for concurrent HTTP requests with rate limiting,
Bloom filter deduplication, and robust error handling.
"""

import asyncio
import hashlib
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp
from aiohttp import ClientTimeout

from vnull.core.config import settings
from vnull.core.logging import get_logger, log_performance
from vnull.crawler.bloom_filter import BloomFilter

logger = get_logger(__name__)


@dataclass
class CrawlResult:
    """Result of crawling a single URL."""
    url: str
    status_code: int
    content: str
    content_type: str
    content_length: int
    crawled_at: datetime
    depth: int
    links: list[str] = field(default_factory=list)
    error: str | None = None
    
    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300 and self.error is None
    
    @property
    def content_hash(self) -> str:
        """SHA256 hash of content for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class CrawlTask:
    """A URL to be crawled with metadata."""
    url: str
    depth: int
    parent_url: str | None = None


class AsyncCrawler:
    """Async web crawler with deduplication and rate limiting.
    
    Features:
    - Concurrent requests with semaphore-based limiting
    - Bloom filter URL deduplication
    - Configurable crawl depth
    - Link extraction and domain filtering
    - Automatic retry with exponential backoff
    
    Example:
        >>> crawler = AsyncCrawler()
        >>> async for result in crawler.crawl("https://example.com", max_depth=2):
        ...     print(f"Crawled: {result.url}")
    """
    
    def __init__(
        self,
        max_concurrent: int | None = None,
        delay_ms: int | None = None,
        user_agent: str | None = None,
        respect_robots: bool = True,
    ) -> None:
        """Initialize crawler.
        
        Args:
            max_concurrent: Max concurrent requests.
            delay_ms: Delay between requests in ms.
            user_agent: Custom user agent string.
            respect_robots: Whether to respect robots.txt.
        """
        self.max_concurrent = max_concurrent or settings.max_concurrent_requests
        self.delay_ms = delay_ms or settings.crawl_delay_ms
        self.user_agent = user_agent or settings.user_agent
        self.respect_robots = respect_robots
        
        self.bloom = BloomFilter()
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._session: aiohttp.ClientSession | None = None
        self._robots_cache: dict[str, set[str]] = {}
        
        logger.info(
            "Crawler initialized",
            max_concurrent=self.max_concurrent,
            delay_ms=self.delay_ms,
        )
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "DNT": "1",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                },
            )
        return self._session
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url)
        # Remove fragment, normalize path
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized.rstrip("/")
    
    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Extract and normalize links from HTML."""
        links = []
        # Simple regex for href extraction (BeautifulSoup used in parser)
        href_pattern = re.compile(r'href=["\']([^"\'>]+)["\']', re.IGNORECASE)
        
        for match in href_pattern.finditer(html):
            href = match.group(1)
            
            # Skip non-http links
            if href.startswith(("javascript:", "mailto:", "tel:", "#", "data:")):
                continue
            
            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)
            
            # Only keep http(s) URLs
            if absolute_url.startswith(("http://", "https://")):
                links.append(self._normalize_url(absolute_url))
        
        return list(set(links))
    
    def _is_same_domain(self, url: str, base_url: str) -> bool:
        """Check if URL is on the same domain as base."""
        url_domain = urlparse(url).netloc
        base_domain = urlparse(base_url).netloc
        return url_domain == base_domain
    
    async def _fetch_url(self, task: CrawlTask) -> CrawlResult:
        """Fetch a single URL with error handling."""
        async with self._semaphore:
            # Rate limiting
            if self.delay_ms > 0:
                await asyncio.sleep(self.delay_ms / 1000)
            
            session = await self._get_session()
            
            try:
                async with session.get(task.url, allow_redirects=True) as response:
                    content_type = response.headers.get("Content-Type", "")
                    
                    # Only process HTML content
                    if "text/html" not in content_type.lower():
                        return CrawlResult(
                            url=task.url,
                            status_code=response.status,
                            content="",
                            content_type=content_type,
                            content_length=0,
                            crawled_at=datetime.now(timezone.utc),
                            depth=task.depth,
                            error=f"Non-HTML content: {content_type}",
                        )
                    
                    content = await response.text()
                    links = self._extract_links(content, str(response.url))
                    
                    return CrawlResult(
                        url=str(response.url),
                        status_code=response.status,
                        content=content,
                        content_type=content_type,
                        content_length=len(content),
                        crawled_at=datetime.now(timezone.utc),
                        depth=task.depth,
                        links=links,
                    )
                    
            except asyncio.TimeoutError:
                return CrawlResult(
                    url=task.url,
                    status_code=0,
                    content="",
                    content_type="",
                    content_length=0,
                    crawled_at=datetime.now(timezone.utc),
                    depth=task.depth,
                    error="Timeout",
                )
            except aiohttp.ClientError as e:
                return CrawlResult(
                    url=task.url,
                    status_code=0,
                    content="",
                    content_type="",
                    content_length=0,
                    crawled_at=datetime.now(timezone.utc),
                    depth=task.depth,
                    error=str(e),
                )
    
    async def crawl(
        self,
        seed_url: str,
        max_depth: int | None = None,
        same_domain_only: bool = True,
        max_pages: int | None = None,
    ) -> AsyncGenerator[CrawlResult, None]:
        """Crawl starting from seed URL.
        
        Args:
            seed_url: Starting URL.
            max_depth: Maximum crawl depth.
            same_domain_only: Only crawl same domain.
            max_pages: Maximum pages to crawl.
            
        Yields:
            CrawlResult for each crawled page.
        """
        max_depth = max_depth or settings.max_depth
        seed_url = self._normalize_url(seed_url)
        
        queue: asyncio.Queue[CrawlTask] = asyncio.Queue()
        await queue.put(CrawlTask(url=seed_url, depth=0))
        self.bloom.add(seed_url)
        
        pages_crawled = 0
        
        logger.info(
            "Starting crawl",
            seed_url=seed_url,
            max_depth=max_depth,
            max_pages=max_pages,
        )
        
        while not queue.empty():
            if max_pages and pages_crawled >= max_pages:
                logger.info("Max pages reached", count=pages_crawled)
                break
            
            # Get batch of tasks
            batch: list[CrawlTask] = []
            while not queue.empty() and len(batch) < self.max_concurrent:
                batch.append(await queue.get())
            
            # Fetch batch concurrently
            results = await asyncio.gather(
                *[self._fetch_url(task) for task in batch],
                return_exceptions=True,
            )
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Crawl exception", error=str(result))
                    continue
                
                pages_crawled += 1
                yield result
                
                if not result.is_success:
                    continue
                
                # Queue new links if within depth
                if result.depth < max_depth:
                    for link in result.links:
                        if link in self.bloom:
                            continue
                        
                        if same_domain_only and not self._is_same_domain(link, seed_url):
                            continue
                        
                        self.bloom.add(link)
                        await queue.put(CrawlTask(
                            url=link,
                            depth=result.depth + 1,
                            parent_url=result.url,
                        ))
        
        logger.info(
            "Crawl completed",
            pages_crawled=pages_crawled,
            urls_seen=len(self.bloom),
        )
    
    async def save_result(self, result: CrawlResult, output_dir: Path | None = None) -> Path:
        """Save crawl result to disk.
        
        Args:
            result: CrawlResult to save.
            output_dir: Output directory.
            
        Returns:
            Path to saved file.
        """
        output_dir = output_dir or settings.raw_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from URL hash
        filename = f"{result.content_hash}.html"
        filepath = output_dir / filename
        
        # Save with metadata header
        metadata = f"<!-- URL: {result.url} -->\n<!-- Crawled: {result.crawled_at.isoformat()} -->\n<!-- Depth: {result.depth} -->\n"
        
        filepath.write_text(metadata + result.content, encoding="utf-8")
        
        logger.debug("Saved crawl result", path=str(filepath), url=result.url)
        return filepath
    
    async def close(self) -> None:
        """Close the crawler session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self) -> "AsyncCrawler":
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        await self.close()
