"""Web documentation downloader.

This module provides functionality to crawl and download web documentation.
"""
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Set, Union
from urllib.parse import urlparse, urljoin
import html2text

import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm

from leo_assist.core.vector_store import Document, DocumentSource, DocumentType
from leo_assist.utils.logger import logger
from leo_assist.ingestion.downloaders.base import BaseDownloader

class WebDocumentationDownloader(BaseDownloader):
    """Downloader for web-based documentation."""
    
    # Common file extensions to skip
    SKIP_EXTENSIONS = [
        # Images
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp', '.bmp', '.tiff',
        # Documents
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp',
        # Archives
        '.zip', '.tar.gz', '.tgz', '.rar', '.7z', '.bz2', '.gz',
        # Executables
        '.exe', '.dmg', '.pkg', '.deb', '.rpm', '.msi', '.bin',
        # Media
        '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.swf',
        # System files
        '.so', '.dll', '.dylib', '.o', '.a', '.lib', '.pdb',
        # Other
        '.css', '.js', '.woff', '.woff2', '.ttf', '.eot', '.otf', '.map'
    ]
    
    def __init__(
        self,
        base_url: str,
        output_dir: Path = None,
        allowed_domains: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        max_pages: Optional[int] = None,
        request_delay: Optional[float] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        user_agent: Optional[str] = None,
        timeout: int = 10
    ):
        """Initialize web documentation downloader.
        
        Args:
            base_url: Base URL of the documentation site
            output_dir: Directory to save downloaded pages (default: settings.data_dir / 'web_docs')
            allowed_domains: List of allowed domains (None for base domain only)
            exclude_paths: List of URL paths to exclude
            max_pages: Maximum number of pages to download (default: settings.llm.max_pages)
            request_delay: Delay between requests in seconds (default: settings.llm.request_delay)
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay between retry attempts in seconds
            user_agent: User agent string to use for requests
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.base_url = base_url.rstrip('/')
        self.output_dir = output_dir
        self.allowed_domains = allowed_domains or [urlparse(base_url).netloc]
        self.exclude_paths = set(exclude_paths or [])
        self.max_pages = max_pages
        self.request_delay = request_delay
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent
        })
        
        # Track visited URLs and documents
        self.visited_urls: Set[str] = set()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _should_visit(self, url: str) -> bool:
        """Check if a URL should be visited.
        
        Args:
            url: URL to check
            
        Returns:
            bool: True if the URL should be visited, False otherwise
        """
        # Skip if already visited
        if url in self.visited_urls:
            return False
        
        parsed = urlparse(url)
        
        # Skip non-HTTP(S) URLs
        if parsed.scheme not in ('http', 'https'):
            return False
        
        # Skip disallowed domains
        if parsed.netloc not in self.allowed_domains:
            return False
        
        # Skip excluded paths
        if any(excluded in parsed.path for excluded in self.exclude_paths):
            return False
        
        # Skip common non-content URLs
        if any(url.lower().endswith(ext) for ext in self.SKIP_EXTENSIONS):
            return False
        
        return True
    
    def _fetch_with_retry(self, url: str) -> Optional[requests.Response]:
        """Fetch a URL with retry logic.
        
        Args:
            url: URL to fetch
            
        Returns:
            Optional[requests.Response]: Response object if successful, None otherwise
        """
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    logger.warning(f"Failed to fetch {url} after {self.max_retries} attempts: {str(e)}")
                    return None
                
                logger.debug(
                    f"Attempt {attempt + 1} failed for {url}. "
                    f"Retrying in {self.retry_delay} seconds..."
                )
                time.sleep(self.retry_delay)
        
        return None
    
    def _get_page_content(self, url: str) -> Optional[Tuple[str, str, str]]:
        """Fetch and parse a web page.
        
        Args:
            url: URL of the page to fetch
            
        Returns:
            Optional[Tuple[str, str, str]]: Tuple of (title, content, html_content) if successful, None otherwise
        """
        response = self._fetch_with_retry(url)
        if not response:
            return None
        
        # Only process HTML content
        content_type = response.headers.get('content-type', '')
        if 'text/html' not in content_type:
            return None
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Create a copy of the soup for markdown conversion
            soup_for_markdown = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements from both soups
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe']):
                element.decompose()
            
            for element in soup_for_markdown.find_all(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe']):
                element.decompose()
            
            # Get page title
            title = soup.title.string if soup.title else url
            
            # Get main content (try to find the main content area)
            main_content = soup.find('main') or soup.find('article') or soup.find('div', role='main') or soup
            
            # Clean up text
            text_elements = []
            for element in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'pre', 'code', 'table']):
                text = element.get_text(separator=' ', strip=True)
                if text:
                    text_elements.append(text)
            
            content = '\n\n'.join(text_elements)
            
            # Get HTML content for markdown conversion
            html_content = str(soup_for_markdown)
            
            return title, content, html_content
            
        except Exception as e:
            logger.warning(f"Error parsing {url}: {str(e)}")
            return None
    
    def _extract_links(self, url: str, soup: BeautifulSoup) -> List[str]:
        """Extract all links from a page.
        
        Args:
            url: URL of the current page
            soup: BeautifulSoup object of the page
            
        Returns:
            List[str]: List of unique, normalized URLs
        """
        links = set()
        base_domain = urlparse(url).netloc
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            
            # Skip empty links and anchors
            if not href or href.startswith('#'):
                continue
            
            # Handle relative URLs
            if href.startswith('/'):
                href = f"{urlparse(url).scheme}://{base_domain}{href}"
            elif not href.startswith(('http://', 'https://')):
                href = urljoin(url, href)
            
            # Normalize URL
            href = href.split('#')[0].rstrip('/')
            
            if self._should_visit(href):
                links.add(href)
                
        return list(links)
    
    def _convert_html_to_markdown(self, html_content: str, url: str) -> str:
        """Convert HTML content to Markdown format.
        
        Args:
            html_content: The HTML content to convert
            url: The URL of the page (for handling relative links)
            
        Returns:
            str: The converted Markdown content
        """
        try:
            # Configure html2text
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.reference_links = True
            h.protect_links = True
            h.body_width = 0  # Don't wrap lines
            
            # Convert to markdown
            markdown = h.handle(html_content)
            
            # Add source URL at the top
            markdown = f"# Source: [{url}]({url})\n\n---\n\n{markdown}"
            
            return markdown.strip()
            
        except Exception as e:
            logger.warning(f"Error converting HTML to Markdown: {str(e)}")
            return ""
    
    def _create_file_path(self, url: str) -> Path:
        """Save web page content to a file in the output directory as Markdown.
        
        Args:
            url: The URL of the page
            title: The title of the page
            markdown_content: The Markdown content to save
            
        Returns:
            Path: The path to the saved file
        """
        # Create a safe filename from the URL
        parsed_url = urlparse(url)
        path = parsed_url.path.strip('/')
        
        # If path is empty, use 'index'
        if not path:
            path = 'index'
        
        # Create directory structure based on URL path
        file_path = self.output_dir / parsed_url.netloc / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Change extension to .json
        file_path = file_path.with_suffix('.json')
        
        # Save to file
        # file_path.write_text(markdown_content, encoding='utf-8')
        return file_path
    
    def download(self) -> List[Document]:
        """Crawl and download documentation pages.
        
        Returns:
            List[Document]: List of downloaded documents
        """
        documents = []
        queue = [self.base_url]
        
        try:
            with tqdm(desc="Crawling documentation", unit="page") as pbar:
                while queue and len(documents) < self.max_pages:
                    if not queue:
                        break
                        
                    url = queue.pop(0)
                    
                    # Skip if already visited
                    if url in self.visited_urls:
                        continue
                    
                    # Mark as visited
                    self.visited_urls.add(url)
                    
                    # Fetch page content
                    result = self._get_page_content(url)
                    if not result:
                        self.failed_downloads += 1
                        continue
                    
                    title, content, html_content = result
                    markdown_content = self._convert_html_to_markdown(html_content, url)
                    
                    # Skip pages with little content
                    if len(content) < 1:  # Minimum content length
                        self.skipped_files += 1
                        continue
                    
                    # Save the page to a file as Markdown
                    file_path = None
                    try:
                        file_path = self._create_file_path(url)
                        logger.debug(f"Saved {url} to {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to save {url} to file: {str(e)}")
                    
                    # Create document with both raw content and markdown
                    doc = Document(
                        id=f"web_{len(documents)}",
                        title=title,
                        content=markdown_content,
                        type=DocumentType.DOCUMENT,
                        source=DocumentSource.WEB,
                        url=url,
                        metadata={
                            "title": title,
                            "file_path": str(file_path.relative_to(self.output_dir)) if file_path else None,
                            "format": "markdown"
                        }
                    )

                    try:
                        file_path.write_text(doc.model_dump_json(), encoding='utf-8')
                    except Exception as e:
                        logger.warning(f"Failed to save {url} to file: {str(e)}")
                    
                    documents.append(doc)
                    self.downloaded_files += 1
                    pbar.update(1)
                    
                    # Extract links for further crawling
                    if len(documents) < self.max_pages:
                        try:
                            response = self._fetch_with_retry(url)
                            if response and 'text/html' in response.headers.get('content-type', ''):
                                soup = BeautifulSoup(response.text, 'html.parser')
                                new_links = self._extract_links(url, soup)
                                
                                # Add new links to queue
                                for link in new_links:
                                    if link not in self.visited_urls and link not in queue:
                                        queue.append(link)
                        except Exception as e:
                            logger.warning(f"Error extracting links from {url}: {str(e)}")
            
            logger.info(
                f"Downloaded {self.downloaded_files} documentation pages, "
                f"skipped {self.skipped_files}, failed {self.failed_downloads}"
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Error crawling documentation: {str(e)}")
            raise
        finally:
            # Clean up session
            self.session.close()
