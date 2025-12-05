"""SQLite-based caching layer for the fact-checking system."""

import sqlite3
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages SQLite-based caching with TTL (time-to-live) support.
    
    Provides caching for:
    - Search results
    - Crawled web content
    - Source credibility scores
    """
    
    def __init__(self, db_path: str = "cache.db", default_ttl_hours: int = 24):
        """
        Initialize cache manager.
        
        Args:
            db_path: Path to SQLite database file
            default_ttl_hours: Default time-to-live for cache entries in hours
        """
        self.db_path = Path(db_path)
        self.default_ttl_hours = default_ttl_hours
        
        # Create database and tables
        self._init_database()
        
        logger.info(f"Cache manager initialized with database: {self.db_path}")
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Search results cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    language TEXT NOT NULL,
                    results TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL
                )
            """)
            
            # Web content cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_cache (
                    url_hash TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL
                )
            """)
            
            # Credibility scores cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS credibility_cache (
                    domain TEXT PRIMARY KEY,
                    score REAL NOT NULL,
                    features TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL
                )
            """)
            
            # Create indices for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_search_expires 
                ON search_cache(expires_at)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_expires 
                ON content_cache(expires_at)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_credibility_expires 
                ON credibility_cache(expires_at)
            """)
            
            conn.commit()
            logger.debug("Database schema initialized")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _compute_hash(self, text: str) -> str:
        """Compute MD5 hash of text for cache key."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _compute_expires_at(self, ttl_hours: Optional[int] = None) -> str:
        """Compute expiration timestamp."""
        ttl = ttl_hours if ttl_hours is not None else self.default_ttl_hours
        expires_at = datetime.now() + timedelta(hours=ttl)
        return expires_at.isoformat()
    
    def _is_expired(self, expires_at: str) -> bool:
        """Check if cache entry is expired."""
        expires_datetime = datetime.fromisoformat(expires_at)
        return datetime.now() > expires_datetime
    
    # Search cache methods
    
    def get_search_results(self, query: str, language: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached search results.
        
        Args:
            query: Search query text
            language: Query language (vi or en)
        
        Returns:
            Cached search results or None if not found/expired
        """
        query_hash = self._compute_hash(f"{query}_{language}")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT results, expires_at FROM search_cache 
                WHERE query_hash = ?
            """, (query_hash,))
            
            row = cursor.fetchone()
            
            if row is None:
                logger.debug(f"Cache miss: search query '{query}' ({language})")
                return None
            
            # Check expiration
            if self._is_expired(row['expires_at']):
                logger.debug(f"Cache expired: search query '{query}' ({language})")
                self._delete_search_cache(query_hash)
                return None
            
            logger.debug(f"Cache hit: search query '{query}' ({language})")
            return json.loads(row['results'])
    
    def set_search_results(
        self,
        query: str,
        language: str,
        results: List[Dict[str, Any]],
        ttl_hours: Optional[int] = None
    ) -> None:
        """
        Cache search results.
        
        Args:
            query: Search query text
            language: Query language
            results: Search results to cache
            ttl_hours: Time-to-live in hours (uses default if None)
        """
        query_hash = self._compute_hash(f"{query}_{language}")
        results_json = json.dumps(results, ensure_ascii=False)
        expires_at = self._compute_expires_at(ttl_hours)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO search_cache 
                (query_hash, query_text, language, results, expires_at)
                VALUES (?, ?, ?, ?, ?)
            """, (query_hash, query, language, results_json, expires_at))
            
            conn.commit()
            logger.debug(f"Cached search results for query '{query}' ({language})")
    
    def _delete_search_cache(self, query_hash: str) -> None:
        """Delete expired search cache entry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM search_cache WHERE query_hash = ?", (query_hash,))
            conn.commit()
    
    # Content cache methods
    
    def get_content(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Get cached web content.
        
        Args:
            url: URL of the content
        
        Returns:
            Cached content or None if not found/expired
        """
        url_hash = self._compute_hash(url)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT content, expires_at FROM content_cache 
                WHERE url_hash = ?
            """, (url_hash,))
            
            row = cursor.fetchone()
            
            if row is None:
                logger.debug(f"Cache miss: content for URL '{url}'")
                return None
            
            # Check expiration
            if self._is_expired(row['expires_at']):
                logger.debug(f"Cache expired: content for URL '{url}'")
                self._delete_content_cache(url_hash)
                return None
            
            logger.debug(f"Cache hit: content for URL '{url}'")
            return json.loads(row['content'])
    
    def set_content(
        self,
        url: str,
        content: Dict[str, Any],
        ttl_hours: Optional[int] = None
    ) -> None:
        """
        Cache web content.
        
        Args:
            url: URL of the content
            content: Content to cache
            ttl_hours: Time-to-live in hours (uses default if None)
        """
        url_hash = self._compute_hash(url)
        content_json = json.dumps(content, ensure_ascii=False)
        expires_at = self._compute_expires_at(ttl_hours)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO content_cache 
                (url_hash, url, content, expires_at)
                VALUES (?, ?, ?, ?)
            """, (url_hash, url, content_json, expires_at))
            
            conn.commit()
            logger.debug(f"Cached content for URL '{url}'")
    
    def _delete_content_cache(self, url_hash: str) -> None:
        """Delete expired content cache entry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM content_cache WHERE url_hash = ?", (url_hash,))
            conn.commit()
    
    # Credibility cache methods
    
    def get_credibility(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Get cached credibility score.
        
        Args:
            domain: Domain name
        
        Returns:
            Cached credibility data or None if not found/expired
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT score, features, expires_at FROM credibility_cache 
                WHERE domain = ?
            """, (domain,))
            
            row = cursor.fetchone()
            
            if row is None:
                logger.debug(f"Cache miss: credibility for domain '{domain}'")
                return None
            
            # Check expiration
            if self._is_expired(row['expires_at']):
                logger.debug(f"Cache expired: credibility for domain '{domain}'")
                self._delete_credibility_cache(domain)
                return None
            
            logger.debug(f"Cache hit: credibility for domain '{domain}'")
            return {
                'score': row['score'],
                'features': json.loads(row['features'])
            }
    
    def set_credibility(
        self,
        domain: str,
        score: float,
        features: Dict[str, Any],
        ttl_hours: Optional[int] = None
    ) -> None:
        """
        Cache credibility score.
        
        Args:
            domain: Domain name
            score: Credibility score
            features: Feature dictionary
            ttl_hours: Time-to-live in hours (uses default if None)
        """
        features_json = json.dumps(features, ensure_ascii=False)
        expires_at = self._compute_expires_at(ttl_hours)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO credibility_cache 
                (domain, score, features, expires_at)
                VALUES (?, ?, ?, ?)
            """, (domain, score, features_json, expires_at))
            
            conn.commit()
            logger.debug(f"Cached credibility for domain '{domain}'")
    
    def _delete_credibility_cache(self, domain: str) -> None:
        """Delete expired credibility cache entry."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM credibility_cache WHERE domain = ?", (domain,))
            conn.commit()
    
    # Cache management methods
    
    def clear_expired(self) -> Dict[str, int]:
        """
        Clear all expired cache entries.
        
        Returns:
            Dictionary with counts of deleted entries per cache type
        """
        now = datetime.now().isoformat()
        counts = {}
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Clear expired search cache
            cursor.execute("DELETE FROM search_cache WHERE expires_at < ?", (now,))
            counts['search'] = cursor.rowcount
            
            # Clear expired content cache
            cursor.execute("DELETE FROM content_cache WHERE expires_at < ?", (now,))
            counts['content'] = cursor.rowcount
            
            # Clear expired credibility cache
            cursor.execute("DELETE FROM credibility_cache WHERE expires_at < ?", (now,))
            counts['credibility'] = cursor.rowcount
            
            conn.commit()
        
        total = sum(counts.values())
        logger.info(f"Cleared {total} expired cache entries: {counts}")
        return counts
    
    def clear_all(self) -> None:
        """Clear all cache entries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM search_cache")
            cursor.execute("DELETE FROM content_cache")
            cursor.execute("DELETE FROM credibility_cache")
            conn.commit()
        
        logger.info("Cleared all cache entries")
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with statistics for each cache type
        """
        stats = {}
        now = datetime.now().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Search cache stats
            cursor.execute("SELECT COUNT(*) as total FROM search_cache")
            total = cursor.fetchone()['total']
            cursor.execute("SELECT COUNT(*) as expired FROM search_cache WHERE expires_at < ?", (now,))
            expired = cursor.fetchone()['expired']
            stats['search'] = {'total': total, 'valid': total - expired, 'expired': expired}
            
            # Content cache stats
            cursor.execute("SELECT COUNT(*) as total FROM content_cache")
            total = cursor.fetchone()['total']
            cursor.execute("SELECT COUNT(*) as expired FROM content_cache WHERE expires_at < ?", (now,))
            expired = cursor.fetchone()['expired']
            stats['content'] = {'total': total, 'valid': total - expired, 'expired': expired}
            
            # Credibility cache stats
            cursor.execute("SELECT COUNT(*) as total FROM credibility_cache")
            total = cursor.fetchone()['total']
            cursor.execute("SELECT COUNT(*) as expired FROM credibility_cache WHERE expires_at < ?", (now,))
            expired = cursor.fetchone()['expired']
            stats['credibility'] = {'total': total, 'valid': total - expired, 'expired': expired}
        
        return stats
    
    def close(self) -> None:
        """Close database connection (cleanup)."""
        logger.info("Cache manager closed")
