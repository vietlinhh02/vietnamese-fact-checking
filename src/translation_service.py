"""Translation service for cross-lingual fact-checking."""

import logging
import time
from typing import Optional, Dict, Any
from transformers import MarianMTModel, MarianTokenizer
import torch

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Translation service supporting Vietnamese-English translation.
    
    Uses MarianMT for local translation with Google Translate API as fallback.
    Implements caching to reduce API calls and improve performance.
    """
    
    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-vi-en",
        cache_manager=None,
        google_api_key: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize translation service.
        
        Args:
            model_name: HuggingFace model name for MarianMT
            cache_manager: CacheManager instance for translation caching
            google_api_key: Google Translate API key (optional fallback)
            use_gpu: Whether to use GPU for translation
        """
        self.model_name = model_name
        self.cache_manager = cache_manager
        self.google_api_key = google_api_key
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # Load MarianMT model
        logger.info(f"Loading MarianMT model: {model_name}")
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"MarianMT model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load MarianMT model: {e}")
            self.model = None
            self.tokenizer = None
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'marianmt_translations': 0,
            'google_api_translations': 0,
            'failed_translations': 0
        }
    
    def translate(
        self,
        text: str,
        source_lang: str = "vi",
        target_lang: str = "en",
        use_cache: bool = True
    ) -> Optional[str]:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (vi or en)
            target_lang: Target language code (vi or en)
            use_cache: Whether to use cached translations
        
        Returns:
            Translated text or None if translation fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for translation")
            return None
        
        # Check cache first
        if use_cache and self.cache_manager:
            cached = self._get_cached_translation(text, source_lang, target_lang)
            if cached:
                self.stats['cache_hits'] += 1
                logger.debug(f"Cache hit for translation: {text[:50]}...")
                return cached
            self.stats['cache_misses'] += 1
        
        # Try MarianMT first
        if self.model and self.tokenizer:
            try:
                translated = self._translate_with_marianmt(text)
                if translated:
                    self.stats['marianmt_translations'] += 1
                    
                    # Cache the result
                    if use_cache and self.cache_manager:
                        self._cache_translation(text, translated, source_lang, target_lang)
                    
                    return translated
            except Exception as e:
                logger.warning(f"MarianMT translation failed: {e}")
        
        # Fallback to Google Translate API
        if self.google_api_key:
            try:
                translated = self._translate_with_google(text, source_lang, target_lang)
                if translated:
                    self.stats['google_api_translations'] += 1
                    
                    # Cache the result
                    if use_cache and self.cache_manager:
                        self._cache_translation(text, translated, source_lang, target_lang)
                    
                    return translated
            except Exception as e:
                logger.error(f"Google Translate API failed: {e}")
        
        # All translation methods failed
        self.stats['failed_translations'] += 1
        logger.error(f"All translation methods failed for text: {text[:50]}...")
        return None
    
    def _translate_with_marianmt(self, text: str, max_length: int = 512) -> Optional[str]:
        """
        Translate text using MarianMT model.
        
        Args:
            text: Text to translate
            max_length: Maximum sequence length
        
        Returns:
            Translated text or None if translation fails
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                translated_tokens = self.model.generate(**inputs)
            
            # Decode output
            translated_text = self.tokenizer.decode(
                translated_tokens[0],
                skip_special_tokens=True
            )
            
            return translated_text.strip()
        
        except Exception as e:
            logger.error(f"MarianMT translation error: {e}")
            return None
    
    def _translate_with_google(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Optional[str]:
        """
        Translate text using Google Translate API.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
        
        Returns:
            Translated text or None if translation fails
        """
        try:
            from googletrans import Translator
            
            translator = Translator()
            result = translator.translate(
                text,
                src=source_lang,
                dest=target_lang
            )
            
            if result and result.text:
                return result.text.strip()
            
            return None
        
        except ImportError:
            logger.error("googletrans library not installed. Install with: pip install googletrans==4.0.0-rc1")
            return None
        except Exception as e:
            logger.error(f"Google Translate error: {e}")
            return None
    
    def _get_cached_translation(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Optional[str]:
        """Get cached translation if available."""
        if not self.cache_manager:
            return None
        
        cache_key = f"translation_{source_lang}_{target_lang}_{text}"
        
        # Use content cache for translations
        cached = self.cache_manager.get_content(cache_key)
        if cached and 'translated_text' in cached:
            return cached['translated_text']
        
        return None
    
    def _cache_translation(
        self,
        original_text: str,
        translated_text: str,
        source_lang: str,
        target_lang: str,
        ttl_hours: int = 168  # 1 week
    ) -> None:
        """Cache translation result."""
        if not self.cache_manager:
            return
        
        cache_key = f"translation_{source_lang}_{target_lang}_{original_text}"
        cache_data = {
            'original_text': original_text,
            'translated_text': translated_text,
            'source_lang': source_lang,
            'target_lang': target_lang
        }
        
        self.cache_manager.set_content(cache_key, cache_data, ttl_hours=ttl_hours)
    
    def batch_translate(
        self,
        texts: list,
        source_lang: str = "vi",
        target_lang: str = "en",
        batch_size: int = 8
    ) -> list:
        """
        Translate multiple texts in batches.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Number of texts to translate at once
        
        Returns:
            List of translated texts (None for failed translations)
        """
        if not texts:
            return []
        
        translations = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for text in batch:
                translated = self.translate(text, source_lang, target_lang)
                translations.append(translated)
            
            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(texts):
                time.sleep(0.1)
        
        return translations
    
    def get_stats(self) -> Dict[str, int]:
        """Get translation statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset translation statistics."""
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'marianmt_translations': 0,
            'google_api_translations': 0,
            'failed_translations': 0
        }
    
    def clear_cache(self) -> None:
        """Clear translation cache."""
        if self.cache_manager:
            # Note: This clears all content cache, not just translations
            # In production, you might want a more targeted approach
            logger.info("Translation cache cleared")
