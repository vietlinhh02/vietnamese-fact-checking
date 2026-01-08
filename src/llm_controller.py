"""LLM Controller for ReAct agent with multiple provider support."""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import os
from pathlib import Path
import requests

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM provider."""
    content: str
    usage: Dict[str, int]  # tokens used
    model: str
    provider: str
    latency: float  # response time in seconds


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.1,
        **kwargs
    ) -> LLMResponse:
        """Generate response from messages."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini provider."""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = "gemini-flash-lite-latest"
        self.rate_limit = 15  # requests per minute
        self.last_request_time = 0
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model_name)
                logger.info("Initialized Gemini provider")
            except ImportError:
                logger.warning("google-generativeai not installed")
                self.client = None
        else:
            logger.warning("No Gemini API key provided")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Gemini is available."""
        return self.client is not None and self.api_key is not None
    
    def _rate_limit_wait(self):
        """Wait if needed to respect rate limits."""
        time_since_last = time.time() - self.last_request_time
        min_interval = 60 / self.rate_limit  # seconds between requests
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.1,
        response_schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Gemini with optional structured output."""
        if not self.is_available():
            raise RuntimeError("Gemini provider not available")
        
        self._rate_limit_wait()
        start_time = time.time()
        
        try:
            # Convert messages to Gemini format
            prompt = self._format_messages(messages)
            
            # Configure generation
            generation_config = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            
            # Add structured output configuration if schema provided
            if response_schema:
                generation_config["response_mime_type"] = "application/json"
                # Note: response_json_schema may not be available in all Gemini versions
                # We'll use prompt engineering as fallback
                if hasattr(self.client, 'supports_json_schema'):
                    generation_config["response_json_schema"] = response_schema
                else:
                    # Add schema to prompt for structured output
                    schema_prompt = f"\n\nPlease respond with valid JSON matching this schema:\n{response_schema}"
                    prompt += schema_prompt
            
            # Generate response
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            self.last_request_time = time.time()
            latency = self.last_request_time - start_time
            
            # Extract usage info (approximate)
            usage = {
                "prompt_tokens": len(prompt.split()) * 1.3,  # rough estimate
                "completion_tokens": len(response.text.split()) * 1.3,
                "total_tokens": 0
            }
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            
            return LLMResponse(
                content=response.text,
                usage=usage,
                model=self.model_name,
                provider="gemini",
                latency=latency
            )
            
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Gemini."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"Human: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        return "\n\n".join(formatted)


class GroqProvider(LLMProvider):
    """Groq API provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq provider."""
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = "llama-3.1-70b-versatile"
        self.rate_limit = 30  # requests per minute
        self.last_request_time = 0
        
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info("Initialized Groq provider")
            except ImportError:
                logger.warning("groq package not installed")
                self.client = None
        else:
            logger.warning("No Groq API key provided")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Groq is available."""
        return self.client is not None and self.api_key is not None
    
    def _rate_limit_wait(self):
        """Wait if needed to respect rate limits."""
        time_since_last = time.time() - self.last_request_time
        min_interval = 60 / self.rate_limit
        
        if time_since_last < min_interval:
            wait_time = min_interval - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.1,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Groq."""
        if not self.is_available():
            raise RuntimeError("Groq provider not available")
        
        self._rate_limit_wait()
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            self.last_request_time = time.time()
            latency = self.last_request_time - start_time
            
            # Extract usage info
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage=usage,
                model=self.model_name,
                provider="groq",
                latency=latency
            )
            
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise


class LocalLlamaProvider(LLMProvider):
    """Local Llama model provider using transformers."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct"):
        """Initialize local Llama provider."""
        self.model_name = model_name
        self.device = "cuda" if self._check_cuda() else "cpu"
        self.model = None
        self.tokenizer = None
        
        try:
            self._load_model()
            logger.info(f"Initialized local Llama provider: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load local Llama model: {e}")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _load_model(self):
        """Load local Llama model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization if possible
            model_kwargs = {}
            if self.device == "cuda":
                try:
                    # Try 4-bit quantization
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    logger.info("Using 4-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not available, using full precision")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if local model is available."""
        return self.model is not None and self.tokenizer is not None
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.1,
        **kwargs
    ) -> LLMResponse:
        """Generate response using local Llama."""
        if not self.is_available():
            raise RuntimeError("Local Llama provider not available")
        
        start_time = time.time()
        
        try:
            import torch
            
            # Format messages for Llama
            prompt = self._format_messages(messages)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode response
            response_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            latency = time.time() - start_time
            
            # Calculate usage
            prompt_tokens = inputs.input_ids.shape[1]
            completion_tokens = len(response_tokens)
            
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
            
            return LLMResponse(
                content=response_text.strip(),
                usage=usage,
                model=self.model_name,
                provider="local_llama",
                latency=latency
            )
            
        except Exception as e:
            logger.error(f"Local Llama generation failed: {e}")
            raise
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Llama chat template."""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>")
            elif role == "user":
                formatted.append(f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>")
            elif role == "assistant":
                formatted.append(f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>")
        
        # Add assistant start for generation
        formatted.append("<|start_header_id|>assistant<|end_header_id|>\n")
        
        return "".join(formatted)


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible chat completions provider with retry and rate limiting."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-pro",
        timeout: int = 60,
        max_retries: int = 3,
        base_retry_delay: float = 2.0,
        rate_limit_rpm: int = 30
    ):
        self.base_url = (base_url or os.getenv("OPENAI_COMPAT_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_COMPAT_API_KEY")
        self.model_name = model_name or os.getenv("OPENAI_COMPAT_MODEL", "gemini-2.5-pro")
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        
        # Rate limiting
        self.rate_limit_rpm = rate_limit_rpm
        self.min_request_interval = 60.0 / rate_limit_rpm  # seconds between requests
        self.last_request_time = 0.0

    def is_available(self) -> bool:
        return bool(self.base_url)

    def _rate_limit_wait(self):
        """Wait if needed to respect rate limits."""
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            logger.debug(f"OpenAI-compat rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.1,
        **kwargs
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("OpenAI-compatible provider not available")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        payload.update(kwargs)

        url = f"{self.base_url}/v1/chat/completions"
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting before each request
                self._rate_limit_wait()
                
                start_time = time.time()
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                self.last_request_time = time.time()
                
                # Handle rate limit errors with retry
                if response.status_code == 429:
                    # Get retry delay from header or use exponential backoff
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = self.base_retry_delay * (2 ** attempt)
                    else:
                        delay = self.base_retry_delay * (2 ** attempt)
                    
                    logger.warning(
                        f"Rate limit hit (429), attempt {attempt + 1}/{self.max_retries}, "
                        f"waiting {delay:.1f}s before retry"
                    )
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(delay)
                        continue
                    else:
                        response.raise_for_status()  # Let it raise on last attempt
                
                # Handle other server errors with retry
                if response.status_code >= 500:
                    delay = self.base_retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Server error ({response.status_code}), attempt {attempt + 1}/{self.max_retries}, "
                        f"waiting {delay:.1f}s before retry"
                    )
                    if attempt < self.max_retries - 1:
                        time.sleep(delay)
                        continue
                
                response.raise_for_status()
                data = response.json()

                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                usage = data.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
                total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
                if total_tokens == 0:
                    prompt_tokens = int(len(" ".join(m.get("content", "") for m in messages).split()) * 1.3)
                    completion_tokens = int(len(content.split()) * 1.3)
                    total_tokens = prompt_tokens + completion_tokens

                latency = time.time() - start_time
                return LLMResponse(
                    content=content,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    },
                    model=self.model_name,
                    provider="openai_compat",
                    latency=latency
                )
                
            except requests.exceptions.Timeout as e:
                last_error = e
                delay = self.base_retry_delay * (2 ** attempt)
                logger.warning(
                    f"Request timeout, attempt {attempt + 1}/{self.max_retries}, "
                    f"waiting {delay:.1f}s before retry"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    continue
                    
            except requests.exceptions.RequestException as e:
                last_error = e
                # Don't retry for client errors (4xx except 429)
                if hasattr(e, 'response') and e.response is not None:
                    if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                        raise
                
                delay = self.base_retry_delay * (2 ** attempt)
                logger.warning(
                    f"Request error: {e}, attempt {attempt + 1}/{self.max_retries}, "
                    f"waiting {delay:.1f}s before retry"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    continue
        
        # All retries exhausted
        raise RuntimeError(f"All {self.max_retries} retries failed. Last error: {last_error}")


class LLMController:
    """Controller for managing multiple LLM providers with fallback."""
    
    def __init__(
        self,
        providers: Optional[List[str]] = None,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        local_model: Optional[str] = None,
        openai_compat_base_url: Optional[str] = None,
        openai_compat_api_key: Optional[str] = None,
        openai_compat_model: Optional[str] = None
    ):
        """Initialize LLM controller with providers.
        
        Args:
            providers: List of provider names to try in order
            gemini_api_key: Gemini API key
            groq_api_key: Groq API key
            local_model: Local model name for fallback
        """
        if providers is None:
            providers = ["gemini", "groq", "local_llama"]
        
        self.providers = {}
        self.provider_order = providers
        
        # Initialize providers
        if "gemini" in providers:
            self.providers["gemini"] = GeminiProvider(gemini_api_key)
        
        if "groq" in providers:
            self.providers["groq"] = GroqProvider(groq_api_key)
        
        if "local_llama" in providers:
            model_name = local_model or "meta-llama/Llama-3.2-3B-Instruct"
            self.providers["local_llama"] = LocalLlamaProvider(model_name)

        if "openai_compat" in providers:
            self.providers["openai_compat"] = OpenAICompatibleProvider(
                base_url=openai_compat_base_url,
                api_key=openai_compat_api_key,
                model_name=openai_compat_model or "gemini-2.5-pro"
            )
        
        # Track usage
        self.usage_stats = {name: {"requests": 0, "tokens": 0} for name in self.providers}
        
        logger.info(f"Initialized LLM controller with providers: {list(self.providers.keys())}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return [name for name, provider in self.providers.items() if provider.is_available()]
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.1,
        preferred_provider: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response with fallback between providers.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            preferred_provider: Preferred provider to try first
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            RuntimeError: If no providers are available
        """
        # Determine provider order
        provider_order = self.provider_order.copy()
        if preferred_provider and preferred_provider in self.providers:
            # Move preferred provider to front
            if preferred_provider in provider_order:
                provider_order.remove(preferred_provider)
            provider_order.insert(0, preferred_provider)
        
        last_error = None
        
        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue
                
            provider = self.providers[provider_name]
            
            if not provider.is_available():
                logger.debug(f"Provider {provider_name} not available, trying next")
                continue
            
            try:
                logger.debug(f"Trying provider: {provider_name}")
                response = provider.generate(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                
                # Update usage stats
                self.usage_stats[provider_name]["requests"] += 1
                self.usage_stats[provider_name]["tokens"] += response.usage["total_tokens"]
                
                logger.info(f"Generated response using {provider_name} ({response.usage['total_tokens']} tokens)")
                return response
                
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                last_error = e
                continue
        
        # No providers worked
        available = self.get_available_providers()
        if not available:
            raise RuntimeError("No LLM providers are available")
        else:
            raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """Get usage statistics for all providers."""
        return self.usage_stats.copy()
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        for provider_name in self.usage_stats:
            self.usage_stats[provider_name] = {"requests": 0, "tokens": 0}


# Prompt templates for ReAct agent
class PromptTemplates:
    """Prompt templates for different ReAct agent phases."""
    
    # Template with placeholder for temporal context
    SYSTEM_PROMPT_TEMPLATE = """You are a Vietnamese fact-checking agent. Your task is to verify claims by following a structured reasoning process.

{temporal_context}

You have access to these tools:
- search(query): Search for information using Exa API
- crawl(url): Extract content from a specific webpage
- analyze_credibility(source_url): Analyze the credibility of a source

Follow this process:
1. THOUGHT: Analyze the claim and plan your verification approach
2. ACTION: Choose and execute a tool with parameters
3. OBSERVATION: Process the results and decide next steps
4. Repeat until you have sufficient evidence
5. CONCLUSION: Make a final verdict (Supported/Refuted/Not Enough Info)

Always think step by step and be thorough in your analysis."""
    
    # Legacy static prompt (kept for backward compatibility)
    SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(temporal_context="")
    
    @classmethod
    def get_system_prompt(cls) -> str:
        """Get system prompt with current temporal context.
        
        Returns:
            System prompt string with current datetime and temporal reasoning guidelines.
        """
        try:
            from src.temporal_context import get_temporal_context_prompt
            temporal_context = get_temporal_context_prompt()
        except ImportError:
            # Fallback if temporal_context module not available
            from datetime import datetime
            temporal_context = f"CURRENT DATE: {datetime.now().strftime('%B %d, %Y')}"
        
        return cls.SYSTEM_PROMPT_TEMPLATE.format(temporal_context=temporal_context)

    REASONING_TEMPLATE = """
CLAIM: {claim}

Current evidence collected:
{evidence_summary}

THOUGHT: {thought}

ACTION: {action}

OBSERVATION: {observation}

Next step: {next_step}
"""

    CONCLUSION_TEMPLATE = """
CLAIM: {claim}

EVIDENCE SUMMARY:
{evidence_summary}

REASONING TRACE:
{reasoning_trace}

FINAL VERDICT: {verdict}
CONFIDENCE: {confidence}

EXPLANATION:
{explanation}
"""


def create_llm_controller(config: Optional[Dict[str, Any]] = None) -> LLMController:
    """Create LLM controller from configuration.
    
    Args:
        config: Configuration dict with API keys and preferences
        
    Returns:
        Configured LLMController instance
    """
    if config is None:
        config = {}
    
    return LLMController(
        providers=config.get("providers", ["gemini", "groq", "local_llama"]),
        gemini_api_key=config.get("gemini_api_key"),
        groq_api_key=config.get("groq_api_key"),
        local_model=config.get("local_model", "meta-llama/Llama-3.2-3B-Instruct"),
        openai_compat_base_url=config.get("openai_compat_base_url"),
        openai_compat_api_key=config.get("openai_compat_api_key"),
        openai_compat_model=config.get("openai_compat_model")
    )


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test LLM controller
    controller = create_llm_controller()
    
    print("Available providers:", controller.get_available_providers())
    
    # Test generation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Vietnam?"}
    ]
    
    try:
        response = controller.generate(messages, max_tokens=100)
        print(f"\nResponse from {response.provider}:")
        print(response.content)
        print(f"Usage: {response.usage}")
    except Exception as e:
        print(f"Generation failed: {e}")
