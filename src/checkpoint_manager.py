"""Checkpoint manager for handling session timeouts in Colab Pro."""

import os
import time
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages automatic checkpointing to prevent data loss from Colab session timeouts.
    
    Implements 30-minute auto-save to Google Drive for persistence.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        interval_minutes: int = 30,
        auto_start: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints (should be on Google Drive)
            interval_minutes: Time interval between automatic checkpoints
            auto_start: Whether to start auto-save thread immediately
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.interval_minutes = interval_minutes
        self.interval_seconds = interval_minutes * 60
        
        self._state: Dict[str, Any] = {}
        self._last_checkpoint_time: Optional[datetime] = None
        self._auto_save_thread: Optional[threading.Thread] = None
        self._stop_auto_save = threading.Event()
        
        if auto_start:
            self.start_auto_save()
    
    def save_checkpoint(self, name: str = "checkpoint", metadata: Optional[Dict] = None) -> str:
        """
        Save current state to checkpoint file.
        
        Args:
            name: Name for the checkpoint file
            metadata: Optional metadata to save with checkpoint
        
        Returns:
            Path to saved checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{name}_{timestamp}.pkl"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint_data = {
            'state': self._state,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self._last_checkpoint_time = datetime.now()
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Clean up old checkpoints (keep last 5)
            self._cleanup_old_checkpoints(name, keep_last=5)
            
            return str(checkpoint_path)
        
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load state from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, loads most recent.
        
        Returns:
            Loaded checkpoint data
        """
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoint files found")
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self._state = checkpoint_data['state']
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            logger.info(f"Checkpoint timestamp: {checkpoint_data['timestamp']}")
            
            return checkpoint_data
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def update_state(self, key: str, value: Any) -> None:
        """
        Update a value in the checkpoint state.
        
        Args:
            key: State key
            value: State value
        """
        self._state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the checkpoint state.
        
        Args:
            key: State key
            default: Default value if key not found
        
        Returns:
            State value
        """
        return self._state.get(key, default)
    
    def get_all_state(self) -> Dict[str, Any]:
        """Get entire state dictionary."""
        return self._state.copy()
    
    def clear_state(self) -> None:
        """Clear all state."""
        self._state.clear()
    
    def start_auto_save(self) -> None:
        """Start automatic checkpoint saving thread."""
        if self._auto_save_thread is not None and self._auto_save_thread.is_alive():
            logger.warning("Auto-save thread already running")
            return
        
        self._stop_auto_save.clear()
        self._auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self._auto_save_thread.start()
        logger.info(f"Auto-save started (interval: {self.interval_minutes} minutes)")
    
    def stop_auto_save(self) -> None:
        """Stop automatic checkpoint saving thread."""
        if self._auto_save_thread is None:
            return
        
        self._stop_auto_save.set()
        self._auto_save_thread.join(timeout=5)
        logger.info("Auto-save stopped")
    
    def _auto_save_loop(self) -> None:
        """Background thread loop for automatic checkpointing."""
        while not self._stop_auto_save.is_set():
            time.sleep(self.interval_seconds)
            
            if not self._stop_auto_save.is_set():
                try:
                    self.save_checkpoint(name="auto_checkpoint")
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")
    
    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get path to most recent checkpoint file."""
        checkpoint_files = list(self.checkpoint_dir.glob("*.pkl"))
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time
        latest = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def _cleanup_old_checkpoints(self, name_prefix: str, keep_last: int = 5) -> None:
        """
        Remove old checkpoint files, keeping only the most recent ones.
        
        Args:
            name_prefix: Prefix of checkpoint files to clean up
            keep_last: Number of recent checkpoints to keep
        """
        checkpoint_files = sorted(
            self.checkpoint_dir.glob(f"{name_prefix}_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove old files
        for old_file in checkpoint_files[keep_last:]:
            try:
                old_file.unlink()
                logger.debug(f"Removed old checkpoint: {old_file}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_file}: {e}")
    
    def time_since_last_checkpoint(self) -> Optional[timedelta]:
        """Get time elapsed since last checkpoint."""
        if self._last_checkpoint_time is None:
            return None
        return datetime.now() - self._last_checkpoint_time
    
    def should_checkpoint(self) -> bool:
        """Check if it's time for a checkpoint based on interval."""
        if self._last_checkpoint_time is None:
            return True
        
        elapsed = self.time_since_last_checkpoint()
        return elapsed.total_seconds() >= self.interval_seconds
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save checkpoint and stop auto-save."""
        try:
            self.save_checkpoint(name="final_checkpoint")
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")
        
        self.stop_auto_save()


def mount_google_drive(mount_path: str = "/content/drive") -> bool:
    """
    Mount Google Drive in Colab environment.
    
    Args:
        mount_path: Path where Drive should be mounted
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from google.colab import drive
        drive.mount(mount_path, force_remount=False)
        logger.info(f"Google Drive mounted at {mount_path}")
        return True
    except ImportError:
        logger.warning("Not running in Colab environment - skipping Drive mount")
        return False
    except Exception as e:
        logger.error(f"Failed to mount Google Drive: {e}")
        return False
