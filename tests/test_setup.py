"""Tests for project setup and configuration."""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import SystemConfig, ModelConfig, SearchConfig, AgentConfig, ColabConfig, load_config
from checkpoint_manager import CheckpointManager
from logging_config import setup_logging, get_logger


class TestConfiguration:
    """Test configuration management."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = SystemConfig()
        
        assert config.model.phobert_model == "vinai/phobert-base"
        assert config.model.xlm_roberta_model == "xlm-roberta-base"
        assert config.search.max_search_results == 10
        assert config.agent.max_iterations == 10
        assert config.colab.checkpoint_interval_minutes == 30
        assert config.log_level == "INFO"
        assert config.random_seed == 42
    
    def test_config_from_env(self):
        """Test loading configuration from environment variables."""
        os.environ['EXA_API_KEY'] = 'test_exa_key'
        os.environ['GEMINI_API_KEY'] = 'test_gemini_key'
        os.environ['LOG_LEVEL'] = 'DEBUG'
        
        config = SystemConfig.from_env()
        
        assert config.search.exa_api_key == 'test_exa_key'
        assert config.agent.gemini_api_key == 'test_gemini_key'
        assert config.log_level == 'DEBUG'
        
        # Cleanup
        del os.environ['EXA_API_KEY']
        del os.environ['GEMINI_API_KEY']
        del os.environ['LOG_LEVEL']
    
    def test_config_yaml_roundtrip(self):
        """Test saving and loading configuration from YAML."""
        config = SystemConfig()
        config.model.batch_size = 16
        config.agent.max_iterations = 15
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.to_yaml(temp_path)
            loaded_config = SystemConfig.from_yaml(temp_path)
            
            assert loaded_config.model.batch_size == 16
            assert loaded_config.agent.max_iterations == 15
            assert loaded_config.model.phobert_model == config.model.phobert_model
        finally:
            os.unlink(temp_path)
    
    def test_config_json_roundtrip(self):
        """Test saving and loading configuration from JSON."""
        config = SystemConfig()
        config.search.max_search_results = 20
        config.colab.checkpoint_interval_minutes = 45
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.to_json(temp_path)
            loaded_config = SystemConfig.from_json(temp_path)
            
            assert loaded_config.search.max_search_results == 20
            assert loaded_config.colab.checkpoint_interval_minutes == 45
        finally:
            os.unlink(temp_path)
    
    def test_approved_sources_list(self):
        """Test that approved sources are properly configured."""
        config = SystemConfig()
        
        expected_sources = [
            "vnexpress.net",
            "vtv.vn",
            "vov.vn",
            "tuoitre.vn",
            "thanhnien.vn",
            "baochinhphu.vn"
        ]
        
        assert config.search.approved_sources == expected_sources


class TestCheckpointManager:
    """Test checkpoint management functionality."""
    
    def test_checkpoint_creation(self):
        """Test creating and saving checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(
                checkpoint_dir=temp_dir,
                interval_minutes=30,
                auto_start=False
            )
            
            manager.update_state('test_key', 'test_value')
            manager.update_state('iteration', 42)
            
            checkpoint_path = manager.save_checkpoint('test')
            
            assert os.path.exists(checkpoint_path)
            assert 'test_' in checkpoint_path
    
    def test_checkpoint_load(self):
        """Test loading checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(
                checkpoint_dir=temp_dir,
                interval_minutes=30,
                auto_start=False
            )
            
            # Save checkpoint
            manager.update_state('model_state', {'layer1': [1, 2, 3]})
            manager.update_state('epoch', 5)
            checkpoint_path = manager.save_checkpoint('training')
            
            # Create new manager and load
            new_manager = CheckpointManager(
                checkpoint_dir=temp_dir,
                interval_minutes=30,
                auto_start=False
            )
            
            loaded_data = new_manager.load_checkpoint(checkpoint_path)
            
            assert loaded_data['state']['model_state'] == {'layer1': [1, 2, 3]}
            assert loaded_data['state']['epoch'] == 5
            assert 'timestamp' in loaded_data
    
    def test_state_operations(self):
        """Test state update and retrieval operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(
                checkpoint_dir=temp_dir,
                interval_minutes=30,
                auto_start=False
            )
            
            # Update state
            manager.update_state('key1', 'value1')
            manager.update_state('key2', 123)
            
            # Get state
            assert manager.get_state('key1') == 'value1'
            assert manager.get_state('key2') == 123
            assert manager.get_state('nonexistent', 'default') == 'default'
            
            # Get all state
            all_state = manager.get_all_state()
            assert all_state == {'key1': 'value1', 'key2': 123}
            
            # Clear state
            manager.clear_state()
            assert manager.get_all_state() == {}
    
    def test_checkpoint_cleanup(self):
        """Test that old checkpoints are cleaned up."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(
                checkpoint_dir=temp_dir,
                interval_minutes=30,
                auto_start=False
            )
            
            # Create multiple checkpoints
            for i in range(10):
                manager.update_state('iteration', i)
                manager.save_checkpoint('test')
            
            # Check that only 5 most recent are kept
            checkpoint_files = list(Path(temp_dir).glob('test_*.pkl'))
            assert len(checkpoint_files) <= 5
    
    def test_context_manager(self):
        """Test checkpoint manager as context manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with CheckpointManager(temp_dir, interval_minutes=30, auto_start=False) as manager:
                manager.update_state('test', 'value')
            
            # Check that final checkpoint was saved
            checkpoint_files = list(Path(temp_dir).glob('final_checkpoint_*.pkl'))
            assert len(checkpoint_files) > 0


class TestLogging:
    """Test logging configuration."""
    
    def test_logging_setup(self):
        """Test basic logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = setup_logging(
                log_level="INFO",
                log_dir=temp_dir,
                log_to_file=True,
                log_to_console=False
            )
            
            assert logger is not None
            
            # Check that log file was created
            log_files = list(Path(temp_dir).glob('factcheck_*.log'))
            assert len(log_files) > 0
            
            # Close all handlers to release file locks
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
    
    def test_get_logger(self):
        """Test getting module-specific logger."""
        logger = get_logger(__name__)
        assert logger is not None
        assert logger.name == __name__
    
    def test_logging_levels(self):
        """Test different logging levels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
                logger = setup_logging(
                    log_level=level,
                    log_dir=temp_dir,
                    log_to_file=False,
                    log_to_console=False
                )
                assert logger is not None


class TestProjectStructure:
    """Test that project structure is correctly set up."""
    
    def test_directories_exist(self):
        """Test that all required directories exist."""
        project_root = Path(__file__).parent.parent
        
        required_dirs = ['src', 'data', 'models', 'experiments', 'notebooks', 'tests']
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Directory {dir_name} does not exist"
            assert dir_path.is_dir(), f"{dir_name} is not a directory"
    
    def test_config_files_exist(self):
        """Test that configuration files exist."""
        project_root = Path(__file__).parent.parent
        
        assert (project_root / 'config.yaml').exists()
        assert (project_root / 'requirements.txt').exists()
        assert (project_root / '.env.example').exists()
    
    def test_requirements_content(self):
        """Test that requirements.txt contains essential packages."""
        project_root = Path(__file__).parent.parent
        requirements_path = project_root / 'requirements.txt'
        
        with open(requirements_path, 'r') as f:
            content = f.read()
        
        essential_packages = [
            'torch',
            'transformers',
            'dgl',
            'hypothesis',
            'beautifulsoup4',
            'selenium',
            'trafilatura',
            'gradio',
            'pytest'
        ]
        
        for package in essential_packages:
            assert package in content, f"Package {package} not in requirements.txt"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
