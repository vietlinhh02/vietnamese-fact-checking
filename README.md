# Vietnamese Autonomous Fact-Checking System

A research-oriented platform for verifying factual claims in Vietnamese text using an agent-based architecture with the ReAct (Reasoning and Acting) framework.

## Project Structure

```
vietnamese-fact-checking/
├── src/                    # Source code
│   ├── config.py          # Configuration management
│   ├── checkpoint_manager.py  # Checkpoint handling for Colab
│   └── logging_config.py  # Logging setup
├── data/                   # Data storage
├── models/                 # Trained models
├── experiments/            # Experiment results
├── notebooks/              # Jupyter notebooks
│   └── setup_colab.ipynb  # Colab setup notebook
├── tests/                  # Test files
├── config.yaml            # Main configuration file
├── requirements.txt       # Python dependencies
└── .env.example          # Environment variables template
```

## Setup Instructions

### For Google Colab Pro

1. **Mount Google Drive**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Upload project files to Drive**
   - Create folder: `/content/drive/MyDrive/vietnamese-fact-checking/`
   - Upload all project files

3. **Open and run** `notebooks/setup_colab.ipynb`

4. **Add API keys to Colab Secrets**
   - Click the key icon in the left sidebar
   - Add: `GOOGLE_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`

### For Local Development

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Run tests**
   ```bash
   pytest tests/
   ```

## Configuration

The system uses YAML configuration files. Edit `config.yaml` to customize:

- **Model settings**: Model names, batch sizes, quantization
- **Search settings**: API keys, rate limits, approved sources
- **Agent settings**: Max iterations, confidence thresholds
- **Colab settings**: Checkpoint intervals, memory limits

You can also load configuration from environment variables or JSON files.

## Checkpoint Management

The system automatically saves checkpoints every 30 minutes to prevent data loss from Colab session timeouts:

```python
from src.checkpoint_manager import CheckpointManager

# Initialize with auto-save
checkpoint_manager = CheckpointManager(
    checkpoint_dir="/content/drive/MyDrive/vietnamese-fact-checking/checkpoints",
    interval_minutes=30,
    auto_start=True
)

# Save state
checkpoint_manager.update_state('model_weights', model.state_dict())
checkpoint_manager.save_checkpoint('training_checkpoint')

# Load state
checkpoint_data = checkpoint_manager.load_checkpoint()
model.load_state_dict(checkpoint_data['state']['model_weights'])
```

## Logging

Logs are saved to both console and file:

```python
from src.logging_config import setup_logging, get_logger

# Setup logging system
setup_logging(log_level="INFO", log_dir="./logs")

# Get logger for your module
logger = get_logger(__name__)
logger.info("Processing claim...")
```

## API Keys Required

- **Google Custom Search API**: For web search (100 queries/day free)
- **Gemini API**: For LLM reasoning (15 RPM free tier)
- **Groq API**: Alternative LLM (30 RPM free tier)
- **Google Translate API** (optional): For translation

## Features

- ✅ Automatic checkpoint saving (30-minute intervals)
- ✅ Google Drive integration for persistence
- ✅ Flexible configuration management (YAML/JSON/ENV)
- ✅ Comprehensive logging system
- ✅ Optimized for Colab Pro constraints
- ✅ Support for multiple LLM providers

## Next Steps

After setup, you can:

1. Implement claim detection module (Task 2)
2. Build web crawling module (Task 3)
3. Develop ReAct agent core (Task 13)
4. Create demo interface (Task 19)

## License

Research project - see LICENSE file for details.
