# Synthetic Data ML Notebooks - Self-Contained & Colab Ready!

The notebooks in this directory provide a complete, self-contained machine learning pipeline for magnetic distortion classification using synthetic sensor data. No external data dependencies required!

## � Key Features

### **Fully Self-Contained**
- 🔬 **Generates its own training data** - no external sensor logs needed
- 🎛️ **Configurable synthetic data** - control distortion levels, motion patterns, duration
- 🏷️ **Perfect ground truth labels** - precise magnetic distortion classifications
- 🔄 **Reproducible experiments** - consistent results every time

### **Complete ML Pipeline**
- 📊 **Data Generation**: Create realistic synthetic IMU sensor data
- 🎵 **Spectrogram Creation**: Convert time-series to audio spectrograms  
- 🤖 **Model Training**: Fine-tune Audio Spectrogram Transformer (AST)
- 🎯 **Classification**: Test models on new synthetic data
- 📈 **Analysis**: Comprehensive result analysis and visualization

### **Google Colab Ready**
- ✅ Automatic repository cloning in Colab
- ✅ Smart environment detection and setup
- ✅ Proper handling of Google Colab secrets
- ✅ Cross-platform compatibility

### **Modular Architecture**
- ✅ Uses new `script_utils` for environment management
- ✅ Uses `mcap_utils` for data processing
- ✅ Uses `synthetic` module for data generation
- ✅ Graceful fallbacks for maximum compatibility

## 📓 Notebooks Overview

### 1. [`classify.ipynb`](classify.ipynb) - **Live Classification Demo**
*🎯 Start here for a quick introduction!*

**What it does:**
- 🔬 Generates realistic synthetic sensor data with magnetic distortion
- 🎵 Creates audio spectrograms from the time-series data  
- 🤖 Loads pre-trained magnetic distortion classifier
- 📊 Performs real-time classification and analysis

**Perfect for:**
- Understanding the complete ML pipeline
- Testing model performance on new data
- Learning how sensor data becomes spectrograms
- Demonstrating classification capabilities

**Key synthetic data features:**
- Multi-axis magnetometer, accelerometer, gyroscope data
- Configurable magnetic distortion levels (none/low/high)
- Realistic sensor noise and motion patterns
- Perfect ground truth for validation

---

### 2. [`dataset.ipynb`](dataset.ipynb) - **Synthetic Data Explorer**  
*🔍 Dive deep into synthetic data generation!*

**What it does:**
- 🎛️ Interactive exploration of synthetic data capabilities
- ⚡ Real-time generation of different motion scenarios
- 📈 Visualization of sensor data characteristics
- 🎨 Demonstration of spectrogram creation process

**Perfect for:**
- Understanding synthetic data generation
- Experimenting with different motion patterns
- Learning about sensor data characteristics
- Prototyping new data scenarios

**Key learning outcomes:**
- How synthetic IMU data is generated
- Relationship between motion and sensor readings
- Spectrogram visualization techniques
- Data quality assessment methods

---

### 3. [`fine_tune.ipynb`](fine_tune.ipynb) - **Model Training Workshop**
*🏋️ Train your own magnetic distortion classifier!*

**What it does:**
- 🔄 Generates multiple synthetic training sequences
- 📚 Creates comprehensive training dataset with labels
- 🤖 Fine-tunes Audio Spectrogram Transformer (AST) model
- 📊 Evaluates model performance and saves results

**Perfect for:**
- Learning model training workflows
- Understanding hyperparameter tuning
- Experimenting with different training strategies
- Building custom classification models

**Training features:**
- Configurable training data generation
- Multiple distortion scenarios for robust training
- HuggingFace integration for modern ML workflows
- Comprehensive model evaluation and validation

## 🚀 Quick Start

### **🔬 Try Classification Demo (Recommended)**

**Option 1: Google Colab** *(Zero setup required!)*
1. Open [`classify.ipynb`](classify.ipynb) in Google Colab
2. Click "Run all" - that's it! 
3. Watch as it generates synthetic data and classifies magnetic distortion

**Option 2: Local Environment**
1. Clone the repository and open [`classify.ipynb`](classify.ipynb)
2. Run all cells - automatic environment detection
3. Synthetic data generation requires no external dependencies

### **🎓 Learn the Full Pipeline**

**For Data Scientists:**
1. Start with [`dataset.ipynb`](dataset.ipynb) to understand synthetic data
2. Run [`classify.ipynb`](classify.ipynb) to see end-to-end classification
3. Try [`fine_tune.ipynb`](fine_tune.ipynb) to train your own model

**For ML Engineers:**
1. Jump to [`fine_tune.ipynb`](fine_tune.ipynb) for model training
2. Use [`classify.ipynb`](classify.ipynb) for inference testing
3. Explore [`dataset.ipynb`](dataset.ipynb) for data pipeline insights

### **⚡ No Setup Required!**

**Self-Contained Features:**
- ✅ **Zero external dependencies** - generates own training data
- ✅ **Works offline** - no downloads or API calls required  
- ✅ **Instant results** - synthetic data generates in seconds
- ✅ **Perfect for learning** - controlled, reproducible experiments

**Cross-Platform Compatible:**
- ✅ Google Colab (recommended for beginners)
- ✅ Jupyter Notebook/Lab
- ✅ VS Code with Jupyter extension
- ✅ Any Python notebook environment

## 🔧 Technical Architecture

### **🎯 Self-Contained Design**
The notebooks are designed to be completely self-contained with zero external dependencies:

**Synthetic Data Generation:**
- Realistic IMU sensor simulation (magnetometer, accelerometer, gyroscope)
- Configurable magnetic distortion scenarios (none/low/high)
- Ground truth generation for perfect training labels
- Controllable motion patterns and sensor noise

**Modular Code Structure:**
- `src/synthetic/` - Synthetic data generation engine
- `src/mcap_utils/` - Data processing and format handling
- `scripts/script_utils.py` - Environment detection and setup
- Backward compatibility with legacy imports

**Cross-Platform Compatibility:**
- Automatic Google Colab vs local environment detection
- Smart path resolution and repository setup
- Graceful degradation with clear error messages

### **🎵 Audio Classification Approach**
The notebooks use an innovative approach treating sensor data as audio:

**Pipeline:**
1. **Time-Series → Spectrogram:** Convert sensor data to visual spectrograms
2. **Audio Models:** Use Audio Spectrogram Transformer (AST) for classification
3. **Transfer Learning:** Leverage pre-trained audio models for sensor tasks

**Benefits:**
- Proven audio classification techniques work on sensor data
- Rich feature extraction through spectrogram analysis
- Access to state-of-the-art pre-trained models

## 📋 Requirements & Setup

### **Zero Setup Required!**
The notebooks automatically install everything needed:

```python
# Auto-installed in notebooks:
datasets[audio]==3.0.1    # HuggingFace datasets with audio support
mcap==1.2.1               # Sensor data format handling  
torch                     # PyTorch for ML
torchaudio               # Audio processing
transformers[torch]       # HuggingFace transformers
numpy                    # Numerical computing
```

### **Optional: External Integration**
For advanced features (not required for basic usage):

- `NSTRUMENTA_API_KEY` - Upload results to Nstrumenta platform
- `HF_TOKEN` - Access private HuggingFace models

### **Data Requirements: None!**
- ✅ **Training data:** Generated synthetically
- ✅ **Test data:** Generated synthetically  
- ✅ **Ground truth:** Perfect synthetic labels
- ✅ **Models:** Downloaded automatically from HuggingFace

## 🐛 Troubleshooting

### **Common Issues (Rare with Synthetic Data!)**

**"Module not found" errors:**
- ✅ The notebooks include automatic fallback imports
- ✅ Environment setup is automatic - just run all cells
- ✅ No manual installation required

**"Synthetic data generation failed":**
- ✅ Check Python environment has basic numpy/scipy
- ✅ Restart notebook kernel and try again
- ✅ All dependencies auto-install in first cell

**"Model loading failed":**
- ✅ Notebooks download models automatically
- ✅ Check internet connection for HuggingFace downloads
- ✅ Models are cached after first download

**"GPU not detected" (for training):**
- ✅ Training works on CPU (just slower)
- ✅ In Colab: Runtime → Change runtime type → GPU
- ✅ Enable debug mode for faster CPU training

### **🔧 Advanced Configurations**

**Customize Synthetic Data:**
```python
# In any notebook, modify these parameters:
duration_seconds = 10        # Length of generated data
distortion_level = "high"    # none/low/high
sample_rate = 100           # Hz
motion_pattern = "complex"   # simple/complex/stationary
```

**Enable Debug Mode (Fine-Tuning):**
```python
DEBUG_MODE = True  # Faster training for testing
# - Fewer training steps (50 vs 500)
# - Smaller datasets (100 vs 1000 samples)  
# - Separate debug directories
```

**Optional External Integration:**
```python
# For uploading results to Nstrumenta platform:
import os
os.environ['NSTRUMENTA_API_KEY'] = 'your_key_here'
```

## 🎓 Educational Value

### **Perfect for Learning ML/AI:**
- 🔬 **Controlled Experiments:** Perfect ground truth for validation
- 📊 **Immediate Results:** No waiting for data downloads
- 🎯 **Clear Pipeline:** See every step from data to prediction
- 🔄 **Reproducible:** Same synthetic data every time

### **Key Learning Outcomes:**
1. **Time-Series Classification:** How sensor data becomes ML predictions
2. **Transfer Learning:** Using audio models for sensor applications  
3. **Data Pipeline:** Complete ML workflow from raw data to results
4. **Model Evaluation:** Understanding accuracy, confusion matrices, etc.

### **Research Applications:**
- Test new classification algorithms
- Prototype sensor fusion techniques
- Validate model robustness  
- Generate training data for edge cases

## 🔄 What's Different from Traditional Approaches

### **❌ Traditional ML Notebooks:**
- Require external datasets
- Complex setup and dependencies  
- Unclear data provenance
- Hard to reproduce results
- Break when data links fail

### **✅ Our Synthetic Data Approach:**
- 🎯 **Zero Dependencies:** Generate perfect training data
- ⚡ **Instant Setup:** Run immediately in any environment
- 🔬 **Perfect Ground Truth:** Know exactly what the model should predict
- 🔄 **Reproducible:** Same synthetic data every time
- 📚 **Educational:** Clear relationship between inputs and outputs

### **🎵 Audio Classification Innovation:**
Traditional sensor classification uses time-series techniques. We use:
- **Spectrograms:** Convert sensor data to images
- **Audio Models:** Leverage state-of-the-art AST (Audio Spectrogram Transformer)
- **Transfer Learning:** Apply audio expertise to sensor problems
- **Better Performance:** Audio models often outperform traditional approaches

## 📚 Additional Resources

### **Project Documentation:**
- 📖 **[Main README](../README.md)** - Complete project overview
- 🔧 **[Scripts Documentation](../scripts/README_script_utils.md)** - Command-line tools
- 🧪 **[Synthetic Data Guide](../docs/synthetic_data.md)** - Deep dive into data generation

### **Code Examples:**
- 💻 **[Basic Example](../examples/basic_example.py)** - Simple classification script  
- 📊 **[Data Generation](../scripts/synthetic_data.py)** - Standalone synthetic data creation
- 🧪 **[Unit Tests](../tests/)** - Code validation and examples

### **Community & Support:**
- 🐛 **Issues:** [GitHub Issues](https://github.com/nstrumenta/time-series-classifier/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/nstrumenta/time-series-classifier/discussions)
- 📧 **Contact:** Open an issue for questions

## 🎯 Next Steps

### **🚀 Ready to Start:**
1. **Try [`classify.ipynb`](classify.ipynb)** for instant gratification
2. **Explore [`dataset.ipynb`](dataset.ipynb)** to understand the data
3. **Train with [`fine_tune.ipynb`](fine_tune.ipynb)** for custom models

### **🔬 For Researchers:**
- Experiment with different synthetic data parameters
- Try new motion patterns and distortion scenarios  
- Compare with traditional time-series approaches
- Adapt techniques to your own sensor classification problems

### **🏭 For Production:**
- Use synthetic data to test model robustness
- Generate edge case training data
- Validate models before deploying on real sensor data
- Create comprehensive test suites

---

**🎉 Happy Learning!** These notebooks represent a new paradigm in ML education - perfect synthetic data that works everywhere, every time!
2. **Use debug mode** for initial testing
3. **Switch to synthetic data** if available
4. **Report any issues** or suggestions

The updated notebooks provide a robust, user-friendly experience for both new users and those migrating from the previous version!
