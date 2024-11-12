# 🎙️ Voice-Based Pronunciation Prediction

An advanced speech recognition and pronunciation assessment system implementing multiple state-of-the-art models: Quartznet, Wav2Vec2, and HuBERT. This project provides real-time voice input analysis and supports pre-recorded audio file processing.

## 🌟 Features

### Real-Time Voice Analysis (Quartznet)
- 🎤 Live voice recording and transcription
- 🔄 Real-time pronunciation feedback
- 📊 Interactive visualization of pronunciation accuracy
- 🗣️ Text-to-speech feedback using pyttsx3

### Pre-recorded Audio Analysis (Wav2Vec2 & HuBERT)
- 📁 Support for WAV file processing
- 🎯 High-accuracy speech recognition
- 📈 Comprehensive performance metrics
- 🖼️ Visual representation of results

## 🛠️ Technologies Used

- **Speech Recognition Models**:
  - Quartznet (NVIDIA NeMo)
  - Wav2Vec2 (Facebook AI)
  - HuBERT (Facebook AI)
- **Audio Processing**:
  - sounddevice
  - torchaudio
- **Visualization**:
  - matplotlib
  - seaborn
- **Metrics**:
  - jiwer
  - scikit-learn

## 📊 Performance Metrics

The system evaluates pronunciation using multiple metrics:
- Word Error Rate (WER)
- Match Error Rate (MER)
- Word Information Lost Rate (WIL)
- Accuracy
- Precision
- Recall
- F1 Score

## 💻 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Real-time Voice Analysis (Quartznet)
```python
from quartznet_analysis import main
main()
```

### Pre-recorded Audio Analysis (Wav2Vec2)
```python
from wav2vec_analysis import process_audio
process_audio("path/to/audio.wav")
```

### HuBERT Analysis
```python
from hubert_analysis import analyze_audio
analyze_audio("path/to/audio.wav")
```

## 📊 Visualization Features

1. **Pronunciation Visualization**
   - Color-coded word comparison
   - Green: Correctly pronounced words
   - Red: Mispronounced words

2. **Audio Waveform Overlay**
   - Time-aligned text visualization
   - Word-level accuracy indicators

3. **Performance Metrics Plot**
   - Bar charts for accuracy metrics
   - Comprehensive score visualization

## 🎯 Use Cases

1. **Language Learning**
   - Real-time pronunciation feedback
   - Detailed accuracy assessment
   - Progress tracking

2. **Speech Therapy**
   - Pronunciation monitoring
   - Progress visualization
   - Performance tracking

3. **Educational Assessment**
   - Automated pronunciation evaluation
   - Detailed performance metrics
   - Visual progress tracking

## 📈 Model Comparison

### Quartznet
- Best for: Real-time analysis
- Features: Live feedback, interactive visualization
- Use case: Immediate pronunciation assessment

### Wav2Vec2
- Best for: Accurate transcription
- Features: High-precision text conversion
- Use case: Detailed audio analysis

### HuBERT
- Best for: Complex speech patterns
- Features: Advanced acoustic modeling
- Use case: Research and detailed analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NVIDIA NeMo team for Quartznet
- Facebook AI Research for Wav2Vec2 and HuBERT
- The open-source community for various tools and libraries


