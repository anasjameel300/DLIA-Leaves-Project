# Medicinal Plant Leaf Classification Pipeline

A comprehensive deep learning pipeline for classifying 20 different medicinal plant leaves using four state-of-the-art CNN architectures: VGG19, ResNet50, DenseNet121, and MobileNet.

## 🌿 Overview

This project implements a complete machine learning pipeline for medicinal plant leaf classification:

1. **Background Removal**: Preprocesses images by removing backgrounds using U2Net
2. **Model Training**: Trains four different CNN architectures using transfer learning
3. **Model Comparison**: Comprehensive analysis and comparison of all models
4. **Results Visualization**: Detailed plots and reports for model performance

## 📁 Project Structure

```
medicinal-plant-classification/
├── medicinal plant leaf set/          # Original dataset (20 plant categories)
├── removed background/                # Preprocessed images (background removed)
├── background_removal_all.py         # Background removal script
├── train_vgg19.py                    # VGG19 training script
├── train_resnet50.py                 # ResNet50 training script
├── train_densenet121.py              # DenseNet121 training script
├── train_mobilenet.py                # MobileNet training script
├── model_comparison.py               # Model comparison and analysis
├── run_all_training.py               # Master script to run all training
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset (Background Removal)

```bash
python background_removal_all.py
```

This will:
- Process all images in the `medicinal plant leaf set/` folder
- Remove backgrounds using GPU-accelerated U2Net
- Save processed images to `removed background/` folder

### 3. Run Complete Training Pipeline

```bash
python run_all_training.py
```

This will:
- Train all four models sequentially (VGG19 → ResNet50 → DenseNet121 → MobileNet)
- Save individual model results
- Generate comprehensive comparison analysis

## 🎯 Individual Model Training

You can also train models individually:

```bash
# Train VGG19
python train_vgg19.py

# Train ResNet50
python train_resnet50.py

# Train DenseNet121
python train_densenet121.py

# Train MobileNet
python train_mobilenet.py
```

## 📊 Model Comparison

After training all models, run the comparison analysis:

```bash
python model_comparison.py
```

This generates:
- **model_comparison_summary.csv**: Detailed metrics comparison
- **comparison_report.txt**: Comprehensive analysis report
- **Visualization plots**: Accuracy, loss, confusion matrices, etc.

## 🌱 Plant Categories

The dataset includes 20 medicinal plant categories:

1. Aloevera
2. Amla
3. Bamboo
4. Bhrami
5. Bringaraja
6. Castor
7. Coffee
8. Coriender
9. Curry
10. Eucalyptus
11. Ginger
12. Guava
13. Henna
14. Hibiscus
15. Lemon
16. Mint
17. Neem
18. Onion
19. Palak(Spinach)
20. Papaya

## 🏗️ Model Architectures

### 1. VGG19
- **Parameters**: ~138M
- **Pros**: Excellent feature extraction, proven architecture
- **Cons**: Large model size, slower training
- **Best for**: High accuracy requirements

### 2. ResNet50
- **Parameters**: ~25M
- **Pros**: Excellent performance, residual connections
- **Cons**: Moderate size
- **Best for**: Production-ready applications

### 3. DenseNet121
- **Parameters**: ~8M
- **Pros**: Feature reuse, efficient parameter usage
- **Cons**: Memory intensive during training
- **Best for**: Efficient feature learning

### 4. MobileNet
- **Parameters**: ~4.2M
- **Pros**: Lightweight, fast training/inference
- **Cons**: Slightly lower accuracy
- **Best for**: Mobile/edge deployment

## ⚙️ Technical Specifications

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: 10GB+ free space

### Software Requirements
- Python 3.8+
- TensorFlow 2.10+
- CUDA 11.0+ (for GPU acceleration)

### Training Configuration
- **Image Size**: 224x224 pixels
- **Batch Size**: 16-64 (varies by model)
- **Epochs**: 50 (with early stopping)
- **Optimizer**: Adam (learning rate: 0.001)
- **Data Augmentation**: Rotation, shifts, zoom, flips

## 📈 Expected Results

Based on similar datasets, expected performance:

| Model | Expected Accuracy | Training Time | Model Size |
|-------|------------------|---------------|------------|
| VGG19 | 85-92% | 2-3 hours | ~550MB |
| ResNet50 | 88-94% | 1-2 hours | ~100MB |
| DenseNet121 | 87-93% | 1-2 hours | ~32MB |
| MobileNet | 82-89% | 30-60 min | ~17MB |

## 📁 Output Structure

After running the complete pipeline:

```
project/
├── vgg19_results/
│   ├── vgg19_final_model.h5
│   ├── vgg19_results.pkl
│   ├── vgg19_results.json
│   ├── vgg19_training_plots.png
│   └── vgg19_classification_report.csv
├── resnet50_results/
│   └── [similar structure]
├── densenet121_results/
│   └── [similar structure]
├── mobilenet_results/
│   └── [similar structure]
└── model_comparison_results/
    ├── model_comparison_summary.csv
    ├── comparison_report.txt
    ├── accuracy_loss_comparison.png
    ├── training_history_comparison.png
    └── confusion_matrices_comparison.png
```

## 🔧 Customization

### Modify Training Parameters

Edit the training scripts to customize:

```python
# In any training script
trainer = ModelTrainer(
    data_path="removed background",
    img_size=(224, 224),      # Change image size
    batch_size=32             # Adjust batch size
)

trainer.run_training_pipeline(epochs=50)  # Change epochs
```

### Add New Models

To add a new model architecture:

1. Create `train_newmodel.py` based on existing templates
2. Update `run_all_training.py` to include the new model
3. Update `model_comparison.py` to include the new model

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size in training scripts
   - Use smaller image size
   - Close other applications

2. **GPU Not Detected**
   - Install CUDA and cuDNN
   - Check TensorFlow GPU installation
   - Verify GPU drivers

3. **Training Too Slow**
   - Enable mixed precision training (already configured)
   - Use smaller models (MobileNet, DenseNet121)
   - Reduce image size

### Performance Optimization

- **GPU Memory**: Enable memory growth (already configured)
- **Mixed Precision**: Uses float16 for faster training
- **Data Loading**: Optimized with ImageDataGenerator
- **Early Stopping**: Prevents overfitting and saves time

## 📚 References

- [VGG19 Paper](https://arxiv.org/abs/1409.1556)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [DenseNet Paper](https://arxiv.org/abs/1608.06993)
- [MobileNet Paper](https://arxiv.org/abs/1704.04861)
- [U2Net Paper](https://arxiv.org/abs/2005.09007)

## 🤝 Contributing

Feel free to contribute by:
- Adding new model architectures
- Improving data augmentation
- Optimizing training parameters
- Adding new evaluation metrics

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- Dataset: Medicinal Plant Leaf Dataset
- Background Removal: U2Net implementation
- Deep Learning Framework: TensorFlow/Keras

---

**Note**: This pipeline is designed for research and educational purposes. For production use, consider additional optimizations like model quantization, pruning, and deployment-specific considerations.
