# Neural Networks & Deep Learning: Multi-Model Fusion For Image Classification

A comprehensive image classification project implementing and comparing multiple state-of-the-art deep learning architectures for binary classification of Bottle and Cow objects from the PASCAL VOC 2012 dataset. This project explores both individual model performance and ensemble fusion techniques.

## 🎯 Project Overview

This project tackles binary image classification using modern neural network architectures and ensemble methods. We implemented transfer learning from ImageNet-pretrained models and evaluated performance using Mean Average Precision (MAP) metrics.

### Key Objectives
- Compare performance of different CNN and Vision Transformer architectures
- Implement early and late fusion ensemble techniques
- Evaluate models on PASCAL VOC 2012 dataset for Bottle and Cow classification
- Analyze architectural strengths and limitations

## 👥 Team Members

- **Huzaifa Bin Tariq** - 25133
- **Hazim Ghulam Farooq** - 25148  
- **Muhammad Wasay** - 24497

## 🏗️ Architecture Overview

### Individual Models Tested
1. **ResNet50** - Residual learning with skip connections
2. **VGG19 with BatchNorm** - Deep CNN with batch normalization
3. **Vision Transformer (ViT_B_16/ViT_L_16)** - Transformer-based image processing
4. **DenseNet121** - Dense connections between layers
5. **ResNeXt50_32x4d** - Enhanced ResNet with grouped convolutions
6. **EfficientNet-B0** - Parameter-efficient CNN

### Fusion Approaches
- **Early Fusion**: Feature-level combination using SVM and Neural Networks
- **Late Fusion**: Prediction-level combination (Average, Max, Min, Majority Voting)

## 📊 Performance Results

### Individual Model Performance (MAP Scores)

#### Bottle Classification
| Model | MAP Score |
|-------|-----------|
| **ResNet50** | **0.6507** |
| Vision Transformer (ViT_B_16) | 0.6079 |
| VGG19 with BatchNorm | 0.6070 |
| DenseNet121 | 0.5788 |
| ResNeXt50_32x4d | 0.5142 |
| EfficientNet-B0 | 0.3946 |

#### Cow Classification
| Model | MAP Score |
|-------|-----------|
| **ResNet50** | **0.8598** |
| Vision Transformer (ViT_B_16) | 0.7927 |
| Vision Transformer (ViT_L_16) | 0.7849 |
| DenseNet121 | 0.6219 |
| ResNeXt50_32x4d | 0.6091 |

### Selected Best Models
- **Bottle Class**: ResNet50, ViT_B_16, VGG19 with BatchNorm
- **Cow Class**: ResNet50, ViT_L_16, ViT_B_16

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch torchvision
pip install transformers
pip install scikit-learn
pip install numpy pandas matplotlib
pip install pillow
```

### Installation
```bash
git clone https://github.com/your-username/nndl-image-classification.git
cd nndl-image-classification
pip install -r requirements.txt
```

### Dataset Setup
```python
# Download PASCAL VOC 2012 dataset
from torchvision.datasets import VOCDetection

# Custom dataset wrappers included
from datasets import VOCBottleClassification, VOCCowClassification
```

### Usage

#### Individual Model Training
```python
from models import ResNet50Classifier, ViTClassifier, VGGClassifier

# Initialize model
model = ResNet50Classifier(num_classes=1, pretrained=True)

# Train model
trainer = ModelTrainer(model, train_loader, val_loader)
trainer.train(epochs=5)

# Evaluate
map_score = trainer.evaluate()
print(f"MAP Score: {map_score}")
```

#### Fusion Methods
```python
from fusion import EarlyFusion, LateFusion

# Early Fusion
early_fusion = EarlyFusion(models=[resnet50, vit, vgg19])
early_fusion.train(train_features, train_labels)

# Late Fusion
late_fusion = LateFusion(models=[resnet50, vit, vgg19])
predictions = late_fusion.predict(test_data, method='average')
```

## 📁 Project Structure

```
nndl-image-classification/
├── models/
│   ├── resnet.py              # ResNet implementation
│   ├── vit.py                 # Vision Transformer
│   ├── vgg.py                 # VGG with BatchNorm
│   ├── densenet.py            # DenseNet implementation
│   ├── resnext.py             # ResNeXt implementation
│   └── efficientnet.py       # EfficientNet implementation
├── fusion/
│   ├── early_fusion.py       # Feature-level fusion
│   └── late_fusion.py        # Prediction-level fusion
├── datasets/
│   ├── voc_bottle.py         # Bottle dataset wrapper
│   └── voc_cow.py            # Cow dataset wrapper
├── utils/
│   ├── metrics.py            # MAP calculation
│   ├── transforms.py         # Data augmentation
│   └── visualizations.py     # Result visualization
├── experiments/
│   ├── train_individual.py   # Individual model training
│   ├── train_fusion.py       # Fusion experiments
│   └── evaluate.py           # Model evaluation
├── results/
│   ├── individual_results.json
│   ├── fusion_results.json
│   └── top_predictions/      # Top-ranked images
├── notebooks/
│   └── analysis.ipynb        # Results analysis
├── requirements.txt
└── README.md
```

## 🔬 Methodology

### Data Preprocessing
- **Image Resizing**: 224×224 pixels
- **Normalization**: ImageNet statistics
- **Augmentation**: Random horizontal flips
- **Binary Labels**: 1 for presence, 0 for absence

### Training Configuration
- **Epochs**: 5 (limited for comparison)
- **Loss Function**: Binary Focal Loss (for class imbalance)
- **Optimization**: Transfer learning from ImageNet weights
- **Evaluation Metric**: Mean Average Precision (MAP)

### Fusion Techniques

#### Early Fusion
1. **Feature Extraction**: Penultimate layer features from each model
2. **Feature Concatenation**: Combined high-dimensional feature vector
3. **Classification**: 
   - RBF SVM with StandardScaler normalization
   - Neural Network (512 hidden units, ReLU, dropout)

#### Late Fusion
1. **Average Fusion**: Mean of prediction probabilities
2. **Maximum Fusion**: Maximum prediction value
3. **Minimum Fusion**: Minimum prediction value
4. **Majority Voting**: Threshold-based voting

## 📈 Key Findings

### Model Performance Insights
- **ResNet50** achieved the highest MAP scores for both classes (0.6507 for Bottle, 0.8598 for Cow)
- **Vision Transformers** performed competitively, especially on Cow classification
- **ViT models** excelled with large, centrally positioned objects but struggled with smaller/edge objects
- **Traditional CNNs** showed better robustness across varied image conditions

### Architecture Strengths
| Architecture | Strengths | Limitations |
|-------------|-----------|-------------|
| **ResNet50** | Robust across conditions, handles occlusions well | - |
| **Vision Transformers** | Excellent for large objects, captures long-range dependencies | Sensitive to object size/position |
| **VGG19** | Simple, reliable | Less effective than modern architectures |
| **DenseNet/ResNeXt** | Architectural innovations | Struggled with class imbalance |

### Fusion Results
- **Early Fusion Neural Network**: Highest sensitivity (268 cow detections, 1345 bottle detections)
- **Late Fusion**: More conservative predictions, better precision control
- **SVM Early Fusion**: Balanced approach for specific use cases

## ⚠️ Limitations

1. **Limited Training**: Only 5 epochs restricted Vision Transformer potential
2. **Class Imbalance**: Particularly affected Bottle classification
3. **Dataset Size**: Limited validation on personal test images
4. **Computational Constraints**: Prevented extensive hyperparameter tuning

## 🔮 Future Work

1. **Extended Training**: Longer fine-tuning for Vision Transformers
2. **Class Balancing**: Address dataset imbalance issues
3. **Ensemble Optimization**: Advanced fusion techniques
4. **Multi-class Extension**: Expand beyond binary classification
5. **Real-time Deployment**: Optimize for inference speed

## 📊 Fusion Performance Summary

### Detection Statistics (Threshold = 0.5)

#### Cow Class
- **Early Fusion SVM**: 1 detection
- **Early Fusion NN**: 268 detections
- **Late Fusion Average**: 0 detections
- **Late Fusion Max**: 1 detection
- **Late Fusion Min**: 0 detections
- **Majority Vote**: 0 detections

#### Bottle Class
- **Early Fusion SVM**: 60 detections
- **Early Fusion NN**: 1345 detections
- **Late Fusion Average**: 19 detections
- **Late Fusion Max**: 120 detections
- **Late Fusion Min**: 1 detection
- **Majority Vote**: 14 detections

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- PASCAL VOC Challenge organizers for the dataset
- PyTorch team for the deep learning framework
- Hugging Face for Vision Transformer implementations
- Our instructors for guidance throughout the project

## 📚 References

1. He, K., et al. "Deep residual learning for image recognition." CVPR 2016.
2. Dosovitskiy, A., et al. "An image is worth 16x16 words: Transformers for image recognition at scale." ICLR 2021.
3. Simonyan, K., & Zisserman, A. "Very deep convolutional networks for large-scale image recognition." ICLR 2015.
4. Huang, G., et al. "Densely connected convolutional networks." CVPR 2017.
5. Tan, M., & Le, Q. "EfficientNet: Rethinking model scaling for convolutional neural networks." ICML 2019.

---

*This project demonstrates the power of ensemble methods and transfer learning in computer vision, providing insights into when different architectures excel and how fusion techniques can be leveraged for improved performance.*
