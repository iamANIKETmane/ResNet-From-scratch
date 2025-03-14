# CIFAR-10 Image Classification with ResNet Architectures

## Project Overview
This repository contains a deep learning project focused on image classification using ResNet architectures on the CIFAR-10 dataset. The project explores various ResNet implementations with different depths and training strategies to optimize the trade-off between model complexity, training time, and classification accuracy.

## Dataset
The CIFAR-10 dataset consists of 60,000 32×32 color images across 10 classes:
- 50,000 training images
- 10,000 testing images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Model Architectures
We implemented and compared several ResNet variants:

### ResNet-34
- Parameters: 4,348,082
- Training: 200 epochs
- Performance:
  - Train Loss: 0.521 | Train Accuracy: 99.980%
  - Test Loss: 0.647 | Test Accuracy: 95.120%
  - Custom Test Dataset Accuracy: 82%

### ResNet-34 (100 epochs)
- Parameters: 4,348,082
- Training: 100 epochs
- Performance:
  - Train Loss: 0.521 | Train Accuracy: 99.980%
  - Test Loss: 0.647 | Test Accuracy: 91%
  - Custom Test Dataset Accuracy: 84%

### ResNet-26
- Parameters: 4,196,906
- Training: 100 epochs
- Performance:
  - Train Loss: 0.580 | Train Accuracy: 96.616%
  - Test Loss: 0.735 | Test Accuracy: 90.290%
  - Custom Test Dataset Accuracy: 79%

### ResNet-50 (Best Model)
- Parameters: 4,815,794
- Training: 300 epochs
- Performance:
  - Train Loss: 0.001 | Train Accuracy: 99.992%
  - Test Loss: 0.203 | Test Accuracy: 95.590%
  - Custom Test Dataset Accuracy: 85.095%

## Key Features
- **Residual Learning**: Implemented skip connections to mitigate vanishing gradient problems
- **Data Augmentation**: Applied random crops, horizontal flips, and other augmentation techniques
- **Optimized Channel Progression**: Customized channel configurations for better efficiency
- **Regularization**: Implemented dropout and weight decay to prevent overfitting
- **Learning Rate Scheduling**: Used cosine annealing to optimize convergence

## Training Methodology
- **Loss Function**: Cross-entropy loss
- **Optimizer**: SGD with Nesterov momentum (0.9)
- **Learning Rate**: Initial rate of 0.1 with scheduled decay
- **Batch Size**: 64
- **Regularization**: Weight decay (5×10^-4) and dropout

## Results Summary
Our experiments demonstrate that:
1. ResNet-50 achieves the best performance with 95.59% accuracy on the test set and 85.095% on a custom test dataset.
2. Increasing training epochs from 100 to 200/300 significantly improves model performance.
3. The trade-off between model complexity and accuracy is optimized with our ResNet-50 implementation.

## Experimentation Papers
Here are some key papers that contributed to optimizing the model and achieving good accuracies:

1. **Sharpness-Aware Minimization (SAM)**  
   - [SAM Optimizer](https://arxiv.org/pdf/2010.01412)
   - [GitHub SAM Implementation](https://github.com/davda54/sam)

2. **Squeeze-Excitation Networks (SENet)**  
   - [SENet Paper](https://arxiv.org/pdf/1709.01507)

3. **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**  
   - [EfficientNet Paper](https://arxiv.org/pdf/1805.09501)

4. **Deep Residual Learning for Image Recognition (ResNet Paper)**  
   - [ResNet Paper](https://arxiv.org/pdf/1610.02915)

5. **AutoAugment: Learning Augmentation Policies from Data**  
   - [AutoAugment Paper](https://arxiv.org/pdf/1811.09030)

## Usage
The repository contains Jupyter notebooks for model training and evaluation. To run the code:

1. Install the required dependencies:
```bash
pip install torch torchvision numpy matplotlib
```

2. Run the training script:
```bash
python train.py --model resnet50 --epochs 300
```

3. Evaluate a trained model:
```bash
python evaluate.py --model_path checkpoints/resnet50_best.pth
```

## Future Work
- Explore more advanced data augmentation techniques
- Implement knowledge distillation to create more compact models
- Test on more complex datasets like CIFAR-100 or Tiny ImageNet
- Experiment with alternative optimization strategies like SAM (Sharpness-Aware Minimization)

## References
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
- Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2019). AutoAugment: Learning Augmentation Policies from Data.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/32095591/6e9ebf26-8778-4d4d-8956-34f5812dac33/ResNetFinal.ipynb  
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/32095591/9633add1-5dcf-4ccb-b886-e0795a44b7f2/Deep_Learning_Mini_Project.pdf

