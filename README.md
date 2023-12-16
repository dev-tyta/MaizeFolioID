# MaizeFolioID

## Project Description

**MaizeFolioID** is a state-of-the-art image classification model specifically developed to identify and classify foliar diseases in maize leaves. Leveraging the advanced technology of pre-trained models from ImageNet, MaizeFolioID is a vital tool for early disease detection, aiming to mitigate crop losses in agriculture.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites and Dependencies](#prerequisites-and-dependencies)
3. [Installation](#installation)
4. [Model Description](#model-description)
5. [Training and Evaluation](#training-and-evaluation)
6. [Usage](#usage)
7. [Future Work](#future-work)
8. [Contributing](#contributing)
9. [Acknowledgments](#acknowledgments)
10. [License](#license)
11. [Contact Information](#contact-information)

### **1. Introduction**

MaizeFolioID combines the latest in deep learning and agricultural science to tackle the challenge of detecting foliar diseases in maize. This tool is instrumental in enabling early interventions and securing crop health.

### **2. Prerequisites and Dependencies**

#### **2.1. Prerequisites**
- Deep understanding of deep learning principles and familiarity with ImageNet's pre-trained models.
- Proficiency in Python for effective programming.

#### **2.2 Dependencies**
- **Deep Learning Libraries**: Utilization of ImageNet pre-trained models.
- **Data Handling**: Employment of Pandas and NumPy for data operations.
- **Image Processing**: Use of specialized tools for image data processing.

### **3. Installation**

Begin using MaizeFolioID by following these steps:

```bash
# Clone the repository
git clone https://github.com/dev-tyta/MaizeFolioID.git

# Navigate to the project directory
cd MaizeFolioID

# Install the required dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### **4. Data Description**

The dataset fueling this endeavor was sourced from Kaggle, shedding light on maize leaf conditions.

- **Dataset Link**: [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)

The dataset is structured with images representing various foliar diseases in maize leaves. Make sure to explore the dataset for a comprehensive understanding.

### **5. Model Description**

MaizeFolioID utilizes an ImageNet pre-trained model, fine-tuned to classify specific maize leaf conditions. We've made meticulous modifications to cater to our classification needs, ensuring the model's sensitivity towards the intricacies of maize leaf diseases.

### **6. Training and Evaluation**

While the model is pre-trained on ImageNet, it has been further trained on the provided dataset for specialized recognition. The evaluation metrics and results will be updated after subsequent retraining sessions, potentially featuring model performance visualizations.

### **7. Usage**

To experience the model in action:

1. Visit the Streamlit deployment: [MaizeFolioID Streamlit App](https://maizefolioid-h.streamlit.app/)
2. Alternatively, access the saved model via [Huggingface Model Hub](https://huggingface.co/Testys/MaizeFolioID).
3. Upload a maize leaf image for immediate disease analysis.


### **8. Contribution Guidelines**

We wholeheartedly welcome contributions!

1. Fork the repository.
2. Make your proposed changes.
3. Submit a pull request.

Your insights could aid in refining the model or introducing new features!

### **9. Future Work & Roadmap**

1. **Model Evolution**: We're planning to harness the capabilities of a VisionTransformer model on our dataset, driving accuracy improvements.
2. **Data Augmentation**: Efforts are in the pipeline to gather more datasets for each class, enriching the model's training foundation.

### **10. Acknowledgments**

A heartfelt thanks to Kaggle and all data providers for their invaluable datasets.

### **11. License**

MaizeFolioID is open-source and available under [License Type]. For more details, see the LICENSE file in the repository.

### **12. Contact Information**

For queries or collaboration intentions, kindly reach out through [GitHub](https://github.com/dev-tyta) or [Email](mailto:testimonyadekoya.02@gmail.com).

---
