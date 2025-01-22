# Predictive Maintenance Analysis

## Overview
This project implements a comprehensive data analysis pipeline for predictive maintenance using various machine learning approaches. The analysis includes exploratory data analysis (EDA), data preprocessing, outlier detection, clustering analysis, predictive modeling for machine failure prediction, and an interactive web interface for real-time predictions.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Features](#features)
- [Analysis Pipeline](#analysis-pipeline)
- [Machine Learning Models](#machine-learning-models)
- [Clustering Analysis](#clustering-analysis)
- [Interactive Interface](#interactive-interface)
- [Usage](#usage)
- [Results](#results)

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- ydata-profiling
- yellowbrick
- ipywidgets
- imbalanced-learn (imblearn)
- gradio
- pickle

## Installation
Install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ydata-profiling yellowbrick ipywidgets imblearn gradio
```

## Dataset
The project uses the AI4I 2020 Predictive Maintenance Dataset with various failure types:
- Tool wear failure (TWF): Occurs between 200-240 mins
- Heat dissipation failure (HDF): When temperature difference < 8.6K and speed < 1380 rpm
- Power failure (PWF): Power outside 3500W-9000W range
- Overstrain failure (OSF): Product of tool wear and torque exceeds variant-specific threshold
- Random failures (RNF): 0.1% random failure chance

## Machine Learning Models
### Decision Tree Classifier
- Implemented with max_depth=8
- Saved model for deployment using pickle
- Used for the interactive interface

### K-Nearest Neighbors (KNN)
- Optimized using GridSearchCV
- Best parameters: n_neighbors=2
- Learning curve analysis

### Random Forest Classifier
- 100 estimators
- Bootstrap sampling
- Parallel processing enabled

### Gradient Boosting Classifier
- Default parameters
- Learning curve analysis
- Confusion matrix visualization

### Gaussian Naive Bayes
- Probabilistic classifier
- Performance evaluation with classification report
- Learning curve analysis

### Model Evaluation Metrics
All models evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Training time
- Prediction time
- Learning curves
- Confusion matrices

## Interactive Interface
Built using Gradio framework for real-time predictions:

### Features
- Slider inputs for:
  - Air temperature (100-350)
  - Process temperature (100-350)
- Numeric inputs for:
  - Rotational speed
  - Torque
  - Tool wear
- Radio button for Type (L/M/H)

### Outputs
- Failure probability predictions
- Maintenance action recommendation
- Top 2 most likely outcomes

### Model Deployment
```python
# Load the saved model
loaded_model = pickle.load(open('finalized_model.pkl', 'rb'))

# Launch the interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(100, 350, label="Air temperature"),
        gr.Slider(100, 350, label="Process temperature"),
        gr.Number(label="Rotational speed"),
        gr.Number(label="Torque"),
        gr.Number(label="Tool wear"),
        gr.Radio(["L", "M", "H"], label="Type")
    ],
    outputs=[
        gr.Label(num_top_classes=2, label="Result"), 
        gr.components.Textbox(label="Action")
    ]
)
demo.launch()
```

## Usage
1. Clone this repository:
```bash
git clone [repository-url]
```

2. Navigate to the project directory:
```bash
cd predictive-maintenance-analysis
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Run the Jupyter notebook for analysis:
```bash
jupyter notebook
```

5. Launch the prediction interface:
```bash
python app.py
```

## Results
- Successfully implemented and compared multiple machine learning models
- Handled class imbalance using SVMSMOTE oversampling
- Created interactive interface for real-time predictions
- Model comparison results stored in model_performance DataFrame
- Deployed best performing model using Gradio interface

### Model Performance Comparison
- Decision Tree: Baseline model with good interpretability
- KNN: Effective for local pattern recognition
- Random Forest: Robust ensemble performance
- Gradient Boosting: High accuracy with gradient optimization
- Gaussian Naive Bayes: Fast training and prediction times

## Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

