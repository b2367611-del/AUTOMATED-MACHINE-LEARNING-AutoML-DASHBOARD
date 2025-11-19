# AutoML Platform

A comprehensive Automated Machine Learning platform built with FastAPI that allows users to upload datasets, perform automated data analysis, and train multiple machine learning models with minimal configuration.

## Features

### 🚀 **Complete AutoML Workflow**
1. **Data Upload & Validation** - Support for CSV and Excel files
2. **Automated Data Analysis** - Column type inference, missing data analysis
3. **Exploratory Data Analysis** - Automated visualizations and statistics
4. **Model Training** - Multiple algorithms with cross-validation
5. **Model Evaluation** - Comprehensive metrics and comparisons
6. **Model Export** - Download trained models and reports

### 🤖 **Supported Algorithms**

**Classification:**
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier
- CatBoost Classifier
- Support Vector Machine (SVM)

**Regression:**
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor
- Support Vector Regression (SVR)

### 📊 **Automated Features**
- **Data Preprocessing**: Automatic handling of missing values, feature scaling, and categorical encoding
- **Data Quality Reports**: Comprehensive quality analysis with missing data %, outlier detection, and preprocessing transparency
- **Target Detection**: Smart suggestions for target columns
- **Problem Type Detection**: Automatic classification vs regression detection
- **Feature Importance**: Visual feature importance analysis
- **Cross-Validation**: Robust model evaluation with k-fold CV
- **Interactive Visualizations**: Plotly-powered charts and graphs
- **Advanced Preprocessing**: Outlier detection, feature engineering, and smart imputation strategies

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git (optional)

### Step 1: Clone or Download
```bash
git clone <repository-url>
cd AutoML
```

### Step 2: Create Virtual Environment
```bash
python -m venv automl_env
automl_env\Scripts\activate  # Windows
# or
source automl_env/bin/activate  # macOS/Linux
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python main.py
```

The application will be available at: **http://localhost:8000**

## Usage Guide

### 1. Upload Dataset
- Navigate to http://localhost:8000
- Drag and drop or click to upload your CSV/Excel file
- Supported formats: `.csv`, `.xlsx`, `.xls`
- Maximum file size: 50MB

### 2. Data Overview
- Review dataset information (shape, columns, data types)
- Check missing values and column types
- View target column suggestions
- **Generate Data Quality Report** - Get comprehensive quality analysis with missing data %, outlier counts, and preprocessing transparency

### 3. Generate EDA (Optional)
- Click "Generate Detailed Analysis"
- View automated visualizations and statistics
- Understand data distributions and correlations

### 4. Configure Training
- **Target Column**: Select the column you want to predict
- **Problem Type**: Choose or let the system auto-detect
- **Test Size**: Set the train/test split ratio (10-50%)
- **CV Folds**: Choose cross-validation folds (3, 5, or 10)

### 5. Train Models
- Click "Start Training"
- Multiple models are trained automatically
- Cross-validation ensures robust evaluation
- Training time varies based on dataset size

### 6. View Results
- **Model Comparison**: See all models ranked by performance
- **Best Model**: Automatically highlighted with crown icon
- **Metrics**: View accuracy, precision, recall, F1-score (classification) or R², MSE, MAE (regression)
- **Feature Importance**: Understand which features matter most

### 7. Download & Export
- **Download Model**: Get a complete model package (.zip)
- **Download Report**: Get detailed training report (.json)
- Models include preprocessing pipelines and usage instructions

## API Endpoints

### Upload
- `POST /api/upload/` - Upload and analyze dataset
- `GET /api/upload/info/{upload_id}` - Get dataset information

### Quality Reports
- `GET /api/quality/report/{upload_id}` - Get HTML quality report
- `GET /api/quality/report/{upload_id}/json` - Get JSON quality report

### Training
- `POST /api/train/eda/{upload_id}` - Generate EDA
- `POST /api/train/start` - Start model training
- `GET /api/train/results/{training_id}` - Get training results

### Evaluation
- `GET /api/evaluate/metrics/{training_id}` - Get model metrics
- `GET /api/evaluate/feature-importance/{training_id}` - Get feature importance
- `POST /api/evaluate/predict/{training_id}` - Make predictions

### Download
- `GET /api/download/model/{training_id}` - Download model package
- `GET /api/download/report/{training_id}` - Download training report
- `GET /api/download/models` - List all trained models

## Project Structure

```
AutoML/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── app/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py      # Application configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py     # Pydantic data models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── upload.py      # File upload endpoints
│   │   ├── train.py       # Training endpoints
│   │   ├── evaluate.py    # Evaluation endpoints
│   │   ├── download.py    # Download endpoints
│   │   └── quality.py     # Data quality endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py           # Data upload & analysis
│   │   ├── data_preprocessing.py       # EDA & preprocessing
│   │   ├── data_quality_reporter.py    # Quality analysis
│   │   └── ml_training.py              # Model training
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── app.js     # Frontend JavaScript
│   └── templates/
│       ├── index.html                 # Main UI template
│       └── data_quality_report.html   # Quality report template
├── uploads/               # Uploaded datasets (auto-created)
└── trained_models/        # Saved models (auto-created)
```

## Configuration

### Environment Variables
Create a `.env` file in the root directory:

```env
# Application settings
DEBUG=True
MAX_FILE_SIZE=52428800  # 50MB in bytes

# ML settings
TEST_SIZE=0.2
CV_FOLDS=5
RANDOM_STATE=42
```

### Default Settings
- **Max File Size**: 50MB
- **Allowed Extensions**: .csv, .xlsx, .xls
- **Test Size**: 20% of data for testing
- **Cross-Validation**: 5-fold CV
- **Models Trained**: Up to 6 algorithms per training session

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Port Already in Use**
   ```bash
   python main.py  # Will automatically find available port
   # or specify custom port
   uvicorn main:app --host 127.0.0.1 --port 8001
   ```

3. **Memory Issues with Large Datasets**
   - Reduce dataset size or use sampling
   - Increase system RAM
   - Use fewer models in training

4. **Slow Training**
   - Large datasets take longer
   - Reduce CV folds
   - Use fewer models
   - Consider cloud deployment

### Performance Tips
- **Datasets < 10MB**: Fastest performance
- **Datasets 10-50MB**: May take 1-5 minutes for training
- **Memory Usage**: ~2-4x dataset size during processing
- **Optimal Features**: 5-50 features for best performance

## Advanced Usage

### Programmatic API Usage
```python
import requests
import pandas as pd

# Upload dataset
with open('data.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/api/upload/', files={'file': f})
    upload_info = response.json()

# Start training
training_config = {
    "upload_id": upload_info["upload_id"],
    "target_column": "target",
    "problem_type": "classification",
    "test_size": 0.2,
    "cv_folds": 5
}
response = requests.post('http://localhost:8000/api/train/start', json=training_config)
results = response.json()

print(f"Best model: {results['best_model']}")
```

### Model Deployment
The downloaded model package includes everything needed for deployment:
- Trained model (joblib)
- Preprocessing pipeline
- Target encoder (if applicable)
- Usage documentation
- Metadata with performance metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For questions, issues, or feature requests:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation

---

**Made with ❤️ using FastAPI, scikit-learn, and modern web technologies**