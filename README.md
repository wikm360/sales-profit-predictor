# Sales Profit Predictor 📊

Welcome to the **Sales Profit Predictor**, a powerful machine learning project designed to forecast sales profit using advanced data analysis and predictive modeling. This project combines exploratory data analysis (EDA), robust feature engineering, a high-performance Random Forest model, and an interactive Streamlit dashboard with model interpretability powered by LIME (Local Interpretable Model-agnostic Explanations).

Whether you're a business analyst, data scientist, or developer, this tool empowers you to predict profits, understand key drivers of profitability, and make data-driven decisions with an intuitive user interface.

---

## ✨ Features

- **Accurate Profit Predictions**: Leverage a Random Forest regressor trained on historical sales data to predict profits with high precision.
- **Interactive Dashboard**: A user-friendly Streamlit interface to input sales details, view predictions, and explore additional metrics like profit margin and profit per unit.
- **Model Interpretability**: Gain insights into model predictions with LIME, visualizing the contribution of each feature to the predicted profit.
- **Robust Preprocessing**: Includes log-transformed features, outlier removal, and KNN imputation for handling missing values.
- **Comprehensive EDA**: Jupyter notebook for in-depth exploratory data analysis to understand data patterns and relationships.
- **Scalable and Modular Code**: Well-organized project structure with separate modules for data preprocessing, model training, and dashboard.

---

## 📈 Dataset

The dataset used in this project is stored in `data/data.csv` and contains historical sales data with the following details:

- **Features**:
  - Numerical: `Sales`, `Unit Price (log)`, `Shipping Cost (log)`, `Product Base Margin`, `Discount`, `Order Quantity`
  - Categorical: `Order Priority`, `Product Category`, `Product Sub-Category`, `Product Container`, `Product Name`, `Province`, `Region`, `Customer Segment`, `Ship Mode`, `Customer Name`
- **Target**: `Profit` (in USD)
- **Preprocessing**:
  - Log transformation applied to `Unit Price` and `Shipping Cost` to handle skewness.
  - Outliers removed using IQR-based filtering.
  - Missing values imputed using KNN imputation.
  - Categorical features encoded with `OneHotEncoder`.

---

## 🗂 Project Structure

```
sales-profit-predictor/
│
├── data/
│   └── data.csv                # Raw dataset
├── notebooks/
│   └── EDA.ipynb               # Jupyter notebook for exploratory data analysis
├── src/
│   └── model.py                # Model training and preprocessing script
├── streamlit_app.py            # Interactive Streamlit dashboard
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore file
```

---

## 🚀 Installation

Follow these steps to set up the project locally:

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/wikm360/sales-profit-predictor.git
   cd sales-profit-predictor
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   Ensure all dependencies (e.g., `pandas`, `scikit-learn`, `streamlit`, `lime`) are installed correctly by running:
   ```bash
   python -m streamlit --version
   ```

---

## 🛠 Usage

### 1. Train the Model
To train the Random Forest model and save it as `src/best_rf_model.pkl`:
```bash
python src/model.py
```

This script:
- Loads and preprocesses the dataset (`data/data.csv`).
- Applies feature engineering (log transformation, outlier removal, KNN imputation).
- Trains a Random Forest regressor with cross-validation.
- Saves the trained model.

### 2. Run the Streamlit Dashboard
To launch the interactive dashboard:
```bash
streamlit run streamlit_app.py
```

The dashboard will open in your default web browser. You can:
- Enter sales details (e.g., sales amount, unit price, shipping cost, etc.) in the form.
- Click **Predict Profit** to view the predicted profit, profit margin, and profit per unit.
- Explore **LIME explanations** to understand which features drive the prediction.

### 3. Explore the Data
Open `notebooks/EDA.ipynb` in Jupyter Notebook to perform exploratory data analysis:
```bash
jupyter notebook notebooks/EDA.ipynb
```

This notebook includes visualizations and statistical analyses to uncover insights from the dataset.

---

## 🧠 Model Details

- **Algorithm**: Random Forest Regressor
- **Hyperparameters**:
  - `n_estimators=200`
  - `max_depth=20`
  - `min_samples_split=2`
  - `min_samples_leaf=1`
  - `max_features=0.5`
  - `random_state=42`
- **Preprocessing**:
  - Numerical features scaled with `StandardScaler`.
  - Categorical features encoded with `OneHotEncoder` (handles unknown categories).
  - Pipeline-based preprocessing for seamless training and inference.
- **Evaluation**:
  - Cross-validation R² scores to ensure model robustness.
  - Metrics on test set: Mean Squared Error (MSE), R-squared (R²), Mean Absolute Error (MAE).

---

## 📊 Streamlit Dashboard

The Streamlit dashboard (`streamlit_app.py`) offers a rich user experience with the following components:

- **Input Form**: Enter sales details across numerical (e.g., Sales, Discount) and categorical (e.g., Product Category, Ship Mode) features.
- **Prediction Results**: Displays predicted profit, profit margin, and profit per unit in a visually appealing format.
- **LIME Explanations**: Visualizes feature contributions to the prediction with a bar chart and detailed table.
- **Tabs**: Separate tabs for predictions and project information.
- **Sidebar**: Quick navigation, developer info, and pro tips for optimizing sales strategies.

---

## 📋 Requirements

Key dependencies listed in `requirements.txt`:
```
pandas
numpy
scikit-learn
streamlit
lime
matplotlib
joblib
```

To install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code follows the project's coding style and includes appropriate tests.

---

## 🐛 Issues and Support

If you encounter any issues or have questions, please:
- Check the [Issues](https://github.com/wikm360/sales-profit-predictor/issues) section for similar problems.
- Open a new issue with a detailed description of the problem and steps to reproduce it.

For additional support, contact the developer at [wikm.ir](https://wikm.ir).

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Developer

**Mohammad Shafiee**
- GitHub: [wikm360](https://github.com/wikm360)
- Website: [wikm.ir](https://wikm.ir)

---

## 🌟 Acknowledgments

- Inspired by real-world sales forecasting challenges.
- Built with powerful open-source libraries: `scikit-learn`, `Streamlit`, `LIME`, and more.
- Special thanks to the data science community for continuous inspiration and support.

---

**Sales Profit Predictor v2.1** | Last updated: May 2025

*Predict smarter, profit better!* 🚀