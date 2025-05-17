# Sales Profit Predictor

This project predicts sales profit using machine learning. It includes exploratory data analysis (EDA), feature engineering with log-transformed features and outlier removal, a RandomForest model, and an interactive Streamlit dashboard.

## Dataset
- **Source**: data.csv
- **Features**: Sales, Unit Price (log), Shipping Cost (log), Product Base Margin, Discount, Order Quantity, Order Priority, Product Category, Product Sub-Category, Product Container, Product Name, Province, Region, Customer Segment, Ship Mode, Customer Name
- **Target**: Profit (in USD)

## Project Structure
- `data/`: Raw datasets (`data.csv`)
- `notebooks/`: Jupyter notebook for EDA (`EDA.ipynb`)
- `src/`: Model training and preprocessing (`model.py`)
- `streamlit_app.py`: Interactive dashboard
- `requirements.txt`: Project dependencies

## Installation
```bash
pip install -r requirements.txt