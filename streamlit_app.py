import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Page configuration
st.set_page_config(
    page_title="Sales Profit Predictor",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .card { background-color: white; border-radius: 0.75rem; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); margin-bottom: 1.5rem; }
    .header { font-size: 2.5rem; font-weight: 700; color: #1e293b; text-align: center; margin-bottom: 0.5rem; }
    .subheader { font-size: 1.25rem; color: #64748b; text-align: center; margin-bottom: 2rem; }
    .stNumberInput, .stSelectbox { margin-bottom: 0.75rem; }
    .stButton>button { background-color: #3b82f6; color: white; border-radius: 0.5rem; padding: 0.5rem 1.5rem; font-weight: 600; border: none; width: 100%; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #2563eb; transform: translateY(-2px); box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
    .result-card { background-color: white; border-radius: 0.75rem; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center; border-left: 4px solid #10b981; }
    .result-value { font-size: 2rem; font-weight: 700; color: #10b981; margin: 1rem 0; }
    .result-label { font-size: 1rem; color: #64748b; }
    .metric-row { display: flex; justify-content: space-between; gap: 1rem; margin-bottom: 1rem; }
    .metric-card { background-color: #f1f5f9; border-radius: 0.5rem; padding: 1rem; text-align: center; flex: 1; }
    .metric-value { font-size: 1.25rem; font-weight: 600; color: #3b82f6; margin-bottom: 0.25rem; }
    .metric-label { font-size: 0.875rem; color: #64748b; }
    .footer { text-align: center; margin-top: 2rem; font-size: 0.875rem; color: #64748b; }
</style>
""", unsafe_allow_html=True)

# Cache the data loading function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/train_clean.csv")
        df["Ship Mode"] = df["Ship Mode"].fillna(df["Ship Mode"].mode()[0]).astype(str)
        df = df[df["Ship Mode"].notnull()]
        return df
    except FileNotFoundError:
        st.error("Data file not found! Please ensure data/train_clean.csv exists.")
        return pd.DataFrame()

# Load the data
df = load_data()

# Load the model
model_path = "./src/best_rf_model.pkl"
model_loaded = False
if not os.path.exists(model_path):
    st.error("⚠️ Model file not found! Please train the model using src/model.py")
else:
    with st.spinner("Loading model..."):
        pipeline = joblib.load(model_path)
        model_loaded = True

# Debug: Print column names
if not df.empty:
    st.write("Columns in df:", df.columns.tolist())

# Define feature names and types
feature_names = [
    'Sales', 'Unit Price (log)', 'Shipping Cost (log)', 'Product Base Margin',
    'Discount', 'Order Priority', 'Product Category', 'Product Sub-Category',
    'Product Container', 'Customer Segment', 'Ship Mode', 'Product Name',
    'Province', 'Region', 'Customer Name'
]

categorical_features = [
    'Order Priority', 'Product Category', 'Product Sub-Category',
    'Product Container', 'Customer Segment', 'Ship Mode', 'Product Name',
    'Province', 'Region', 'Customer Name'
]

numerical_features = [
    'Sales', 'Unit Price (log)', 'Shipping Cost (log)', 'Product Base Margin',
    'Discount'
]

# Initialize LIME explainer
lime_explainer = None
preprocessor = None

# Create LIME explainer if data is available
if not df.empty and model_loaded:
    # Extract the preprocessor from the pipeline
    try:
        # Try to get preprocessor from pipeline
        if hasattr(pipeline, 'named_steps') and 'preprocessor' in pipeline.named_steps:
            preprocessor = pipeline.named_steps['preprocessor']
        else:
            # If not found, create a new one (same as in the model training)
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
                ]
            )
            
        # Sample data for training the explainer
        X_train = df[feature_names].copy()
        
        # Fill missing values
        X_train[numerical_features] = X_train[numerical_features].fillna(X_train[numerical_features].median())
        for col in categorical_features:
            X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
            
        # Fit preprocessor if it's a new one
        if 'preprocessor' not in pipeline.named_steps:
            preprocessor.fit(X_train)
        
        # Create mapping dictionaries for each categorical feature
        cat_mappings = {}
        categorical_names = {}
        
        for feature in categorical_features:
            # Get unique values
            unique_values = sorted(X_train[feature].unique())
            # Create mapping
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            cat_mappings[feature] = mapping
            # Create reverse mapping for display
            categorical_names[feature_names.index(feature)] = list(mapping.keys())
            # Convert to numerical in the dataframe
            X_train[feature] = X_train[feature].map(mapping)
        
        # Make sure all columns are numeric and handle NaN values
        for col in X_train.columns:
            if col in numerical_features:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
                X_train[col] = X_train[col].fillna(0)
        
        # Create a wrapper function for predictions that handles data conversion
        def predict_fn(x_to_predict):
            # Convert numpy array back to DataFrame with proper column names
            x_df = pd.DataFrame(x_to_predict, columns=feature_names)
            
            # Convert categorical columns back to string values for the pipeline
            for i, feature in enumerate(categorical_features):
                # Get the mapping for this feature
                mapping = cat_mappings[feature]
                # Get the reverse mapping (index to value)
                reverse_map = {idx: val for val, idx in mapping.items()}
                # Convert indices back to original string values
                if feature in x_df.columns:
                    x_df[feature] = x_df[feature].apply(
                        lambda x: reverse_map.get(int(x) if not np.isnan(x) else 0, list(reverse_map.values())[0])
                    )
            
            return pipeline.predict(x_df)
        
        # Convert training data to numpy array
        X_train_array = X_train.values.astype(float)
        
        # Create categorical feature indices
        categorical_indices = [feature_names.index(f) for f in categorical_features]
        
        # Create LIME explainer with training data
        lime_explainer = LimeTabularExplainer(
            training_data=X_train_array,
            feature_names=feature_names,
            categorical_features=categorical_indices,
            categorical_names=categorical_names,
            mode='regression',
            random_state=42
        )
        
    except Exception as e:
        st.error(f"Error setting up LIME explainer: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    except Exception as e:
        st.error(f"Error setting up LIME explainer: {str(e)}")

# App header
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="header">Sales Profit Predictor 📊</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Predict your sales profit with our machine learning model</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["Prediction", "About"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Enter Sales Details")
        
        with st.form("profit_form"):
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                order_quantity = st.number_input("Order Quantity", min_value=1, value=10, step=1)
                sales = st.number_input("Sales ($)", min_value=0.0, value=1000.0, step=100.0)
                unit_price = st.number_input("Unit Price ($)", min_value=0.0, value=50.0, step=5.0)
                shipping_cost = st.number_input("Shipping Cost ($)", min_value=0.0, value=10.0, step=1.0)
                product_margin = st.number_input("Product Base Margin (%)", min_value=0.0, max_value=100.0, value=30.0, step=5.0)
                discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)

            with form_col2:
                if not df.empty:
                    product_category = st.selectbox("Product Category", sorted(df['Product Category'].dropna().astype(str).unique()))
                    product_subcategory = st.selectbox("Product Sub-Category", sorted(df['Product Sub-Category'].dropna().astype(str).unique()))
                    product_container = st.selectbox("Product Container", sorted(df['Product Container'].dropna().astype(str).unique()))
                    customer_segment = st.selectbox("Customer Segment", sorted(df['Customer Segment'].dropna().astype(str).unique()))
                    ship_mode = st.selectbox("Ship Mode", sorted(df['Ship Mode'].dropna().astype(str).unique()))
                    order_priority = st.selectbox("Order Priority", sorted(df['Order Priority'].dropna().astype(str).unique()))
                else:
                    st.error("Cannot load form fields because data is missing.")
                    product_category = product_subcategory = product_container = ""
                    customer_segment = ship_mode = order_priority = ""

            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("✨ Predict Profit")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Prediction Results")
        
        if submit_button and model_loaded and not df.empty:
            with st.spinner("Calculating prediction..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i == 99:
                        break
                
                # Prepare input data for prediction
                input_data = pd.DataFrame({
                    'Sales': [sales],
                    'Unit Price (log)': [np.log1p(unit_price)],
                    'Shipping Cost (log)': [np.log1p(shipping_cost)],
                    'Product Base Margin': [product_margin],
                    'Discount': [discount],
                    'Order Priority': [order_priority],
                    'Product Category': [product_category],
                    'Product Sub-Category': [product_subcategory],
                    'Product Container': [product_container],
                    'Customer Segment': [customer_segment],
                    'Ship Mode': [ship_mode],
                    'Product Name': [df['Product Name'].dropna().iloc[0]],
                    'Province': [df['Province'].dropna().iloc[0]],
                    'Region': [df['Region'].dropna().iloc[0]],
                    'Customer Name': [df['Customer Name'].dropna().iloc[0]]
                })
                
                # Make prediction using the pipeline
                prediction = pipeline.predict(input_data)[0]
                
                # Show the result
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<div class="result-label">Predicted Profit</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-value">${prediction:,.2f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show additional metrics
                st.markdown('<div class="metric-row">', unsafe_allow_html=True)
                profit_margin = (prediction / sales) * 100 if sales > 0 else 0
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{profit_margin:.1f}%</div>
                    <div class="metric-label">Profit Margin</div>
                </div>
                ''', unsafe_allow_html=True)
                profit_per_unit = prediction / order_quantity if order_quantity > 0 else 0
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">${profit_per_unit:.2f}</div>
                    <div class="metric-label">Profit per Unit</div>
                </div>
                ''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # LIME Explanation
                st.subheader("🔍 Model Explanation (LIME)")
                
                if lime_explainer is not None:
                    with st.spinner("Generating LIME explanation..."):
                        try:
                            # Convert input data for LIME
                            input_for_lime = input_data.copy()
                            
                            # Convert categorical features to numerical using the same mapping
                            for feature in categorical_features:
                                if feature in cat_mappings:
                                    mapping = cat_mappings[feature]
                                    # Convert to numerical
                                    input_for_lime[feature] = input_for_lime[feature].map(
                                        lambda x: mapping.get(x, 0)  # Use 0 as default
                                    )
                            
                            # Convert to numeric and ensure floats
                            for col in input_for_lime.columns:
                                input_for_lime[col] = pd.to_numeric(input_for_lime[col], errors='coerce')
                                input_for_lime[col] = input_for_lime[col].fillna(0)
                            
                            # Convert to numpy array with float type
                            input_array = input_for_lime.values[0].astype(float)
                            
                            # Generate LIME explanation
                            exp = lime_explainer.explain_instance(
                                data_row=input_array,
                                predict_fn=predict_fn,
                                num_features=10
                            )
                            
                            # Plot LIME explanation
                            fig = exp.as_pyplot_figure()
                            st.pyplot(fig)
                            
                            # Display LIME results as a table
                            lime_results = exp.as_list()
                            lime_df = pd.DataFrame(lime_results, columns=['Feature', 'Contribution'])
                            st.write("**Feature Contributions to Prediction**")
                            st.dataframe(lime_df.style.format({'Contribution': '{:.4f}'}))
                            
                        except Exception as e:
                            st.error(f"Error generating LIME explanation: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.warning("LIME explainer not available. Unable to generate explanation.")

                # Timestamp
                st.markdown(f'''
                <div style="text-align: center; font-size: 0.8rem; color: #64748b; margin-top: 1rem;">
                    Prediction made on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
                ''', unsafe_allow_html=True)
        else:
            if not submit_button:
                st.info("👈 Fill out the form and click 'Predict Profit' to see results")
            elif not model_loaded:
                st.error("⚠️ Model not loaded! Cannot make predictions.")
            elif df.empty:
                st.error("⚠️ Data not loaded! Cannot make predictions.")
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 About This App")
    
    about_col1, about_col2 = st.columns(2)
    
    with about_col1:
        st.markdown("""
        ### What is Sales Profit Predictor?
        This application uses machine learning to predict the profit from sales based on various input parameters. The model has been trained on historical sales data to identify patterns and relationships that affect profitability. Now with LIME for model interpretability!
        
        ### How to Use
        1. Enter your sales details in the form
        2. Click "Predict Profit" button
        3. View your predicted profit, metrics, and LIME explanations
        
        ### Features
        - Fast and accurate profit predictions
        - User-friendly interface
        - Detailed metrics and analysis
        - Model interpretability with LIME
        - Based on Random Forest machine learning model
        """)
    
    with about_col2:
        st.markdown("""
        ### Technology Stack
        - **Frontend**: Streamlit
        - **Backend**: Python
        - **Machine Learning**: Scikit-learn (Random Forest)
        - **Interpretability**: LIME
        - **Data Processing**: Pandas, NumPy
        
        ### Model Information
        The prediction model is a Random Forest regressor trained on historical sales data. The model takes into account various factors like product category, shipping costs, discounts, and more to predict the final profit. LIME provides insights into which features drive the predictions.
        
        ### Privacy Notice
        All data entered into this application is used only for making predictions and is not stored or shared with third parties.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100?text=Sales+Predictor", use_column_width=True)
    st.markdown("### 📌 Quick Navigation")
    st.markdown("* [Prediction](#prediction)")
    st.markdown("* [About](#about)")
    st.markdown("---")
    st.markdown("### 🛠️ Developer Info")
    st.markdown("Built by Mohammad Shafiee")
    st.markdown("[GitHub Repository](https://github.com/wikm360/sales-profit-predictor)")
    st.markdown("[Visit Website](https://wikm.ir)")
    st.markdown("---")
    st.markdown("### 💡 Pro Tips")
    tips = [
        "Higher product margins generally lead to better profits",
        "Standard shipping often balances cost and delivery time",
        "Corporate customer segments typically have larger order quantities",
        "Low-priority orders might save on shipping costs"
    ]
    tip = st.selectbox("Sales tips:", tips)
    st.markdown("---")
    if model_loaded:
        st.success("✅ Model loaded successfully")
    else:
        st.error("❌ Model not loaded")
    if not df.empty:
        st.success(f"✅ Data loaded: {len(df)} records")
    else:
        st.error("❌ Data not loaded")

# Footer
st.markdown("""
<div class="footer">
    Sales Profit Predictor v2.1 with LIME | Last updated: May 2025 | © Mohammad Shafiee
</div>
""", unsafe_allow_html=True)