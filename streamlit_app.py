import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Page configuration with improved title and layout
st.set_page_config(
    page_title="Sales Profit Predictor",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern, professional look
st.markdown("""
<style>
    /* Main theme colors */
    
    /* Main layout */
    .main {
        background-color: var(--background);
        padding: 1rem;
    }
    
    /* Card styling */
    .card {
        background-color: var(--card-bg);
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
    }
    
    /* Header styling */
    .header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text);
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subheader {
        font-size: 1.25rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Form controls */
    .stNumberInput, .stSelectbox {
        margin-bottom: 0.75rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--primary-dark);
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Results styling */
    .result-card {
        background-color: var(--card-bg);
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 4px solid var(--secondary);
    }
    
    .result-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--secondary);
        margin: 1rem 0;
    }
    
    .result-label {
        font-size: 1rem;
        color: #64748b;
    }
    
    /* Metric cards */
    .metric-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: var(--light-gray);
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        flex: 1;
    }
    
    .metric-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--light-gray);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.875rem;
        color: #64748b;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        margin-left: 0.5rem;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading {
        animation: pulse 1.5s infinite;
        background-color: var(--light-gray);
        border-radius: 0.5rem;
        height: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache the data loading function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/data.csv")
        df["Ship Mode"] = df["Ship Mode"].fillna(df["Ship Mode"].mode()[0]).astype(str)
        df = df[df["Ship Mode"].notnull()]
        return df
    except FileNotFoundError:
        st.error("Data file not found! Please ensure data/data.csv exists.")
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
        model = joblib.load(model_path)
        model_loaded = True

# App header
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="header">Sales Profit Predictor 📊</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Predict your sales profit with our machine learning model</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2 = st.tabs(["Prediction", "About"])

with tab1:
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📝 Enter Sales Details")
        
        with st.form("profit_form"):
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                order_quantity = st.number_input("Order Quantity", 
                                                min_value=1, 
                                                value=10, 
                                                step=1,
                                                help="Number of items ordered")
                
                sales = st.number_input("Sales ($)", 
                                        min_value=0.0, 
                                        value=1000.0, 
                                        step=100.0,
                                        help="Total sales amount before expenses")
                
                unit_price = st.number_input("Unit Price ($)", 
                                           min_value=0.0, 
                                           value=50.0, 
                                           step=5.0,
                                           help="Price per individual unit")
                
                shipping_cost = st.number_input("Shipping Cost ($)", 
                                              min_value=0.0, 
                                              value=10.0, 
                                              step=1.0,
                                              help="Cost to ship the order")
                
                product_margin = st.number_input("Product Base Margin (%)", 
                                               min_value=0.0, 
                                               max_value=100.0, 
                                               value=30.0, 
                                               step=5.0,
                                               help="Base profit margin percentage")
                
                discount = st.number_input("Discount (%)", 
                                         min_value=0.0, 
                                         max_value=100.0, 
                                         value=5.0, 
                                         step=1.0,
                                         help="Discount percentage offered")

            with form_col2:
                if not df.empty:
                    product_category = st.selectbox("Product Category", 
                                                  sorted(df['Product Category'].dropna().astype(str).unique()),
                                                  help="Main category of the product")
                    
                    product_subcategory = st.selectbox("Product Sub-Category", 
                                                     sorted(df['Product Sub-Category'].dropna().astype(str).unique()),
                                                     help="Specific subcategory of the product")
                    
                    product_container = st.selectbox("Product Container", 
                                                   sorted(df['Product Container'].dropna().astype(str).unique()),
                                                   help="Container type used for the product")
                    
                    customer_segment = st.selectbox("Customer Segment", 
                                                  sorted(df['Customer Segment'].dropna().astype(str).unique()),
                                                  help="Market segment of the customer")
                    
                    ship_mode = st.selectbox("Ship Mode", 
                                           sorted(df['Ship Mode'].dropna().astype(str).unique()),
                                           help="Shipping method used")
                    
                    order_priority = st.selectbox("Order Priority", 
                                                sorted(df['Order Priority'].dropna().astype(str).unique()),
                                                help="Priority level assigned to the order")
                else:
                    st.error("Cannot load form fields because data is missing.")
                    product_category = product_subcategory = product_container = ""
                    customer_segment = ship_mode = order_priority = ""

            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("✨ Predict Profit")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Results display section
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Prediction Results")
        
        if submit_button and model_loaded and not df.empty:
            with st.spinner("Calculating prediction..."):
                # Create a progress animation
                progress_bar = st.progress(0)
                for i in range(100):
                    # Update progress bar
                    progress_bar.progress(i + 1)
                    if i == 99:
                        break
                
                # Prepare input data for prediction
                input_data = pd.DataFrame({
                    'Sales': [sales],
                    'Unit Price (log)': [np.log1p(unit_price)],
                    'Shipping Cost (log)': [np.log1p(shipping_cost)],
                    'Product Base Margin': [product_margin],
                    'Discount_knn': [discount],
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
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Show the result
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<div class="result-label">Predicted Profit</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-value">${prediction:,.2f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show some additional metrics
                st.markdown('<div class="metric-row">', unsafe_allow_html=True)
                
                # Profit margin calculation
                profit_margin = (prediction / sales) * 100 if sales > 0 else 0
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">{profit_margin:.1f}%</div>
                    <div class="metric-label">Profit Margin</div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Profit per unit calculation
                profit_per_unit = prediction / order_quantity if order_quantity > 0 else 0
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value">${profit_per_unit:.2f}</div>
                    <div class="metric-label">Profit per Unit</div>
                </div>
                ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
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
    # About section with information cards
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 About This App")
    
    about_col1, about_col2 = st.columns(2)
    
    with about_col1:
        st.markdown("""
        ### What is Sales Profit Predictor?
        
        This application uses machine learning to predict the profit from sales based on various input parameters. The model has been trained on historical sales data to identify patterns and relationships that affect profitability.
        
        ### How to Use
        
        1. Enter your sales details in the form
        2. Click "Predict Profit" button
        3. View your predicted profit and related metrics
        
        ### Features
        
        - Fast and accurate profit predictions
        - User-friendly interface
        - Detailed metrics and analysis
        - Based on Random Forest machine learning model
        """)
    
    with about_col2:
        st.markdown("""
        ### Technology Stack
        
        - **Frontend**: Streamlit
        - **Backend**: Python
        - **Machine Learning**: Scikit-learn (Random Forest)
        - **Data Processing**: Pandas, NumPy
        
        ### Model Information
        
        The prediction model is a Random Forest regressor trained on historical sales data. The model takes into account various factors like product category, shipping costs, discounts, and more to predict the final profit.
        
        ### Privacy Notice
        
        All data entered into this application is used only for making predictions and is not stored or shared with third parties.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar content
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
    
    # Add a small widget to the sidebar
    st.markdown("### 💡 Pro Tips")
    tips = [
        "Higher product margins generally lead to better profits",
        "Standard shipping often balances cost and delivery time",
        "Corporate customer segments typically have larger order quantities",
        "Low-priority orders might save on shipping costs"
    ]
    tip = st.selectbox("Sales tips:", tips)
    
    st.markdown("---")
    
    # Add current status
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
    Sales Profit Predictor v2.0 | Last updated: May 2025 | &copy; Mohammad Shafiee
</div>
""", unsafe_allow_html=True)