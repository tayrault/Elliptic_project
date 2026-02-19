import os
import streamlit as st

# Set the browser tab title and icon. Must be called before any other Streamlit API usage.
st.set_page_config(page_title="Fraud Busters", page_icon="🛡️", layout='centered')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import f1_score
import networkx as nx

# ============= DATA LOADING FUNCTIONS =============
@st.cache_data
def load_data(classes_path, edgelist_path, features_path):
    """Load the Elliptic dataset from CSV files"""
    try:
        df_classes = pd.read_csv(classes_path)
        df_edgelist = pd.read_csv(edgelist_path)
        df_features = pd.read_csv(features_path, header=None)
        return df_classes, df_edgelist, df_features
    except Exception as e:
        st.error(f"Cannot load data files: {e}")
        raise

def prepare_data(df_classes, df_features):
    """Merge classes with features and prepare the dataset"""
    # Merge class and feature tables
    df = df_classes.merge(
        df_features,
        how='left',
        left_on='txId',
        right_on=0
    )
    
    # Drop unknown class
    df_labeled = df[df['class'].isin(['1', '2'])]
    
    return df, df_labeled

def split_train_test(df_labeled):
    """Split data into train and test sets based on timesteps"""
    # Split according to time step (1-36 for train, 37-49 for test)
    train_df = df_labeled[df_labeled[1].between(1, 36)]
    test_df = df_labeled[df_labeled[1].between(37, 49)]
    
    # Prepare features and labels
    X_train = train_df.drop(train_df.columns[0:4], axis=1)
    X_test = test_df.drop(test_df.columns[0:4], axis=1)
    y_train = (train_df['class'] == '1').astype(int)
    y_test = (test_df['class'] == '1').astype(int)
    
    return X_train, X_test, y_train, y_test, test_df

# ============= MODEL TRAINING FUNCTIONS =============
def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model with class imbalance handling"""
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Create and train XGBoost model
    clf_xgb = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    clf_xgb.fit(X_train, y_train)
    
    return clf_xgb

# ============= EVALUATION FUNCTIONS =============
def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, and F1 score"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy, precision, recall, f1

def calculate_f1_by_timestep(clf_xgb, test_df, threshold):
    """Calculate F1 scores for each timestep in the test set"""
    test_timesteps = range(37, 50)
    f1_scores = []
    num_illicit = []
    
    for t in test_timesteps:
        # Get data for this time step
        t_data = test_df[test_df[1] == t]
        X_t = t_data.drop(t_data.columns[0:4], axis=1)
        y_t = (t_data['class'] == '1').astype(int)
        
        # Count illicit samples
        illicit_count = y_t.sum()
        num_illicit.append(illicit_count)
        
        # Calculate F1
        if illicit_count > 0 and len(y_t) > 0:
            y_prob_t = clf_xgb.predict_proba(X_t)[:, 1]
            y_pred_t = (y_prob_t >= threshold).astype(int)
            f1 = f1_score(y_t, y_pred_t, zero_division=0)
        else:
            f1 = 0
        f1_scores.append(f1)
    
    return test_timesteps, f1_scores, num_illicit

def calculate_chain_lengths(edgelist, transaction_ids, k=10):
    """Calculate chain lengths for given transaction IDs using edge list"""
    # Create directed graph from edge list
    G = nx.DiGraph()
    G.add_edges_from(edgelist.itertuples(index=False, name=None))
    
    def chain_len_k(G, source, k=10):
        try:
            lengths = nx.single_source_shortest_path_length(G, source, cutoff=k)
            return max(lengths.values(), default=0)
        except:
            return 0
    
    # Calculate chain lengths for each transaction ID
    chain_lengths = [chain_len_k(G, txid, k=k) for txid in transaction_ids]
    
    chain_length_df = pd.DataFrame({
        0: transaction_ids,
        'chain_length_k10': chain_lengths
    })
    
    return chain_length_df

def calculate_min_hops_to_illicit(edgelist, transaction_ids, actual_labels):
    """Calculate minimum hops to illicit transactions for each transaction ID"""
    # Create illicit_nodes list from actual labels
    illicit_nodes = [txid for txid, label in zip(transaction_ids, actual_labels) if (label) == '1']
    
    if len(illicit_nodes) == 0:
        st.warning("No illicit nodes found in the data. Cannot calculate hops to illicit transactions.")
        return None
    else:
        st.info(f"Found {len(illicit_nodes)} illicit nodes for hop calculation.")
    
    # Create directed graph from edge list
    G = nx.DiGraph()
    G.add_edges_from(edgelist.itertuples(index=False, name=None))
    
    # Calculate minimum hops to illicit nodes
    exposure_dict = {}
    for source_node in illicit_nodes:
        try:
            # Compute shortest path lengths from the current source node
            lengths = nx.single_source_shortest_path_length(G, source_node, cutoff=10)
            for target_node, dist in lengths.items():
                # Update the exposure_dict with the minimum distance found so far
                if target_node not in exposure_dict or dist < exposure_dict[target_node]:
                    exposure_dict[target_node] = dist
        except:
            pass  # Node not in graph
    
    exposure_df = (
        pd.DataFrame.from_dict(exposure_dict, orient="index", columns=["min_hops_to_illicit"])
        .reset_index()
        .rename(columns={"index": "txId"})
    )

    # Ensure all transaction IDs are included
    all_nodes = pd.DataFrame({"txId": list(G.nodes)})
    exposure_df = all_nodes.merge(exposure_df, on="txId", how="left")

    # Fill nodes with no exposure as 11
    exposure_df["min_hops_to_illicit"] = exposure_df["min_hops_to_illicit"].fillna(11)
    
    return exposure_df

def plot_f1_by_timestep(test_timesteps, f1_scores, num_illicit):
    """Create a plot showing F1 scores and number of illicit samples by timestep"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Bar chart for number of illicit samples
    ax2 = ax1.twinx()
    ax2.bar(test_timesteps, num_illicit, alpha=0.3, color='blue', label='# Illicit')
    ax2.set_ylabel('Num samples', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Line chart for F1 scores
    ax1.plot(test_timesteps, f1_scores, 'ro-', label='XGBoost', linewidth=2, markersize=8)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Illicit F1')
    ax1.set_ylim(0, 1)
    ax1.set_xticks(list(test_timesteps))
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    plt.title('F1 Score by Time Step (Test Set: 37-49)')
    plt.tight_layout()
    
    return fig

# ============= MAIN APP =============
def main():
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'main'
    
    if st.session_state['current_page'] == 'thank_you':
        show_thank_you_page()
    else:
        show_main_app()

def show_thank_you_page():
    """Display the Thank You page with image"""
    st.markdown(
        """
        <div style='background-color: #1f77b4; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;'>
            <h1 style='color: white; margin: 0; font-size: 2.5em;'>🛡️ Fraud Busters 🛡️</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.title("🙏 Thank You!")
    st.markdown("""
    Thank you for using the **Fraud Busters** Bitcoin Transaction Detection System!
    
    We appreciate your trust in our machine learning solution for identifying illicit transactions.
    """)
    
    # Display image from file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "thankyou.png")
    
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.info("📸 Place your 'thankyou.png' image in the streamlit folder to display it here.")
        st.markdown(f"**Expected path:** `{image_path}`")
    
    # Bottom navigation
    st.divider()
    if st.button("← Back to Main App", use_container_width=True, type="primary"):
        st.session_state['current_page'] = 'main'
        st.rerun()

def show_main_app():
    """Display the main application page"""
    # Team name banner at the top
    st.markdown(
        """
        <div style='background-color: #1f77b4; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;'>
            <h1 style='color: white; margin: 0; font-size: 2.5em;'>🛡️ Fraud Busters 🛡️</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.title("XGBoost for Elliptic Bitcoin Transactions")
    #st.markdown("Detection of illicit Bitcoin transactions using XGBoost classifier")
    
    # File paths - Construct absolute path to data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, '..', 'data')
    
    classes_path = os.path.join(data_folder, "elliptic_txs_classes.csv")
    edgelist_path = os.path.join(data_folder, "elliptic_txs_edgelist.csv")
    features_path = os.path.join(data_folder, "elliptic_txs_features.csv")
    
    # Load data automatically (cached after first load)
    if 'data_loaded' not in st.session_state:
        try:
            st.info(f"📁 Loading data from: {data_folder}")
            
            # Check if files exist
            for path, name in [(classes_path, "classes"), (features_path, "features"), (edgelist_path, "edgelist")]:
                if not os.path.exists(path):
                    st.error(f"File not found: {path}")
                    st.stop()
            
            with st.spinner("Loading data..."):
                df_classes, df_edgelist, df_features = load_data(
                    classes_path, edgelist_path, features_path
                )
                
                # Prepare data
                df, df_labeled = prepare_data(df_classes, df_features)
                
                st.session_state['df_classes'] = df_classes
                st.session_state['df_edgelist'] = df_edgelist
                st.session_state['df_features'] = df_features
                st.session_state['df'] = df
                st.session_state['df_labeled'] = df_labeled
                st.session_state['data_loaded'] = True
                
                st.success("✅ Data loaded successfully!")
                
        except Exception as e:
            st.error(f"Error loading data: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
    
    # Display data summary
    st.header("📊 Data Overview")
    df = st.session_state['df']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Transactions", len(df))
    with col2:
        st.metric("Illicit (class 1)", (df['class'] == '1').sum())
    with col3:
        st.metric("Licit (class 2)", (df['class'] == '2').sum())
    with col4:
        st.metric("Unknown", (df['class'] == 'unknown').sum())
    with col5:
        st.metric("Suspicious", (df['class'] == 'suspicious').sum())

    # Model Training Section
    st.header("🤖 Model Training")
    
    # Create two columns for Train and Clear buttons
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Train XGBoost Model", use_container_width=True):
            try:
                with st.spinner("Training XGBoost model..."):
                    progress_bar = st.progress(0)
                    
                    st.write("**Step 1/4:** Splitting data by timesteps...")
                    df_labeled = st.session_state['df_labeled']
                    X_train, X_test, y_train, y_test, test_df = split_train_test(df_labeled)
                    progress_bar.progress(25)
                    
                    st.write(f"**Step 2/4:** Training data: {len(X_train)} samples (timesteps 1-36)")
                    st.write(f"**Step 2/4:** Testing data: {len(X_test)} samples (timesteps 37-49)")
                    progress_bar.progress(50)
                    
                    st.write("**Step 3/4:** Training XGBoost classifier...")
                    clf_xgb = train_xgboost_model(X_train, y_train, X_test, y_test)
                    progress_bar.progress(75)
                    
                    st.write("**Step 4/4:** Saving model to session...")
                    st.session_state['model'] = clf_xgb
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    st.session_state['test_df'] = test_df
                    progress_bar.progress(100)
                    
                    st.success("✅ Model trained successfully!")
                    
            except Exception as e:
                st.error(f"Error training model: {e}")
                st.exception(e)
    
    with col2:
        if st.button("🗑️ Clear Model", use_container_width=True, type="secondary"):
            # Clear model and training-related session state
            keys_to_clear = ['model', 'X_train', 'X_test', 'y_train', 'y_test', 'test_df', 'edgelist', 
                            'uploaded_full_data', 'last_uploaded_file_id', 'prediction_results', 'prediction_correctness']
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("✅ Model cleared! Data remains loaded.")
            st.rerun()
    
    # Model Evaluation Section
    if 'model' in st.session_state:
        st.header("📈 Model Evaluation")
        
        # Prediction threshold slider
        pred_threshold = st.slider("Prediction Threshold", 0.05, 0.95, 0.7, 0.05, key='pred_threshold')
        
        clf_xgb = st.session_state['model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        test_df = st.session_state['test_df']
        
        # Calculate predictions with current threshold
        y_prob = clf_xgb.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= pred_threshold).astype(int)
        
        # Calculate overall metrics
        accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
        with col4:
            st.metric("F1 Score", f"{f1:.4f}")
        
        # Prediction Section
        st.header("🔮 Make Predictions on New Data")
        st.markdown("Upload a CSV file or paste raw transaction features to get predictions.")
        
        # Optional edge file uploader
        st.subheader("📊 Upload Edge List for Path Analysis")
        edge_file = st.file_uploader("Upload edge CSV file (2 columns: from_txId, to_txId)", type=['csv'], key='edge_uploader')
        
        if edge_file is not None:
            try:
                edgelist = pd.read_csv(edge_file, header=None)
                st.success(f"✅ Loaded {len(edgelist)} edges")
                st.session_state['edgelist'] = edgelist
                
                # Show preview
                with st.expander("Preview edge list data"):
                    st.dataframe(edgelist.head())
            except Exception as e:
                st.error(f"Error loading edge file: {e}")
        
        # Upload Feature Table
        st.subheader("📊 Upload Feature Table for XGBoost Prediction")
        uploaded_file = st.file_uploader("Upload CSV file with 167 columns (col 0=label, col 1=txID, cols 2-166=features)", type=['csv'])
        
        # Store uploaded file data in session state
        if uploaded_file is not None:
            try:
                full_data = pd.read_csv(uploaded_file, header=None)
                current_file_id = f"{uploaded_file.file_id}_{len(full_data)}"
                
                # Only update if it's a new file
                if 'last_uploaded_file_id' not in st.session_state or st.session_state['last_uploaded_file_id'] != current_file_id:
                    st.session_state['uploaded_full_data'] = full_data
                    st.session_state['last_uploaded_file_id'] = current_file_id
                    # Clear previous results when new file is uploaded
                    if 'prediction_results' in st.session_state:
                        del st.session_state['prediction_results']
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        # Check if we have uploaded data (either just uploaded or from previous session)
        if 'uploaded_full_data' in st.session_state:
            full_data = st.session_state['uploaded_full_data']
            try:
                st.success(f"Loaded {len(full_data)} transactions with {len(full_data.columns)} columns")
                
                # Show preview
                with st.expander("Preview uploaded data"):
                    st.dataframe(full_data.head())
                
                if st.button("Predict", key="predict_btn_main"):
                    with st.spinner("Making predictions using XGBoost..."):
                        st.info("Making predictions using XGBoost......")
                        # Extract actual labels, transaction IDs and features
                        actual_labels = full_data[0]
                        transaction_ids = full_data[1]
                        input_data = full_data.iloc[:, 2:]  # Columns 2-166 are features
                            
                        # Make predictions
                        y_prob_new = clf_xgb.predict_proba(input_data)[:, 1]
                        y_pred_new = (y_prob_new >= pred_threshold).astype(int)
                        
                        # Determine correctness
                        correctness = []
                        correctness_symbol = []
                        
                        for pred, actual in zip(y_pred_new, actual_labels.astype(str)):
                            if actual.lower() == 'unknown':
                                correctness.append('Unknown')
                                correctness_symbol.append('❔')
                            elif (pred == 1 and actual == '1') or (pred == 0 and actual == '2'):
                                correctness.append('Correct')
                                correctness_symbol.append('✅')
                            else:
                                correctness.append('Wrong')
                                correctness_symbol.append('❌')
                        
                        # Create results dataframe with transaction ID as first column
                        results = pd.DataFrame({
                            'Transaction_ID': transaction_ids.astype(str),
                            'Probability_Illicit': y_prob_new,
                            'Prediction': ['Illicit' if pred == 1 else 'Licit' for pred in y_pred_new],
                            'Status': correctness_symbol
                        })
                        
                        # Calculate and merge chain lengths and min hops if edge file is available
                        if 'edgelist' in st.session_state:
                            try:
                                st.info("Calculating chain lengths...")
                                chain_length_df = calculate_chain_lengths(
                                    st.session_state['edgelist'], 
                                    transaction_ids, 
                                    k=10
                                )
                                # Convert column 0 to string for consistent merge
                                chain_length_df[0] = chain_length_df[0].astype(str)
                                
                                # Merge with results using inner join on transaction ID
                                results = results.merge(
                                    chain_length_df, 
                                    left_on='Transaction_ID', 
                                    right_on=0, 
                                    how='inner'
                                )
                                results = results.drop(columns=[0])  # Remove duplicate column
                                st.success(f"✅ Added chain length analysis ({len(results)} transactions matched)")
                                
                                # Calculate minimum hops to illicit transactions
                                st.info("Calculating minimum hops to illicit transactions...")
                                exposure_df = calculate_min_hops_to_illicit(
                                    st.session_state['edgelist'],
                                    transaction_ids,
                                    actual_labels
                                )
                                
                                if exposure_df is not None:
                                    # Convert txId to string for consistent merge
                                    exposure_df['txId'] = exposure_df['txId'].astype(str)
                                    
                                    # Merge with results using inner join on transaction ID
                                    results = results.merge(
                                        exposure_df,
                                        left_on='Transaction_ID',
                                        right_on='txId',
                                        how='inner'
                                    )
                                    results = results.drop(columns=['txId'])  # Remove duplicate column
                                    st.success(f"✅ Added min hops to illicit analysis ({len(results)} transactions matched)")
                                    
                            except Exception as e:
                                st.warning(f"Could not calculate network features: {e}")
                        
                        # Add Tier classification if both chain_length_k10 and min_hops_to_illicit exist
                        if 'chain_length_k10' in results.columns and 'min_hops_to_illicit' in results.columns:
                            def assign_tier(row):
                                chain_len = row['chain_length_k10']
                                min_hops = row['min_hops_to_illicit']
                                
                                # Tier 1: Deep AND close to illicit
                                if chain_len >= 6 and pd.notna(min_hops) and min_hops <= 3:
                                    return '1'
                                # Tier 2: Deep but not directly exposed
                                elif chain_len >= 6 and (pd.isna(min_hops) or min_hops > 3):
                                    return '2'
                                # Tier 3: Close to illicit but shallow
                                elif chain_len < 6 and pd.notna(min_hops) and min_hops <= 3:
                                    return '3'
                                # Tier 4: Low Risk
                                else:
                                    return '4'
                            
                            results['Tier'] = results.apply(assign_tier, axis=1)
                            st.success("✅ Added Tier classification")
                        
                        # Store results in session state
                        st.session_state['prediction_results'] = results
                        st.session_state['prediction_correctness'] = correctness
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.exception(e)
            
            # Display results from session state if they exist
            if 'prediction_results' in st.session_state:
                results = st.session_state['prediction_results']
                correctness = st.session_state['prediction_correctness']
                
                # Display summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Predictions", len(results))
                with col2:
                    st.metric("Correct ✅", (pd.Series(correctness) == 'Correct').sum())
                with col3:
                    st.metric("Wrong ❌", (pd.Series(correctness) == 'Wrong').sum())
                with col4:
                    st.metric("Unknown ❔", (pd.Series(correctness) == 'Unknown').sum())
                
                # Display tier summary if Tier column exists
                if 'Tier' in results.columns:
                    st.subheader("Tier Distribution")
                    tier_cols = st.columns(4)
                    with tier_cols[0]:
                        st.metric("Tier 1", (results['Tier'] == '1').sum())
                    with tier_cols[1]:
                        st.metric("Tier 2", (results['Tier'] == '2').sum())
                    with tier_cols[2]:
                        st.metric("Tier 3", (results['Tier'] == '3').sum())
                    with tier_cols[3]:
                        st.metric("Tier 4", (results['Tier'] == '4').sum())
                
                # Display results
                st.subheader("Prediction Results")
                st.dataframe(results, use_container_width=True)
                
                # Download button
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                    key="download_csv"
                )
    
    # Bottom navigation
    st.divider()
    if st.button("🙏 Go to Thank You Page", use_container_width=True, type="primary"):
        st.session_state['current_page'] = 'thank_you'
        st.rerun()

if __name__ == "__main__":
    main()