import streamlit as st

# Function to Recommend ML Algorithm
def recommend_algorithm(problem_type, ml_task, dataset_size, num_features, special_case):
    recommendation = "### 🏆 Recommended Algorithm:\n"

    if problem_type == "Supervised Learning":
        if ml_task == "Binary Classification":
            if special_case == "Image Data":
                if "Small" in dataset_size or "Medium" in dataset_size:
                    recommendation += (
                        "- **Feature Extraction** using pretrained CNNs (e.g., ResNet, EfficientNet, VGG, MobileNet)\n"
                        "  followed by traditional ML models (k-NN, SVM, Random Forest, XGBoost)\n"
                    )
                else:
                    recommendation += "- **Custom CNNs (end-to-end deep learning)** for image classification\n"
            else:
                if "Small" in dataset_size:
                    recommendation += "- Logistic Regression, Decision Tree, k-NN\n"
                elif "Medium" in dataset_size:
                    recommendation += "- Random Forest, SVM, XGBoost\n"
                else:
                    recommendation += "- Neural Networks, LightGBM, XGBoost\n"

        elif ml_task == "Multi-class Classification":
            if "Small" in dataset_size:
                recommendation += "- Logistic Regression (OvR), Decision Tree\n"
            elif "Medium" in dataset_size:
                recommendation += "- Random Forest, XGBoost, LightGBM\n"
            else:
                recommendation += "- Deep Learning (MLP, CNNs for images, LSTMs for text)\n"

        elif ml_task == "Regression":
            if "Small" in dataset_size:
                recommendation += "- Linear Regression, Ridge/Lasso\n"
            elif "Medium" in dataset_size:
                recommendation += "- Random Forest, XGBoost, SVR\n"
            else:
                recommendation += "- LightGBM, Neural Networks (CNNs for images, RNNs for sequential data)\n"

    elif problem_type == "Unsupervised Learning":
        if ml_task == "Clustering":
            recommendation += "- k-Means, GMM, DBSCAN\n"
        elif ml_task == "Anomaly Detection":
            recommendation += "- Isolation Forest, One-Class SVM, Autoencoders\n"
        elif ml_task == "Feature Selection":
            recommendation += "- PCA, Autoencoders, Recursive Feature Elimination (RFE), SelectKBest\n"
        elif ml_task == "Dimensionality Reduction":
            recommendation += "- PCA, Autoencoders, t-SNE, UMAP\n"

    elif problem_type == "Reinforcement Learning":
        if ml_task == "Value-Based Methods":
            recommendation += "- Q-Learning, Deep Q Networks (DQN)\n"
        elif ml_task == "Policy-Based Methods":
            recommendation += "- REINFORCE Algorithm, Proximal Policy Optimization (PPO)\n"
        elif ml_task == "Actor-Critic Methods":
            recommendation += "- Advantage Actor-Critic (A2C), Deep Deterministic Policy Gradient (DDPG)\n"

    # Special Cases
    if special_case == "Imbalanced Data":
        recommendation += "\n🔹 Use SMOTER/SMOGN + XGBoost, Weighted Random Forest, or Quantile Regression\n"
    elif special_case == "Text Data":
        recommendation += "\n🔹 Use TF-IDF + Naïve Bayes (small), Transformer Models (BERT, GPT) (large)\n"
    elif special_case == "Time-Series Data":
        recommendation += "\n🔹 Use ARIMA, XGBoost with time-based features, or LSTMs/GRUs\n"
    elif special_case == "Sequential Data":
        recommendation += "\n🔹 Use RNNs (LSTM, GRU) for sequences\n"

    # Consider Number of Features
    if num_features == "High (≥100 features)":
        recommendation += "\n🔹 Consider Feature Selection (PCA, Lasso Regression, Tree-based Feature Importance)\n"
    elif num_features == "Low (<10 features)":
        recommendation += "\n🔹 Use simpler models like Logistic Regression, k-NN, Decision Tree\n"

    return recommendation

# Main function to structure the Streamlit app
def main():
    st.title("🤖 Machine Learning Algo. Selector")
    st.markdown("(Developed @DIT YCCE Nagpur)")
    st.markdown(" *** Warning: This is a preliminary tool and we do not claim total correctness of the outcome ! ")
    # Step 1: Problem Type
    problem_type = st.selectbox("What type of learning?", ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning"])

    # Step 2: ML Task Selection
    if problem_type == "Supervised Learning":
        ml_task = st.selectbox("Select the ML Task:", ["Binary Classification", "Multi-class Classification", "Regression"])
    elif problem_type == "Unsupervised Learning":
        ml_task = st.selectbox("Select the ML Task:", ["Clustering", "Anomaly Detection", "Feature Selection", "Dimensionality Reduction"])
    else:  # Reinforcement Learning
        ml_task = st.selectbox("Select the RL Method:", ["Value-Based Methods", "Policy-Based Methods", "Actor-Critic Methods"])

    # Step 3: Dataset Size
    dataset_size = st.selectbox("What is the dataset size?", ["Small (<1,000 samples)", "Medium (1,000-100,000 samples)", "Large (>100,000 samples)"])

    # Step 4: Number of Features (Predictors)
    num_features = st.selectbox("How many features (predictors) does your dataset have?", ["Low (<10 features)", "Medium (10-100 features)", "High (≥100 features)"])

    # Step 5: Special Considerations
    special_case = st.selectbox("Any special considerations?", ["None", "Imbalanced Data", "Text Data", "Time-Series Data", "Image Data", "Sequential Data"])

    # Recommendation Button
    if st.button("Recommend Algorithm"):
        result = recommend_algorithm(problem_type, ml_task, dataset_size, num_features, special_case)
        st.markdown(result)

# Run the Streamlit app
if __name__ == "__main__":
    main()
