import streamlit as st

# Function to Recommend ML Algorithm
def recommend_algorithm(problem_type, ml_task, dataset_size, special_case):
    recommendation = "### 🏆 Recommended Algorithm:\n"

    if problem_type == "Supervised Learning":
        if ml_task == "Classification":
            if special_case == "Image Data":
                if "Small" in dataset_size or "Medium" in dataset_size:
                    recommendation += (
                        "- **Feature Extraction** using pretrained CNNs (e.g., ResNet, EfficientNet, VGG, MobileNet)\n"
                        "  followed by traditional ML models (k-NN, SVM, Random Forest, XGBoost)\n"
                    )
                else:  # Large image dataset
                    recommendation += "- **Custom CNNs (end-to-end deep learning)** for image classification\n"
            else:
                if "Small" in dataset_size:
                    recommendation += "- Logistic Regression, k-NN, Decision Tree\n"
                elif "Medium" in dataset_size:
                    recommendation += "- Random Forest, SVM, XGBoost\n"
                else:
                    recommendation += "- Neural Networks, LightGBM, XGBoost\n"

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

    # Special Cases
    if special_case == "Imbalanced Data":
        recommendation += "\n🔹 Use SMOTER/SMOGN + XGBoost, Weighted Random Forest, or Quantile Regression\n"
    elif special_case == "Text Data":
        recommendation += "\n🔹 Use TF-IDF + Naïve Bayes (small), Transformer Models (BERT, GPT) (large)\n"
    elif special_case == "Time-Series Data":
        recommendation += "\n🔹 Use ARIMA, XGBoost with time-based features, or LSTMs/GRUs\n"
    elif special_case == "Sequential Data":
        recommendation += "\n🔹 Use RNNs (LSTM, GRU) for sequences\n"

    return recommendation

# Main function to structure the Streamlit app
def main():
    st.title("🤖 Machine Learning Algorithm Selector")

    # Step 1: Problem Type
    problem_type = st.selectbox("What type of learning?", ["Supervised Learning", "Unsupervised Learning"])

    # Step 2: ML Task Selection
    if problem_type == "Supervised Learning":
        ml_task = st.selectbox("Select the ML Task:", ["Classification", "Regression"])
    else:
        ml_task = st.selectbox("Select the ML Task:", ["Clustering", "Anomaly Detection", "Feature Selection", "Dimensionality Reduction"])

    # Step 3: Dataset Size
    dataset_size = st.selectbox("What is the dataset size?", ["Small (<1,000 samples)", "Medium (1,000-100,000 samples)", "Large (>100,000 samples)"])

    # Step 4: Special Considerations
    special_case = st.selectbox("Any special considerations?", ["None", "Imbalanced Data", "Text Data", "Time-Series Data", "Image Data", "Sequential Data"])

    # Recommendation Button
    if st.button("Recommend Algorithm"):
        result = recommend_algorithm(problem_type, ml_task, dataset_size, special_case)
        st.markdown(result)

# Run the Streamlit app
if __name__ == "__main__":
    main()
