import streamlit as st
#/Users/mac/PycharmProjects/PythonProject1
def main():
    # Title of the Web App
    st.title("🤖 Machine Learning Algo. Selector")

    # Step 1: Problem Type
    problem_type = st.selectbox("What type of learning?", ["Supervised Learning", "Unsupervised Learning"])

    # Step 2: Target Variable Type
    if problem_type == "Supervised Learning":
        target_type = st.selectbox("What is the target variable type?", ["Categorical", "Continuous"])
    else:
        target_type = st.selectbox("What is the goal?", ["Clustering", "Dimensionality Reduction"])

    # Step 3: Dataset Size
    dataset_size = st.selectbox("What is the dataset size?", ["Small (<1,000 samples)", "Medium (1,000-100,000 samples)", "Large (>100,000 samples)"])

    # Step 4: Special Considerations
    special_case = st.selectbox("Any special considerations?", ["None", "Imbalanced Data", "Text Data", "Time-Series Data", "Image Data", "Sequential Data"])

    # Button to Generate Recommendation
    if st.button("Recommend Algorithm"):
        result = recommend_algorithm(problem_type, target_type, dataset_size, special_case)
        st.markdown(result)

# Function to Recommend ML Algorithm
def recommend_algorithm(problem_type, target_type, dataset_size, special_case):
    recommendation = "### 🏆 Recommended Algorithm:\n"

    if problem_type == "Supervised Learning":
        if target_type == "Categorical":
            if "Small" in dataset_size:
                recommendation += "- Logistic Regression, k-NN, Decision Tree\n"
            elif "Medium" in dataset_size:
                recommendation += "- Random Forest, SVM, XGBoost\n"
            else:
                recommendation += "- Neural Networks, LightGBM, XGBoost\n"
        else:  # Regression
            if "Small" in dataset_size:
                recommendation += "- Linear Regression, Ridge/Lasso\n"
            elif "Medium" in dataset_size:
                recommendation += "- Random Forest, XGBoost, SVR\n"
            else:
                recommendation += "- LightGBM, Neural Networks\n"

    else:  # Unsupervised Learning
        if target_type == "Clustering":
            recommendation += "- k-Means, GMM, DBSCAN\n"
        else:  # Dimensionality Reduction
            recommendation += "- PCA, Autoencoders, t-SNE, UMAP\n"

    # Special Cases
    if special_case == "Imbalanced Data":
        recommendation += "\n🔹 Use SMOTE + XGBoost, Weighted Random Forest\n"
    elif special_case == "Text Data":
        recommendation += "\n🔹 Use TF-IDF + Naïve Bayes, Transformer Models (BERT, GPT)\n"
    elif special_case == "Time-Series Data":
        recommendation += "\n🔹 Use ARIMA, LSTMs, XGBoost with time-based features\n"
    elif special_case == "Image Data":
        recommendation += "\n🔹 Use CNNs (ResNet, EfficientNet, VGG)\n"
    elif special_case == "Sequential Data":
        recommendation += "\n🔹 Use RNNs (LSTM, GRU) for sequences\n"

    return recommendation

# Ensuring Streamlit Runs in the Main Thread
if __name__ == "__main__":
    main()
