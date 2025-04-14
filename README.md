🚀 Predicting AI Model Resource Usage using Regression Techniques

Author: Rohan Pratap Reddy Ravula
Course: DATA 6250 - Machine Learning for Data Science
Instructor: Dr. Memo Ergezer
Term: Spring 2025
🧠 Introduction

With the explosive growth of AI models such as GPT-4, Claude, and Gemini, deployment decisions now face major challenges due to model size, hardware demands, latency, and energy costs. This project aims to predict critical system-level metrics such as:

    Inference latency

    Memory usage

    Compute cost

By applying various regression techniques on AI model metadata and training stats sourced from Epoch AI, we aim to estimate deployment and efficiency metrics, facilitating informed hardware-model pairing for real-world deployment.
📊 Dataset
📌 Source

We used datasets from the Epoch AI project (https://epoch.ai/), including:

    notable_ai_models.csv

    large_scale_ai_models.csv

    ml_hardware.csv

🔍 Structure

    Contains 900+ rows of major AI models with architectural specs, training compute, and hardware details.

    Hardware metadata includes TDP, transistor count, and fabrication details.

    Some features were multi-labeled (comma-separated), necessitating normalization.

🔧 Preprocessing Steps

    Redundant Columns Removed: URLs, citations, notes

    Normalization: Exploded categorical values (e.g., hardware list) into separate rows

    Missing Value Imputation:

        IterativeImputer + RandomForest for numeric values

        RAG (Dragon-Qwen-7B-OV) + Sentence Transformers (all-mpnet-base-v2) for semantic filling

        External linking between datasets (model, hardware)

📈 Baseline Performance

Models trained on the original dataset using StandardScaler:

    🔹 Linear Regression

    🔹 Support Vector Regressor (SVR)

    🔹 Random Forest Regressor

    🔹 Gradient Boosting Regressor

Baseline results revealed that Gradient Boosting consistently achieved the best R² scores and generalization across targets.
🔬 Experiments
⚙️ Experiment 1: Feature Scaling

    Applied MinMaxScaler

    Compared results to Standard Scaler baseline

    Gradient Boosting and SVR showed improved stability with MinMaxScaler

🧪 Experiment 2: Feature Engineering

    Sentence Transformers: Encoded categorical columns using nplr/gte-small

    HDBSCAN Clustering + all-MiniLM-L6-v2 for text feature embedding

    UMAP used for dimensionality reduction

    Resulted in feature-rich vectors improving model discrimination power

🔄 Experiment 3: Feature Transformation

    Applied PCA to reduce the feature space to 15 dimensions

    Helped in combating overfitting, especially for linear models

    Gradient Boosting retained performance post-transformation

🔊 Experiment 4: Noisy Features

    Introduced:

        One synthetic categorical noise column

        One continuous noise feature

    Tree-based models were more resilient to noise than linear models

    SVR and Ridge regressor performance dropped with noise

🧠 Experiment 5: Interpretability (Optional)

    Used SHAP and LIME to interpret feature importance

    Key findings:

        Hardware metrics like TDP, Process size, and Transistors were top contributors

        Metadata (e.g., Task type, Country) were less influential

⏱️ Experiment 6: Time & Memory Profiling

    Training Time:

        Random Forest = Fastest

        Gradient Boosting = More variable but efficient

    Inference Time: GBM fastest

    Memory Usage: 0.0 MB reported due to notebook environment limitations

        Suggestion: Use memory_profiler in future work

📊 Results and Discussion
Experiment	Best Model	R² Score Range	Observations
Original	Gradient Boosting	~0.94–0.95	Good baseline accuracy
Scaling	Gradient Boosting	~0.95+	Slight improvements with normalized scale
Feature Engg	Gradient Boosting	~0.96+	Strong performance gain with semantic embeddings
PCA	Gradient Boosting	~0.93	Reduced complexity, slight accuracy drop
Noise	Random Forest	~0.92	Maintained robustness under noise
Interpretability	–	–	SHAP + LIME explained importance of hardware factors
✅ Recommended Model

🎯 Gradient Boosting Regressor is the most consistent and high-performing model across all preprocessing scenarios.

    Pros:

        Resilient to transformations

        Fast inference

        Well-suited for tabular + embedded features

    Recommendation: Combine with Random Forest for ensemble robustness

🔭 Future Scope

    📦 Model Ensembling: Gradient Boosting + Random Forest

    🔧 Hyperparameter Optimization: Use Optuna or Bayesian tuning

    🧹 Feature Reduction: SHAP-guided selection to prune features

    📱 Deployment: Containerize for web-based inference or API integration

📚 Learning Outcomes

    Solidified skills in:

        ML pipelines

        Dataset normalization and feature engineering

        Model interpretability (SHAP/LIME)

        Trade-offs in model selection (accuracy vs efficiency)
