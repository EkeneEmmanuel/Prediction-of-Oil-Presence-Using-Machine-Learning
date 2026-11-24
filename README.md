# GeoSpark: Physics-Informed Machine Learning for Oil Presence Prediction

**SPE DSEATS Africa Region Datathon 2026**

**Team:** GeoSpark

A machine learning approach to predicting hydrocarbon presence by integrating conventional predictive modelling with petroleum engineering principles and physics-informed feature engineering.

---

## 1. Project Overview

Hydrocarbon exploration involves evaluating geological, reservoir, and seismic information to determine the likelihood of finding oil in a prospect. Making these decisions requires integrating multiple petroleum system elements, including reservoir quality, trap integrity, hydrocarbon charge, and seismic evidence.

This project develops two complementary machine learning approaches to predict oil presence at exploration locations using geological and geophysical measurements.

The first approach implements a conventional machine learning workflow, combining multiple tree-based algorithms through a soft-voting ensemble.

The second approach introduces petroleum engineering knowledge into the modelling process through physics-informed feature engineering and a LightGBM classifier.

The objective is to investigate whether incorporating domain knowledge into machine learning can improve predictive performance, geological consistency, and the interpretability of hydrocarbon prospect evaluation.

## 2. Problem Statement

The project addresses a binary classification problem:

**Given geological and geophysical measurements from an exploration location, can machine learning predict whether the location contains oil?**

The target variable is:

| Target | Meaning |
|---|---|
| 0 | No oil (Dry) |
| 1 | Oil present |

The modelling problem is particularly interesting because hydrocarbon discovery is not determined by a single geological variable. It involves the interaction of several petroleum system elements.

The project therefore approaches the problem from the perspective of petroleum systems.

### Petroleum Engineering Context

| Petroleum system element | Geological interpretation | Dataset features |
|---|---|---|
| Reservoir | Rock capable of storing and transmitting hydrocarbons | Rock_Type, Porosity, Permeability |
| Trap | Geological structure that prevents hydrocarbon migration | Trap_Type |
| Charge | Availability and proximity of hydrocarbon sources | Proximity_to_Oil_Field |
| Seismic evidence | Quality of seismic information supporting prospect evaluation | Seismic_Score |

The target variable, `Oil_Presence`, represents the observed hydrocarbon outcome.

---

## 3. Dataset Description

The project uses two CSV datasets:

- `oil_presence_trainset.csv`
- `oil_presence_testset.csv`

The training dataset contains the geological and geophysical measurements alongside the target variable. The test dataset contains measurements for which oil presence must be predicted.

### Input Features

| Feature | Description | Unit |
|---|---|---|
| Rock_Type | Lithological classification | Categorical |
| Porosity | Percentage of pore space in the rock | % |
| Permeability | Ability of the rock to transmit fluids | mD |
| Seismic_Score | Seismic evidence score | 0–1 |
| Trap_Type | Type of geological trap | Categorical |
| Proximity_to_Oil_Field | Distance to an existing oil field | km |
| Estimated_Reservoir_Depth | Estimated reservoir depth | m |

### Target Variable

`Oil_Presence`: Binary indicator of hydrocarbon presence.

### Dataset Challenges

The exploratory analysis identified several important characteristics:

1. Class imbalance: approximately 28% of the training observations contain oil.
2. Missing values: some training features contain substantial missing observations.
3. Geological inconsistencies: porosity and permeability exhibit an unexpectedly weak relationship.
4. Lithological anomalies: some shale observations contain unusually high permeability values.
5. Seismic information: Seismic_Score provides the strongest individual predictive signal in the dataset.

These observations informed the subsequent feature engineering and modelling strategies.

---

## 4. Methodology

The project is organized into four major stages.

### Stage 1: Exploratory Data Analysis (EDA)

The first stage investigates the statistical and geological characteristics of the dataset.

Key investigations include:

- Target distribution and class imbalance.
- Missing-value analysis.
- Physical-range validation of numerical measurements.
- Oil occurrence across different rock and trap types.
- Relationship between numerical features and hydrocarbon presence.
- Correlation analysis.
- Porosity-permeability consistency.
- Relationship between seismic scores and oil occurrence.
- Combined influence of reservoir quality, trap presence, and seismic evidence.

An important finding was that Seismic_Score showed the strongest individual relationship with oil presence.

The analysis also revealed that a combination of good reservoir rock, an existing trap, and strong seismic evidence was associated with substantially higher oil occurrence.

### Stage 2: Conventional Machine Learning Workflow

The first modelling approach establishes a conventional machine learning baseline.

#### 4.2.1 Data Preprocessing

The preprocessing pipeline includes:

- Reconciliation of missing and inconsistent trap labels.
- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- One-hot encoding of categorical variables.
- Standardization for algorithms sensitive to feature scales.

Scikit-learn pipelines and column transformers are used to integrate preprocessing with model training.

#### 4.2.2 Feature Engineering

Four additional features were introduced:

**1. Has Trap**

Indicates whether a geological trap is present.

**2. Is Shale**

Identifies shale formations.

**3. Seismic × Trap**

Captures the interaction between seismic evidence and trap presence.

$$
\text{Seismic x Trap} = \text{Seismic Score} \times \text{Has Trap}
$$


**4. Seismic Squared**

Captures potential nonlinear relationships in seismic evidence.

$$
\text{Seismic Squared} = (\text{Seismic Score})^2
$$

#### 4.2.3 Model Development

The following algorithms were evaluated:

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Linear classification baseline |
| Random Forest | Ensemble-based nonlinear modelling |
| Extra Trees | Randomized tree ensemble |
| HistGradientBoosting | Gradient boosting |
| LightGBM | Gradient-boosted decision trees |
| Multi-Layer Perceptron | Neural network baseline |
| XGBoost | Gradient boosting, when available |

Models were evaluated using the same five-fold stratified cross-validation strategy.

#### 4.2.4 Soft-Voting Ensemble

The stronger tree-based models were combined into a soft-voting ensemble.

Rather than relying on a single classifier, the ensemble averages the predicted class probabilities from its constituent models.

Logistic Regression and the Multi-Layer Perceptron were excluded from the final ensemble because of their comparatively weaker performance during evaluation.

### Stage 3: Physics-Informed Machine Learning

The second approach integrates petroleum engineering knowledge into the machine learning pipeline.

Rather than relying exclusively on the original dataset features, additional features were engineered to represent geological relationships and petroleum system characteristics.

#### 4.3.1 Log Permeability

Permeability values exhibited a wide numerical range. A logarithmic transformation was introduced to compress the distribution.

$$
\text{Log Permeability} = \ln(1 + \text{Permeability})
$$


#### 4.3.2 Compaction Resistance Index (CRI)

Reservoir porosity can decrease with increasing burial depth because of compaction and cementation.

The following feature was introduced:

$$
CRI = \frac{\text{Porosity (fraction)}}{\text{Estimated Reservoir Depth}}
$$

This feature represents the relationship between porosity and estimated reservoir depth.

#### 4.3.3 Trap Type Missing Indicator

A binary feature was introduced to distinguish missing trap information from observations with known trap classifications.

#### 4.3.4 Trap Type Weight

Trap categories were assigned weights based on their observed oil occurrence rates in the reference training data.

The procedure involves:

1. Grouping observations by trap type.
2. Calculating the oil occurrence rate for each category.
3. Ranking categories according to their observed occurrence rates.
4. Normalizing the rankings.
5. Using the mean weight for previously unseen or missing categories.

This approach allows the model to incorporate information about trap effectiveness.

#### 4.3.5 Rock Type Weight

A similar approach was applied to rock classifications.

Additionally, shale observations with permeability values exceeding 100 mD were relabelled as `Misclassified_Lithology` to distinguish anomalous observations from conventional shale formations.

#### 4.3.6 Trap Confidence Index

The Trap Confidence Index integrates seismic evidence with the assigned trap-type weight.

$$
\text{Trap Confidence Index} = (\text{Seismic Score}) \cdot (W_{\text{Trap Type}})
$$


This feature captures the combined contribution of seismic support and trap classification.

#### 4.3.7 Charge Potential and Chance of Success

Charge potential is represented using an inverse-distance transformation:

$$
\text{Charge Potential} = \frac{1}{\text{Proximity to Oil Field}  + 1}
$$

The final Chance of Success feature is a multiplication of rock type weight, trap type weight, and charge potential:

The categorical weights are learned from the reference training data rather than being manually assigned.

**Important:** During cross-validation, category weights are calculated using only the corresponding training fold to prevent information leakage into validation observations.

#### 4.3.8 LightGBM Classification

The physics-informed features are supplied to a LightGBM classifier.

The final model uses the following selected hyperparameters:

| Hyperparameter | Value |
|---|---:|
| n_estimators | 161 |
| learning_rate | 0.0209225 |
| max_depth | 26 |
| num_leaves | 12 |
| min_child_samples | 13 |
| random_state | 42 |

The hyperparameters were selected through an Optuna optimization process during model development.

### Stage 4: Model Comparison and Drilling Prioritization

Accuracy alone is insufficient for evaluating an imbalanced hydrocarbon classification problem.

The project therefore uses several complementary evaluation metrics.

| Metric | Interpretation |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | Proportion of predicted oil locations that actually contain oil |
| Recall | Proportion of actual oil locations correctly identified |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ability to distinguish between oil and dry locations |
| PR-AUC | Precision-recall performance, particularly useful for imbalanced data |
**Standard Machine Learning Results:**

|	Accuracy | Precision	Recall |F1	| ROC-AUC	| PR-AUC | Accuracy (rows with seismic)
|---|---|---|---|---|---|
| HistGradientBoosting	| 0.8130 | 0.7403	| 0.5030	| 0.5990	| 0.8089	| 0.6796	| 0.8779 |
| XGBoost	| 0.8100	| 0.7240	| 0.5102	| 0.5986	| 0.8132	| 0.6807	| 0.8815 |
| Random Forest	| 0.8097	| 0.6985 | 0.5534	| 0.6175	| 0.8033	| 0.6686	| 0.8872 |
| Logistic Regression	| 0.8067	| 0.7976	| 0.4070 | 0.5390	| 0.7917	| 0.6546	| 0.8651 |
| Extra Trees	| 0.8043	| 0.6847	| 0.5474	| 0.6084	| 0.7877	| 0.6448	| 0.8759 |
| LightGBM	| 0.8043	| 0.6952	| 0.5258	| 0.5988	| 0.8084	| 0.6762	| 0.8826 |
| Neural Net (MLP) |	0.8020	| 0.7786	| 0.4010	| 0.5293	| 0.7963	| 0.6558	| 0.8564 |


**Physics Informed Machine Learning Results:**

| Metrics | TRAIN (all rows)	| VALIDATION (all rows) |
|---|---|---|
| Accuracy	| 0.8517	| 0.8450 |
| Precision	| 0.8555	| 0.8558
| Recall | 0.5601	| 0.5329 |
| Specificity	| 0.9637	| 0.9654 |
| F1 Score	| 0.6770	| 0.6568 |
| TP	| 373.0000	| 89.0000 |
| TN	| 1671.0000	| 418.0000 |
| FP	| 63.0000	| 15.0000 |
| FN	| 293.0000 | 78.0000 |
| ROC-AUC	| 0.9064	| 0.8396 |
| PR-AUC | 0.8078	| 0.7628 |

#### Drilling Prioritization

The physics-informed LightGBM model generates oil-presence probabilities for the test observations.

These probabilities are subsequently classified into five drilling-priority tiers:

| Predicted probability of oil | Priority |
|---|---|
| 0–20% | Very Low |
| 20–40% | Low |
| 40–60% | Moderate |
| 60–80% | High |
| 80–100% | Very High |

This provides a probability-based framework for organizing exploration prospects and supporting further evaluation.

---

## 5. Model Interpretability

SHAP (SHapley Additive exPlanations) is used to investigate the features influencing model predictions.

Interpretability analysis is performed for both modelling approaches.

For the conventional workflow, the analysis investigates the contributions of the original geological and geophysical measurements.

For the physics-informed workflow, the analysis examines the contributions of engineered features such as:

- Seismic_Score
- Trap_Confidence_Index
- Chance_of_Success
- Rock_Type
- Log_Permeability

Feature importance and SHAP visualizations provide additional insight into the relationships learned by the models.

---

## 6. Key Findings

The project produced several important geological and machine learning observations.

1. **Seismic evidence is highly informative:** Seismic_Score emerged as the strongest individual predictive feature in the exploratory analysis.

2. **Hydrocarbon discovery involves interacting geological conditions:** Reservoir quality, trap presence, and seismic evidence collectively provide a stronger indication of oil presence than considering these factors independently.

3. **Geological inconsistencies require attention:** The unexpected porosity-permeability relationship and unusually high shale permeability values highlighted the importance of domain knowledge during data preparation.

4. **Physics-informed feature engineering provides an alternative modelling strategy:** Petroleum engineering relationships can be incorporated into machine learning through carefully constructed features.

5. **Model interpretability matters:** SHAP analysis provides a means of investigating whether the model relies on meaningful geological information.

6. **Probability-based predictions can support prospect prioritization:** Predicted oil probabilities can be translated into drilling-priority categories for further assessment.

### Overall Conclusion

Both modelling approaches produced useful predictive results. The physics-informed LightGBM approach performed better than the conventional ensemble in the project's final comparison.

The results demonstrate the potential value of integrating petroleum engineering knowledge into machine learning workflows for subsurface exploration applications.

---

## 7. Technologies and Libraries

The project was implemented in Python using the following technologies:

| Technology | Application |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Scikit-learn | Preprocessing, model development and evaluation |
| LightGBM | Gradient-boosted classification |
| XGBoost | Additional model comparison, when available |
| Optuna | Hyperparameter optimization |
| SHAP | Model interpretability |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| SciPy | Statistical analysis |


## 13. Acknowledgement

This project was developed as part of the **Society of Petroleum Engineers (SPE) DSEATS Africa Region Datathon 2026**.

The work reflects a collaborative effort to integrate petroleum engineering knowledge, geological interpretation, and data science in addressing a practical subsurface exploration problem.

## 14. Disclaimer

This project is an academic and datathon machine learning exercise. Its predictions should not be interpreted as definitive evidence of hydrocarbon presence or used as standalone inputs for actual drilling investment decisions.

All model predictions are subject to the limitations of the available dataset, feature assumptions, and model generalization.
