  # Blinded Clinical Trial Simulation and Machine Learning Models  
  Longevity InTime – Stage 1 Technical Assignment  
  Author: Hemanth Kumar Kottapalli  

  ---

  ## 1. Project Overview

  This repository contains a blinded clinical trial simulation pipeline and two machine learning models (Logistic Regression and MLP).  
  The objective is to simulate a randomized clinical trial using real patient data, generate blinded outcomes, and train predictive models without any access to counterfactual information.

  The implementation follows a potential-outcomes-based causal modeling framework, ensuring a cheat-secure and unbiased evaluation.

  ---

  ## 2. Repository Structure

  ```
  clinical-trial-simulation-ML-Hemanth/
  │
  ├── data/
  │   └── heart.csv
  │
  ├── src/
  │   ├── simulate_trial.py
  │   ├── train_blinded_model.py
  │   └── train_blinded_mlp.py
  │
  ├── results/
  │   ├── trial_train_val.csv
  │   └── trial_test_blinded.csv
  │
  └── README.md
  ```

  ---

  ## 3. Dataset Description

  The dataset (`data/heart.csv`) is a structured clinical dataset that includes variables such as:

  - Age  
  - Resting blood pressure  
  - Cholesterol  
  - Maximum heart rate  
  - ST depression  
  - Chest pain type  
  - Exercise-induced angina  
  - Thalassemia  
  - Calcium score  
  - Other relevant clinical indicators  

  All categorical variables were one-hot encoded, and numerical variables were standardized during the simulation process.

  ---

  ## 4. Clinical Trial Simulation Methodology

  The simulation is implemented in `simulate_trial.py` and follows standard principles used in randomized controlled trials.

  ### 4.1 Potential Outcomes Model
  For each patient:

  - A hidden coefficient vector β is sampled internally.  
  - Control outcome probability:  
    `p0 = sigmoid(Xβ)`  
  - Treatment outcome probability:  
    `p1 = sigmoid(Xβ + δ)`  
    where δ < 0 represents a protective treatment effect.

  ### 4.2 Outcome Sampling
  Counterfactual outcomes are generated:

  - `Y0 ~ Bernoulli(p0)`  
  - `Y1 ~ Bernoulli(p1)`  

  These values are never saved to maintain blinding.

  ### 4.3 Treatment Assignment
  Treatment is assigned randomly:

  `T ~ Bernoulli(0.5)`

  ### 4.4 Observed Outcome
  Only one outcome is revealed:

  `Y = T * Y1 + (1 - T) * Y0`

  ### 4.5 Blinded Data Split
  The final dataset is split into:

  - Training/Validation: `trial_train_val.csv`  
  - Blinded Test Set: `trial_test_blinded.csv`  

  No information about Y0, Y1, β, or δ is ever exposed.

  ---

  ## 5. Cheat-Secure Design

  The simulation ensures complete blinding:

  1. True effects (β and δ) are never exposed.  
  2. Counterfactuals (Y0, Y1) are discarded and cannot be recovered.  
  3. Only observed outcomes under randomized treatment are used.  
  4. No label leakage from the original dataset.  
  5. Metadata and identifiers are removed.  
  6. The test set remains entirely unseen during model development.

  This structure prevents reverse-engineering and mirrors real clinical trial constraints.

  ---

  ## 6. Machine Learning Models

  Two independent models are implemented to evaluate predictive performance.

  ### 6.1 Logistic Regression (Baseline Model)
  Script: `train_blinded_model.py`

  - Cross-validation ROC-AUC: ~0.879  
  - Blinded test ROC-AUC: ~0.871  
  - Brier score: ~0.143  

  ### 6.2 MLP Neural Network
  Script: `train_blinded_mlp.py`

  - Cross-validation ROC-AUC: ~0.862  
  - Blinded test ROC-AUC: ~0.853  
  - Brier score: ~0.154  

  Performance is consistent with expectations for nonlinear vs. linear classifiers on simulated clinical outcomes.

  ---

  ## 7. Running the Pipeline

  ### Step 1: Generate the simulated clinical trial data
  ```bash
  python src/simulate_trial.py
  ```

  ### Step 2: Train the Logistic Regression model
  ```bash
  python src/train_blinded_model.py
  ```

  ### Step 3: Train the MLP Neural Network
  ```bash
  python src/train_blinded_mlp.py
  ```

  Each script prints validation and blinded test performance.

  ---

  ## 8. Summary

  This project demonstrates:

  - End-to-end blinded clinical trial simulation  
  - Use of randomized treatment assignment  
  - Strict enforcement of blinding and no counterfactual leakage  
  - Implementation of potential-outcomes causal modeling  
  - Training of two predictive models with evaluation on a blinded test set  

  This pipeline closely reflects real clinical trial modeling workflows used in biotechnology and longevity research.

  ---

  ## 9. Contact

  Hemanth Kumar Kottapalli  
  Machine Learning Engineer  
  Submitted for Longevity InTime – Stage 1  
