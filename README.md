# Optimizing an ML Pipeline in Azure

This project is part of my **Udacity Azure Machine Learning Nanodegree**. The idea was to solve the same classification problem in two different ways and see which one comes out ahead:

1. A hand-built **Scikit-learn Logistic Regression** model, with its hyperparameters tuned by **Azure HyperDrive**.
2. An **Azure AutoML** run that tries out different algorithms and hyperparameters on its own.

Once both were trained, I compared them to figure out which approach gave the better model.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Scikit-learn Pipeline](#scikit-learn-pipeline)
- [AutoML](#automl)
- [Pipeline Comparison](#pipeline-comparison)
- [Results Summary](#results-summary)
- [Future Work](#future-work)
- [Project Structure](#project-structure)
- [References](#references)

---

## Overview

The goal here is to build and tune an Azure ML pipeline with the Python SDK and a provided Scikit-learn model, then put it head-to-head against an Azure AutoML run on the exact same task.

It's a **binary classification problem**: predict whether a client will subscribe to a bank term deposit (`y = yes/no`), using data collected during a phone-based marketing campaign.

---

## Architecture

![Architectural Diagram](https://user-images.githubusercontent.com/68206315/115319967-d6fd3d00-a178-11eb-9c95-5e6bf5eca35e.png)

At a high level, one branch trains a Scikit-learn script and lets HyperDrive tune it, while the other branch runs an AutoML experiment. The two results are then compared so we can pick the winner.

---

## Dataset

The [Bank Marketing dataset](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) holds records from a direct marketing campaign that reached out to clients by phone.

| Attribute | Description |
| --- | --- |
| **Source** | Azure Open Datasets (Bank Marketing) |
| **Task** | Binary classification |
| **Target variable** | `y` — whether the client subscribed to a term deposit (yes/no) |
| **Features** | Client demographics, contact details, and campaign attributes |

---

## Scikit-learn Pipeline

Everything for this pipeline lives in [train.py](train.py). Here's what it does, step by step:

1. **Load the data** — The training data is pulled straight from the source URL into an unregistered `TabularDataset` with `from_delimited_files()`.
2. **Clean and prep** — Categorical columns are cleaned up and one-hot encoded, and the target is turned into a simple 0/1 label.
3. **Train the model** — A **Logistic Regression** classifier is trained with two hyperparameters worth tuning: `C` (inverse regularization strength) and `max_iter` (maximum iterations).
4. **Tune with HyperDrive** — **Azure HyperDrive** then searches for the settings that push **accuracy** as high as possible.

### Hyperparameter Search Space

| Hyperparameter | Type | Values / Range |
| --- | --- | --- |
| `max_iter` | Discrete (`choice`) | 25, 50, 100, 200 |
| `C` | Continuous (`uniform`) | 0.08 – 0.1 |

### Why Random Sampling?

I went with **random sampling** to pick hyperparameter values from the search space.

The nice thing about random sampling is that it handles both discrete and continuous hyperparameters, and it plays well with early stopping. That means you get a good spread of the search space without burning compute on runs that clearly aren't going anywhere.

### Why the Bandit Policy?

For early stopping I used a **Bandit policy**. It leaves the first 5 runs alone (`delay_evaluation`) so nothing gets cut off too early, then checks in every 2 iterations (`evaluation_interval`) and kills any run whose accuracy drops outside the top 10% of the best run so far.

In practice this saves a lot of time — instead of letting weak runs finish, the policy stops them once they fall too far behind the leader, without hurting the quality of the final model.

### Best HyperDrive Result

The best run landed at an accuracy of **0.91492**, using **100 maximum iterations** and a regularization value of **C ≈ 0.08703**.

---

## AutoML

I ran the AutoML experiment on local compute with 5-fold cross-validation and let it loose on a whole range of algorithms and hyperparameter combinations.

The model it settled on was a **VotingEnsemble**, which blends several models together using weighted averaging.

| Attribute | Value |
| --- | --- |
| **Best algorithm** | VotingEnsemble |
| **Accuracy** | 0.91671 |
| **Cross-validation folds** | 5 |
| **Ensembled iterations** | (1, 0, 22, 28, 19, 15, 21) |

Across the algorithms that made it into the ensemble, the regularization strengths and maximum iterations were the parameters that mattered most.

---

## Pipeline Comparison

| Criteria | HyperDrive (Logistic Regression) | AutoML (VotingEnsemble) |
| --- | --- | --- |
| **Accuracy** | 0.91492 | **0.91671** |
| **Algorithm selection** | Manual (Logistic Regression) | Automated |
| **Hyperparameter tuning** | Automated (defined search space) | Automated |
| **Model type** | Single model | Ensemble of models |

The gap is small, but the **AutoML VotingEnsemble came out on top** of the HyperDrive-tuned Logistic Regression. The workflows are pretty different too: with HyperDrive I had to decide on the algorithm and the search space up front, while AutoML figured out both the algorithm and the hyperparameters by itself.

My takeaway is that the ensemble is doing the heavy lifting here — pooling the predictions of several models usually gives you something a bit more robust and accurate than any single tuned model on its own.

---

## Results Summary

| Model | Method | Accuracy |
| --- | --- | --- |
| Logistic Regression | HyperDrive | 0.91492 |
| VotingEnsemble | AutoML | **0.91671** |

---

## Future Work

A few things I'd like to try next:

- **Widen the hyperparameter search** — Test more combinations of `C` and `max_iter` and bump up the number of HyperDrive runs to cover more ground.
- **Move AutoML to remote compute** — Running AutoML on a proper cluster with the full feature set (instead of the trimmed-down local run) should squeeze out a bit more accuracy.
- **Deal with class imbalance** — Take a closer look at how the target classes are split and try resampling or class weights to help the model catch the minority class.
- **Do more feature engineering** — Build some new features and trim the ones that don't help, and see if that moves the needle.

---

## Project Structure

```text
.
├── README.md               # Project documentation
├── train.py                # Scikit-learn training script (used by HyperDrive)
└── udacity-project.ipynb   # Notebook orchestrating HyperDrive and AutoML runs
```

---

## References

- [Bank Marketing Dataset](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv)
- [Azure Machine Learning Documentation](https://learn.microsoft.com/azure/machine-learning/)
- [HyperDrive (Hyperparameter Tuning)](https://learn.microsoft.com/azure/machine-learning/how-to-tune-hyperparameters)
- [Azure AutoML](https://learn.microsoft.com/azure/machine-learning/concept-automated-ml)

---

# Rubric Q&A

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**

This dataset contains records from a phone-based bank marketing campaign, including client demographics, contact details, and campaign attributes. We seek to predict whether a client will subscribe to a bank term deposit (`y = yes/no`) — a binary classification problem.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

The best performing model was an Azure AutoML **VotingEnsemble** with an accuracy of **0.91671**, which narrowly outperformed the HyperDrive-tuned Scikit-learn Logistic Regression (accuracy **0.91492**).

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

The pipeline is defined in `train.py`. It loads the Bank Marketing data from the source URL into a `TabularDataset` using `from_delimited_files()`, cleans and one-hot encodes the categorical columns, and converts the target into a 0/1 label. A **Logistic Regression** classifier is then trained with two tunable hyperparameters — `C` (inverse regularization strength) and `max_iter` (maximum iterations). **Azure HyperDrive** searches the hyperparameter space to maximize **accuracy**, sampling `max_iter` from a discrete set (25, 50, 100, 200) and `C` from a uniform range (0.08 – 0.1).

**What are the benefits of the parameter sampler you chose?**

I chose **random sampling**. It supports both discrete and continuous hyperparameters, provides a good spread across the search space without exhaustively testing every combination, and works well with early stopping — so compute isn't wasted on runs that clearly aren't improving.

**What are the benefits of the early stopping policy you chose?**

I used a **Bandit policy**. It leaves the first 5 runs alone (`delay_evaluation`) so nothing is cut off prematurely, then checks every 2 iterations (`evaluation_interval`) and terminates any run whose accuracy falls outside the top 10% of the best run so far. This saves significant time by stopping weak runs early without hurting the quality of the final model.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

AutoML ran on local compute with 5-fold cross-validation and selected a **VotingEnsemble** (accuracy **0.91671**), which blends several models using weighted averaging over the ensembled iterations (1, 0, 22, 28, 19, 15, 21). Across the ensembled algorithms, the regularization strengths and maximum iterations were the most influential hyperparameters.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

The two models were very close in accuracy: the HyperDrive Logistic Regression reached **0.91492** while the AutoML VotingEnsemble reached **0.91671** — a small edge to AutoML. The architectures differ significantly: HyperDrive tuned a single, manually chosen model (Logistic Regression) over a predefined search space, whereas AutoML automatically selected both the algorithm and hyperparameters, ultimately producing an ensemble of models. The likely reason for the difference is that the ensemble pools predictions from several models via weighted averaging, which tends to be more robust and slightly more accurate than any single tuned model.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

- **Widen the hyperparameter search** — Test more combinations of `C` and `max_iter` and increase the number of HyperDrive runs to explore more of the search space, which may uncover better settings.
- **Move AutoML to remote compute** — Running AutoML on a full cluster with the complete feature set (instead of a trimmed-down local run) could squeeze out additional accuracy.
- **Address class imbalance** — The target classes are likely imbalanced; applying resampling or class weights would help the model better detect the minority class and improve metrics beyond raw accuracy.
- **More feature engineering** — Building new features and dropping unhelpful ones may give the models a stronger signal to learn from.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

The compute cluster was deleted in code at the end of the notebook via `compute_target.delete()`, so no manual cleanup image is required.
