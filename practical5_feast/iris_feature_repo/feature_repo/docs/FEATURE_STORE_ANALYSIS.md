# Feature Store Analysis

## 1. Elimination of Training-Serving Skew

Feast provides a centralized definition of features that can be used
during both training and online inference.

In this experiment, the original Iris measurements and engineered
features are defined in the feature repository and registered as
FeatureViews.

The same feature definitions can therefore be retrieved from the
offline store for historical training data and from the online store
for real-time inference.

This reduces the possibility of training-serving skew caused by
different feature calculation logic.

## 2. Feature Reusability

The engineered features created in this experiment include:

- sepal_area
- petal_area
- sepal_to_petal_length_ratio
- petal_length_bin

These features are registered in Feast and grouped into the
iris_engineered_features FeatureView.

The iris_feature_service provides a reusable collection of the
registered features.

This allows multiple machine learning models or applications to
reuse the same feature definitions instead of implementing the
feature engineering logic repeatedly.

## 3. Centralized Feature Governance

The feature repository provides a centralized location for defining
and managing features.

Feature definitions, entities, feature views, data sources, and
feature services are maintained in one repository.

This improves consistency and makes feature definitions easier to
inspect, maintain, and reuse across machine learning workflows.

## 4. Conclusion

This experiment demonstrated the use of Feast as a feature store for
the Iris dataset.

The experiment successfully demonstrated:

- Feature repository creation
- Entity definition
- FeatureView creation
- Feature registration
- Online feature materialization
- Online feature retrieval
- Historical feature retrieval
- Feature Service based feature reuse

The feature store provides a structured approach for managing
machine learning features and supporting both training and inference
workflows.
