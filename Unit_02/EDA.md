# Data exploration, preprocessing, labels, and dimensionality reduction

### Importance of Data

If you had to prioritize one of the following areas below to improve the performance of your neural network, which would have the most impact?

- [ ] A) Using the latest optimization algorithm
- [x] B) The quality and size of your data
- [ ] C) A deeper neural network
- [ ] D) A more advanced loss function

In your class project, how much time will typically be spent on data preparation and transformation?

- [ ] A) Less than half
- [x] B) More than half

The quality and size of your dataset will be much more impactful (in most cases) than the type of neural network you choose. A lot of time will be spent on data preprocessing, which is important to ensure that your neural network learns meaningful and true patterns.

# What is your data type?

The nature of your "inputs" (i.e., predictors or x) and "labels" (i.e., predictand or y) can help give you an idea about what type of neural network you will need to use:

- Pixel (i.e., grid cell or point value), e.g., artificial neural network (dense layers).
- Image-based (i.e., spatial region), e.g., convolutional neural network.
- Temporal (i.e., time series), e.g., recurrent neural network.

# Some questions to ask yourself about your data

- How many features (i.e., variables) should I use?
  - When starting on a new project, focus on a few features that have strong predictive power
  - As we add more dimensions or features, we also increase the computational demands and amount of training data needed to train skillful models. This is known as the curse of dimensionality (i.e., Hughes Phenomenon)
  - Various feature selection methods can be employed, e.g., variance threshold, univariate feature selection, recursive feature elimination, and sequential feature selection. [source](https://scikit-learn.org/stable/modules/feature_selection.html)
  - It may be wise not to use highly correlated variables. What new information would be provided by a variable with a strong correlation to an existing variable?
  - When dealing with autocorrelated data, the effective sample size is smaller than the actual number of samples due to a lack of independence among observations. 
>As a rough rule of thumb, your model should train on at least an order of magnitude more examples than trainable parameters [source](https://developers.google.com/machine-learning/data-prep)

- Is my data reliable, representative, or skewed? 
  - Reliability: Is your data free of errors? Make sure to visualize your data!
  - Feature representation: Are your features useful for learning?
  - Minimize skew: Does your training data distribution align with your test data distribution?
  - [Read More](https://developers.google.com/machine-learning/data-prep)

- Will I use direct labels or derived labels?
  - Machine learning is easier when your labels are well-defined.
  - Direct label is what you want to predict (e.g., tornado reports: did a tornado happen?).
  - Derived label is a proxy of what you want to predict (e.g., storm updraft rotation: did a tornado ___likely___ happen?).

- Do I have too much data or imbalanced data?
  - What happens if you have TOO MUCH data? Consider how you would sample your dataset to capture the population distribution sufficiently.
  - A data set with skewed class proportions is called imbalanced (e.g., thunderstorms that produce tornadoes).
  - Classes that make up a large proportion of the data set are called majority classes (e.g., non-severe thunderstorms).
  - Those that make up a smaller proportion are minority classes (e.g., severe thunderstorms).
  - Class imbalance can be handled using downsampling or class weights.
  - Downsampling (in this context) means training on a disproportionately low subset of the majority class examples.
  - Adding minority class weights means adding weight(s) to penalize poor performance on the minority class (and reduce focus on the majority class).
  - Upweighting can be done when dealing with too much data, and it is subsampled using a smaller subset of the total dataset, but weights are added to the majority class to simulate the larger relative frequency to the respective minority class.

- How should I split my data into train and test sets?
  - If your data doesn’t change much over time, random splits are okay!
  - If there is any autocorrelation or clustering within your data, random splitting is usually not a good idea. This is common when dealing with time series data. Perform a temporal split instead.
  - Seasonality, trends, or cyclical effects can be an issue! Consider removing these effects during preprocessing or using the date as a feature so the model can learn these effects.

- How and when should I transform and/or rescale my data?
  - Mandatory transformations are applied for data compatibility, e.g., converting non-numeric features into numeric classes (`{cold front: 1}`).
  - Optional quality transformations that may help the model perform better, e.g., normalized numeric features (most models perform better afterwards).
  - Transformations or rescaling can be done before or during training.
  - Before training is good when the population data statistics are needed. Full data statistics can be dangerous during training due to possible leakage of training or testing data, which can artificially inflate skill scores.
  - During training, it is useful when transformation/rescaling is easy and fast to iterate. If complicated, transforms per minibatch can slow training.
  - Standardization: centering the respective data to have a mean of zero and a standard deviation of one, e.g., z-score.
  - Normalization: scaling of data to a set range, such as from 0 to +1 or -1 to +1, by using the minimum and maximum values
  - Scaling to a range is a good choice when both of the following conditions are met: 1) You know the approximate upper and lower bounds on your data with few or no outliers, and 2) Your data is approximately uniformly distributed across that range.
  - If your data set contains extreme outliers, you might try feature clipping, which caps all feature values above (or below) a certain value to fixed values.
  - When using various features, normalizing or standardizing the variables is needed to ensure they have a similar numeric range. Otherwise, some features may have much larger magnitudes than others, limiting their contributions to the learning process. By normalizing/standardizing our features, we facilitate meaningful comparisons between features. This practice can also improve machine learning model convergence.
  - Strongly recommend normalizing a data set that has numeric features covering distinctly different numeric ranges.
  - Recommend normalizing a single numeric feature that covers a wide range.
  Transformations can also be done in other dimensions, e.g., space or two dimensions. Taking the square root of data can also tamper data distribution peaks (like log-scale).

- How do I encode my classes?
  - We can frame our problem as a binary classification (two classes; e.g., True or False, or Yes or No problems).
  - Classification problems can also be multiclass in nature.
  Continuous data could be grouped or binned into certain classes; thus, it can be converted to classes.
  - Bucketing: transform numeric (usually continuous) data to categorical data.
  Integer encoding involves assigning an integer value to each unique category. This method can work well for data with an underlying natural ordered relationship.
  For one-hot encoding, a binary variable (e.g., zero or one) is added at a certain position in each sample's vector, which is meant to represent the respective sample's class, e.g., [0, 0, 1] vs. [1, 0, 0].
  - Out of Vocab (OOV): Just as numerical data contains outliers, categorical data does, as well.
By using OOV, the system won't waste time training on each of those rare words.
  - Another option is to hash every string (category) into your available index space. Hashing often causes collisions, but you rely on the model learning some shared representation of the categories in the same index that works well for the given problem.
  - You can take a hybrid approach and combine hashing with a vocabulary. Use a vocabulary for the most important categories in your data, but replace the OOV bucket with multiple OOV buckets, and use hashing to assign categories to buckets.
  - An embedding layer is a learned transformation of your input data rather than a preprocessing step done beforehand. Instead of manually encoding categorical data (e.g., one-hot encoding), an embedding layer learns your input's dense, lower-dimensional representation during training.

---

# Recommended Order of Operations:

1. Filter or Fill Missing Values (e.g., imputation, interpolation)
2. Product Harmonization or Alignment (for multi-product problems)
3. Exploratory Data Analysis (e.g., autocorrelation, covariance, distribution)
4. Label Creation for Classification (if relevant, e.g., integer or one-hot encoding)
5. Train, Validation, and Test Data Split (including cross-validation strategy)
6. Feature Selection (e.g., wrapper methods, feature extraction)
7. Anomalies and Detrending (e.g., annual or seasonal, global or local)
8. Handle Outliers and Distribution Transformations (e.g., pruning, Box-Cox)
9. Standardization or Normalization (e.g., z-score, min-max scaling)

# Common data normalization, standardization, and transformations

- Min-Max Normalization: Scales the data to a fixed range, usually [0, 1].

$X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$

- Maximum Absolute Scaling: Rescales each feature by its maximum absolute value. It is useful for data that is already centered at zero.

$X' = \frac{X}{\max(\lvert X \rvert)}$

- Z-score Standardization: Standardizes the features by removing the mean and scaling to unit variance.

$X' = \frac{X - \mu}{\sigma}$, where ( $\mu$ ) is the mean and ( $\sigma$ ) is the standard deviation of the feature.

- Log Transformation: Reduces skewness in the data. It's applicable to data that follows a power law distribution.

$X' = \log(X + c)$, where c is a constant added to avoid a value of zero.

- Box-Cox Transformation: Used to stabilize variance and make the data more normally distributed.

When ( $\lambda$ $\neq$ 0 ):

$X' = \frac{X^{\lambda} - 1}{\lambda}$

When ( $\lambda$ = 0 ):

$X' = \log(X)$

Where $\lambda$ is a parameter that needs to be estimated from the data. The goal is to find a value that makes the transformed data as close to normally distributed as possible.

- Yeo-Johnson Transformation: Similar to Box-Cox but can handle zero and negative values.

When ( $X$ $\geq$ 0 ) and ( $\lambda$ $\neq$ 0 ):

$X' = \frac{((X + 1)^{\lambda} - 1)}{\lambda}$

When ( $X$ $\geq$ 0 ) and ( $\lambda$ = 0 ):

$X' = \log(X + 1)$

When ( $X$ < 0 ) and ( $\lambda$ $\neq$ 2 ):

$X' = -\frac{((-X + 1)^{2 - \lambda} - 1)}{2 - \lambda}$

When ( $X$ < 0 ) and ( $\lambda$ = 2 ):

$X' = -\log(-X + 1)$

- Quantile Transformation: Transforms the data to follow a uniform or normal distribution.

$X' = F^{-1}(F(X))$, where ( F ) is the cumulative distribution function (CDF) of the data and ( $F^{-1}$ ) is the inverse CDF of the desired distribution.

---

Open source resource: [sklearn](https://scikit-learn.org/0.16/modules/preprocessing.html#preprocessing)

 




