## Description

This Markdown file contains lecture notes for [Data-Centric AI course from MIT](https://dcai.csail.mit.edu/).

Laboratory assignments can be found in labs directory. Labs are cloned from https://github.com/dcai-course/dcai-lab

Table of content:
* [Lecture 1 Data-Centric AI vs. Model-Centric AI](#lecture-1-data-centric-ai-vs-model-centric-ai)
  * Data-Centric AI vs. Model-Centric AI
  * Examples of data-centric AI
* [Lecture 2 Label Errors and Confident Learning](#lecture-2-label-errors-and-confident-learning)
  * Confident learning (CL) 
* [Lecture 3 Dataset Creation and Curation](#lecture-3-dataset-creation-and-curation)
  * Selection Bias
  * Labeling data with crowdsourced workers
  * Curating a dataset labeled by multiple annotators
* [Lecture 4 Data-centric Evaluation of ML Models](#lecture-4-data-centric-evaluation-of-ml-models)
  * Evaluation of ML models
  * Underperforming Subpopulations
  * Why did my model get a particular prediction wrong?
  * Quantifying the influence of individual datapoints on a model
* [Lecture 5 Class Imbalance, Outliers, and Distribution Shift](#lecture-5-class-imbalance-outliers-and-distribution-shift)
  * Class imbalance
  * Outliers
  * Distribution shift
* [Lecture 6 Growing or Compressing Datasets](#lecture-6-growing-or-compressing-datasets)
  * Active learning
  * Core-set selection
* [Lecture 7 Interpretability in Data-Centric ML](#lecture-7-interpretability-in-data-centric-ml)
  * Introduction to Interpretable ML
  * What are interpretable features really
  * How do we get interpretable features?
* [Lecture 8 Encoding Human Priors: Data Augmentation and Prompt Engineering](#lecture-8-encoding-human-priors-data-augmentation-and-prompt-engineering)
  * Human Priors to Augment Training Data
  * Human Priors at Test-Time (LLMs)
* [Lecture 9 Data Privacy and Security](#lecture-9-data-privacy-and-security)
  * Defining security: security goals and threat models
  * Membership inference attacks
  * Metrics-based attacks
  * Data extraction attacks
  * Defending against privacy leaks: empirical defenses and evaluation
  * Differential privacy

## Lecture 1 Data-Centric AI vs. Model-Centric AI

### Data-Centric AI vs. Model-Centric AI

* Data-centric approach tends to overperform the model-centric approach in real-life scenarios
* Tesla is considered to have more advanced autonomous driving systems than similar competitors because of their Data engine. When they notice a problem with their self-driving system (eg. a problem in tunnels), they tend to extend the dataset by collecting more frames of driving in the tunnel.
![Tesla Data Engine](https://dcai.csail.mit.edu/lectures/data-centric-model-centric/dataengine.png)

* In practice the best approach is an iterative process of combining data-centric and model-centric approaches in the following manner (one can iterate Steps 3 and 4 multiple times):
1. Explore the data, fix fundamental issues, and transform it to be ML-appropriate.
2. Train a baseline ML model on the properly formatted dataset.
3. Utilize this model to help us improve the dataset (techniques taught in this class).
4. Try different modelling techniques to improve the model on the improved dataset and obtain the best model.

### Examples of data-centric AI
* Outlier detection and removal (handling abnormal examples in the dataset)
* Error detection and correction (handling incorrect values/labels in the dataset)
* Establishing consensus (determining truth from many crowdsourced annotations)
* Data augmentation (adding examples to data to encode prior knowledge)
* Feature engineering and selection (manipulating how data are represented)
* Active learning (selecting the most informative data to label next)
* Curriculum learning (ordering the examples in the dataset from easiest to hardest)

## Lecture 2 Label Errors and Confident Learning

### Confident learning (CL) 

Even commonly used datasets (MNIST, CIFAR, and ImageNet) have label issues. Examples of label issues in those datasets can be found at https://labelerrors.com. 

Confident learning (CL) has emerged as a subfield within supervised learning and weak supervision to:
* Characterize label noise
* Find label errors
* Learn with noisy labels
* Find ontological issues

CL is based on the principles of [pruning noisy data](https://arxiv.org/abs/1705.01936) (as opposed to [fixing label errors](https://en.wikipedia.org/wiki/Error_correction_code) or [modifying the loss function](https://papers.nips.cc/paper_files/paper/2013/hash/3871bd64012152bfb53fdf04b401193f-Abstract.html)), [counting to estimate noise](https://www.semanticscholar.org/paper/Business-Optimization-St-Hilaire/b00ce7c4d8a29071dd70ef9d944bdc73e53d4f78) (as opposed to [jointly learning noise rates](https://arxiv.org/abs/1406.2080) during training), and [ranking examples](https://en.wikipedia.org/wiki/Learning_to_rank) to train with confidence (as opposed to [weighting by exact probabilities](https://www.semanticscholar.org/paper/Training-deep-neural-networks-using-a-noise-layer-Goldberger-Ben-Reuven/bc550ee45f4194f86c52152c10d302965c3563ca)). Here, we generalize CL, building on the assumption of [Angluin and Laird's classification noise process](https://homepages.math.uic.edu/~lreyzin/papers/angluin88b.pdf), to directly estimate the joint distribution between noisy (given) labels and uncorrupted (unknown) labels.
![Confident learning process](https://dcai.csail.mit.edu/lectures/label-errors/confident_learning_digram_final.jpg)

$\tilde{y}$ - noisy data sample
<br>
$y*$ - correct label (not known)

This is the ideal matrix when correct labels $y*$ are known. In reality, we want to estimate this matrix by making predictions of the model using noisy labels. This process can be described in the following steps:
* Pass noisy images through the model (use out-of-sample data, the model should be trained on different images - we can use k-fold cross-validation
* Analyse the distribution of output probabilities of the model - calculate the average confidence score of the model for each class and set it as a threshold for each class (eg. $t_{dog}=0.7$ is the threshold for class dog at the level of 0.7 confidence score of the model)
* Our noisy labels $\tilde{y}$ become $y*$ and model predictions become $\tilde{y}$
* Choose samples from the dataset above the given threshold defined two steps before filling the matrix. Samples below the threshold are outliers
* Calculate joint distribution (normalize) from filled values.

Above process can be performed in one line of code using [cleanlab library](https://github.com/cleanlab/cleanlab) using [find_label_issues function](https://docs.cleanlab.ai/stable/cleanlab/classification.html#cleanlab.classification.CleanLearning.find_label_issues):

```python
from cleanlab.filter import find_label_issues

ordered_label_issues = find_label_issues(
    labels=labels,
    pred_probs=pred_probs,  # out-of-sample predicted probabilities
    return_indices_ranked_by="self_confidence",
)
```

Self-confidence is the measure how much confident is the model in its prediction for given label. If it is equal to 1, then the model is 100% confident that given label is the right label. For multi-class classification this is the same as cross-enropy loss in some settings. By sorting values from smallest the first values are most likely to be label erorrs.

Another option is to use normalized margin. For each sample we are not taking into consideration only confidence score for the predicted class but we subtract from this confidence score the second biggest confidence score. 

Suppose we have have 3 classes: dog, cat, fox. We have 3 samples and model predicts for them following vectors given in the table, then values of self-confidence and normalized margin are as following:
| # Sample | Model prediction vector | Self-confidence value | Normalized margin value |
|----------|-------------------------|-----------------------|-------------------------|
| 1        | [0.7, 0.1, 0.2]         | 0.7                   | 0.5                     |
| 2        | [0.4, 0.3, 0.3]         | 0.4                   | 0.1                     |
| 3        | [0.8. 0.05, 0.15]       | 0.8                   | 0.65                    |

In both cases (self-confidence and normalized margin), when we sort by values from smalles we get feedback that for Sample 2 there is biggest probability of label error.

Confidence Learning works as long as exaples in class *i* are labeled *i* more time than any other class. This means that those are realistic sufficient conditions (allowing sigfinicant error in all model outputs).

**Benefits of Confident Learning**:

Unlike most machine learning approaches, confident learning requires no hyperparameters. We use cross-validation to obtain predicted probabilities out-of-sample. Confident learning features a number of other benefits. CL:
* directly estimates the joint distribution of noisy and true labels
* works for multi-class datasets
* finds the label errors (errors are ordered from most likely to least likely)
* is non-iterative (finding training label errors in ImageNet takes 3 minutes)
* is theoretically justified (realistic conditions exactly find label errors and consistent estimation of the joint distribution)
does not assume randomly uniform label noise (often unrealistic in practice)
only requires predicted probabilities and noisy labels (any model can be used)
* does not require any true (guaranteed uncorrupted) labels
* extends naturally to multi-label datasets
* is free and open-sourced as the cleanlab Python package for characterizing, finding, and learning with label errors.

**The Principles of Confident Learning**

CL builds on principles developed across the literature dealing with noisy labels:

* Prune to search for label errors, using soft-pruning via loss-reweighting, to avoid the convergence pitfalls of iterative re-labeling.
* Count to train on clean data, avoiding error-propagation in learned model weights from reweighting the loss with imperfect predicted probabilities.
* Rank which examples to use during training, to allow learning with unnormalized probabilities or SVM decision boundary distances, building on well-known robustness findings of PageRank (Page et al., 1997) and ideas of curriculum learning in [MentorNet (Jiang et al.,2018)](https://arxiv.org/abs/1712.05055).

A Bigger Neural Network will usually outperform a smaller NN, but a bigger NN will tend to overfit much stronger than a smaller one. As a result in real-world scenarios, smaller models with lower accuracy on our dataset can overperform more complex models on real data. 

This is the reason why we should ensure that our test set is of good quality and does not have errors in labels. Then even if our train set is noisy, we will notice that a smaller model can perform better on real-world data. Distribution of the test set should be as close as possible to the real data we expect to have.


## Lecture 3 Dataset Creation and Curation

### Selection Bias
* Sometimes it is beneficial to combine classes where the model mistakes them for another. If it will be required to distinguish between classes, maybe a smaller or other model would perform better in distinguishing between them
* When data is evenly distributed for few classes and there are very few examples of other classes it may be beneficial to combine classes with a small amount of data and create a cluttered class so that our model will perform well at data where we have enough data
* Spurious Correlation - Neural Networks are taking shortcuts when it is possible. If learning the background of a given object is easier for the model than learning the object itself, they will use the background for prediction


* Selection bias - training data is not fully representative of the real-world data or deployment conditions (also called distribution shift). Selection bias has many cases such as:
  * Time - e.g... on stock data, we can not use future data where the model will be used. To handle this we should use the most recent data as a test set
  * Over-filtering - if we over-curate our data, the model will not work well on some hard examples, which we have filtered, because we thought they would be too hard to learn for the model. If this data can occur in production, we should not filter this data
  * Rare events - it is hard to collect enough data for rare events
  * Convenience - we gather data that is easy to collect. Eg. survey only our friends. To deal with rare events we can over-sample rare events data in a test set
  * Location - we use only location-specific data eg. data from 3 hospitals, while the model should be used in all hospitals in the country. To handle it we should hold out all data from some specific location not in the train set. 
How to estimate how much data we need to improve our model to have some level of accuracy (e.g. 90%)?
* We can use our current dataset and subsample it iteratively training the model on the bigger and bigger parts of the data
* We calculate accuracy for each subset and plot a graph of accuracy vs. dataset size
* Since some subsets of the data can be harder or easier than the remaining data, we should use cross-validation and average results
* Then we extrapolate based on our results
* The biggest problem in this approach is that classical ML models eg linear regression can not accurately extrapolate in the range bigger than observed in one
* That's why commonly we tend to use empirical knowledge that this curve can be approximated by the equation:
  
    $log(error)= -a \cdot log(n) + b$

We can estimate scalar parameters $a$, $b$ of this simple model by minimizing the MSE over observations.

### Labeling data with crowdsourced workers

**Setting of labeling data**
* For data labeling it is good practice to have one or more annotators annotate each example
* On the other hand annotations are time-consuming and costly, thus we may not want to duplicate the work of another annotator
* To allocate labeling efforts most wisely we would like to collect more annotations for harder examples, while easier examples can be annotated by one or two annotators
* To grade the quality of annotator annotations, when we have multiple annotators to annotate each image we can check, how often the annotator disagrees with other annotators (e.g.. we can have 3 annotators, and some examples are annotated by all of them, some by two of them and some by only one annotator)
* We can slip a few 'quality control' examples into the dataset, for which we already know the ground-truth label. The subset of data provided to each annotator for labeling should contain some quality control examples
* Potential problems in data labeling are:
  * Low accuracy of annotators
  * Copycat (one annotator can have multiple accounts) - then even if we have multiple annotators to annotate each image, it can be labeled wrong in all examples

### Curating a dataset labeled by multiple annotators
**Multi-annotators estimates**:
* Consensus label = single best label (use majority vote)
* Confidence in consensus - how likely is it wrong? Agreement between annotators can be our confidence score
* Quality score of annotator - overall accuracy of the labeler. We can measure it by averaging how often the label given by the annotator is consistent with the consensus label. For this average, we count only examples that are annotated by graded annotators and are annotated by at least two annotators.

A better method for analyzing such data (CROWDLAB algorithm - Classifier Refinement Of croWDsourced LABels):
* How to deal with ties between annotators? We can use trained classifiers to break ties 
* When we have only one annotation for a given image we can use our classifier for consensus. If the annotator and model give the same label with high confidence it is probably the correct label
* Use weighted average for consensus label - a bad annotator can have a lower weight than the best annotator and model weight can be modified accordingly to its performance on given data
* The bigger the number of annotators that labeled the sample is the bigger the annotator's impact on the consensus label (smaller impact of the classifier)

In practice it can be an iterative process:
* We can start with a majority vote for the consensus label
* Train a classifier on consensus labels
* Use the trained classifier to break ties
* Use Crowdlab algorithm to improve labels
* Retrain classifier on new consensus labels
* Eventually they will all converge (labels will not be changed in the next iterations)

## Lecture 4 Data-centric Evaluation of ML Models

Most Machine Learning applications involve these steps:

1. Collect data and define the appropriate ML task for our application.
2. Explore the data to see if it exhibits any fundamental problems.
3. Preprocess the data into a format suitable for ML modeling.
4. Train a straightforward ML model that is expected to perform reasonably.
5. **Investigate shortcomings of the model and the dataset.**
6. Improve the dataset to address its shortcomings.
7. Improve the model (architecture search, regularization, hyperparameter tuning, ensembling different models).
8. Deploy the model and monitor subsequent data for new issues.

### Evaluation of ML models
Common pitfalls when evaluating models include:

* Failing to use truly held-out data (data leakage). For unbiased evaluation without any overfitting, the examples used for evaluation should only be used for computing evaluation scores, not for any modeling decisions like choosing: model parameters/hyperparameters, which type of model to use, which subset of features to use, how to preprocess the data etc.
* Some observed labels may be incorrect due to annotation errors.
* Reporting only the average evaluation score over many examples may under-represent severe failure cases for rare examples/subpopulations.

### Underperforming Subpopulations
Another problem in ML models is underperforming subpopulations. Subpopulation/data slice/cohorts/subgroup is a subset of the dataset that shares common characteristics. Examples of subgroups are:
* data captured via one sensor vs. another, one location vs. another
* factors in human-centric data like race, gender, socioeconomics, age, etc.

Model prediction should not depend on which slice a data point belongs to. The key challenge is that even if we delete this information from feature values directly it can still be correlated with other features influencing model predictions.

To boost performance for a particular slice we can: 
* Over-sample or up-weight examples from minority subgroup that is receiving poor predictions
* Collect additional data from the subgroup of interest. To assess whether this is a promising approach: we can re-fit our model to many alternative versions of our dataset in which we have down-subsampled this subgroup to varying degrees, and then extrapolate the resulting model performance that would be expected if we had more data from this subgroup
* Measure or engineer additional features that allow our model to perform better for a particular subgroup. Sometimes the provided features in the original dataset may bias results for a particular subgroup. Consider the example of classifying if a customer will purchase some product or not, based on customer & product features. Here predictions for young customers may be worse due to less available historical observations. We could add an extra feature to the dataset specifically tailored to improve predictions for this subgroup such as: 'Popularity of this product among young customers'.

**Discovering underperforming subpopulations**

Some data may not have obvious slices, for example, a collection of documents or images. In this case, how can we identify subgroups where our model underperforms?

Here is one general strategy:
1. Sort examples in the validation data by their loss value, and look at the examples with high loss for which our model is making the worst predictions (Error Analysis).
2. Apply clustering to these examples with high loss to uncover clusters that share common themes amongst these examples.

Many clustering techniques only require that we define a distance metric between two examples. By inspecting the resulting clusters, we may be able to identify patterns that our model struggles with. Step 2 can also use clustering algorithms that are label or loss-value aware, which is done in the Domino slice discovery method pictured below.
![Domino slice discovery method](https://dcai.csail.mit.edu/lectures/data-centric-evaluation/slicediscovery.png)

### Why did my model get a particular prediction wrong?

Reasons a classifier might output an erroneous prediction for example:
1. The given label is incorrect (and our model made the right prediction).
2. This example does not belong to any of the classes (or is fundamentally not predictable, e.g. a blurry image).
3. This example is an outlier (there are no similar examples in the training data).
4. This type of model is suboptimal for such examples. To diagnose this, we can up-weight this example or duplicate it many times in the dataset, and then see if the model is still unable to correctly predict it after being retrained. This scenario is hard to resolve via dataset improvement, instead try: fitting different types of models, hyperparameter tuning, and feature engineering.
5. The dataset contains other examples with (nearly) identical features that have a different label. In this scenario, there is little we can do to improve model accuracy besides measuring additional features to enrich the data. Calibration techniques may be useful to obtain more useful predicted probabilities from our model.

Recommended actions to construct a better train/test dataset under the first three scenarios above include:
1. Correct the labels of incorrectly annotated examples.
2. Toss unpredictable examples and those that do not belong to any of the classes (or consider adding an 'Other ' class if there are many such examples).
3. Toss examples that are outliers in training data if similar examples would never be encountered during deployment. For other outliers, collect additional training data that looks similar if we can. If not, consider a data preprocessing operation that makes outliers' features more similar to other examples (e.g. quantile normalization of a numeric feature, or deleting a feature). We can also use Data Augmentation to encourage our model to be invariant to the difference that makes this outlier stand out from other examples. If these options are infeasible, we can emphasize an outlier in the dataset by up-weighting it or duplicating it multiple times (perhaps with slight variants of its feature values). To catch outliers encountered during deployment, include Out-of-Distribution detection in our ML pipeline.
4. The type of model we are using is suboptimal for such examples. To diagnose we can do the following:
* Up-weight similar examples or duplicate them many times in the dataset
* Retrain model
* See if the new model can classify this example correctly
5. Dataset has other examples with (nearly) identical features but different labels. In this case recommended actions are:
* Defining classes more distinctly
* Measure extra features to enrich the data

### Quantifying the influence of individual data points on a model

How would my ML model change if I retrain it after omitting a particular datapoint $(x, y)$ from the dataset?

This question is answered by the influence function $I(x)$. Many variants of influence quantify slightly different things, depending on how we define change. For instance, the change in the model's predictions vs. the change in its loss (i.e. predictive performance), is typically evaluated over held-out data. Here we focus on the latter type of change.

The above is called **Leave-one-out (LOO) influence**, but another form of influence exists called the **Data Shapely value** which asks: What is the LOO influence of datapoint $(x, y)$ in any subset of the dataset that contains $(x, y)$? Averaging this quantity over all possible data subsets leads to an influence value that may better reflect the value of $(x, y)$ in broader contexts. For instance, if there are two identical data points in a dataset where omitting both severely harms model accuracy, LOO influence may still conclude that neither is too important (unlike the Data Shapely value).

Influence reveals which data points have the greatest impact on the model. For instance, correcting the label of a mislabeled data point with high influence can produce much better model improvement than correcting a mislabeled data point that has low influence. $I(x)$ can also be used to assign literal value to data as illustrated in the following figure:
![$I(x)$ used to assign literal value to data](https://dcai.csail.mit.edu/lectures/data-centric-evaluation/datavaluation.png)

Unfortunately, influence can be expensive to compute for an arbitrary ML model. For an arbitrary black-box classifier, we can **approximate influence via these Monte-Carlo sampling steps**:
1. Subsample $T$ different data subsets $D_t$ from the original training dataset (without replacement).
2. Train a separate copy of the model $M_t$ on each subset $D_t$ 
 and report its accuracy on held-out validation data $a_t$.
3. To assess the value of a datapoint $(x_i, y_i)$, compare the average accuracy of models for those subsets that contained $(x_i, y_i)$ vs. those that did not. Accuracy here could be replaced by any other loss of interest.

For special families of models, we can efficiently compute the exact influence function. In a regression setting where we use a linear regression model and the mean squared error loss to evaluate predictions, the fitted parameters of the trained model are a closed-form function of the dataset. Thus for linear regression, the LOO influence $I(x)$ can be calculated via a simple formula and is also known as Cook's Distance in this special case.

In classification, the influence function can be computed in reasonable $O(n\:log\:n)$ time for a K Nearest Neighbors (KNN) model. For the valuation of unstructured data, a general recipe is to use a pre-trained neural network to embed all the data, and then apply a KNN classifier on the embeddings, such that the influence of each data point can be efficiently computed. These two steps are illustrated in the following figure:
![KNN classifier using pretrained NN extracted features](https://dcai.csail.mit.edu/lectures/data-centric-evaluation/embedneighbors.png)

Reviewing Influential Samples:
* Influence reveals which data points have greatest impact on the model
* Correcting mislabeled datapoit with high influence can boost model accuracy more than correcting a mislabeled data point with low influence
* Finding mislabeled data may be hard sorting only by influence instead of using confident learning as well

## Lecture 5 Class Imbalance, Outliers, and Distribution Shift

### Class imbalance

Many real-world classification problems have the property that certain classes are more prevalent than others. For example:
* Medical diagnosis (e.g. COVID infection): among all patients, only 10% might have COVID
* Fraud detection: among all credit card transactions, fraud might make up 0.2% of the transactions
* Manufacturing defect classification: different types of manufacturing defects might have different prevalence
* Self-driving car object detection: different types of objects have different prevalence (cars vs. trucks vs. pedestrians)

**Evaluation metrics**

If we are splitting a dataset into train/test splits, make sure to use stratified data splitting to ensure that the train distribution matches the test distribution (otherwise, we are creating a distribution shift) problem.

With imbalanced data, standard metrics like accuracy might not make sense. For example, a classifier that always predicts 'NOT FRAUD' would have 99.8% accuracy in detecting credit card fraud.

There is no one-size-fits-all solution for choosing an evaluation metric: the choice should depend on the problem. For example, an evaluation metric for credit card fraud detection might be a weighted average of the precision and recall scores (the **F-beta score**), with the weights determined by weighing the relative costs of failing to block a fraudulent transaction and incorrectly blocking a genuine transaction:

$F_\beta=(1+\beta^2 ) \cdot\frac{Precision \cdot Recall}{\beta^2 \cdot Precision + Recall}$

where:

$Precision=\frac{TP}{TP+FP}$

$Recall=\frac{TP}{TP+FN}$

**Techniques for class imbalance**:
* **Sample weights**. Many models can be fit to a dataset with per-sample weights. Instead of optimizing an objective function that's a uniform average of per-datapoint losses, this optimizes a weighted average of losses, putting more emphasis on certain data points. While simple and conceptually appealing, this often does not work well in practice. For classifiers trained using mini-batches, using sample weights results in varying the effective learning rate between mini-batches, which can make learning unstable.
* **Over-sampling**. Related to sample weights, we can simply replicate data points in the minority class, even multiple times, to make the dataset more balanced. In simpler settings (e.g., least-squares regression, this might be equivalent to sample weights), in other settings (e.g., training a neural network with mini-batch gradient descent), this is not equivalent and often performs better than sample weights. This solution is often unstable, and it can result in overfitting.
* **Under-sampling**. Another way to balance a dataset is to remove data points from the majority class. While discarding data might seem unintuitive, this approach can work surprisingly well in practice. In some situations, it can result in throwing away a lot of data when we have highly imbalanced datasets, resulting in poor performance.
* **SMOTE** (Synthetic Minority Oversampling Technique). Rather than over-sampling by copying data points, we can use dataset augmentation to create new examples of minority classes by combining or perturbing minority examples. The SMOTE algorithm is sensible for certain data types, where interpolation in feature space makes sense, but doesn't make sense for certain other data types: averaging pixel values of one picture of a dog with another picture of a dog is unlikely to produce a picture of a dog. Depending on the application, other data augmentation methods could work better.
* **Balanced mini-batch training**. For models trained with mini-batches, like neural networks, when assembling the random subset of data for each mini-batch, we can include data points from minority classes with higher probability, such that the mini-batch is balanced. This approach is similar to over-sampling, and it does not throw away data.
* **Synthetic data** - we can use data augmentation or generate synthetic examples of the minority classes.

These techniques can be combined. For example, the SMOTE authors note that the combination of SMOTE and under-sampling performs better than plain under-sampling.

### Outliers

Outliers are data points that differ significantly from other data points. Causes include:
* Errors in measurement (e.g., a damaged air quality sensor)
* Bad data collection (e.g., missing fields in a tabular dataset)
* Malicious inputs (e.g., adversarial examples)
* Rare events (statistical outliers, e.g., an albino animal in an image classification dataset)

Outlier identification is of interest because outliers can cause issues during model training, at inference time, or when applying statistical techniques to a dataset. Outliers can harm model training, and certain machine learning models (e.g., vanilla SVM) can be particularly sensitive to outliers in the training set. A model, at deployment time, may not produce reasonable output if given outlier data as input (a form of distribution shift). If data has outliers, data analysis techniques might yield bad results.

Once found, what do we do with outliers? It depends. For example, if we find outliers in the training set, we don't want to blindly discard them: they might be rare events rather than invalid data points. We could, for example, have a domain expert manually review outliers to check whether they are rare data or bad data.

Tasks related to outliers:
* **Outlier detection**. In this task, we are not given a clean dataset containing only in-distribution examples. Instead, we get a single un-labeled dataset, and the goal is to detect outliers in the dataset, data points that are unlike the others. This task comes up, for example, when cleaning a training dataset that is to be used for ML.
* **Anomaly detection**. In this task, we are given an unlabeled dataset of only in-distribution examples. Given a new data point not in the dataset, the goal is to identify whether it belongs to the same distribution as the dataset. This task comes up, for example, when trying to identify whether a data point, at inference time, is drawn from the same distribution as a model's training set. The difference between anomaly detection and binary classification is that for anomaly detection we don't have our anomalies. We have our data distribution and we have to tell if a new sample belongs to this distribution. The difference between outlier detection and anomaly detection is that for outlier detection we don't know what is our distribution (no training data). We just have a dataset and we want to diagnose if some samples are outliers.

**Identifying outliers**

Outlier detection is a heavily studied field, with many algorithms and lots of published research. Some used techniques are:
* **Tukey's fences** (for tabular data). A simple method for scalar real-valued data. If $Q_1$ and $Q_3$ are the lower and upper quartiles, then this test says that any observation outside the following range is considered an outlier: $[Q_1 - k(Q_1-Q_3), Q_3+k(Q_3-Q_1)]$ . A multiplier of $k=1.5$ was proposed by John Tukey. Range between $Q_1$ and $Q_3$ is also called Inter-Quartile Range (IQR) thus $k$ multiplayer is multiplied by IQR
* **Z-score** (for tabular data). The Z-score is the number of standard deviations by which a value is above or below the mean. For one-dimensional or low-dimensional data, assuming a Gaussian distribution of data: calculate the Z-score as $z_i = \frac{x_1-\mu}{\sigma}$, where $\mu$ is the mean of all the data and $\sigma$ is the standard deviation. An outlier is a data point that has a high-magnitude Z-score, $\left \| z_i \right \| > z_{thr}$. A commonly used threshold is $z_{thr}=3$. We can apply this technique to individual features as well.
* **Isolation forest** (for tabular data). This technique is related to decision trees. Intuitively, the method creates a 'random decision tree' and scores data points according to how many nodes are required to isolate them. The algorithm recursively divides (a subset of) a dataset by randomly selecting a feature and a split value until the subset has only one instance. The idea is that outlier data points will require fewer splits to become isolated.
* **KNN distance** (for tabular and unstructured data e.g. images). In-distribution data is likely to be closer to its neighbors. We can use the mean distance (choosing an appropriate distance metric, like cosine distance) to a datapoint's k nearest neighbors as a score. For high-dimensional data like images, we can use embeddings from a trained model and do KNN in the embedding space.
* **Reconstruction-based methods**. Autoencoders are generative models that are trained to compress high-dimensional data into a low-dimensional representation and then reconstruct the original data. If an autoencoder learns a data distribution, then it should be able to encode and then decode an in-distribution data point back into a data point that is close to the original input data. However, for out-of-distribution data, the reconstruction will be worse, so we can use reconstruction loss as a score for detecting outliers. Thus it can be used for anomaly detection by training the autoencoder on the train set and feeding the autoencoder with new samples. If the $L_2$ distance is greater than in the train set then it is probably an anomaly. It can be used for outlier detection as well since in-distribution data will have lower reconstruction loss compared to out-of-distribution data.

We can notice that many outlier detection techniques involve computing a score for every data point and then thresholding to select outliers. Outlier detection methods can be evaluated by looking at the ROC curve, or if we want a single summary number to compare methods, looking at the AUROC.

### Distribution shift

![An example of extreme distribution shift (in particular, covariate shift/data shift) in a hand-written digit classification task.](https://dcai.csail.mit.edu/lectures/imbalance-outliers-shift/distribution-shift.svg)

Distribution shift is a challenging problem that occurs when the joint distribution of inputs and outputs differs between training and test stages, i.e., $p_{train}(x, y) \neq p_{test}(x,y)$. This issue is present, to varying degrees, in nearly every practical ML application, in part because it is hard to perfectly reproduce testing conditions at training time.

**Types of distribution shift**

1. **Covariate shift / data shift / data drift** - occurs when $p(x)$ changes between train and test, but does not. In other words, the distribution of inputs changes between train and test, but $p(y\:|\:x)$ the relationship between inputs and outputs does not change.
![Data drift](https://dcai.csail.mit.edu/lectures/imbalance-outliers-shift/covariate-shift.svg)
Examples of covariate shift:
* Self-driving car trained on the sunny streets of San Francisco and deployed in the snowy streets of Boston
* Speech recognition model trained on native English speakers and then deployed for all English speakers
* Diabetes prediction model trained on hospital data from Boston and deployed in India

To solve the problem with covariate shift we should use various types of data in the train set (or at least test set) to reflect the deployment environment.

2. **Concept shift / concept drift** - occurs when $p(y\:|\:x)$
changes between train and test, but $p(x)$ does not. In other words, the input distribution does not change, but the relationship between inputs and outputs does. This can be one of the most difficult types of distribution shift to detect and correct.
![Concept shift in a two-class dataset with two-dimensional features](https://dcai.csail.mit.edu/lectures/imbalance-outliers-shift/concept-shift.svg)

It is tricky to come up with real-world examples of concept shift where there is no change in $p(x)$. Here are some examples of concept shifts in the real world:
* Predicting a stock price based on company fundamentals, trained on data from 1975 and deployed in 2023. Company fundamentals include statistics like earnings per share. While these numbers ($p(x)$) themselves did change over time, so did the relationship between these numbers and valuation. The P/E ratio (ratio of stock price, $y$, to earnings per share, $x$) changed significantly over time. The S&P500 P/E ratio was 8.30 in 1975, while by 2023, it had risen to about 20. This is concept shift, where $p(y\:|\:x)$ 
 has changed: people are valuing a company more highly (by more than a factor of 2x) for the same earnings per share.
* Making purchase recommendations based on web browsing behavior, trained on pre-pandemic data, and deployed in March 2020. While web browsing behavior ($x$) did not change much (e.g., most individuals browsed the same websites, watched the same YouTube videos, etc.) before the pandemic vs. during the pandemic, the relationship between browsing behavior and purchases did (e.g., someone who watched lots of travel videos on YouTube before the pandemic might buy plane or hotel tickets, while during the pandemic they might pay for nature documentary movies).

**Prior probability shift/label shift**

Prior probability shift appears only in $y \rightarrow x$ problems (when we believe $y$ causes $x$). It occurs when $p(y)$ changes between train and test, but $p(x\:|\:y)$ does not. One can imagine it as the converse of covariate shift.

To understand prior probability shifts, consider the example of spam classification, where a commonly used model is Naive Bayes. If the model is trained on a balanced dataset of 50% spam and 50% non-spam emails, and then it's deployed in a real-world setting where 90% of emails are spam, that is an example of a prior probability shift.

Another example is when training a classifier to predict diagnoses given symptoms, as the relative prevalence of diseases is changing over time. Prior probability shift (rather than covariate shift) is the appropriate assumption to make here because diseases cause symptoms.

**Detecting and addressing distribution shift**

Some ways we can detect distribution shifts in deployments:
* **Monitoring the performance of the model**. Monitor accuracy, precision, statistical measures, or other evaluation metrics. If these change over time, it may be due to a distribution shift.
* **Monitoring the data**. We can detect data shifts by comparing the statistical properties of training data and data seen in a deployment.
At a high level, distribution shift can be addressed by fixing the data and re-training the model. In some situations, the best solution is to collect a better training set.

If unlabeled testing data are available while training, then one way to address covariate shift is to assign individual sample weights to training data points to weigh their feature distribution such that the weighted distribution resembles the feature distribution of test data. In this setting, even though test labels are unknown, label shift can similarly be addressed by employing shared sample weights for all training examples with the same class label, to make the weighted feature distribution in training data resemble the feature distribution in the test data. However, concept shift cannot be addressed without knowledge of its form in this setting, because there is no way to quantify it from unlabeled test data.

## Lecture 6 Growing or Compressing Datasets

### Active learning

The goal of active learning is to select the best examples to label next to improve our model the most. Suppose our examples have feature values $x$ which are inputs to model $A$ that is trained to output accurate predictions $A(x)$. For instance, in image classification applications, we might have: examples which are images, feature values which are pixel intensities, a model that is some neural network classifier, and model outputs which are predicted class probabilities.

Often we can use these outputs from an already-trained model to adaptively decide what additional data should be labeled and added to the training dataset, such that retraining this model on the expanded dataset will lead to the greatest boost in model accuracy. Using active learning, we can train a model with much fewer labeled data and still achieve the same accuracy as a model trained on a much larger dataset where what data to label was selected randomly.

In pool-based active learning, there exists a pool of currently unlabeled examples $U$. We label this data iteratively in multiple rounds $r=1,2,...,T$, where round $T$ is when we have reached our budget. Each round involves the following steps:
1. Compute outputs from our model trained on the currently-labeled data from the previous round $A_r$ (for example obtaining probabilistic predictions for each unlabeled example).
2. Use these model outputs together with an active learning algorithm 
 that scores each unlabeled example $x\in U$ to determine which data would be most informative to label next. Here the *acquisition function*/selection criterium $\phi(x, A_r)$ estimates the potential value of labeling a particular datapoint (i.e. how much is adding this datapoint to our current training set expected to improve the model), based on its feature value and the output of the trained model from the previous round.
3. Collect the labels for the data suggested by our active learning algorithm and add these label examples to our training data for the next round: $D_{r+1}$ (removing these examples from 
$U$ once they are labeled).
4. Train our model on the expanded dataset $D_{r+1}$ to get a new model $M_{r+1}$.
![Active learning process](https://dcai.csail.mit.edu/lectures/growing-compressing-datasets/lec6.001.png)

In classification tasks with $K$ classes, the model outputs a vector of predicted class probabilities $p=[p_1, p_2, ..., p_k]=A_r(x)$ for each example $x$. Here a common *acquisition function* is the entropy of these predictions as a form of uncertainty sampling:

$\phi(x, A_r)=-\sum_{k}p_k \log p_k$
 
*Acquisition function* $\phi(x, A_r)$ takes the largest values for those unlabeled examples that our current model has the most uncertainty about. Labeling these examples is potentially much more informative than others since our model is currently very unsure of what to predict for them.

The active learning process is typically carried out until a labeling budget is reached, or until the model has achieved the desired level of accuracy (evaluated on a separate held-out validation set).

**Passive vs Active Learning**

Here we present a simple 1-dimensional example that illustrates the value of active learning. We compare against a 'passive learning' algorithm that randomly selects which data to label next in each round.
![1-dimensional active learning example](https://dcai.csail.mit.edu/lectures/growing-compressing-datasets/lec6.006.png)

In this simple 1-D example, active learning quickly selects samples to adjust actual decision boundary (represented by the dashed gray line), effectively performing a binary search to reach the boundary in about 6 iterations. On the other hand, passive learning (or random sampling) takes much longer because it relies on randomness to get examples close to the decision boundary, taking close to 100 iterations to reach a similar point as the active approach! Theoretically, active learning can exponentially speed up data efficiency compared to passive learning, in terms of the amount of data $n$ needed to reach a goal model error rate ($2^{-n}$
 for active vs. $n^{-1}$ for passive in this case).

**Practical Challenge 1: Big Models**

Unfortunately, things are more challenging when we go from theory to practice. In modern ML, models used for image/text data have become quite large (with very high numbers of parameters). Here training the model each time we have collected more labels is computationally expensive.

![Active learning challenge with big models](https://dcai.csail.mit.edu/lectures/growing-compressing-datasets/lec6.002.png)

In such settings, we can instead employ batch active learning, where we select a batch of examples to label in each round rather than only a single example. A simple approach to decide on a batch is merely to select the $J$ unlabeled examples with top values according to the acquisition function $\phi(x, A_r)$ from above.

![Batch active learning](https://dcai.csail.mit.edu/lectures/growing-compressing-datasets/lec6.004.png)

However, this approach may fail to consider the diversity of the batch of examples being labeled next, because the acquisition function may take top values for unlabeled data points that all look similar. To ensure the batch of examples to label next are more representative of the remaining unlabeled pool, batch active learning strategies select $J$ examples with high information value that are also jointly diverse. For example, the greedy k-centers approach aims to find a small subset of examples that covers the dataset and minimizes the maximum distance from any unlabeled point to its closest labeled example.

**Practical Challenge 2: Big Data**

Active learning is also challenging with large amounts of unlabeled data, which has become commonplace in the era of big data. Many approaches search globally for the optimal examples to label and scale linearly or even quadratically with representation-based methods like the k-centers approach above. This quickly becomes intractable as we get to web-scale datasets with millions or billions of examples.

![Active learning challenge with big data](https://dcai.csail.mit.edu/lectures/growing-compressing-datasets/lec6.003.png)

Luckily, one option to speed things up is to only compute the model outputs and acquisition function for a subset of the unlabeled pool. Many classes only make up a small fraction of the overall data in practice, and we can leverage the latent representations from pre-trained models to cluster these concepts. We can exploit this latent structure with methods like Similarity Search for Efficient Active Learning and Search of Rare Concepts (SEALS) to improve the computational efficiency of active learning methods by only considering the nearest neighbors of the currently labeled examples in each selection round rather than scanning over all of the unlabeled data.

![Similarity Search for Efficient Active Learning and Search of Rare Concepts (SEALS)](https://dcai.csail.mit.edu/lectures/growing-compressing-datasets/lec6.007.png)

Finding the nearest neighbors for each labeled example in the unlabeled data can be performed efficiently with sublinear retrieval times and sub-second latency on datasets with millions or even billions of examples. While this restricted candidate pool of unlabeled examples impacts theoretical sample complexity, SEALS still achieves the optimal logarithmic dependence on the desired error for active learning. As a result, SEALS maintains similar label efficiency and enables selection to scale with the size of the labeled data and only sublinearly with the size of the unlabeled data, making active learning and search tractable on web-scale datasets with billions of examples!

### Core-set selection

What if we already have too much-labeled data? Situations with systematic feedback (e.g., users tag their friends in photos or mark emails as spam) or self-supervision approaches like BERT, SimCLR, and DINO can generate an unbounded amount of data. In these cases, processing all of the potential data would require a great deal of time and computational resources, which can be cumbersome or prohibitively expensive. This computation is often unnecessary because much of the data is redundant. This is where core-set selection helps.

The goal of core-set selection is to find a representative subset of data such that the quality of a model trained on the subset is similar or equivalent to the performance of training on the full data. The resulting core set can dramatically reduce training time, energy, and computational costs.

There is a wide variety of core-set selection methods in the literature, but many of the techniques for core-set selection are dependent on the type of model and don't generalize well to deep learning models. For example, the greedy k centers approach to core-set selection from above depends on the target model we want to train to quantify the distances between examples. Training the model to perform core-set selection would defeat the purpose and nullify any training time improvements we would get from reducing the data.

Luckily we don't need to use the target model for the initial core-set selection process. Instead, we can make the selection with a smaller, less resource-hungry model, as proposed in Selection via Proxy. For example, by simply reducing the number of layers in a model, we can create a proxy that is much faster to train but still provides a helpful signal for filtering data, leading to end-to-end training-time speed-ups:
![CIFAR10 with and without data selection proxy](https://dcai.csail.mit.edu/lectures/growing-compressing-datasets/lec6.005.png)

Training curves of a ResNet164 convolutional neural network classifier with pre-activation on CIFAR10 with and without data selection via proxy. The light red line shows training the proxy model (a smaller ResNet20 network). The solid red line shows training the target model (ResNet164) on a subset of images selected by the proxy. Using the proxy, we removed 50% of the data without impacting the final accuracy of ResNet164, reducing the end-to-end training time from 3 hours and 49 minutes to 2 hours and 23 minutes.

Things covered in this lecture are only the tip of the Iceberg in terms of the question What to label? **In terms of growing or compressing datasets we can distinguish the following areas**:
* **Active learning for growing datasets**
* **Core-set selection for compressing datasets**
* Generative active learning - generating examples that are the most informative for us to label
* Active search for drug discovery - we don't care about model performance, we just want to find as many positive examples/successful drugs as possible. There is a special field of study and set of selection strategies, ways for framing that problem
* Hard example mining - we use other heuristics or methods to find what are most difficult examples or most valuable things to include in our dataset, which is commonly used in recommendation and search problems 
* Curriculum learning - selecting samples to label considering what order we train on those examples
* And much more

## Lecture 7 Interpretability in Data-Centric ML

### Introduction to Interpretable ML

**Interpretability** is the degree to which a human can understand the cause of a decision of an ML model.

Interpretability is important for three reasons:
* **Debugging a model to improve performance** and validating that the model will perform correctly in the real world.
* **Reviewing incorrect decisions** that were made after deployment
* **Improving usability of models** when being actively used by decision-makers.

We need interpretable ML:
* When the problem formulation is incomplete - We don't know enough about the entire world to perfectly measure the quality of our ML model (most cases)
* When there is an associated risk (e.g. self-driving cars)
* When humans are involved in decision-making

### Why do we care about interpretable features?

Consider the following examples of explanations on the California Housing Dataset. In this dataset, each row represents one block of houses in California. Our goal is to use ML models to predict the median house price of houses in each block.

Take a look at the image below, which visualizes a decision tree model trained for this purpose. Decision trees are usually considered very interpretable models because we can follow their logic through a series of simple yes/no questions. Do we feel like this visualization offers us anything towards understanding how the model makes predictions about house prices?

![Decision tree model on the California Housing dataset with PCA features](https://dcai.csail.mit.edu/lectures/interpretable-features/decision_tree.png)

The visualization above displays features generated from running a Principal Component Analysis (PCA) on the original feature set. This is a powerful algorithm that reduces the size of a feature space and can improve the generalizability of models but also reduces the interpretability of their features.
Next, consider the two images below, both offering feature-importance explanations of models trained on the California Housing Dataset. The first was trained on a set of automatically engineered features, while the second uses the basic features that come with the dataset. Both models have similar performance (r2 ~ 0.85), but we may find the second explanation much easier to reason about.

![Automatic feature engineering model's feature importance](https://dcai.csail.mit.edu/lectures/interpretable-features/engineered_importance.png)

![Basic features model's feature importance](https://dcai.csail.mit.edu/lectures/interpretable-features/interpret_importance.png)

**Performance and Interpretability**

There is some theory that there is a 'performance/interpretability tradeoff'. In other words, as we modify our features to be more interpretable, we might expect the performance of our model to decrease. In theory, this is a logical assumption - how could adding additional constraints on our feature space not result in a performance reduction?

In reality, we don't have access to infinite resources or data - we can't try every possible configuration of data. When we look at how machine learning behaves in real-world examples, we see that adding interpretability to features and models tends to lead to more efficient training, better generalization, and fewer adversarial examples.

Having interpretable features ensures that our model is using information that we know is relevant to our target, thus reducing the chance of the model picking up on spurious correlations.

In the situations where we do see a performance/interpretability trade-off, it's up to us - the ML engineers - to consider the domain at hand to understand the relative importance of performance vs. interpretability, which changes from case to case.

### What are interpretable features really

Interpretable features are those that are most useful and meaningful to the user (i.e., whoever will be using the explanations). Unfortunately, what this means varies heavily between user groups and domains.

Many properties relate to feature interpretability. A few to consider include:
1. Readability: Do users understand what the feature refers to at all? Codes like X12 are not readable and should be replaced with natural-language descriptions.
2. Understandability: Can our users reason about the feature value? It's often easier to reason directly about real-world values than values that have been standardized or otherwise engineered. For example, thinking about income in direct dollar values ($50,000) is easier than thinking about a normalized measure of income (.67).
3. Meaningfulness/Relevancy: Users have an easier time trusting, and are therefore more likely to use, models that use information that they believe is important.
4. Abstract Concepts: Depending on our user base and their expertise, it may be valuable to condense features into digestible abstract concepts - for example, combine detailed information about the area into a single metric of 'neighborhood quality'. There is a tradeoff here - using abstract concepts can (sometimes nefariously) hide important information.

The table below shows some examples of features and their properties.
![Features and their properties](https://dcai.csail.mit.edu/lectures/interpretable-features/table.png)

### How do we get interpretable features?

  *Feature engineering is the first step to making an interpretable model, even if we don't have any model yet (Hong et al., 2020)*

There are three main approaches to interpretable feature generation:
1. Including users (or at least humans of some kind) in the feature generation process
2. Transforming explanations separate from data
3. Using automated feature generation algorithms that factor in interpretability

**Including the User**

Traditionally in design tasks, users are included through the iterative design process. In short: include our users in every step of the process (including feature engineering!) and iterate based on their feedback.

There are ways to make including the user in feature engineering easier, such as through *collaborative feature engineering* - systems that allow multiple people to easily collaborate in feature generation.

In this lecture, we introduce two systems for Collaborative Feature Engineering - Flock and Ballet.

**Flock** 

Flock takes advantage of the fact that people have an easier time highlighting important features when comparing and contrasting, rather than just describing. Choosing features through comparisons can be described in the following steps:
1. Machine-generate features for a prediction task
2. Crowd-generate features
3. Cluster crowd-generated features
4. Iterate on inaccurate model nodes

An example of this method can be distinguishing between a painting by Monet vs. a painting by other people, who have similar paintings to Monet. Flock shows users ('the crowd') two instances of different examples from the database, and asks them to:
1. Classify them
2. Describe in natural language why they chose their classification. 
3. These natural language sentences are separated into phrases by conjunctions and clustered. 
4. Clusters are once again shown to the crowd to be labeled.

Let's walk through an example, where our task is to differentiate paintings by two artists with similar styles: Monet versus Sisley.

1. Show users one painting by each artist, and ask them to identify which artist they believe painted each: 
![Two paintings, one by Monet](https://dcai.csail.mit.edu/lectures/interpretable-features/monet_lillies_vs_sisley.jpg)

2. Ask for a natural-language description of why they chose their classification:

    *The first painting is probably a Monet because it has lilies in it, and looks like Monet's style. The second probably isn't Monet because Monet doesn't normally put people in his paintings.*

3. Split up the description at conjunctions (and/or) and punctuation, and cluster the resulting phrases: 

* The first painting is probably a Monet because it has lilies in it
* It has flowers
* The painting includes lilies
* There are flowers and lilies in the painting

4. Show users the clusters and ask for a single question that would represent the cluster: 

    *Does the painting have flowers/lilies in it?*

These questions represent crowd-generated features, that should be very interpretable because they directly represent the information users used to classify the instances.

**Flock results**

Flock outperforms:
* Original features/data used directly
* Machine-engineered features only
* Crowd classifications

Moreover, Flock generates interpretable features, i.e.:
* Contains flowers
* Is abstract
* Does not contain people

**Ballet**

Ballet is a collaborative feature engineering system that abstracts away most of the ML training pipeline to allow users to focus only on feature engineering. With Ballet, users write just enough Python code to generate a feature from a dataset. These features are then incorporated into the feature engineering pipeline and evaluated. In general, it uses AutoML tactics and evaluation to get feedback on the quality of the created feature. Then it automatically uses feature selections to select the ones that are the most important.

**Explanation Transforms**

To generate more interpretable features even in situations where models require some degree of uninterpretability in their features (for example, models often require standardization, one-hot encoding, and imputing data), we can apply post-processing transformations to the explanations themselves. One tool that helps with this is [Pyreal](https://dtail.gitbook.io/pyreal/).

Pyreal automatically 'undoes' data transformations that reduce the interpretability of features at the explanation level, as well as adding additional transformations that can improve the interpretability of features in explanations. For example, consider the image below, which shows an explanation before and after being transformed by Pyreal. In particular, notice that one-hot encoded features are combined into a single categorical feature (ocean proximity), less interpretable features are transformed into more readable versions (lat/long -> city), and features are unstandardized (median income).

![raw features vs Pyreal features](https://dcai.csail.mit.edu/lectures/interpretable-features/pyreal_transforms.png)

**Interpretable Feature Generation**

Some automated feature engineering algorithms are specially formulated to generate more interpretable features. These algorithms often consider things like what information is most contrastive, meaning it specifically separates classes, or focus on reducing the feature number to a more easily parsable subset.

One example of such an algorithm is the Mind the Gap (MTG) algorithm. This algorithm aims to reduce a set of boolean features to a smaller, interpretable subset.

MTG begins by randomly assigning features to 'feature groups', defined by one or more features concatenated with 'AND' or 'OR'. It then determines which of these groups results in the largest 'gap' between true and false instances, and iterates. For example, in the image below, the yellow and blue dots represent instances that are True or False for a given feature group. Here, the left feature group results in a bigger gap than the right feature group. Through this process, MTG creates a small set of features that maximally separate classes.
![MTG creates a small set of features that maximally separate classes](https://dcai.csail.mit.edu/lectures/interpretable-features/gap_example.png)

Here are some examples of the kinds of feature groups we could generate using MTG. For a specific set of values for each of these groups, we get a cluster that we could describe as 'mammals'.

![MTG cluster example](https://dcai.csail.mit.edu/lectures/interpretable-features/mtg.png)

**Conclusions**

* If we care about the interpretability of our machine learning, we should also care about the interpretability of our features.
* Features are interpretable if they are useful and meaningful to our users - different users have different needs.
* We can generate interpretable features by including humans (ideally, our users) in the feature generation process. We can also transform the explanation itself, or generate features using an algorithm that considers interpretability.


## Lecture 8 Encoding Human Priors: Data Augmentation and Prompt Engineering

**Human priors**

Human priors are prior knowledge we have about the world, about the data, about the task. And we often take them for granted, like the rotated dog.

In the case of the rotated dog, it's a special type of human prior that is particularly useful. That's an invariance. A change to the input data that doesn't change its output. This is useful because we can find smart ways to encode them in the input training data without needing to gather more data.

**Encoding**

Encoding means finding a function to represent the invariance. So for rotating the image, it's just a function for rotation.

Specifically, we're looking at adapting the data today. It's a very effective place to be doing this and much easier than making architectural or loss function adaptations. It's a common technique that we ML researchers and practitioners all do.

When a model overfits, it overindexes on the features it has seen a lot of and starts to memorize patterns as opposed to learning and generalizing. This is also relevant to underfitting as underfitting often means a lack of data in the first place, so adding more data when collecting that data is hard can be very valuable.

### Human Priors to Augment Training Data

Data augmentation comes from the observation that collecting enough data is often very difficult. Sometimes, we think we have a lot of data but we don't have enough coverage of certain classes. This is called class imbalance. Alternatively, we could have a lot of data, but since it's skewed, biased, or simulated, it doesn't generalize to our test set.

What data augmentation does is it enables us to add more data to the data we already have. This is an easy win to improve our model without needing to collect more data when collecting more data is hard or expensive, or time-consuming. For example, if we are in a medical use case, we need labels from a doctor, that doctor might be extremely expensive.

As a concrete example, I had a case where we needed several board-certified pathologists to be able to label our data and even then they didn't always agree with each other. Getting their time and accurate labels over and over again (since we couldn't necessarily get the labeling scheme right the first couple of times) was really expensive and hard to coordinate, delaying the project for months if that wasn't possible.

So what data augmentation can do is enable us to encode our human priors over invariances that we know about in our data and we're able to augment our dataset further, such as using flip and rotation on those dog pictures that we saw previously.

Now, those are pretty simple. There are far more advanced methods, such as Mobius transformations. If we have classes, we can also use an effective method called Mixup, where we can mix our different classes to be used as interpolated examples in alpha space. What does that mean? If we have dog pictures and cat pictures, we can overlay these images together (e.g. by varying the alpha or A parameter in RGBA). For example, we can change the alpha of a cat image to 60% and the dog image to 40%. We would get a blended cat-dog, and as a human, we would agree that there is a cat and dog in it. Then, we could change our class label to be 60% cat and 40% dog for our model to predict. We can vary this however we want across our data to produce more training examples with precise labels. This is a very effective technique and is used pretty widely now.

![MixUp augmentation](https://dcai.csail.mit.edu/lectures/human-priors/lec8.015.jpeg)

Data augmentation can also be taken to the extreme of synthetic data augmentation. This means using the data we already have, we can even train a model to generate more of that kind or class of data. For this, we can train our model or we can use foundation models, such as DALL-E or Stable Diffusion in the image scenario, to generate more data from them. Just know that we have to think about how this impacts our test set if the foundation model has been trained on samples in our test set.

Data augmentation can also be useful for robotics. Because it is so expensive to run experiments in the physical world, we often use simulation to train robotics algorithms. So, we can transfer styles from a simulated environment into the styles of a real environment using a generative model. In this case, this was work from Google on RetinaGAN.

![Sim-to-real transfer for robotics](https://dcai.csail.mit.edu/lectures/human-priors/lec8.017.jpeg)

Data augmentation works across a lot of different modalities. It's not just on images. For text, there's a really interesting technique called back-translation, where we can take an English sentence, such as: 
  
  *'I have no time* 

and use a translation model in French 

  *'je n'ai pas le temps'*
  
then translate it back into English for 

*'I don't have time.'*

What's interesting is now our translation back into English aren't in the same words that we used before, but it has the same meaning. So we can use this new example as augmentation on our data set to help the model understand different ways of phrasing the same thing, and avoid overfitting on the original example. Of course, we can use this on any source and target language.

### Human Priors at Test-Time (LLMs)

Now, that's encoding human priors into training data before we train our model. However, we can also encode human priors into our model at test time. One popular method is called prompt engineering. It is used for large language models (LLMs). What this means is that we are changing the input at test time to elicit certain results at output time. For example, we can ask an LLM to write a letter of recommendation for a student. It'll write a letter of recommendation that's pretty average. But if we ask it to write a letter of recommendation for a student who gets into MIT, then it does much better because it assumes our letter will get into MIT. Interesting, right?

LLMs are special because they have an easy interface for humans to use, and that is language. Human language is something that we are all very comfortable using to prompt the model and to provide input. This is somewhat taken for granted and is not a very well-known thing. In the past, research has focused on understanding how to find those secret knobs inside of a model, called disentanglement. Entire PhDs have been completed around this method. but adding an LLM gives us an interface into the original models, such as an image model. The models themselves don't have to change, but adding the language model does change how we interact with the model.

Prompt engineering depends on the model. Different models have been trained to do different things. For example, we can see here that GPT-3 (Brown et al., 2020) has been trained to just predict the next thing and right here it is just assuming, for example, that we are in a form and we are writing different questions for a form. It's likely seen a lot of forms. Now, GPT-3.5 (ChatGPT) acts very differently: it's able to take in commands because it's been trained additionally on a lot of dialogue and command data. So when asked a question, it answers it rather than proposes a new question.
![Prompt engineering depends on the model](https://dcai.csail.mit.edu/lectures/human-priors/lec8.023.jpeg)

A very powerful method for adapting these models is giving them examples. Examples help nudge not only what we want but also provide context into what type of scenario the model should be operating under. For example, we can give GPT-3 some context that we are answering questions, as opposed to writing questions for a form. We can give examples of asking and answering questions to then be able to answer our questions now.

![Giving examples in the prompt as technique for prompt engineering](https://dcai.csail.mit.edu/lectures/human-priors/lec8.024.jpeg)

## Lecture 9 Data Privacy and Security

Machine learning models are sometimes trained on sensitive data, such as healthcare records, and oftentimes, these models are made publicly available, even though the data they are trained on is sensitive and not suitable for release. The model architecture/weights might be made available for download, or the model might be deployed as an inference endpoint, available for anyone to make predictions on any data in a black-box way.

A natural question arises: do these public models leak private information about the data on which they are trained? It turns out that they do, and this gives rise to a variety of attacks on ML models, including:

* **Membership inference attacks** - given a data point, infer whether it was in the training set of an ML model. For example, consider an ML model trained on a dataset of patients with HIV. If an adversary can identify whether or not a particular person was included in the model's training set, then they'll be able to infer that person's HIV status (because the model is only trained on people with the condition).
* **Data extraction attacks** - given a model, extract some of its training data. For example, consider a large language model (LLM) like OpenAI Codex, the model that powers GitHub Copilot. If the model is trained on a corpus of code including private repositories that contain production secrets like API keys, and an adversary can extract some training data by probing the model, then the adversary might learn some private API keys.
There are many other types of attacks on data / ML models, including adversarial examples, data poisoning attacks, model inversion attacks, and model extraction attacks. ML security is an active area of research, and there are many thousands of papers on the topic, as well as recently-discovered issues like prompt injection in LLMs that haven't even received a systematic treatment from researchers yet.

This lecture covers security-oriented thinking in the context of machine learning, and it focuses on inference and extraction attacks. Finally, it touches on defenses against privacy attacks, including empirical defenses and differential privacy.

### Defining security: security goals and threat models

To be able to reason about whether a system is secure, we must first define:
* **A security goal**, which defines what the system is trying to accomplish. What should/shouldn't happen?
* **A threat model**, which constrains the adversary. What can/can't the adversary do? What does the adversary know? What does the adversary not know? How much computing power does the adversary have? What aspects of the system does the adversary have access to?

Once these are defined, then we can decide whether a system is secure by thinking about whether any possible adversary (that follows the threat model) could violate the security goal of our system. Only if the answer is 'no' is the system secure.

Quantifying overall adversaries is one aspect of what makes security challenging, and why it's important to specify good threat models (without sufficiently constraining the adversary, most security goals cannot be achieved).

For any given system, fully specifying a security goal and threat model can require a lot of thought. Let's build some intuition by doing some threat modeling.

**Image prediction API**

![Prediction class distribution](https://dcai.csail.mit.edu/lectures/data-privacy-security/gcv.png)

Consider a cloud-based image prediction API like the Google Cloud Vision API, which takes in an image and returns a prediction of labels and associated probabilities. A security goal might include that an attacker should not be able to extract the ML model. The model is proprietary, and collecting a large dataset and training a model is expensive, so a cloud provider might not want an adversary to be able to extract model architecture/weights. 

In threat modeling, the cloud provider might assume that the **adversary can**:
* **Make queries to the prediction API**, obtaining for any adversary-chosen input, a probability distribution over output classes
These inputs do not have to be 'natural' / 'real' images, but can be any valid images, i.e., a rectangular grid of pixels of arbitrary colors
* **Know the model architecture (but not weights)**: perhaps the cloud provider has written an academic paper describing their latest and greatest neural network model, so this is public information

The cloud provider might assume that the **adversary can't**:
* **Access activations of hidden layers of the model**: the API exposes only predictions, not intermediate activations
* **Compute gradients through the model**
* **Make more than 1000 queries per hour**: the API is rate-limited
* **Perform more than $100k worth of computing**: at this point, it's not worth it for an adversary to extract this particular model (a high-quality but generic image classification model)

**Patient risk prediction model**

Suppose that a hospital trained an ML model on patient data to predict the likelihood of the patient needing intensive care, and the hospital made the model publicly available for download, so that other hospitals as well as researchers could use it. A security goal might include that the model should not reveal any private information about any patient who was treated at the hospital. For example, given a particular patient, an adversary should not be able to tell whether that patient was in the training dataset of the model. The hospital might assume that the adversary can:
* Have full white-box access to the model: full knowledge of architecture and weights
* Make use of publicly available de-anonymized medical datasets from other institutions as representative data
  * Know the rough distribution of certain features, e.g., patient age
  * Know the range of certain features

The hospital might assume that the adversary can't:
* Obtain, by any other means, any subset of patient data from the hospital (e.g., by compromising its servers)
* Obtain intermediate checkpoints of model weights the hospital made after each epoch while training the model

### Membership inference attacks

Membership inference attacks determine whether a data point is in the training set of an ML model. Consider an ML model $M$ trained on a dataset $D$ of data points $(x_i, y_i)$, where $M$ produces probability distributions over the set of labels, $M: x \rightarrow y$. The goal of a membership inference attack is to determine whether a data point $(x, y)$  is in the training set of $M$, i.e., whether $(x, y) \in D$, given access to $M$.

Different attacks consider different settings where the adversary has varying access to $M$. In this lecture, we focus on black-box access: the attacker can query $M$ for attacker-chosen inputs, and the attacker obtains the model output (e.g., predicted probabilities for all classes).

**Shadow training**

Shadow training attack uses machine learning to break the privacy of machine learning, by training a model to infer whether or not a data point is in the training set of the target model. The approach trains an attack model $A$ that takes in a data point's class label $y$ and a target model's output $y^*$ and performs binary classification, whether or not the data point is in the training set: $A:(y, y^*) \rightarrow \left \{in, out  \right \}$. It collects a training dataset for this attack model with a collection of 'shadow models'.

**Step 1: collecting training data**

The attack assumes that the attacker has access to additional data $D_{shadow}$ that fits the format of the target model. For example, if the target model is an image classifier, then the attacker needs to have access to a bunch of images. The attack trains 'shadow models' on this dataset to produce training data for the attack model.

1. Partition the dataset $D_{shadow}$ into a $D_{in}$ set and $D_{out}$ set
2. Choose a model architecture and train a shadow model $M_{shadow}$ on $D_{in}$
3. For each $(x, y) \in D_{in}$, compute $y=M_{shadow}(x)$, the model's  output on $x$, and use that to create a training data point $((y, y^*), in)$ (label True).
4. For each $(x, y) \in D_{out}$, compute $y=M_{shadow}(x)$, the model's output on $x$, and use that to create a training data point $((y, y^*), out)$ (label False).

This process can be repeated many times, using different partitionings of $D_{shadow}$ and different shadow models to create a large training dataset we call $D_{inout}$.

**Step 2: train the attack model**

Next, the attack trains a binary classifier $A$ on $D_{inout}$. We can use any classification algorithm for $A$; one possible choice is a multi-layer perceptron.

**Step 3: perform the attack**

Now, given a new data point $(x, y)$, the attack computes $y^*=M(x)$ by feeding the data point to the black-box target model $M$, and then predicts whether the data point is in the training set by evaluating $A(y, y^*)$.

**Shadow model downsides**:

* Assumption that attacker has a similar dataset to the one used for training the original model
* Computationally expensive

### Metrics-based attacks

Unlike attacks based on training a binary classifier, metric-based attacks are a lot more simple to implement and computationally inexpensive.

**Prediction correctness-based attack**

Given a data point $(x, y)$, this simple heuristic returns $in$ if the model's prediction $M(x)$ is correct (equal to $y$).

Intuition: This exploits the gap between training and test accuracy, where the model will likely correctly classify most or all of its test data but a much smaller fraction of training data.

**Prediction loss-based attack**

This approach assigns a score to a data point $(x, y)$ using the model's loss for that particular data point, $L(M(x), y)$. The score can be converted into a binary label $in/out$ by thresholding at some chosen threshold $\tau$, outputting $in$ if the loss is below the threshold.

For this metric, one way to choose a threshold might be to use the average or maximum loss of the model on the training data (which might be reported by the publisher of the model).

Intuition: This exploits the property that models are trained to minimize loss, and they can often achieve zero loss for training data.

**Prediction confidence-based**

This approach assigns a score to a data point based on the model's confidence in its predicted class, $max(M(x))$. One way to choose a threshold would be to use the model's average or minimum confidence on the training data, if available.

Intuition: This exploits the property that models are often more confident about the predictions for training data (even when that prediction doesn't match the true label of the data point).

**Prediction entropy-based**

This approach assigns a score to a data point based on the entropy 
 of the model's output, $y=M(x)$:

 $H(y) = - \sum_i y[i] \cdot \log(y[i])$

Intuition: similar to the above, this exploits the property that models are often more confident about predictions for training data.

### Data extraction attacks

Extraction attacks extract training data directly from a trained model. Neural networks unintentionally memorize portions of their input data, and there are techniques for extracting this data, for example, from large language models [(Carlini et al., 2021)](https://www.usenix.org/system/files/sec21-carlini-extracting.pdf).

![Data extraction attack on GPT-2](https://dcai.csail.mit.edu/lectures/data-privacy-security/extraction.png)

At its core, the attack works as follows:

Sample many sequences from the model. These are sampled by initializing the model with a start-of-sentence token and repeatedly sampling in an autoregressive fashion.
Perform a membership inference attack to determine which generated sequences were likely part of the training set. A simple membership inference attack uses the *perplexity* of a sequence to measure how well the LLM 'predicts' the tokens in that sequence. Given a model $\hat{p}$ that predicts the probability of the next token and sequence of tokens $x_1, x_2, ..., x_n$ , the *perplexity* is:
 
$P = \exp \left (-\frac{1}{n} \sum^n_{i=1} \log \hat{p}(x_i | x_1, ..., x_{i-1}) \right )$
 
The more refined version of this basic attack, presented by [Carlini et al., 2021](https://www.usenix.org/system/files/sec21-carlini-extracting.pdf), is successful in extracting hundreds of memorized examples from GPT-2.

**Defending against privacy leaks: empirical defenses and evaluation**

There are a variety of ideas for defenses against privacy attacks that seem plausible. For example, to defend against membership inference attacks, the following techniques sound like reasonable ideas to try:
* Restrict the prediction to the top $k$ classes
* Quantize the model's output to reduce precision
* Add random noise to the model's output
* Rather than outputting probabilities, output classes in sorted order
* Modify the training procedure, e.g., by adding regularization (maybe the privacy leakage is due to overfitting)

Many such ideas have been proposed, and many such ideas do not work. How might we break some of the ideas proposed above?

Security, especially with empirical solutions, can be a cat-and-mouse game. How should one evaluate a proposed defense?

**Evaluating defenses**

The first step is to fix the security goal, threat model, and evaluation metric (e.g., for a model inference attack, one might evaluate an attack based on the F1 score).

When evaluating an empirical defense, it's important to keep [Kerckhoffs's principle](https://en.wikipedia.org/wiki/Kerckhoffs%27s_principle) in mind: a defense should be secure even if the adversary knows the defense. The opposite of this is 'security through obscurity.'

For a defense to be secure, it must be robust (i.e., the system must satisfy its security goals) for all possible attacks within the threat model. Usually, in an evaluation, it's not possible to quantify overall possible attacks. For this reason, in practice, we evaluate defenses by putting on the hat of an attacker and trying as hard as possible to break them. Only if we fail do we conclude that the defense might be a good one.

### Differential privacy

For the issue of data privacy in machine learning, one promising approach that avoids the cat-and-mouse game of empirical defenses involves differential privacy (DP).

At a high level, DP is a definition of privacy that constrains how much an algorithm's output can depend on individual data points within its input dataset. A randomized algorithm $A$ operating on a dataset $D$ is $(\epsilon, \delta)$ -differentially private if:

$Pr[A(D) \in S] \le \exp(\epsilon) \cdot Pr[A({D}') \in S] + \delta$

for any set $S$ of possible outputs of $A$, and any two data sets $D$, ${D}'$ that differ in at most one element.

In the context of ML, the algorithm $A$ is the model training algorithm, the input is the dataset $D$, and the output of the algorithm, which is in $S$, and constrained by the definition of DP, is the model.

A differentially private training algorithm (like DP-SGD, [Abadi et al., 2016](https://arxiv.org/abs/1607.00133)) ensures that the result (a trained model) does not change by much if a data point is added or removed, which intuitively provides some sort of privacy: the model can't depend too much on specific data points.

There are challenges with applying DP in practice. One challenge is that the definition of DP includes two parameters, $\epsilon$ and $\delta$, that can be hard to set. Applied to a particular dataset, it can be hard to understand exactly what these parameters mean in terms of real-world privacy implications, and sometimes, choices of $\epsilon$ and $\delta$ that are good for privacy result in low-quality results (e.g., a model with poor performance).

**Resources**
* [Membership Inference Attacks on Machine Learning: A Survey](https://arxiv.org/abs/2103.07853)
* [A Survey of Privacy Attacks in Machine Learning](https://arxiv.org/abs/2007.07646)
* [Awesome Attacks on Machine Learning Privacy (big list of papers)](https://github.com/stratosphereips/awesome-ml-privacy-attacks)


## Course resources

**Open-Source Software Tools for Data-Centric AI**

* [cleanlab](https://github.com/cleanlab/cleanlab) - automatically detect problems in a dataset to facilitate ML with messy, real-world data
* [refinery](https://github.com/code-kern-ai/refinery) - assess and maintain natural language data
* [great expectations](https://github.com/great-expectations/great_expectations) - validate, document, and profile data for quality testing
* [ydata-profiling](https://github.com/ydataai/ydata-profiling) - generate summary reports of tabular datasets stored as pandas DataFrame
* [cleanvision](https://github.com/cleanlab/cleanvision) - automatically detect low-quality images in computer vision datasets
* [albumentations](https://github.com/albumentations-team/albumentations) - data augmentation for computer vision
* [label-studio](https://github.com/heartexlabs/label-studio) - interfaces to label and annotate data for many ML tasks

**Short Articles**

* [Unbiggen AI](https://spectrum.ieee.org/andrew-ng-data-centric-ai)
* [Andrew Ng Launches A Campaign For Data-Centric AI](https://www.forbes.com/sites/gilpress/2021/06/16/andrew-ng-launches-a-campaign-for-data-centric-ai/)
* [Tips for a Data-Centric AI Approach](https://landing.ai/tips-for-a-data-centric-ai-approach/)
* [Data-Centric Approach vs Model-Centric Approach in Machine Learning](https://neptune.ai/blog/data-centric-vs-model-centric-machine-learning)
* [A Linter for ML Datasets](https://cleanlab.ai/blog/datalab/)
* [Handling Mislabeled Data to Improve Your Model](https://cleanlab.ai/blog/label-errors-tabular-datasets/)

**Papers**

* [A Data Quality-Driven View of MLOps](https://arxiv.org/abs/2102.07750)
* [Advances in Exploratory Data Analysis, Visualisation and Quality for Data Centric AI Systems](https://dl.acm.org/doi/abs/10.1145/3534678.3542604)

**Books**

* [Human-in-the-Loop Machine Learning: Active Learning and Annotation for Human-centered AI](https://books.google.com/books/about/Human_in_the_Loop_Machine_Learning.html?id=LCh0zQEACAAJ)
* [Best Practices in Data Cleaning: A Complete Guide to Everything You Need to Do Before and After Collecting Your Data](https://www.google.com/books/edition/Best_Practices_in_Data_Cleaning/-5-9GDCQPHoC?hl=en&gbpv=1&dq=Best+Practices+in+Data+Cleaning+by+Jason+Osborne&printsec=frontcover)

**Links**

* [Data-Centric AI Competition 2023](https://machinehack.com/tournaments/data_centric_ai_competition_2023)
* [Data-centric AI Resource Hub](https://datacentricai.org/)
* [Label Errors in ML Benchmarks](https://labelerrors.com/)
