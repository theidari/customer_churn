<p align="center">
<img src="https://github.com/theidari/customer_churn/blob/main/assets/churn_header_light.png">
</p>
<h3>1. Project Overview</h3>
<p align="justify">
Predicting customer churn is a vital economic focus for many companies<sup>[1]</sup>. Simply put, customer churn happens when people stop using a company's services, often turning to competitors. Acquiring a new customer can cost up to seven times more than keeping an existing one<sup>[2]</sup>. Therefore, for companies relying on regular subscription fees, like those in banking, telecom, or online services, it's essential to keep their customers. As such, identifying which customers might leave has become a priority in many industries.
</p>
<p align="justify">
For KKBOX, a subscription-based music streaming platform, maintaining user loyalty is essential. Given that users can opt for manual or auto-renewal upon sign-up and can cancel memberships anytime, this project aims to leverage four machine learning algorithms, supplemented by GCP Auto-ML methods, to accurately identify and predict potential customer churn for KKBOX.</p>
<img src="https://img.shields.io/badge/ -223337.svg?style=for-the-badge" width="1500px" height="1px">
<h3>2. Methods and Steps</h3>
<p align="justify">The following steps outline the customer churn prediction process for this notebook:</p>
<h6 align="center">Fig 1-Workflow Diagram</h6>
<img src="https://github.com/theidari/customer_churn/blob/main/assets/workflowfix.png">

<ol>
<b><li><a href="https://github.com/theidari/customer_churn/tree/main/data">Dataset and Data Preprocessing</a></li></b>
<p align="justify">The notebook gathers essential customer details like age, purchase history, usage frequency, and feedback. Then, using data preprocessing, it cleans the data by fixing errors, filling in gaps, and removing unusual data points.</p>
<b><li><a href="https://theidari.github.io/customer_churn/results/data_profiling">Data Profiling</a></li></b>
  Data Preprocessing: Cleanse and preprocess the collected data to remove any inconsistencies, missing values, or outliers. This step may involve data transformation, feature engineering, and scaling.
Feature Selection: Identify the most relevant features that can potentially influence churn. This step helps reduce noise and improve the accuracy of the predictive models.
<b><li>Feature Engineering and Modeling</li></b>
Model Selection: Choose an appropriate predictive modeling technique based on the nature of the data and the problem at hand. Commonly used techniques include logistic regression, decision trees, random forests, support vector machines (SVM), and artificial neural networks.
Model Training: Split the dataset into training and testing sets. Use the training set to train the chosen model by fitting it to the historical data and adjusting the model's parameters to minimize the prediction error.
Model Evaluation: Evaluate the trained model's performance using the testing set. Common evaluation metrics include accuracy, precision, recall, F1 score, and area under the receiver operating characteristic curve (AUC-ROC).
Predictive Analysis: Apply the trained model to new, unseen data to predict the likelihood of churn for individual customers. This step helps identify customers who are at high risk of churn and require targeted retention efforts.
<b><li>Monitoring</li></b>
Customer Retention Strategies: Based on the churn predictions, design and implement personalized retention strategies for at-risk customers. These strategies might include special offers, discounts, personalized communication, loyalty programs, or improved customer service.
Monitor and Iterate: Continuously monitor the performance of the churn prediction model and retention strategies. Collect feedback, measure the effectiveness of the implemented measures, and refine the predictive models and retention strategies over time.
</ol>
<img src="https://img.shields.io/badge/ -223337.svg?style=for-the-badge" width="1500px" height="1px">
<h3>4. Results</h3>
<ul><li><b>member model df</b>
<p align="justify">Analyzing the model outcomes, the Decision Tree achieves a decent accuracy of 92.73%, but its AUC of 35.54% reveals a limitation in distinguishing between classes. Random Forest's accuracy is close at 92.68%, but its higher AUC of 82.06% shows better class differentiation. The Gradient-Boosted Trees model leads with a 92.89% accuracy and an AUC of 85.11%. In contrast, the Linear SVM has an accuracy of 92.19% with a moderate AUC of 50%. Overall, Gradient-Boosted Trees is the standout performer, especially in AUC, followed closely by Random Forest. Both significantly surpass the Decision Tree and Linear SVM in class differentiation capabilities. Validating these findings, Gradient-Boosted Trees maintain their strong performance in the test set, notably with an AUC of 85.11%, emphasizing its effectiveness in class separation.</p></li> 
<li align="justify"></li></ul>
<img src="https://img.shields.io/badge/ -223337.svg?style=for-the-badge" width="1500px" height="1px">
<h3>5. Conclusions</h3>

<img src="https://img.shields.io/badge/ -223337.svg?style=for-the-badge" width="1500px" height="1px">
<h3>6. Future Improvement and Discussion</h3>

<img src="https://img.shields.io/badge/ -223337.svg?style=for-the-badge" width="1500px" height="1px">
<h3>References</h3>

[1] https://www.sciencedirect.com/science/article/pii/S0169023X2200091X <br>
[2] https://www.forbes.com/sites/forbesbusinesscouncil/2022/12/12/customer-retention-versus-customer-acquisition/?sh=3ad964471c7d

