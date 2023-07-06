<p align="center">
<img src="https://github.com/theidari/customer_churn/blob/main/assets/churn_header_light.png">
</p>
<h3>1. Project Overview</h3>
<p align="justify">
Customer churn prediction and profiling are significant economic considerations for numerous companies.<sup>[1]</sup>
Customer churn, also known as customer attrition, refers to the phenomenon where customers or subscribers end their relationship with a company or service provider. It occurs when customers discontinue using a company's products or services, switch to a competitor, or simply stop engaging with the business altogether.
It has become evident that the expenses associated with acquiring a new customer can be significantly higher than (five to seven times) the costs of retaining an existing one[2] As a result, preventing customer churn or attrition has become crucial for subscription-based service companies that depend on steady and recurring membership fees.
 ((( This holds true in various sectors such as insurance [6], banking [7], online gambling [8], online video games [9], music streaming [10], online services [11], and telecommunications [12-15]. Consequently, accurately predicting which customers are likely to churn has become a top priority across multiple industries.)))
</p>
<p align="justify">
KKBOX offers subscription based music streaming service. When users signs up for our service, users can choose to either manual renew or auto-renew the service. Users can actively cancel their membership at any time.
</p>
<img src="https://img.shields.io/badge/ -223337.svg?style=for-the-badge" width="1500px" height="1px">
<h3>2. Methods and Steps</h3>
<h6 align="center">Fig 1-Workflow Diagram</h6>
<img src="https://github.com/theidari/customer_churn/blob/main/assets/workflowfix.png">
Here are the steps typically involved in customer churn prediction:

<ol>
<b><li>2. Dataset</li></b> Gather relevant data about customers, including demographic information, purchase history, usage patterns, customer interactions, and any other data points that might be indicative of churn.
<li>Data Profiling</li>
  Data Preprocessing: Cleanse and preprocess the collected data to remove any inconsistencies, missing values, or outliers. This step may involve data transformation, feature engineering, and scaling.
Feature Selection: Identify the most relevant features that can potentially influence churn. This step helps reduce noise and improve the accuracy of the predictive models.
<li>Modeling</li>
Model Selection: Choose an appropriate predictive modeling technique based on the nature of the data and the problem at hand. Commonly used techniques include logistic regression, decision trees, random forests, support vector machines (SVM), and artificial neural networks.
Model Training: Split the dataset into training and testing sets. Use the training set to train the chosen model by fitting it to the historical data and adjusting the model's parameters to minimize the prediction error.
Model Evaluation: Evaluate the trained model's performance using the testing set. Common evaluation metrics include accuracy, precision, recall, F1 score, and area under the receiver operating characteristic curve (AUC-ROC).
Predictive Analysis: Apply the trained model to new, unseen data to predict the likelihood of churn for individual customers. This step helps identify customers who are at high risk of churn and require targeted retention efforts.
<li>Monitoring</li>
Customer Retention Strategies: Based on the churn predictions, design and implement personalized retention strategies for at-risk customers. These strategies might include special offers, discounts, personalized communication, loyalty programs, or improved customer service.
Monitor and Iterate: Continuously monitor the performance of the churn prediction model and retention strategies. Collect feedback, measure the effectiveness of the implemented measures, and refine the predictive models and retention strategies over time.
</ol>
<img src="https://img.shields.io/badge/ -223337.svg?style=for-the-badge" width="1500px" height="1px">
<h3>3. Dataset</h3>
<h5>Data Storage</h5>
All of the data is stored and written to the GCS bucket.
<img src="https://img.shields.io/badge/ -223337.svg?style=for-the-badge" width="1500px" height="1px">
<h3>4. Results</h3>

<img src="https://img.shields.io/badge/ -223337.svg?style=for-the-badge" width="1500px" height="1px">
<h3>5. Conclusions</h3>

<img src="https://img.shields.io/badge/ -223337.svg?style=for-the-badge" width="1500px" height="1px">
<h3>6. Future Improvement and Discussion</h3>

<img src="https://img.shields.io/badge/ -223337.svg?style=for-the-badge" width="1500px" height="1px">
<h3>References</h3>

[1] https://www.sciencedirect.com/science/article/pii/S0169023X2200091X <br>
[2] https://www.forbes.com/sites/forbesbusinesscouncil/2022/12/12/customer-retention-versus-customer-acquisition/?sh=3ad964471c7d

