<p align="center">
<img src="https://github.com/theidari/customer_churn/blob/main/assets/churn_header_light.png">
</p>
<h3>1. Project Overview</h3>
<p align="justify">
Predicting customer churn is a vital economic focus for many companies<sup>[1]</sup>. Simply put, customer churn happens when people stop using a company's services, often turning to competitors. Acquiring a new customer can cost up to seven times more than keeping an existing one<sup>[2]</sup>. Therefore, for companies relying on regular subscription fees, like those in banking, telecom, or online services, it's essential to keep their customers. As such, identifying which customers might leave has become a priority in many industries.
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
<b><li><a href="https://github.com/theidari/customer_churn/tree/main/data">Dataset</a></li></b> Gather relevant data about customers, including demographic information, purchase history, usage patterns, customer interactions, and any other data points that might be indicative of churn.
<b><li>Data Profiling</li></b>
  Data Preprocessing: Cleanse and preprocess the collected data to remove any inconsistencies, missing values, or outliers. This step may involve data transformation, feature engineering, and scaling.
Feature Selection: Identify the most relevant features that can potentially influence churn. This step helps reduce noise and improve the accuracy of the predictive models.
<b><li>Modeling</li></b>
Model Selection: Choose an appropriate predictive modeling technique based on the nature of the data and the problem at hand. Commonly used techniques include logistic regression, decision trees, random forests, support vector machines (SVM), and artificial neural networks.
Model Training: Split the dataset into training and testing sets. Use the training set to train the chosen model by fitting it to the historical data and adjusting the model's parameters to minimize the prediction error.
Model Evaluation: Evaluate the trained model's performance using the testing set. Common evaluation metrics include accuracy, precision, recall, F1 score, and area under the receiver operating characteristic curve (AUC-ROC).
Predictive Analysis: Apply the trained model to new, unseen data to predict the likelihood of churn for individual customers. This step helps identify customers who are at high risk of churn and require targeted retention efforts.
<b><li>Monitoring</li></b>
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

