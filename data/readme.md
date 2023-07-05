Tables
train.csv
the train set, containing the user ids and whether they have churned.

msno: user id
is_churn: This is the target variable. Churn is defined as whether the user did not continue the subscription within 30 days of expiration. is_churn = 1 means churn,is_churn = 0 means renewal.
train_v2.csv
same format as train.csv, refreshed 11/06/2017, contains the churn data for March, 2017.

sample_submission_zero.csv
the test set, containing the user ids, in the format that we expect you to submit

msno: user id
is_churn: This is what you will predict. Churn is defined as whether the user did not continue the subscription within 30 days of expiration. is_churn = 1 means churn,is_churn = 0 means renewal.
sample_submission_v2.csv
same format as sample_submission_zero.csv, refreshed 11/06/2017, contains the test data for April, 2017.

transactions.csv
transactions of users up until 2/28/2017.

msno: user id
payment_method_id: payment method
payment_plan_days: length of membership plan in days
plan_list_price: in New Taiwan Dollar (NTD)
actual_amount_paid: in New Taiwan Dollar (NTD)
is_auto_renew
transaction_date: format %Y%m%d
membership_expire_date: format %Y%m%d
is_cancel: whether or not the user canceled the membership in this transaction.
transactions_v2.csv
same format as transactions.csv, refreshed 11/06/2017, contains the transactions data until 3/31/2017.

user_logs.csv
daily user logs describing listening behaviors of a user. Data collected until 2/28/2017.

msno: user id
date: format %Y%m%d
num_25: # of songs played less than 25% of the song length
num_50: # of songs played between 25% to 50% of the song length
num_75: # of songs played between 50% to 75% of of the song length
num_985: # of songs played between 75% to 98.5% of the song length
num_100: # of songs played over 98.5% of the song length
num_unq: # of unique songs played
total_secs: total seconds played
user_logs_v2.csv
same format as user_logs.csv, refreshed 11/06/2017, contains the user logs data until 3/31/2017.

members.csv
user information. Note that not every user in the dataset is available.

msno
city
bd: age. Note: this column has outlier values ranging from -7000 to 2015, please use your judgement.
gender
registered_via: registration method
registration_init_time: format %Y%m%d
expiration_date: format %Y%m%d, taken as a snapshot at which the member.csv is extracted. Not representing the actual churn behavior.
members_v3.csv
Refreshed 11/13/2017, replaces members.csv data with the expiration date data removed.
