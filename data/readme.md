<img src="https://github.com/theidari/customer_churn/blob/main/assets/dataset.png">
<h3>Tables Information</h3>
<ul>
  <b><li>members.csv</li></b> 
  Details: <i>user information. Note that not every user in the dataset is available.</i><br>
  Columns:<br> 
  <code>msno</code> user id<br>
  <code>city</code><br> 
  <code>bd</code> age<br>
  <code>gender</code><br> 
  <code>registered_via</code> registration method<br> 
  <code>registration_init_time</code> format %Y%M%D.
  <b><li>transactions.csv</li></b>
  Details: <i>transactions of users up until 2/28/2017.</i><br>
  Columns:<br>
  <code>msno</code> user id<br>
  <code>payment_method_id</code> payment method<br>
  <code>payment_plan_days</code> length of membership plan in days<br>
  <code>plan_list_price</code> in New Taiwan Dollar (NTD)<br>
  <code>actual_amount_paid</code> in New Taiwan Dollar (NTD)<br>
  <code>is_auto_renew</code><br>
  <code>transaction_date</code> format %Y%M%D<br>
  <code>membership_expire_date</code> format %Y%M%D<br>
  <code>is_cancel</code> whether or not the user canceled the membership in this transaction.
  <b><li>user_logs.csv</li></b>
  Details: <i>daily user logs describing the listening behaviors of a user. Data was collected until 2/28/2017.</i><br>
  Columns:<br>
  <code>msno</code> user id<br>
  <code>date</code> format %Y%M%D<br>
  <code>num_25</code> number of songs played less than 25% of the song length<br>
  <code>num_50</code> number of songs played between 25% to 50% of the song length<br>
  <code>num_75</code> number of songs played between 50% to 75% of the song length<br>
  <code>num_985</code> number of songs played between 75% to 98.5% of the song length<br>
  <code>num_100</code> number of songs played over 98.5% of the song length<br>
  <code>num_unq</code> number of unique songs played<br>
  <code>total_secs</code> total seconds played.
</ul>
