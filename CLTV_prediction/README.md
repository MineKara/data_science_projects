# CLTV Prediction with BG-NBD and Gamma-Gamma

## Business Problem

FLO wants to determine a roadmap for sales and marketing activities. In order for the company to make medium-long term plans, it is necessary to estimate the potential value that existing customers will provide to the company in the future.

<p align="center" width="100%">
    <img width="40%" src="cltv.png">
</p>

## Dataset

The data set consists of information obtained from the past shopping behavior of customers who made their last purchases from Flo via OmniChannel (both online and offline shopping) in 2020 - 2021.

| Column                            | Description                                                                                                                      |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| master_id                         | Unique customer id                                                                                                               |
| order_channel                     | Which channel of the shopping platform is used  (Android, iOS, Desktop, Mobile)                                                  |
| last_order_channel                | Channel where last purchase was made                                                                                             |
| first_order_date                  | Date of the customer's first purchase                                                                                            |
| last_order_date                   | Customer's last purchase date                                                                                                    |
| last_order_date_online            | The customer's last shopping date  on the online platform                                                                        |
| last_order_date_offline           | The last shopping date of the customer  on the offline platform                                                                  |
| order_num_total_ever_online       | Total number of purchases made by  the customer on the online platform                                                           |
| order_num_total_ever_offline      | Total number of purchases made by  the customer offline                                                                          |
| customer_value_total_ever_offline | Total fee paid by the customer for offline purchases                                                                             |
| customer_value_total_ever_online  | Total fee paid by the customer for online shopping                                                                               |
| interested_in_categories_12       | List of categories the customer has shopped  in the last 12 months                                                               |
| store_type                        | It represents 3 different companies. If the person who shopped from company A also shopped from company B, it is written as A,B. |
