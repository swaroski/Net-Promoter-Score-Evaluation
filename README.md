# Net-Promoter-Score-Evaluation
This model showcases how to evaluate customer satisfaction from online reviews at scale. One can identify information about the market's sentiment for a product or service from survey results.  Process: a product was selected at random from those containing over 75 reviews(Amazon's dataset). Reviews were then segmented for investigation by sentiment analysis score. Exploring the review score's top and bottom tiers, clear insights emerge for actionable improvements.


![image](https://user-images.githubusercontent.com/25379742/98488499-8eb81700-21f7-11eb-9fe6-abd1cd100766.png)

This data is readily available on Amazon reviews. For this project, the Electronics dataset is used but you can use any data which encopasses similar user sentiments. The following code extracts a user-defined number of records from the electronics reviews collection, scans for null values, and creates a new dataframe for the columns of interest.

Reviews are scaled to Net Promoter Scores with the following assumptions: 5 stars equates to a promoter, 4 stars is neutral, and 1-3 stars is a detractor.
