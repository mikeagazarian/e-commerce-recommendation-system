# e-commerce-recommendation-system
Introduction
We are constantly caught in an information overload due to the recent rapid development of information technologies and the internet. A number of recommender algorithms have been presented in an effort to address this particular difficulty. Many e-commerce websites, including amazon, Netflix, and others, use recommender algorithms to constantly make product recommendations based on browsing history. In contrast to conventional tools like search engines, which require users to enter keywords, and category navigation, where contents are organized into groups based on common sense, recommender systems automatically display the product layout depending on browser user behavior.[1]
Building an e-commerce recommendation system that makes item recommendations to a customer based on his interests is the main goal of our project. This project seeks to make product recommendations based entirely on user interaction, regardless of the ratings for the products.
Data
For this project we used the dataset from the following resource : [2] . The dataset that we chose consisted of around 11 million records and had 19 features. Each record had the features such as EventTime, EventType, UserID, ProductID, UserSession, Brand, Price, Category0, Category1, Category2, Category3, Target, TimeStamp, Hour, Minute, Weekday, Day, Month and Year.
Data Preprocessing
Our dataset had redundancies and categorical data. First we got rid of the redundant features such as the time. Second, the user session was dropped which was the session that the user was currently on. Since we were not basing our model on the specific time and session of users this was not a feature that would influence our model. We had a total of 4 category features ranked from 1 to 4 with 1 being the most general category. The last 2 categories were dropped as the number of NAN values were too high so the columns were dropped as a whole.
Next part of the preprocessing stage was taking care of NAN values. The brand and category features had missing values and these were replaced with “unknown”. We didn’t use the most frequent since this may categorize the features incorrectly and going with a separate category of unknown would be a safer approach.
For our features that weren’t numerical, they had to be converted in order to be ran through the model. The two categories and the brand features were not numerical and needed conversion. The LabelEncoder method from the sklearn-preprocessing library is used to convert the features to numerical. The label encoder numbers the categories, if the first category value is ran through, then that would be labeled as 1 and the categories with the same name would be 1 as well. This will allow the algorithm to read the data.
The target category in our data is renamed to purchase_or_cart for better understanding of the data.
For the matrix factorization algorithm, the data types of the features were all made to int64 just to attempt to fix memory issues. Last
Data Visualization

<img width="215" alt="image" src="https://github.com/mikeagazarian/e-commerce-recommendation-system/assets/105412677/cbb74161-6e66-4ec9-a7d0-5125b6b6016b">

 
The graph shows the distribution of purchased items and not purchased. 1 representing purchased and 0 representing items in cart that weren’t purchased.

<img width="218" alt="image" src="https://github.com/mikeagazarian/e-commerce-recommendation-system/assets/105412677/ad05b8cd-fee5-45b4-8e21-7acc2e314304">

 
The graph represents the distribution of items based on the price. It is apparent that there are much cheaper items than the more expensive.

<img width="229" alt="image" src="https://github.com/mikeagazarian/e-commerce-recommendation-system/assets/105412677/22a88318-8c59-49e0-a803-a00f767d208b">

 
The graph represents the different categories in the data set and the number of items in each category. Some categories are much smaller amounts compared to others.

![image](https://github.com/mikeagazarian/e-commerce-recommendation-system/assets/105412677/9612165e-96cf-4989-b13d-fbe120dd2542)

 
The correlation matrix with our data is shown to be non-correlated. This graph doesn’t represent our data well since we have many categories for our features and shows that we can’t use this graph to determine which features will influence our model.

<img width="278" alt="image" src="https://github.com/mikeagazarian/e-commerce-recommendation-system/assets/105412677/509d9d17-d74d-4de1-8520-bcba4e2adae4">

 
This graph shows the popularity of the day of the week where the majority of the purchases were.

![image](https://github.com/mikeagazarian/e-commerce-recommendation-system/assets/105412677/be831ed1-b995-4707-93ef-a52e0feeddfa)

 
This graph shows the RMSE score vs the number of latent factors that were used for the SVD model. The more latent factors used the smaller our RMSE score becomes. We were unable to visualize with more latent factors as we ran into memory issues.
Methods
Collaborative Filtering
Collaborative filtering is an algorithm used for recommending items to users by analyzing their past behavior and preferences. It looks for patterns and similarities in the behavior of different users and uses this information to suggest items that are likely to be appealing to a particular user. There are two main types of collaborative filtering : item based and user based. For this project KNN Item-Based Collaborative Filtering was used. Item-based collaborative filtering identifies items that are similar to each other based on user preferences and recommends similar items to users who have liked or used similar items. It works by analyzing user-item interaction data, such as ratings, reviews, or purchase history, to identify patterns and similarities between items. In the context of item-based collaborative filtering, KNN is used to calculate similarity between items based on their user-item interaction data. The algorithm finds the k nearest neighbors of a target item based on similarity metrics such as cosine similarity, Pearson correlation, or Mean Squared Difference. These similarity metrics measure how similar items are to each other based on their user-item interaction patterns.
For implementing this, the preprocessed dataset had to be processed again according to the algorithm requirements. The unnecessary columns such as Weekday, Day, Month had to be dropped. To determine the item-item similarity based on a user's interaction with a particular product, the algorithm used three primary features: UserID, ProductID, and PurchaseOrCart. While generating recommendations, features like Brand, Price, Category0, and Category1 were withheld in order to provide product description. As the memory usage exceeded the level that the system could tolerate, the dataset size had to be decreased. Records for 15k users were used for this, and each user's individual product interactions were taken into consideration. A total of about 51k records were produced as a result. So the final shape of the dataset used for the implementation of this algorithm was 51986 rows x 7 columns.
The algorithm first calculated the item-item similarity using the UserID, ProductID and PurchaseOrCart. The similarity metric used here is Mean Squared Difference.As the data values for PurchaseOrCart column was 0 and 1 other similarity metrics returned a zero division error, only Mean Squared Difference fitted the best. Using the similarity values calculated, recommendations were generated. Initially, the products that a given user has not interacted with were gathered. Then the similarity for these products was fetched and they were sorted based on descending similarity. From these, top N products were recommended to the user along with the product descriptions such as Brand, Price, Category0 and Category1.
Following is an example of recommendations generated for a specific user with specific product:

![image](https://github.com/mikeagazarian/e-commerce-recommendation-system/assets/105412677/2f99d47d-a2dd-48ba-afcf-ed944db8be48)

 
In this example we can see that the user has interacted with products that belong to the electronics-smartphone category. The recommendations that were generated consisted mostly of the same category products.
Matrix Factorization
Matrix Factorization is intended to reduce dimensions of a user-item matrix. For our model, we used the purchase_or_cart feature to fill in the values of the matrix. 1 represents that the item was purchased and a 0 for otherwise. The specific python library used for this model is the surprise library. The surprise library is specifically made for recommendation systems and works to model recommendation problems. The reader method from the surprise library is used to identify the rating as a 0 to 1 scale for the dataset. The Matrix factorization variant used for our model is Singular Value Decomposition (SVD). We were able to create our SVD matrix by importing SVD in from surprise. The n factors in our model consisted from 10 to 100 factors in increments of 10. Using more factors would potentially create better results but the amount of memory used would crash the process running. 100 would take reasonable time to run therefore that was used. The number of epochs used in the model is 20. The epoch is the number of times the data is ran through the algorithm, for our case the model ran the data through the SVD 20 times. To train the data the fit method is used from scikit-learn to train the model. The fit method basically updates parameters iteratively. After the model has been trained, the test method is used on that trained data to make predictions and is stored in a variable called predictions.
Evaluation
The evaluation metrics for Collaborative Filtering and Matrix Factorization show that Matrix Factorization has performed better than Collaborative Filtering in terms of RMSE, MSE, and
MAE.	The	RMSE	for Matrix Factorization is 0.3166, which is lower than the RMSE for
Collaborative Filtering (0.5322). Similarly, the MSE for Matrix Factorization is 0.1002, which is lower than the MSE for Collaborative Filtering (0.2832). Additionally, the MAE for Matrix Factorization is 0.2372, which is also lower than the MAE for Collaborative Filtering (0.4518).
Conclusion
The results for both the methods show that Matrix Factorization can be the better choice for implementing an E-commerce Recommendation System. But this cannot be said for sure as Matrix Factorization uses the entire dataset whereas Collaborative Filtering is using the data for only 15K users due to system limitations.
Future Works
For future works, performance of Collaborative Filtering on the entire dataset can be analyzed and improved. Both the methods can be implemented on a dataset containing ratings for the products and then their performance can be evaluated. The research done for this project suggests that a better recommendation system can be built using the ratings for a product, this aspect of the project can be explored further.
References
[1]	J. Yu et al., "Collaborative Filtering Recommendation with Fluctuations of User’ Preference," 2021 IEEE International Conference on Information Communication and Software Engineering	(ICICSE),	Chengdu,	China,	2021,	pp.	222-226,	doi:
10.1109/ICICSE52190.2021.9404120.
[2]	Schettler, Darien. “Recommender System - e-Commerce Dataset - 2020.” Kaggle, 1 Jan. 2021, https://www.kaggle.com/datasets/dschettler8845/recsys-2020-ecommerce-dataset.
[3]	Hao Wang, "Fair recommendation by geometric interpretation and analysis of matrix factorization", International Symposium on Robotics, Artificial Intelligence, and Information Engineering (RAIIE 2022), pp.80, 2022.
