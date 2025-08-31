# Predicting Restaurant Revenue Using Machine Learning

This project utilized a Kaggle dataset and machine learning to predict restaurant revenue, focusing on regression modeling.

Dataset Features:
 Restaurant Attributes: Ratings, Seating Capacity, Average Meal Price, Marketing Budget, Social Media Followers, Chef Experience, Service Quality, Location, Cuisine, and Parking Availability.

Model and Performance (Linear Regression with numeric and categorical features):

 R² Score: 0.957 (Explaining approximately 96% of revenue variance)
 RMSE: 54,487 (Indicating strong predictive performance)

Key Insights on Feature Impact (Derived from correlation and analysis):

* Strong Positive Correlation:
 1. Average Meal Price and Seating Capacity: Larger restaurants with higher prices generally earn more.

 2. Positive Correlation:
 Ratings, Chef Experience and Service Quality: Higher quality experiences lead to increased customer satisfaction and sales.
 
3. Moderate Correlation:
 Marketing Budget and Social Media Followers:  Visibility is important, particularly for newer establishments.

 4. Location and Cuisine Influence:
 Urban Locations and Popular Cuisines: Certain urban areas and popular cuisines (e.g., Italian, Indian) correlate with higher revenue.

 5. Seasonality and Festive Impact:
 Holiday and Festival Spikes: Revenue increases during holidays and festivals, emphasizing the importance of strategic timing for promotions and capacity planning.

* Feature Selection Significance: A simplified model using only the strongest positive features performed nearly as well as the full model, demonstrating the importance of effective feature selection. This suggests that Average Meal Price, Seating Capacity, Ratings, Chef Experience, and Service Quality are key drivers of revenue.  While other factors like Marketing and Location play a role, their impact is less pronounced.  This finding can help restaurant owners prioritize their efforts and investments for maximal revenue growth.  
For example, focusing on improving food quality and service might yield better returns than increasing marketing spend.  Similarly, optimizing seating capacity and pricing strategies could significantly impact revenue.  
This insight underscores the value of data-driven decision-making in the restaurant industry.
