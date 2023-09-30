# Armut Association Rule Based Recommender System

## Business Problem:
Armut, Turkey's largest online service platform, brings together service providers and those who want to receive service. It allows you to easily access services such as cleaning, renovation and transportation with a few taps on your computer or smartphone. It is desired to create a product recommendation system with Association Rule Learning, using the data set containing service users and the services and categories they receive.

## Data set:
The data set consists of the services received by customers and the categories of these services. It contains date and time information of each service received.

## Variables:

UserId: Customer number

ServiceId: Anonymized services belonging to each category. (Example: Sofa washing service under the cleaning category) A ServiceId can be found under different categories and represents different services under different categories. (Example: The service with a CategoryId of 7 and a ServiceId of 4 is radiator cleaning, while the service with a CategoryId of 2 and ServiceId of 4 is radiator cleaning) service furniture assembly)

CategoryId: Anonymized categories. (Example: Cleaning, transportation, renovation category)

CreateDate: Date the service was purchased
