library(sparklyr)
library(dplyr)
library(dbplot)

#To connect to Spark local
sc <- spark_connect(master = "local")

#sdf_copy_to is used to copy data into Spark from R
iris_tbl <- sdf_copy_to(sc, iris, name = "iris_tbl", overwrite = TRUE)

#To get an overview of the data in the Spark table
glimpse(iris_tbl)

#dbplot is used to access data in spark and plot it using ggplot like functions
dbplot_histogram(iris_tbl,Sepal_Width)
dbplot_histogram(iris_tbl,Petal_Width)


#Splitting the dataset into test and train
partitions <- iris_tbl %>%
  sdf_random_split(training = 0.7, test = 0.3, seed = 1111)

iris_training <- partitions$training
iris_test <- partitions$test

#Description of the training set to understand the distribution of data
sdf_describe(iris_training)


#Applying Random Forest on the dataset
rf_model <- iris_training %>%
  ml_random_forest(Species ~ ., type = "classification")

#Predicting values on the test set using the model created earlier
pred <- ml_predict(rf_model, iris_test)
pred

#Confusion Matrix to evaluate classification accuracy
confusionmatrix <- pred %>% 
  sdf_crosstab("prediction", "Species")
confusionmatrix

#Performance Evaluator to calculate accuracy
ml_multiclass_classification_evaluator(pred)
