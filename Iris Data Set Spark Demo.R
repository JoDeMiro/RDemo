Sys.setenv(SPARK_HOME = "/usr/local/spark-2.4.0-bin-hadoop2.7")


# install.packages("sparklyr")
# install.packages("devtools")
# install.packages("caret")
# install.packages("e1071")
# install.packages("ggplot2")
# devtools::install_github("rstudio/sparklyr")


library(sparklyr)
library(dplyr)
library(ggplot2)


conf <- spark_config()
conf$spark.executor.memory <- "3GB"
#conf$spark.memory.fraction <- 0.9
conf$spark.executor.cores <- 8
conf$spark.driver.memory <- "4GB"
conf$spark.dynamicAllocation.enabled <- "false"


sc <- spark_connect(master="spark://193.224.59.115:7077", 
                    version = "2.4.0",
                    app_name = "SparklyR-Nov17-LiveDemo",
                    config = conf)



## Spark Web User Interface
spark_web(sc)
## spark_log(sc)


## ~18 sec - irisDataSet464 - 550Mb - 20 million lines
irisDF = spark_read_csv(sc,
                        "irisDataSet",
                        "hdfs://193.224.59.115:9000/input/irisDataSet464Random.csv",
                        header = TRUE,
                        infer_schema = TRUE,
                        repartition = 28)

## Number of Partitions
sdf_num_partitions(irisDF)


## Repartition
irisDF <- sdf_repartition(irisDF, 80)
#80 VCPU: this is the optimal for our cluster


## Number of rows ~ 20 million lines
sdf_nrow(irisDF)


## Avaiable tables
src_tbls(sc)


## Iris data (dataFrame)
head(irisDF)

## Example for select dplyr
sepal_lengthDF = select(irisDF, sepal_length)

head(sepal_lengthDF)


## Example for filter dplyr
filtered_irisDF <- filter(irisDF, sepal_length > 6)

head(filtered_irisDF)


## Example for pipe
pipe_irisDF <- irisDF %>%
  filter(sepal_length > 6.1)

head(pipe_irisDF)


## Example for select
special_selected_irisDF <- irisDF %>%
  select(species, ends_with("length"))

head(special_selected_irisDF)


## Example mutate
mutate_irisDF <- irisDF %>%
  mutate(species, ratio = sepal_length / sepal_width) %>%
  select(species:sepal_length, sepal_width, ratio)

head(mutate_irisDF)


## Example arrange
arranged_irisDF <- irisDF %>%
  arrange(desc(sepal_length))

head(arranged_irisDF)


## Example group_by
groupby_irisDF <- irisDF %>%
  group_by(species) %>%
  summarise(
    n = n(),
    sepal_mean_length = mean(sepal_length, na.rm = TRUE)
  ) %>%
  filter(n > 1)

head(groupby_irisDF)

## End of dplyr examples

##---------------------------------------------------starts here
## Split data
partitions <- irisDF %>%
  sdf_partition(training = 0.7, test = 0.3, seed = 1111)

## partitions
mode(partitions)
names(partitions)
head(partitions$training)

## Split data
iris_training <- partitions$training
iris_test <- partitions$test

## Machine learing part <--------------------------

## Model - ~ 27 sec
dt_model <- iris_training %>%
  ml_decision_tree(species ~ ., max_depth = 2)

## Prediction
pred_train <- sdf_predict(iris_training, dt_model)
pred_test <- sdf_predict(iris_test, dt_model)

## Evaluation
ml_multiclass_classification_evaluator(pred_train)
ml_multiclass_classification_evaluator(pred_test)

## Number of rows
sdf_nrow(pred)



## Sampling

## <---------------------------------------------- downsampling

sampleDataFrame = sample_frac(pred, 0.0001)

head(sampleDataFrame)


## Collect downsampled data

## <---------------------------------------------- downsampling

sdf = sdf_collect(sampleDataFrame)


## Confusion matrix

## <---------------------------------------------- confusion matrix

table(sdf$species, sdf$predicted_label)


## Feature importance
ml_tree_feature_importance(dt_model)


## Evaluation - on the original dataset ~20 million lines
ml_multiclass_classification_evaluator(pred, metric_name = "accuracy")


## ---------------------- other machine learning models

# We can use the same formula for every models
ml_formula <- formula(species ~ sepal_length + sepal_width + petal_length + petal_width)

# Logistic Regression
ml_lr <- ml_logistic_regression(iris_training, ml_formula, max_iter = 2)

# Decision Tree
ml_dt <- ml_decision_tree(iris_training, ml_formula, max_depth = 3)

# Random Forest
ml_rf <- ml_random_forest(iris_training, ml_formula, max_depth = 2, subsampling_rate = 0.001)

# Neural Network - 4 variables, 15 node number in a layer, 3 number of classes
ml_nn <- ml_multilayer_perceptron_classifier(iris_training, ml_formula, layers = c(4,15,3), max_iter = 10)



## Predictions
pred_lr <- sdf_predict(iris_test, ml_lr)
pred_dt <- sdf_predict(iris_test, ml_dt)
pred_rf <- sdf_predict(iris_test, ml_rf)
pred_nn <- sdf_predict(iris_test, ml_nn)

## Evaluation
ml_multiclass_classification_evaluator(pred_lr)
ml_multiclass_classification_evaluator(pred_dt)
ml_multiclass_classification_evaluator(pred_rf)
ml_multiclass_classification_evaluator(pred_nn)




## Graphs

## <---------------------------------------------- aggregation

iris_summary <- irisDF %>% 
  mutate(sepal_width = ROUND(sepal_width * 2) / 2) %>% # Bucketizing Sepal_Width
  group_by(species, sepal_width) %>% 
  summarize(count = n(), sepal_length = mean(sepal_length, na.rm = T), stdev = sd(sepal_length)) %>% collect

iris_summary

## <---------------------------------------------- graphs

ggplot(iris_summary, aes(sepal_width, sepal_length, color = species)) + 
  geom_line(size = 1.2) +
  geom_errorbar(aes(ymin = sepal_length - stdev, ymax = sepal_length + stdev), width = 0.05) +
  geom_text(aes(label = count), vjust = -0.2, hjust = 1.2, color = "black") +
  theme(legend.position="top")




## Close Spark Connection <---------------------------- close spark session

## Close Spark Connection
spark_disconnect(sc)

## ToDo <---------------------------------------------- session info

# devtools::session_info()



