#import necessary libraries
library(neuralnet)   
library(readxl)
library(ggplot2)
library(reshape2)
library(gridExtra)


#read the dataset from excel file
exchange_data<-read_excel("/Users/saraperera/Desktop/ML/ExchangeUSD (2).xlsx")

# Display the structure of the dataset
str(exchange_data)

# Check the first few rows of the dataset
head(exchange_data)

# Check for missing values in the exchange data 
sum(is.na(exchange_data))


# Extract USD/EUR exchange rates (3rd column)
exchange_rate <- exchange_data$`USD/EUR`

# Define the number of training samples
train_samples <- 400

# Split data into training and testing sets
train_data <- exchange_rate[1:train_samples]
test_data <- exchange_rate[(train_samples + 1):length(exchange_rate)]

# Function to normalize data
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# Normalize training and testing data
train_data_norm<- normalize(train_data)
test_data_norm<- normalize(test_data)


# Function to create lagged input vectors
create_input_vectors <- function(data, time_delays) {
  input_matrix <- matrix(NA, nrow = length(data) - max(time_delays), ncol = length(time_delays))
  for (i in 1:length(time_delays)) {
    input_matrix[, i] <- data[i:(length(data) - max(time_delays) + i - 1)]
  }
  return(input_matrix)
}


# Specify time delays 
time_delays <- 1:4

# Create input/output matrices for training/testing
input_train <- create_input_vectors(train_data_norm, time_delays)
input_test <- create_input_vectors(test_data_norm, time_delays)
head(input_train)

# Function to create output vectors
create_output_vectors <- function(data, time_delays) {
  output_vector <- data[(max(time_delays) + 1):length(data)]
  return(output_vector)
}

# Create input/output matrices for training/testing
output_train <- create_output_vectors(train_data_norm, time_delays)
output_test <- create_output_vectors(test_data_norm, time_delays)

# Reshape output_train to match the structure of input_train
output_train <- matrix(output_train, nrow = nrow(input_train), ncol = 1)

# Check the structure of output_train_matrix
head(output_train)

colnames(input_train)
colnames(output_train)
colnames(input_train) <- paste0("Lag", 1:ncol(input_train))
colnames(output_train) <- "output_data"
head(input_train)
head (output_train)

#Function to train the neural network
train_neural_network <- function(input_data, output_data, hidden_layers,
                                 activation_function, learning_rate) {
  formula <- as.formula(paste("output_data ~", paste(colnames(input_data), 
                                                     collapse = " + "), sep=""))
  model <- neuralnet(formula, data = cbind(input_data, output_data), 
                     hidden = hidden_layers,
                     linear.output = FALSE, 
                     act.fct = activation_function,
                     learningrate = learning_rate,
                     algorithm = "sag")
  
  return(model)
}

# Train the neural network model
mlp_model <- train_neural_network(
  input_data = input_train,
  output_data = output_train,
  hidden_layers = c(4,5), 
  activation_function = "tanh",
  learning_rate = 0.04 
)


#Predictions
predictions <- predict(mlp_model, input_test) #model is the mlp model


# Denormalize data (reverse min-max scaling) cus we used min max scaling at the beginning
denormalize <- function(x, original_data) {
  minOriginal <- min(original_data)
  maxOriginal <- max(original_data)
  denormalizedData <- x * (maxOriginal - minOriginal) + minOriginal
  return(denormalizedData)
}

# Denormalize predictions and test output
denormalized_predictions <- denormalize(predictions, test_data)
denormalized_test_output <- denormalize(output_test, test_data)

denormalized_predictions
denormalized_test_output

# Function to calculate RMSE
rmse <- function(actual, predicted) {
  return(sqrt(mean((actual - predicted)^2)))
}

# Function to calculate MAE
mae <- function(actual, predicted) {
  return(mean(abs(actual - predicted)))
}

# Function to calculate MAPE
mape <- function(actual, predicted) {
  return(mean(abs((actual - predicted) / actual)) * 100)
}

# Function to calculate sMAPE
smape <- function(actual, predicted) {
  return(2 * mean(abs(actual - predicted) / (abs(actual) + abs(predicted))) * 100)
}

# Calculate performance metrics
rmse_value <- rmse(denormalized_test_output, denormalized_predictions)
mae_value <- mae(denormalized_test_output, denormalized_predictions)
mape_value <- mape(denormalized_test_output, denormalized_predictions)
smape_value <- smape(denormalized_test_output, denormalized_predictions)

# Print performance metrics
print(paste("RMSE:", rmse_value))
print(paste("MAE:", mae_value))
print(paste("MAPE:", mape_value))
print(paste("sMAPE:", smape_value))


# Plot predicted vs actual values
plot(denormalized_test_output, type = "l", col = "darkgreen", xlab = "Index", 
     ylab = "Value", main = "Predicted vs Actual Values")
lines(denormalized_predictions, col = "red")
legend("topleft", legend = c("Actual", "Predicted"), col = c("darkgreen", "red"), lty = 1)

# Visualize the neural network structure
plot(mlp_model)
summary(mlp_model)

# Plot predicted vs actual values (scatter plot)
plot(denormalized_test_output, denormalized_predictions, col = "blue", 
     xlab = "Desired Output", ylab = "Predicted Output", 
     main = "Predicted vs Actual Values (Scatter Plot)")
abline(0, 1, col = "red", lty = 2)  # Add a diagonal line for reference
legend("topleft", legend = "Ideal Prediction", col = "red", lty = 2)



