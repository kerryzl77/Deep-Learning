library(dplyr)
library(GGally)
library(aplpack)
library(corrplot)
library(rpart)
library(rpart.plot)
library(factoextra)

# 2. Data Preparation
# Set working directory
folder_path <- file.path(dirname(rstudioapi::getSourceEditorContext()$path), "/Group Project ")
setwd(folder_path)

# Load and preprocess data
Crab_Data <- read.csv('CrabAgePrediction.csv') 
Crab_Data <- Crab_Data %>%
  mutate(Sex = as.factor(Sex)) %>%  # Ensure Sex is a factor
  mutate(Female = as.integer(Sex == 'F'),  # Create binary column for Female
         Male = as.integer(Sex == 'M'),    # Create binary column for Male
         Indeterminate = as.integer(Sex == 'I'))  # Create binary column for Infant

# 3. Exploratory Data Analysis (EDA)
# Correlation Matrix
Crab_Data_Corr <- Crab_Data[, sapply(Crab_Data, is.numeric)]
corrplot(cor(Crab_Data_Corr), method = 'circle')

# Scatterplot Matrix 
set.seed(123)
sample_size <- 30 # Sample
Crab_Data_Sample <- Crab_Data_Corr[sample(1:nrow(Crab_Data), sample_size), ]
pairs(Crab_Data_Sample)
# pairs(Crab_Data_Corr) # Whole Dataset

# Parallel Coordinates & Chernoff Faces
Crab_Data_Sample_Norm <- as.data.frame(scale(Crab_Data_Sample))

ggparcoord(data = Crab_Data_Sample_Norm %>% select_if(is.numeric), 
           groupColumn = NULL) 
faces(Crab_Data_Sample_Norm)

# MDS
dist_matrix <- dist(Crab_Data_Sample_Norm)  # Distance matrix
mds_result <- cmdscale(dist_matrix)  # Apply MDS
plot(mds_result[,1], mds_result[,2], xlab="Dimension 1", ylab="Dimension 2", main="Sample MDS Plot", type="n")
text(mds_result[,1], mds_result[,2])

Crab_Data_PCA_Sample <- prcomp(Crab_Data_Sample, scale. = TRUE)
pca_scores <- Crab_Data_PCA_Sample$x[, 1:2] 
plot(pca_scores, main = 'Sample PCA Biplot', type="n")  
text(pca_scores[,1], pca_scores[,2])

# 4. Dimensionality Reduction and Clustering
# PCA on Full Data
Crab_Data_PCA <- prcomp(Crab_Data_Corr, scale. = TRUE)
biplot(Crab_Data_PCA, main = 'PCA Biplot')

# Calculate the proportion of variance explained by each principal component
variance_explained <- Crab_Data_PCA$sdev^2 / sum(Crab_Data_PCA$sdev^2)
cumulative_variance <- cumsum(variance_explained)

# Plot the cumulative variance to determine how many components to retain
plot(cumulative_variance, xlab = "Number of Principal Components", ylab = "Cumulative Variance Explained",
     type = "b", pch = 19, main = "Cumulative Variance Explained by PCA Components")
abline(h = 0.9, col = "red", lty = 2)  # Add a line at 90% variance explained for reference

# K-means Clustering based on PCA
Crab_Data_PCA_Reduced <- Crab_Data_PCA$x[, 1:2] # 2 PC
kmeans_result <- kmeans(Crab_Data_PCA_Reduced, centers = 2)
plot(Crab_Data_PCA_Reduced[, 1], Crab_Data_PCA_Reduced[, 2], col = kmeans_result$cluster, main = "K-means Clustering on PCA Results")

# Visualize the average silhouette method
fviz_nbclust(Crab_Data_PCA_Reduced, kmeans, method = "silhouette", k.max = 20) +
  ggtitle("Silhouette Method for Optimal Clusters - PCA")
fviz_nbclust(Crab_Data_Corr, kmeans, method = "silhouette", k.max = 20) +
  ggtitle("Silhouette Method for Optimal Clusters")
# Visualize the within-cluster sum of squares method (elbow method)
fviz_nbclust(Crab_Data_PCA_Reduced, kmeans, method = "wss", k.max = 20) + 
  ggtitle("Elbow Method for Optimal Clusters - PCA")
fviz_nbclust(Crab_Data_Corr, kmeans, method = "wss", k.max = 20) +
  ggtitle("Elbow Method for Optimal Clusters")

# Sample compare
# Hierarchical clustering
hc_result <- hclust(dist_matrix)  # Apply hierarchical clustering
plot(hc_result)  # Plot the dendrogram
cutree(hc_result, k=3)  # Assuming you want to create 3 clusters

fviz_nbclust(Crab_Data_Sample_Norm, kmeans, method = "silhouette", k.max = 20) +
  ggtitle("Silhouette Method for Optimal Clusters - Sample")

# 5. Model Building: Decision Tree
# Split Data into Training and Testing
set.seed(123)
train_indices <- sample(1:nrow(Crab_Data), 0.7 * nrow(Crab_Data))
train_data <- Crab_Data[train_indices, ]
test_data <- Crab_Data[-train_indices, ]

# Build Decision Tree Model
model <- rpart(Age ~ ., data = train_data, method = "anova")

# Visualize the Decision Tree
rpart.plot(model, main="Decision Tree for Crab Age Prediction")

# Model Evaluation
predictions <- predict(model, test_data)
rmse <- sqrt(mean((predictions - test_data$Age)^2))
cat("RMSE on test data:", rmse, "\n")
