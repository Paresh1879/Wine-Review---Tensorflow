# Wine Review Sentiment Analysis

This project aims to perform sentiment analysis on wine reviews using deep learning techniques, specifically using TensorFlow and TensorFlow Hub for embedding, and LSTM for sequence modeling.

## Dataset
The dataset used for this analysis is the Wine Reviews dataset, obtained from [Kaggle](https://www.kaggle.com/zynicide/wine-reviews). It contains various attributes of wine reviews, including the description, points, price, variety, and winery.

## Preprocessing
- **Missing Values**: Rows with missing values in the 'description' and 'points' columns were removed.
- **Label Creation**: A binary label was created based on the 'points' column, where wines with points >= 90 were labeled as positive sentiment (1), and others as negative sentiment (0).
- **Train-Validation-Test Split**: The dataset was split into training (80%), validation (10%), and test (10%) sets.

## TensorFlow Model Architecture
### Embedding and Layering
- A pre-trained embedding layer from TensorFlow Hub (NNLM-EN-DIM128) was used to convert text descriptions into fixed-size dense vectors.
- The embedded vectors were passed through two dense layers with ReLU activation.
- The final output layer used a sigmoid activation function for binary classification.

### LSTM
- A TextVectorization layer was used to tokenize the text descriptions and convert them into sequences of integers.
- An embedding layer was added to map the integer sequences to dense vectors.
- An LSTM layer with 32 units was used for sequence modeling.
- Two dense layers with ReLU activation and dropout regularization were added for classification.
- The final output layer used a sigmoid activation function.

## Model Evaluation
- Both models were compiled using binary cross-entropy loss and Adam optimizer.
- Model performance was evaluated on accuracy metrics.
- The models were trained and evaluated on training, validation, and test datasets.

## Results
- The TensorFlow Hub model achieved an accuracy of approximately 82.49% on the test dataset.
- The LSTM model achieved an accuracy of approximately 82.54% on the test dataset.

## Conclusion
Both models performed comparably well in sentiment analysis on wine reviews, with LSTM slightly outperforming the TensorFlow Hub model. Further fine-tuning and experimentation could potentially improve model performance.

For detailed implementation, refer to the Jupyter Notebook in this repository.

