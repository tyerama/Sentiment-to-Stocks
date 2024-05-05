# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel) (Local)
#     language: python
#     name: conda-root-py
# ---

# +
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from gcloud import storage
from io import BytesIO, StringIO
from textblob import TextBlob
import pickle
from flask import render_template

app = Flask(__name__)
# -

# Step 1: Load stock price data
companies = ['CHKP', 'LUMI.TA', 'MBLY', 'NICE', 'POLI.TA']
stock_data = {}
# Create a GCS client
client = storage.Client()


def read_csv_from_gcs(bucket_name, file_path):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = storage.Blob(file_path, bucket)
    content = blob.download_as_text()
    return pd.read_csv(StringIO(content))


for company in companies:
    file_path = f'{company}.csv'
    stock_data[company] = read_csv_from_gcs('datasets_project_group_20', file_path)

    # Display the first 5 rows of the DataFrame for each company
    print(f"First 5 rows of {company}:\n{stock_data[company].head()}\n")

# +

# Step 2: Load sentiment analysis data
file_path = 'reddit_data.csv'
reddit_data = read_csv_from_gcs('datasets_project_group_20', file_path)

# Display the first 5 rows of the DataFrame
print(f"First 5 rows of reddit_data:\n{reddit_data.head()}")


# +
# Save stock_data and reddit_data using pickle
with open('stock_data.pkl', 'wb') as stock_file:
    pickle.dump(stock_data, stock_file)

with open('reddit_data.pkl', 'wb') as reddit_file:
    pickle.dump(reddit_data, reddit_file)

# Load stock_data and reddit_data using pickle
with open('stock_data.pkl', 'rb') as stock_file:
    loaded_stock_data = pickle.load(stock_file)

with open('reddit_data.pkl', 'rb') as reddit_file:
    loaded_reddit_data = pickle.load(reddit_file)

# +
from textblob import TextBlob

# Assuming 'self_text' is the column containing comments
def analyze_sentiment(text):
    if isinstance(text, str):
        analysis = TextBlob(text)
        # Assign sentiment labels (0 for neutral, 1 for positive, -1 for negative)
        return 0 if analysis.sentiment.polarity == 0 else (1 if analysis.sentiment.polarity > 0 else -1)
    else:
        return 0  # Assuming NaN values are neutral

# Apply sentiment analysis to the Reddit data
reddit_data['sentiment'] = reddit_data['self_text'].apply(analyze_sentiment)

# You can drop the 'self_text' column if it's no longer needed
reddit_data = reddit_data.drop(columns=['self_text'])

# Display the first 5 rows of the DataFrame
print(f"First 5 rows of reddit_data:\n{reddit_data.head()}")
# Convert 'created_time' to datetime and extract date
reddit_data['created_time'] = pd.to_datetime(reddit_data['created_time'])
reddit_data['Date'] = reddit_data['created_time'].dt.strftime('%Y-%m-%d')

# +
# Group by date and sentiment, then calculate the percentage for each sentiment
sentiment_percentage = reddit_data.groupby(['Date', 'sentiment']).size().unstack(fill_value=0)
sentiment_percentage = sentiment_percentage.div(sentiment_percentage.sum(axis=1), axis=0)

# Rename columns for clarity
sentiment_percentage.columns = ['negative_sentiment', 'neutral_sentiment', 'positive_sentiment']

# Reset index to make 'date' a regular column
sentiment_percentage = sentiment_percentage.reset_index()

# Display the result
print(sentiment_percentage)
# Convert 'Date' to datetime in sentiment_percentage
sentiment_percentage['Date'] = pd.to_datetime(sentiment_percentage['Date'])
# -

# Create an empty list to store merged dataframes for all companies
all_merged_data = []

# Merge each company's stock data with sentiment_percentage
for company in companies:
    # Convert 'Date' to datetime in each company's stock data
    stock_data[company]['Date'] = pd.to_datetime(stock_data[company]['Date'])
    
    # Merge with sentiment_percentage on 'Date'
    merged_data = pd.merge(stock_data[company], sentiment_percentage, how='left', left_on='Date', right_on='Date')
    
    # Fill missing values with zeros (if there are dates without sentiment data)
    merged_data = merged_data.fillna(0)
    
    # Add a new column 'Company' with the company name
    merged_data['Company'] = company
    
    # Append the merged dataframe to the list
    all_merged_data.append(merged_data)

# +
# Concatenate all dataframes in the list to create a single dataframe
final_merged_data = pd.concat(all_merged_data, ignore_index=True)

# Display the result for all companies
print("Merged data for all companies:\n", final_merged_data.head())


# -

# Define a function to train and predict using the specified model
def predict_stock_prices(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)


# +
# Dictionary to store predicted dataframes for each company
predicted_dfs = {}

# Loop through each company
for company in companies:
    # Select features and target variable for the current company
    company_data = final_merged_data[final_merged_data['Company'] == company]
    features = company_data[['negative_sentiment', 'neutral_sentiment', 'positive_sentiment']]
    target = company_data['Close']  # Assuming 'Close' is the target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Random Forest Regression
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_predictions = predict_stock_prices(rf_model, X_train, y_train, X_test)

    # Linear Regression
    lr_model = LinearRegression()
    lr_predictions = predict_stock_prices(lr_model, X_train, y_train, X_test)

    # Support Vector Regression
    svr_model = SVR(kernel='linear')
    svr_predictions = predict_stock_prices(svr_model, X_train, y_train, X_test)

    # Create a dataframe with actual and predicted values
    predicted_df = pd.DataFrame({
        'Date': company_data.loc[X_test.index, 'Date'].values,
        'Actual': y_test.values,
        f'Predicted_RF_{company}': rf_predictions,
        f'Predicted_LR_{company}': lr_predictions,
        f'Predicted_SVR_{company}': svr_predictions
    })

    # Sort the dataframe by 'Date' in ascending order
    predicted_df = predicted_df.sort_values('Date')

    # Add the dataframe to the dictionary
    predicted_dfs[company] = predicted_df

# Print or use the dataframes as needed
for company, predicted_df in predicted_dfs.items():
    print(f"\nPredicted {company}:")
    print(predicted_df)



# +
# Function to plot the data for a specific company
def plot_predictions(df, company_name):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Actual'], label='Actual', marker='o')
    plt.plot(df['Date'], df[f'Predicted_RF_{company_name}'], label=f'Predicted_RF_{company_name}', marker='o')
    plt.plot(df['Date'], df[f'Predicted_LR_{company_name}'], label=f'Predicted_LR_{company_name}', marker='o')
    plt.plot(df['Date'], df[f'Predicted_SVR_{company_name}'], label=f'Predicted_SVR_{company_name}', marker='o')
    
    plt.title(f"Stock Price Predictions for {company_name}")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# Plot for CHKP
plot_predictions(predicted_dfs['CHKP'], 'CHKP')

# Plot for LUMI.TA
plot_predictions(predicted_dfs['LUMI.TA'], 'LUMI.TA')

# Plot for MBLY
plot_predictions(predicted_dfs['MBLY'], 'MBLY')

# Plot for NICE
plot_predictions(predicted_dfs['NICE'], 'NICE')

# Plot for POLI.TA
plot_predictions(predicted_dfs['POLI.TA'], 'POLI.TA')
# -

# Loop through each company
for company in companies:
    # Select features and target variable for the current company
    company_data = final_merged_data[final_merged_data['Company'] == company]
    features = company_data[['negative_sentiment', 'neutral_sentiment', 'positive_sentiment']]
    target = company_data['Close']  # Assuming 'Close' is the target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Random Forest Regression
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_predictions = predict_stock_prices(rf_model, X_train, y_train, X_test)
    rf_mse = mean_squared_error(y_test, rf_predictions)

    # Linear Regression
    lr_model = LinearRegression()
    lr_predictions = predict_stock_prices(lr_model, X_train, y_train, X_test)
    lr_mse = mean_squared_error(y_test, lr_predictions)

    # Support Vector Regression
    svr_model = SVR(kernel='linear')
    svr_predictions = predict_stock_prices(svr_model, X_train, y_train, X_test)
    svr_mse = mean_squared_error(y_test, svr_predictions)

    # Dynamically divide MSE values by 10 until they become less than 1
    scaling_factor = 1
    while max(rf_mse, lr_mse, svr_mse) >= 1:
        rf_mse /= 10
        lr_mse /= 10
        svr_mse /= 10
        scaling_factor *= 10

    # Print scaled MSE values for each model
    print(f'Mean Squared Error for {company}:')
    print(f'Random Forest: {rf_mse}')
    print(f'Linear Regression: {lr_mse}')
    print(f'SVR: {svr_mse}')
    print('---------------------------')

    # Create a dataframe with actual and predicted values
    predicted_df = pd.DataFrame({
        'Date': company_data.loc[X_test.index, 'Date'].values,
        'Actual': y_test.values,
        f'Predicted_RF_{company}': rf_predictions,
        f'Predicted_LR_{company}': lr_predictions,
        f'Predicted_SVR_{company}': svr_predictions
    })

    # Sort the dataframe by 'Date' in ascending order
    predicted_df = predicted_df.sort_values('Date')

    # Add the dataframe to the dictionary
    predicted_dfs[company] = predicted_df


@app.route('/get_plot', methods=['POST'])
def get_plot():
    data = request.get_json()

    company = data.get('company')
    model = data.get('model')

    # Select features and target variable for the selected company
    company_data = final_merged_data[final_merged_data['Company'] == company]
    features = company_data[['negative_sentiment', 'neutral_sentiment', 'positive_sentiment']]
    target = company_data['Close']  # Assuming 'Close' is the target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    if model == 'RandomForest':
        model_instance = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model == 'LinearRegression':
        model_instance = LinearRegression()
    elif model == 'SVR':
        model_instance = SVR(kernel='linear')
    else:
        return jsonify({'error': 'Invalid model selection'})

    # Train the model and make predictions
    model_instance.fit(X_train, y_train)
    predictions = model_instance.predict(X_test)

    # Create a dataframe with actual and predicted values
    predicted_df = pd.DataFrame({
        'Date': company_data.loc[X_test.index, 'Date'].values,
        'Actual': y_test.values,
        f'Predicted_{model}_{company}': predictions
    })

    # Sort the dataframe by 'Date' in ascending order
    predicted_df = predicted_df.sort_values('Date')

    # Plot the data
    plot_predictions(predicted_df, company)

    # You can also save the plot to an image file and return the file path
    # plt.savefig('path/to/your/image.png')

    return jsonify({'status': 'success'})


if __name__ == '__main__':
    # Use gunicorn as the production server
    import os
    from gunicorn.app.base import BaseApplication
    
    class StandaloneApplication(BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()

        def load_config(self):
            config = {key: value for key, value in self.options.items() if key in self.cfg.settings and value is not None}
            for key, value in config.items():
                self.cfg.set(key.lower(), value)

        def load(self):
            return self.application

    options = {
        'bind': f'0.0.0.0:{os.environ.get("PORT", 8080)}',
        'workers': 1,
        'timeout': 120,
    }

    StandaloneApplication(app, options).run()


