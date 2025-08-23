import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import joblib


def load_preprocess_data(filePath):
    data = pd.read_csv("./laptopPrice.csv")

    categorical_columns = [
        "brand",
        "processor_brand",
        "processor_name",
        "processor_gnrtn",
        "ram_type",
        "os",
        "weight",
        "rating",
        "warranty",
        "Touchscreen",
        "msoffice",
    ]

    df = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    df = clean_numeric_data(df)

    df["Number_of_Ratings"] = np.log1p(df["Number_of_Ratings"])
    df["Number_of_Reviews"] = np.log1p(df["Number_of_Reviews"])

    df = df.astype(int)

    return df


def clean_numeric_data(df):
    df["ram_gb"] = df["ram_gb"].str.replace("GB", "").astype(int)

    df["ssd"] = df["ssd"].str.replace("GB", "")
    df["ssd"] = df["ssd"].str.replace("TB", "000")
    df["ssd"] = df["ssd"].astype(int)

    df["hdd"] = df["hdd"].str.replace("TB", "000")
    df["hdd"] = df["hdd"].str.replace("GB", "")
    df["hdd"] = df["hdd"].astype(int)

    df["graphic_card_gb"] = df["graphic_card_gb"].str.replace("TB", "000")
    df["graphic_card_gb"] = df["graphic_card_gb"].str.replace("GB", "")
    df["graphic_card_gb"] = df["graphic_card_gb"].astype(int)

    df["os_bit"] = df["os_bit"].str.replace("-bit", "").astype(int)

    return df


def train_model(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models


def evaluate_model(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R2_Score": r2_score(y_test, y_pred),
        }

    return results, y_pred


# print(df);


# One hot encoding the columns brand and processor brand


# print(X_test.head)


def predict_custom_sample(model, X_test, sample_index=0):
    test_data = X_test.iloc[sample_index].values.reshape(1, -1)
    sample = pd.DataFrame(test_data, columns=X_test.columns)
    predicted_price = model.predict(sample)

    return predicted_price[0]


def plot_data(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Laptop Price Prediction")
    plt.show()
    
def grid_search_accuracy(X_train, y_train):
    # hyperparameter tuning
    rf = RandomForestRegressor()
    
    params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }
    
    grid = GridSearchCV(rf, params, scoring='r2', n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)
    
    return grid

def best_model_plot(best_model, X):
    important = best_model.feature_importances_
    indices = np.argsort(important)[::-1]
    
    plt.figure(figsize=(8,5))
    plt.bar(range(X.shape[1]), important[indices])
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.title("Feature Importance in Laptop Price Prediction")
    plt.show()
    
    return 0


def main():
    print("Loading Data...")
    df = load_preprocess_data("./laptopPrice.csv")

    X = df.drop("Price", axis=1)
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining model...")
    models = train_model(X_train, y_train)

    print("\nevaluating model...")
    results, y_pred = evaluate_model(models, X_test, y_test)

    print("\nModel performance metrics: ")
    for name, model_evaluation in results.items():
        print(f"\nEvaluation for the model {name}: ")
        for error_name, value in model_evaluation.items():
            print(f"\t{error_name}: {value}")

    tabled_results = pd.DataFrame(results).T
    print(tabled_results)
    # print("MAE: ", mean_absolute_error(y_test, y_pred))
    # print("MSE: ", mean_squared_error(y_test, y_pred))
    # print("RE Score: ", r2_score(y_test, y_pred))
    # print(df.dtypes)

    # sample_index = input("Enter the number of sample: ")
    # sample_index = int(sample_index) - 1
    # print(sample_index)

    # sample_price = predict_custom_sample(lr, X_test, sample_index)
    # print(f"Predicted Price for Sample: {sample_price}")

    # print("\nGenerating visualization: ")
    # plot_data(y_test, y_pred)

    # print("Linear Regression R2: ", lr.score(X_test, y_test))
    # print("Decision Tree R2: ", dt.score(X_test, y_test))
    # print("Random Forest R2: ", rf.score(X_test, y_test))
    
    
    # hyperparameter tuning
    grid = grid_search_accuracy(X_train, y_train)
    
    print("Best Parameters: ", grid.best_params_)
    print("Best Score: " , grid.best_score_)
    # print("Best Score: " , grid.best_estimator_)
    
    best_model = grid.best_estimator_
    
    best_model_plot(best_model, X)
    
    joblib.dump(best_model, "laptop_price_model.pkl")

    return results


if __name__ == "__main__":
    results = main()
