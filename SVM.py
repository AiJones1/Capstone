import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('BitcoinHeistData.csv')

# Define ransomware list and apply binary labels
ransomware_list = [
    'princetonCerber', 'princetonLocky', 'montrealCryptoLocker', 'montrealCryptXXX',
    'montrealWannaCry', 'montrealDMALockerv3', 'montrealCryptoTorLocker2015', 'montrealSamSam',
    'montrealFlyper', 'montrealNoobCrypt', 'montrealDMALocker', 'montrealGlobe', 'montrealEDA2',
    'montrealVenusLocker', 'montrealXTPLocker', 'montrealGlobev3', 'montrealJigSaw',
    'montrealXLockerv5.0', 'montrealXLocker', 'montrealRazy', 'montrealCryptConsole',
    'montrealGlobeImposter', 'montrealSam', 'montrealComradeCircle', 'montrealAPT',
    'paduaCryptoWall', 'paduaKeRanger', 'paduaJigsaw'
]
data['label'] = data['label'].apply(lambda x: 1 if x in ransomware_list else 0)

# Feature selection and scaling
features = [col for col in data.columns if col not in ['label', 'address', 'year', 'day']]
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])
X = data[features]
y = data['label']

# Function to apply undersampling and SMOTE
def apply_undersample_and_smote(X, y):
    print("Applying random undersampling...")
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42) 
    X_res, y_res = rus.fit_resample(X, y)
    print(f"After undersampling: {X_res.shape}, {y_res.shape}")

    print("Applying SMOTE for oversampling...")
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_res, y_res = smote.fit_resample(X_res, y_res)
    print(f"After SMOTE: {X_res.shape}, {y_res.shape}")
    return X_res, y_res

# Evaluation function with Grid Search and undersampling/SMOTE
def train_and_evaluate_svm_with_sampling(X_train, y_train, X_test, y_test, label):
    custom_f1_scorer = make_scorer(f1_score, zero_division=0) 
    param_grid = {'C': [0.1, 1, 10], 'gamma': [ 0.1, 0.01], 'kernel': ['linear', 'rbf']}
    cv = StratifiedKFold(n_splits=5)
    model = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, scoring=custom_f1_scorer, cv=cv)

    # Apply undersampling and SMOTE to training data
    X_train_resampled, y_train_resampled = apply_undersample_and_smote(X_train, y_train)

    print("Starting Grid Search with resampled data...")
    model.fit(X_train_resampled, y_train_resampled)
    print(f"\nBest parameters from Grid Search: {model.best_params_}")

    # Use best model from grid search to make predictions
    best_model = model.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calculate and display metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\nMetrics for {label} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")

    return {'Label': label, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

# Control test on the entire dataset (80-20 split)
def control_test():
    print("\n--- Control Test on Entire Dataset (80-20 Split with Stratification) ---")
    
    # Perform an 80-20 split on the full dataset with stratification
    training_set, test_set = train_test_split(
        data, 
        test_size=0.2, 
        random_state=2, 
        stratify=data['label'] 
    )

    # Drop the 'label' column for feature matrices and 'address' for compatibility
    X_train = training_set.drop(columns=['label', 'address']).values
    y_train = training_set['label'].values
    X_test = test_set.drop(columns=['label', 'address']).values
    y_test = test_set['label'].values

    label = "Control Test (80-20 Split)"
    metrics = train_and_evaluate_svm_with_sampling(X_train, y_train, X_test, y_test, label)
    
    return pd.DataFrame([metrics])

# Function to test individual years
def test_individual_years():
    results = []
    for year in range(2011, 2017):
        print(f"\n--- Training Model for Year: {year} and Testing on 2017-2018 ---")
        train_data = data[data['year'] == year]
        X_train = train_data[features]
        y_train = train_data['label']
        X_test = data[data['year'] >= 2017][features]
        y_test = data[data['year'] >= 2017]['label']

        label = f"Train {year} - Test 2017-2018"
        metrics = train_and_evaluate_svm_with_sampling(X_train, y_train, X_test, y_test, label)
        results.append(metrics)
    
    return pd.DataFrame(results)

# Function to test pair of years
def test_pair_years():
    results = []
    for start_year in range(2011, 2016, 2):
        end_year = start_year + 1
        print(f"\n--- Training Model for Years: {start_year}-{end_year} and Testing on 2017-2018 ---")
        
        train_data = data[(data['year'] == start_year) | (data['year'] == end_year)]
        X_train = train_data[features]
        y_train = train_data['label']
        
        X_test = data[data['year'] >= 2017][features]
        y_test = data[data['year'] >= 2017]['label']

        label = f"Train {start_year}-{end_year} - Test 2017-2018"
        metrics = train_and_evaluate_svm_with_sampling(X_train, y_train, X_test, y_test, label)
        results.append(metrics)
    
    return pd.DataFrame(results)

# Function to test six years combined (2011-2016)
def test_six_years():
    print("\n--- Training Model for Years 2011-2016 and Testing on 2017-2018 ---")
    
    train_data = data[data['year'] <= 2016]
    X_train = train_data[features]
    y_train = train_data['label']
    
    X_test = data[data['year'] >= 2017][features]
    y_test = data[data['year'] >= 2017]['label']

    label = "Train 2011-2016 - Test 2017-2018"
    metrics = train_and_evaluate_svm_with_sampling(X_train, y_train, X_test, y_test, label)
    
    return pd.DataFrame([metrics])

# Visualization function
def visualize_results(results_df):
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Label'], results_df['Accuracy'], label='Accuracy', marker='o')
    plt.plot(results_df['Label'], results_df['Precision'], label='Precision', marker='o')
    plt.plot(results_df['Label'], results_df['Recall'], label='Recall', marker='o')
    plt.plot(results_df['Label'], results_df['F1 Score'], label='F1 Score', marker='o')
    plt.xlabel('Training Data')
    plt.ylabel('Score')
    plt.title('Model Performance on Test Years (2017-2018)')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main execution block
if __name__ == "__main__":
    print("\n--- Starting Experiment with SVM using Undersampling and SMOTE ---\n")
    
    # Run control test first
    control_result = control_test()
    print("\nCompleted control test.")
    
    # Run individual year tests
    individual_year_results = test_individual_years()
    print("\nCompleted individual year tests.")
    
    # Run pair year tests
    pair_year_results = test_pair_years()
    print("\nCompleted pair year tests.")
    
    # Run six years combined test
    six_year_result = test_six_years()
    print("\nCompleted six-year combined test.")

    # Combine all results for visualization
    all_results = pd.concat([control_result, individual_year_results, pair_year_results, six_year_result], ignore_index=True)
    
    # Visualize all results
    visualize_results(all_results)
