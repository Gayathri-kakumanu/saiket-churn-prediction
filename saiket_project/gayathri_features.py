import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def configure_features_and_algorithm():
    print("\n=============================================")
    print("  GAYATHRI'S PROJECT - PHASE 3 & 4: SETUP    ")
    print("=============================================")
    
    # Load training splits
    training_features = pd.read_csv("train_features.csv")
    
    # Process text column into binary indicator metrics
    encoded_training = pd.get_dummies(training_features, columns=['Contract'])
    encoded_training.to_csv("finalized_train_inputs.csv", index=False)
    
    # Establish algorithm backbone structure
    classifier_engine = RandomForestClassifier(random_state=42)
    print(">>> Success: Configuration loaded.")
    
    return classifier_engine

if __name__ == "__main__":
    configure_features_and_algorithm()