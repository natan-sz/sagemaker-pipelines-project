import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer

# Create the required arguments that are needed to run the procesing job
parser = argparse.ArgumentParser()
parser.add_argument('--train-test-split-ratio', type=float, default=0.3)

args, _ = parser.parse_known_args()



# Print out the arguments recieved 
print(f"Recieved Arguments {args}")
split_ratio = args.train_test_split_ratio

# Join up path for csvv file
input_data = os.path.join("/opt/ml/processing/input","inflation interest unemployment.csv")

# Load in the data to DF
df = pd.read_csv(input_data)



#############  TRANSFORM THE DATA HERE #########

# Select only some rows
df = df[["country","Inflation, GDP deflator (annual %)","Unemployment, total (% of total labor force) (modeled ILO estimate)","incomeLevel"]]

#Split the data
X_train, X_test, y_train, y_test = train_test_split(df.drop("incomeLevel",axis=1),
                                                    df["incomeLevel"],
                                                    test_size=split_ratio,
                                                    random_state=0)

# Create a column transfer for Standard Scaling of Inflation and Unemployment
preprocess = make_column_transformer(
    (StandardScaler(), ["Inflation, GDP deflator (annual %)","Unemployment, total (% of total labor force) (modeled ILO estimate)"])
)

train_features = preprocess.fit_transform(X_train)
test_features = preprocess.transform(X_test)

###############################################



############# OUTPUT OF THE DATA ############

train_features_output_path = os.path.join("/opt/ml/processing/train","train_features.csv")
train_labels_output_path = os.path.join("/opt/ml/processing/train","train_labels.csv")
test_features_output_path = os.path.join("/opt/ml/processing/test","test_features.csv")
test_label_output_path = os.path.join("/opt/ml/processing/test","test_labels.csv")

pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)
pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)

y_train.to_csv(train_labels_output_path, header=False, index=False)
y_test.to_csv(test_label_output_path, header=False, index=False)



