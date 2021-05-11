from python_utils import load_spam_dataset

df_train, df_test = load_spam_dataset()

# We pull out the label vectors for ease of use later
#Y_test = df_test.label.values
print(df_train)