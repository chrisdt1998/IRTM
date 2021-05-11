from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import LabelModel
import pickle
import pandas as pd

class TrainingData():
	def __init__(self, data, size):
		self.data = data
		self.size = size

	def initialize(self):
		pass

	def cut_data(self):
		self.data = self.data[:self.size]

	def apply_labels(self):
		@labeling_function()
		def lf_keyword_john(x):
			return 1 if "John" in x else -1

		@labeling_function()
		def lf_keyword_jesus(x):
			return 1 if "Jesus" in x else -1

		@labeling_function()
		def lf_keyword_Lord(x):
			return 1 if "Lord" in x else -1

		@labeling_function()
		def lf_keyword_whom(x):
			return 0 if "whom" in x else -1

		lfs = [lf_keyword_john, lf_keyword_jesus, lf_keyword_Lord, lf_keyword_whom]
		applier = PandasLFApplier(lfs=lfs)
		self.df_train = pd.DataFrame(data=self.data, dtype="string")
		print(self.df_train)
		print(self.df_train.dtypes)
		return applier.apply(df=self.df_train)

	def label_model(self):
		L_train = self.apply_labels()
		label_model = LabelModel(cardinality=2, verbose=True)
		label_model.fit(L_train, n_epochs=500, log_freq=50, seed=123)
		self.df_train["label"] = label_model.predict(L=L_train, tie_break_policy="abstain")
		self.df_train = self.df_train[self.df_train.label != -1]
		print(self.df_train)
		print(len(self.df_train))



with open('processed_text.pkl', 'rb') as out:
	tokens = pickle.load(out)

df_train = TrainingData(tokens, 1000)
df_train.cut_data()
df_train.label_model()