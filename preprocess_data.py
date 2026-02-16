import pandas as pd
import numpy as np

data = pd.read_csv("clean_data.csv")

data["ProdID"] = data["ProdID"].replace("2147483648", np.nan)
data["User ID"] = data["User ID"].replace("2147483648", np.nan)

data.dropna(subset=["User ID"])

data["User ID"] = data["User ID"].astype("int64")

data = data.dropna(subset=["ProdID"])
data["ProdID"] = data["ProdID"].astype("int64")

data["Review Count"] = data["Review Count"].astype("int64")

data["Category"] = data["Category"].fillna("")
data["Brand"] = data["Brand"].fillna("")
data["Description"] = data["Description"].fillna("")
data["Tags"] = data["Tags"].fillna("")

# to clean image data by choose only first url before the '|'

return cleaned_data
