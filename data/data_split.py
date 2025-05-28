import pandas as pd
import numpy as np

# 파일 로드
df_X = pd.read_csv("admet_X.tsv", sep="\t")
df_Y = pd.read_csv("admet_Y.tsv", sep="\t")

# id 기준으로 정렬
df_X = df_X.sort_values(by="id").reset_index(drop=True)
df_Y = df_Y.sort_values(by="id").reset_index(drop=True)

# 랜덤 샘플링 기반으로 3개의 그룹 생성
# np.random.seed(42)
# np.random.seed(5630)
np.random.seed(714)
groups = np.random.choice([1, 2, 3], size=len(df_X))

for i in [1, 2, 3]:
    idxs = np.where(groups == i)[0]
    df_X.iloc[idxs].to_csv(f"client{i}_3/admet_X.tsv", sep="\t", index=False)
    df_Y.iloc[idxs].to_csv(f"client{i}_3/admet_Y.tsv", sep="\t", index=False)