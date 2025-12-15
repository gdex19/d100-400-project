# %%
from final_project.data import read_data, split_data

# from final_project.preprocessing import NUM_FEATURES, CAT_FEATURES

# %%
df = read_data("clean_data")

df.head()
# %%
# Perform train/test/val split: 0.6, 0.2, 0.2
split = [0.6, 0.2, 0.2]
df_train, df_test, df_val = split_data(df, split)
print(f"Lengths {len(df_train)}, {len(df_test)}, {len(df_val)}")
# %%
