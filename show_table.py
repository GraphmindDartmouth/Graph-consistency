import pandas as pd

res = pd.read_csv('./results.csv', index_col=0)
res_dist = pd.read_csv('./results_dist.csv', index_col=0)
final_df = pd.DataFrame(columns=res.columns, index=[x for i in res.index for x in (i, '+L_consistency')])

for n in range(len(res.index)):
    final_df.iloc[2*n,:] = res.iloc[n,:]
    final_df.iloc[2*n+1,:] = res_dist.iloc[n,:]

final_df.to_csv('./final_results.csv')
print(final_df)

