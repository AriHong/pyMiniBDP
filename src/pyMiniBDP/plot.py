import numpy as np
import pandas as pd
import seaborn as sns

def get_clustermap(filt_ad, targetid, sampleid=None, ysize=12, colsort=None, proba=None, col_colors=None):

    df = filt_ad.to_df('log10')[targetid].T
    df['gene_name'] = filt_ad.var.loc[targetid]['EntrezGeneSymbol']
    df = df.set_index('gene_name')
    obs = filt_ad.obs.copy()

    if sampleid is not None:
        df = df.loc[:, sampleid]
        obs = obs.loc[sampleid]
        col_colors =col_colors.loc[sampleid]
    
    if proba is not None:
        colsort = np.argsort(proba)
        df = df.iloc[:,colsort]
        col_cluster=False
        
        obs.loc['proba'] = proba
        proba_colors = {p: c for (p, c) in zip(np.sort(proba), sns.color_palette('flare', len(proba)))}
        proba_colors = obs['proba'].map(proba_colors)
        if col_colors is None:
            col_colors = pd.DataFrame(proba_colorS)
        else:
            col_colors['proba'] = proba_colors
        
        
    elif colsort is not None:

        df = df.iloc[:, colsort]
        col_cluster=False

    else:
        col_cluster=True
    


    sns.clustermap(df,
               cmap='coolwarm', center=0,
               col_colors=col_colors, col_cluster=col_cluster,
                   #row_colors=rowcolors,
                   yticklabels=True, method='ward', z_score=0, figsize=(12, ysize), vmax=2, vmin=-2
                  )