import seaborn as sns
import plotly.express as px
import umap
import numpy as np
import pandas as pd
import scipy.stats as ss

from .data_processing import PlotUMAPError, ImmunoPhenoData

def _correlation_ab(classified_df: pd.DataFrame,
                    z_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters:
        classified_df (pd.DataFrame): DataFrame of all cells and their
            classification as background or signal for a set of antibodies
            Format: rows: cells, col: antibodies
        z_scores_df (pd.DataFrame): DataFrame of z-scores of protein counts
            Format: rows: cells, col: antibodies

    Returns:
        correlation_df (pd.DataFrame): DataFrame containing pearson's correlation
            coefficients for each pair of antibodies
    """

    classified_transpose = classified_df.copy(deep=True).T
    z_transpose = z_scores_df.copy(deep=True).T

    # Before we start, we want to remove any rows that contain inf or NaN
    z_transpose.replace([np.inf, -np.inf], np.nan, inplace=True)
    z_transpose.dropna(axis=0, how='any', inplace=True)

    # Remove same rows from classified
    classified_transpose = classified_transpose.loc[z_transpose.index]

    ab_names = list(classified_transpose.index)
    num_ab = len(ab_names)

    correlation_matrix = np.zeros([num_ab, num_ab])

    for i in range(0, num_ab):
        for j in range(i + 1, num_ab):

            ab_a_zscores = []
            ab_b_zscores = []

            ab_a = list(classified_transpose.iloc[i])
            ab_b = list(classified_transpose.iloc[j])

            ab_bool = [x == 0 and y == 0 for x, y in zip(ab_a, ab_b)]
            ab_pair_bool = list(np.where(ab_bool)[0])

            z_score_a = np.array(z_transpose.iloc[i])
            z_score_b = np.array(z_transpose.iloc[j])

            # Add all z scores that belong to True in ab_pair_bool
            ab_a_zscores.extend(z_score_a[ab_pair_bool])
            # Ignore constant arrays that break the pearson correlation
            if ab_a_zscores.count(ab_a_zscores[0]) == len(ab_a_zscores):
                break

            ab_b_zscores.extend(z_score_b[ab_pair_bool])
            if ab_b_zscores.count(ab_b_zscores[0]) == len(ab_b_zscores):
                break

            # Calculate pearson correlation from both z score lists
            correlation = ss.pearsonr(ab_a_zscores, ab_b_zscores)

            # Add value to matrix
            correlation_matrix[i][j] = correlation[0]
            correlation_matrix[j][i] = correlation[0]
            correlation_matrix[i][i] = 1
            correlation_matrix[i + 1][i + 1] = 1

    correlation_df = pd.DataFrame(correlation_matrix,
                                  index=ab_names,
                                  columns=ab_names)
    return correlation_df

def plot_antibody_correlation(IPD: ImmunoPhenoData):
    """
    Plots a correlation heatmap for each antibody in the data

    Parameters:
        IPD (ImmunoPhenoData Object): Object containing protein data,
            gene data, and cell types
    
    Returns:
        seaborn clustermap for a heatmap of the antibodies
    """
    # Calculate correlation dataframe
    corr_df = _correlation_ab(IPD._classified_filt_df, IPD._z_scores_df)
    g = sns.clustermap(corr_df, vmin=-1, vmax=1, cmap='BrBG')
    g.ax_cbar.set_position((1, .2, .03, .4))

def plot_UMAP(IPD: ImmunoPhenoData,
              normalized: bool = False,
              **kwargs):
    """ Plots a UMAP of protein expression

    Plots a UMAP for the non-normalized protein values or normalized protein
    values

    Args:
        IPD (ImmunoPhenoData): Object containing protein data,
            gene data, and cell types
        normalized (bool): option to plot normalized values
        **kwargs: various arguments to UMAP class constructor, including default values:
            n_neighbors (int): 15
            min_dist (float): 0.1 
            n_components (int): 2
            metric (str): "euclidean"

    Returns:
        UMAP projection of non/normalized protein values with a corresponding
        legend of cell type (if available)

    """
    # Check if existing UMAP is present in class AND UMAP parameters have not changed
    if (IPD._umap_kwargs == kwargs and IPD._raw_umap is not None) and normalized is False:
        # If so, return the stored UMAP
        return IPD._raw_umap
    elif (IPD._umap_kwargs == kwargs and IPD._norm_umap is not None) and normalized is True:
        return IPD._norm_umap
    else:
        # If no UMAP or kwargs are different, generate a new one and store in class
        umap_plot = umap.UMAP(random_state=0, **kwargs)

        # Store new kwargs in class
        IPD._umap_kwargs = kwargs
        
        if normalized:
            try:
                norm_projections = umap_plot.fit_transform(IPD.normalized_counts)
            except:
                raise PlotUMAPError("Cannot plot normalized UMAP. "
                                "normalize_all_antibodies() must be called first.")
        else:
            raw_projections = umap_plot.fit_transform(IPD.protein)
        
        # Normalized UMAP without cell labels
        if IPD.labels is None and normalized:
            norm_plot = px.scatter(
                norm_projections, x=0, y=1,
            )
            IPD._norm_umap = norm_plot
            return norm_plot
        
        # Un-normalized UMAP without cell labels
        elif IPD._cell_labels is None and not normalized:
            raw_plot = px.scatter(
                raw_projections, x=0, y=1,
            )
            IPD._raw_umap = raw_plot
            return raw_plot

        # Normalized UMAP plot with cell labels
        if IPD.labels is not None and normalized:
            # NOTE: if the provided labels contains more cells than present in normalized_counts
            # Find shared index in the IPD.labels based on cells ONLY in normalized_counts
            common_indices = IPD.normalized_counts.index.intersection(IPD.labels.index)

            # Check the number of columns
            num_columns = IPD.labels.shape[1]
            # Check if there is at least one column and if the second column is not empty
            if num_columns > 1 and not IPD.labels.iloc[:, 1].isnull().all():
                # Use the values from the second column
                norm_types = IPD.labels.iloc[:, 1].loc[common_indices].tolist()

            else:
                # If there is no second column or it is empty, use the values from the first column
                norm_types = IPD.labels.iloc[:, 0].loc[common_indices].tolist()

            norm_plot = px.scatter(
                norm_projections, x=0, y=1,
                color_discrete_sequence=px.colors.qualitative.Dark24,
                color=[str(cell_type) for cell_type in norm_types],
                labels={'color':'cell type'}
            )
            IPD._norm_umap = norm_plot
            return norm_plot
        
        # Not normalized UMAP plot with cell labels
        elif IPD._cell_labels is not None and not normalized:
            # Check the number of columns
            num_columns = IPD._cell_labels.shape[1]
            # Check if there is at least one column and if the second column is not empty
            if num_columns > 1 and not IPD._cell_labels.iloc[:, 1].isnull().all():
                # Use the values from the second column
                raw_types = IPD._cell_labels.iloc[:, 1].tolist()
            else:
                # If there is no second column or it is empty, use the values from the first column
                raw_types = IPD._cell_labels.iloc[:, 0].tolist()
                
            reg_plot = px.scatter(
                raw_projections, x=0, y=1,
                color_discrete_sequence=px.colors.qualitative.Dark24,
                color=[str(cell_type) for cell_type in raw_types],
                labels={'color':'cell type'}
            )
            IPD._raw_umap = reg_plot
            return reg_plot