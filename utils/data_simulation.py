# --------------------------------------------------------------------------
#          FUNCTION FOR CREATING SIMULATED DATA
#        WILL BE CALLED FROM MORE THAN ONE SCRIPT
# --------------------------------------------------------------------------
# (C) Nicolas Rost, 2022

from sklearn.datasets import make_classification

def create_simulated_data(n_samples = 1000,
                          n_classes = 2,
                          n_features = 125,
                          n_informative = 25,
                          n_redundant = 50,
                          n_clusters_per_class = 1,
                          random_state = 1897):
    """
    Function that simluates the classification data. This function will likely be called from more than one script,
    so we make it an extra function to keep it consistent. It only calls scikit-learns make_classification() function.

    Parameters
    ----------
    see make_classification() from sklearn.

    Defaults:
    n_samples = 1000
    n_classes = 2
    n_features = 125
    n_informative = 25
    n_redundant = 50
    n_clusters_per_class = 1
    random_state = 1897

    Returns
    -------
    Two data matrices: features (X) and target (y).
    """

    # make classification using inputs
    X, y = make_classification(n_samples = n_samples,
                               n_classes = n_classes,
                               n_features = n_features,
                               n_informative = n_informative,
                               n_redundant = n_redundant,
                               n_clusters_per_class = n_clusters_per_class,
                               random_state = random_state)

    return X, y
