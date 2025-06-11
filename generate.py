import numpy as np
# import os
# from tqdm import tqdm, trange
import argparse

from sklearn.neighbors import NearestNeighbors

from statistics import mode

def sum_ratings(ratings_a, ratings_b):
    """
    Compute the disjoint union of datasets ratings_a and ratings_b expressed as matrices of the same size
    """
    assert ratings_a[np.logical_not(np.logical_or(np.isnan(ratings_a), np.isnan(ratings_b)))].size == 0
    # Check that there are no conflicts between the two ratings matrices

    ratings_c = np.nan_to_num(ratings_a) + np.nan_to_num(ratings_b)
    ratings_c[np.logical_not(ratings_c)] = np.NaN
    return ratings_c


def compute_Re(I, U):
    """
    Helper function to compute Re in the (now unused) case of I and U being 3-dimensional arrays
    """
    return (I @ np.transpose(U, [0,2,1]))


def create_folds_masks(R_m, folds):
    non_nan_lines, non_nan_columns = np.nonzero(~R_m.mask)
    nb_non_nan = non_nan_lines.size

    indices = np.arange(nb_non_nan)
    np.random.shuffle(indices)

    for fold in range(folds):
        valid_mask = np.ones(R.shape, dtype=bool)

        slice_start = int(nb_non_nan*fold/folds)
        slice_end = int(nb_non_nan*(fold+1)/folds)

        fold_indices = indices[slice_start:slice_end]

        valid_mask[non_nan_lines[fold_indices],non_nan_columns[fold_indices]] = False
        train_mask = np.logical_or(nan_mask_R, ~valid_mask)

        yield np.ma.MaskedArray(R, train_mask), np.ma.MaskedArray(R, valid_mask)


def train_MF(R_m, K=1, lr=5e-5, reg_lambda=0.1, reg_mu=0.1, epochs=1):
    """
    Train one (Items, Users) tuple with hidden dim K on masked dataset R_m
    The training is done epochs times, with learning rate lr, and regularization constants reg_lambda and reg_mu

    Returns (Items, Users)
    """
    I = np.random.rand(R_m.shape[0], K)+0.5
    U = np.random.rand(R_m.shape[1], K)+0.5

    for _ in range(epochs):
        Re = I @ U.T
        S = R_m-Re

        S = np.where(R_m.mask, 0, S)     # Set S NaNs to 0 because gradient is undefined in these points, thus ignored
        dI = S @ U
        dU = S.T @ I

        I += 2*lr*(dI - reg_lambda * I)     # -= -2*grad from initial formulation
        U += 2*lr*(dU - reg_mu * U)         # -= -2*grad from initial formulation
    return I, U


def train_kNN(R_m, K=1, axis=1, metric='cosine'):
    movie_means = np.mean(R_m, axis=axis, keepdims=True)
    R_filled = np.where(R_m.mask, movie_means, R_m)

    knn = NearestNeighbors(metric=metric, algorithm='auto', n_neighbors=K, n_jobs=-1)
    knn.fit(R_filled)

    _, indices = knn.kneighbors(R_filled)   # Get indices of the k nearest neighbors

    predicted_ratings = np.where(R_m.mask, np.mean(R_filled[indices], axis=1), R_m)
    return predicted_ratings


def train_SVD(R_m, K=1, axis=1, threshold=0.1):
    means = np.mean(R_m, axis=axis, keepdims=True)
    R_filled = np.where(R_m.mask, 0, R_m - means)
    U, s, Vt = np.linalg.svd(R_filled, full_matrices=False)
    s[s < threshold] = 0

    S_k = np.diag(s[:K])
    U_k = U[:, :K]
    Vt_k = Vt[:K, :]
    return (U_k @ S_k) @ Vt_k + means


def compute_round(ratings):
    return 0.5 * np.round(np.clip(2*ratings, 1, 10))


def aggregate(list_Re, scheme='average', round=True):
    """
    Aggregate values of list_Re along the 0 axis, according to the chosen mode.
    Output values can be rounded to enable accuracy computation.
    """
    assert scheme in ['average', 'voting']

    if scheme == 'average':
        Re_flat = np.mean(list_Re, axis=0)
    elif scheme == 'voting':
        Re_flat = np.apply_along_axis(mode, 0, list_Re)

    if round: return compute_round(Re_flat)
    else: return Re_flat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_eval.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()

    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    # table_test = np.load("..\Assignment 1 Data\\ratings_test.npy")
    print('Ratings Loaded.')
    
    
    # Any method you want

    # Values in the table range from 0.5 to 5 with .5 increments
    
    R = table
    # R = sum_ratings(table, table_test)
    nan_mask_R = np.isnan(R)
    R_m = np.ma.MaskedArray(R, nan_mask_R)
    
    # Hyperparameters
    folds = 5

    K_mf = 5           # We need to explore what the best HP are for each K
    parallel_IU = 3
    reg_lambda = 1e-1
    reg_mu = 1
    lr = 14e-5      # Would be great to adapt lr to dataset size
    epochs = 70
    # We should modify lambda and mu depending on K
    # We should modify spread of I and U depending on K

    K_knn = 50
    metrics = ['cosine', 'manhattan', 'euclidean', 'minkowski']
    metric_knn = metrics[0]

    K_svd = 5


    # k_folds validation
    non_nan_lines, non_nan_columns = np.nonzero(~nan_mask_R)
    nb_non_nan = non_nan_lines.size
    folds_indices = np.random.shuffle(np.arange(nb_non_nan))

    folds_table = np.zeros((folds, *R.shape))

    for fold, (fold_train, fold_valid) in enumerate(create_folds_masks(R_m, folds)):
        
        Re_table = np.zeros((parallel_IU, *R.shape))
        Re_knn = compute_round(train_kNN(R_m=fold_train, K=K_knn, metric=metric_knn))
        
        Re_svd = train_SVD(fold_train, K=K_svd)
        
        for para in range(parallel_IU):
            I, U = train_MF(R_m=fold_train, K=K_mf, lr=lr, reg_lambda=reg_lambda, reg_mu=reg_mu, epochs=epochs)
            Re_table[para] = I @ U.T
        
        Re_mf = aggregate(Re_table, 'average')

        Re_flat = aggregate(np.array([Re_knn, Re_svd, Re_mf]), 'average')
        folds_table[fold] = Re_flat
    
        # difference = fold_valid - Re_flat
        # is_equal = difference == 0

        # print("Correct :", np.sum(is_equal))
        # print("Incorrect :", np.sum(~is_equal))
        # print("Accuracy :", np.mean(is_equal))
        # print("RMSE :", {np.sqrt(np.mean(difference*difference))})

    Re = aggregate(folds_table, 'average', True)

    table = Re * nan_mask_R + np.nan_to_num(R)


    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE