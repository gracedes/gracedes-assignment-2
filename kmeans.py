import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

cmap = 'Set3'

"""
kmeans steps:
1. define number of clusters
2. initialize centroid to each cluster
3. calculate distance between each point and centroid
4. assign each point to the nearest centroid
5. update centroid location
6. repeat steps 3 to 5 until convergence
"""

def kmeans(dset, k=3, init='random', gen_steps=False, tol=1e-4):
    '''
    K-means implementationd for a 
    `dset`:  DataFrame with observations
    `k`: number of clusters, default k=2
    `tol`: tolerance=1E-4
    '''

    working_dset = dset.copy()
    err = []
    goahead = True
    j = 0
    
    # Step 2
    centroids = initiate_centroids(k, dset, init)

    while(goahead):
        # Step 3 and 4
        working_dset['centroid'], j_err = centroid_assignment(working_dset, centroids) 
        err.append(sum(j_err))
        
        # Step 5
        centroids = working_dset.groupby('centroid').agg('mean').reset_index(drop = True)

        # Step 6
        if j>0:
            # Is the error less than a tolerance
            if err[j-1]-err[j]<=tol:
                goahead = False
        j+=1

        if gen_steps: gen_plot(working_dset, centroids)

    working_dset['centroid'], j_err = centroid_assignment(working_dset, centroids)
    centroids = working_dset.groupby('centroid').agg('mean').reset_index(drop = True)
    gen_plot(working_dset, centroids)
    return working_dset['centroid'], j_err, centroids

def initiate_centroids(k, dset, init):
    '''
    Select k data points as centroids
    k: number of centroids
    dset: pandas dataframe
    '''
    centroids = dset.sample(k)
    return centroids

def calc_dist(a,b):
    '''
    Calculate the root of sum of squared errors. 
    a and b are numpy arrays
    '''
    return np.square(np.sum((a-b)**2)) 

def centroid_assignment(dset, centroids):
    '''
    Given a dataframe `dset` and a set of `centroids`, we assign each
    data point in `dset` to a centroid. 
    - dset - pandas dataframe with observations
    - centroids - pa das dataframe with centroids
    '''
    k = centroids.shape[0]
    n = dset.shape[0]
    assignation = []
    assign_errors = []

    for obs in range(n):
        # Estimate error
        all_errors = np.array([])
        for centroid in range(k):
            err = calc_dist(centroids.iloc[centroid, :], dset.iloc[obs,:])
            all_errors = np.append(all_errors, err)

        # Get the nearest centroid and the error
        nearest_centroid =  np.where(all_errors==np.amin(all_errors))[0].tolist()[0]
        nearest_centroid_error = np.amin(all_errors)

        # Add values to corresponding lists
        assignation.append(nearest_centroid)
        assign_errors.append(nearest_centroid_error)

    return assignation, assign_errors

def gen_plot(df, centroids):
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(df.iloc[:,0], df.iloc[:,1],  marker = 'o', 
                c=df['centroid'].astype('category'), 
                cmap = cmap, s=80, alpha=0.5)
    plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  
                marker = 's', s=200, c=range(centroids.shape[0]), 
                cmap = cmap)
    ax.set_xlabel(r'x', fontsize=14)
    ax.set_ylabel(r'y', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

blobs = pd.read_csv('kmeans_blobs.csv')
colnames = list(blobs.columns[1:-1])
kmeans(blobs[colnames], k=4, gen_steps=True)

# sorry, I did not have time to finish the interactive webpage, this kmeans implementation is as far as I got :(