import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, cluster, metrics
from scipy.special import comb
from itertools import combinations
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as sms
from sklearn import tree
import math
import joblib
import sklearn.manifold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report
from scipy.cluster.hierarchy import dendrogram
from kneed import KneeLocator
from sklearn.model_selection import train_test_split
from sklearn import datasets


test_set = np.loadtxt("fashion-mnist_test.csv",delimiter=",",skiprows=1)
train_set = np.loadtxt("fashion-mnist_train.csv",delimiter=",",skiprows=1)
test_label = test_set[:, 0]
train_label = train_set[:, 0]
test_set = test_set[:,1:]
train_set = train_set[:,1:]

#attempts to find optimal epsilon for DBScan method by finding an elbow in distances produced by NearestNeighbors
def optimizeEps(set):
    nbrs = NearestNeighbors().fit(set)
    distances, indices = nbrs.kneighbors(set)
    distance_desc = sorted(distances[:,5-1], reverse=True)
    kneedle = KneeLocator(range(1,len(distance_desc)+1),  #x values
                      distance_desc, # y values
                      S=1.0, #parameter suggested from paper
                      curve="convex", #parameter from figure
                      direction="decreasing") #parameter from figure
    return kneedle.knee_y


#calculates the precision for each label, given the confusion matrix as input
def calc_precision(matrix):
    return [ matrix[i][i] / sum( matrix[:,i] )  for i in range(len(matrix))]




plt.figure(figsize=(18,18),dpi=200.)



pca = decomposition.PCA(n_components=25)
train_set_25dim = pca.fit_transform(train_set)
test_set_25dim = pca.transform(test_set)
D = pca.explained_variance_ratio_
E = range(len(pca.explained_variance_))
plt.bar(E,D)
plt.plot(E,D,'r')
plt.savefig("overview_25dim.png")
plt.clf()
#-----------------------------------


#
#
#
# Dataset dimensionality reduction
#
#
#


train_set_8dim = decomposition.PCA(n_components=8).fit_transform(train_set)
train_set_5dim = decomposition.PCA(n_components=5).fit_transform(train_set)
train_set_4dim = decomposition.PCA(n_components=4).fit_transform(train_set)
train_set_3dim = decomposition.PCA(n_components=3).fit_transform(train_set)
train_set_2dim = decomposition.PCA(n_components=2).fit_transform(train_set)


# MDS method never finished calculating
#rain_set_2dim_mds = sklearn.manifold.MDS(n_components=2, n_jobs=-1).fit_transform(train_set)
#train_set_3dim_mds = sklearn.manifold.MDS(n_components=3, n_jobs=-1).fit_transform(train_set)


#commented-out becuse of sheer amount of time necessary to compute
'''
#testing for optimal parameters for DBScan clustering; 4 dimensions

for u in range(7,9):
    for i in range(140, 280, 1):
        DB = cluster.DBSCAN(eps=i, min_samples=u, n_jobs=-1).fit(train_set_4dim)
        score = metrics.rand_score (train_label, DB.labels_)
        unassigned = -sum(filter( lambda x : int(x) == -1 , DB.labels_ ))
        if(max(DB.labels_) == 10):
            print((i,u,score, unassigned, max(DB.labels_)))
    
#results for 2 dimensions:
        #(45, 12, 0.14267291343744617, 1448, 10)
        #(49, 11, 0.13024026622665932, 997, 10)

#testing for optimal parameters for DBScan clustering; 25 dimensions
for u in range(40,50):
    for i in range(20, 280, 1):
        DB = cluster.DBSCAN(eps=i, min_samples=u, n_jobs=-1).fit(train_set_25dim)
        score = metrics.rand_score (train_label, DB.labels_)
        unassigned = -sum(filter( lambda x : int(x) == -1 , DB.labels_ ))
        if(max(DB.labels_) == 10):
            print((i,u,score, unassigned, max(DB.labels_)))

#testing for optimal parameters accross dimensions
for num in range(2, 25):
    pca = decomposition.PCA(n_components=num)
    train_set_dim = pca.fit_transform(train_set)
    Eps = optimizeEps(train_set_dim)
    DB = cluster.DBSCAN(eps=Eps, min_samples=int(1.8*num), n_jobs=-1).fit(train_set_dim)
    #HDB = cluster.HDBSCAN(min_samples =int(1.8* num), n_jobs=-1).fit(train_set_dim)
    print(metrics.rand_score (train_label, DB.labels_), str(num) + " dim, eps="+ str(Eps)+ ", determined labels for DBScan")




'''





#
#
#
# clustering stage
#
#
#


#AgglomerativeClustering never finished, or if it did - did it with a memory error 
'''
AC5 = cluster.AgglomerativeClustering(n_clusters=10, linkage="average").fit(train_set_5dim)
print(metrics.rand_score (train_label, AC5.labels_), "5dim, Agglomerative")
ac_mrx = confusion_matrix(train_label,AC5.labels_)
print(ac_mrx)
print(calc_precision(ac_mrx), " Precision")'''




#failed attempt for DBScan
DB4 = cluster.DBSCAN(eps=171, min_samples=8, n_jobs=-1).fit(train_set_4dim)
print(metrics.rand_score (train_label, DB4.labels_), "4dim, determined labels for DBScan")



#failed attempt for OPTICS
OP5 = cluster.OPTICS(eps=406, min_samples=int(5*1.8), n_jobs=-1).fit(train_set_5dim)
print(metrics.rand_score (train_label, OP5.labels_), "5dim, determined labels for OPTICS")


#failed attempt for DBScan
DB25 = cluster.DBSCAN(eps=362, min_samples=40, n_jobs=-1).fit(train_set_25dim)
print(metrics.rand_score (train_label, DB25.labels_), "25dim, determined labels for DBScan")

#HDBSCAN
HDB5 = cluster.HDBSCAN(min_samples = 10,min_cluster_size=120, cluster_selection_epsilon=80., cluster_selection_method="eom", max_cluster_size=15000, n_jobs=-1).fit(train_set_5dim)
print(metrics.rand_score (train_label, HDB5.labels_), "5dim, determined labels for HDBScan")
#5, 100, 18 for 33% coverage

BIR5 = cluster.Birch(n_clusters=10,compute_labels=True).fit(train_set_5dim)
print(metrics.rand_score (train_label, BIR5.labels_), "5dim, Birch")

bir_predicted = BIR5.predict(train_set_5dim)
bir_mrx = confusion_matrix(train_label,bir_predicted)
print(bir_mrx)
print(calc_precision(bir_mrx), " Precision")



BIR8 = cluster.Birch(n_clusters=10,compute_labels=True).fit(train_set_8dim)
print(metrics.rand_score (train_label, BIR8.labels_), "5dim, Birch")


bir_mrx = confusion_matrix(train_label, BIR8.labels_)
print(bir_mrx)
print(calc_precision(bir_mrx), " Precision")

#obtaining rand_indexes for Birch clustering on dataset reduced to u dimensions

for u in range(1,10):
    train_set_dim = decomposition.PCA(n_components=u).fit_transform(train_set)

    BIR = cluster.Birch( n_clusters=10,compute_labels=True).fit(train_set_dim)
    score = metrics.rand_score (train_label, BIR.labels_)
    print((u,score))





#failure due to insufficient memory
#SC4 = cluster.SpectralClustering(n_clusters=10).fit(train_set_4dim_small)
#print(metrics.rand_score (train_label, SC4.labels_), "4dim, unknown labels for SpectralClustering") #not enough emory



print("Unassigned 25dim DBSCAN: " + str(-sum(filter( lambda x : x == -1 , DB25.labels_ ))))
print("Unassigned 5dim OPTICS: " + str(-sum(filter( lambda x : x == -1 , OP5.labels_ ))))
print("Unassigned 5dim HDBSCAN: " + str(-sum(filter( lambda x : x == -1 , HDB5.labels_ ))))



#creates the animation for the 4d scatter plot. Takes a significant amount of time to complete
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], c=[], cmap=plt.hot())
plt.colorbar(sc)

# Set the axis labels (you can customize these)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

def update(frame):
    # Clear the previous plot
    ax.clear()

    # Scatter plot with updated data for each frame
    sc = ax.scatter(train_set_4dim[:, 0], train_set_4dim[:, 1], train_set_4dim[:, 2],
                    c=train_set_4dim[:, 3], cmap=plt.hot(), marker='o', s = 2)

    # Set the axis labels (you can customize these)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.view_init(elev=20, azim=frame/2)

    return sc,

# Create the animation
an = animation.FuncAnimation(fig, update, frames=720, interval=100, blit=False)
an.save('scatter_4d.gif', writer='imagemagick', fps=30)





plt.clf()

plt.scatter(train_set_3dim[:, 0], train_set_3dim[:, 1], 1.5 ,c= train_set_3dim[:, 2], cmap="jet")
plt.colorbar()
plt.savefig("scatter_3d.png")

plt.clf()
plt.scatter(train_set_3dim[:, 0], train_set_3dim[:, 1], 1.5 ,c= train_label, cmap="tab10")
plt.colorbar()
plt.savefig("scatter_3d_labeled.png")

plt.clf()
plt.scatter(train_set_2dim[:, 0], train_set_2dim[:, 1], 1.5 ,c= train_label, cmap="tab10")
plt.colorbar()
plt.savefig("scatter_2d_labeled.png")
plt.clf()

'''
plt.clf()
plt.scatter(train_set_2dim[:, 0], train_set_2dim[:, 1], 1.5 ,c= AC5.labels_, cmap="jet")
plt.colorbar()
plt.savefig("scatter_5d_agglomerative.png")
'''


plt.clf()
plt.scatter(train_set_2dim[:, 0], train_set_2dim[:, 1], 1.5 ,c= DB25.labels_, cmap="jet")
plt.colorbar()
plt.savefig("scatter_25d_dbscan.png")


plt.clf()
plt.scatter(train_set_2dim[:, 0], train_set_2dim[:, 1], 1.5 ,c= DB4.labels_, cmap="brg")
plt.colorbar()
plt.savefig("scatter_4d_dbscan.png")


plt.clf()
plt.scatter(train_set_2dim[:, 0], train_set_2dim[:, 1], 1.5 ,c= OP5.labels_, cmap="brg")
plt.colorbar()
plt.savefig("scatter_5d_optics.png")


plt.clf()
plt.scatter(train_set_2dim[:, 0], train_set_2dim[:, 1], 1.5 ,c= HDB5.labels_, cmap="brg")
plt.colorbar()
plt.savefig("scatter_5d_hdbscan.png")

plt.clf()
plt.scatter(train_set_2dim[:, 0], train_set_2dim[:, 1], 1.5 ,c= BIR5.labels_, cmap="jet")
plt.colorbar()
plt.savefig("scatter_5d_birch.png")

plt.clf()
plt.scatter(train_set_2dim[:, 0], train_set_2dim[:, 1], 1.5 ,c= BIR8.labels_, cmap="jet")
plt.colorbar()
plt.savefig("scatter_8d_birch.png")





#
#
#   Points 5-6: Classification 
#
#


#
#
#   Classification for 25 dimension dataset
#
#

neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
rbf = make_pipeline(StandardScaler(), SVC(gamma='auto', cache_size=1000)) 


kf = sms.KFold()
for i, (train_i, test_i) in enumerate(kf.split(train_label)):
    neigh.fit(train_set_25dim[train_i],train_label[train_i])
    rbf.fit(train_set_25dim[train_i],train_label[train_i])
    
    kncavg = neigh.score(train_set_25dim[test_i],train_label[test_i])
    rbfavg = rbf.score(train_set_25dim[test_i],train_label[test_i])
    print(kncavg, rbfavg)


neigh_rand = neigh.score(test_set_25dim,test_label)
rbf_rand = rbf.score(test_set_25dim,test_label)

neigh_predicted = neigh.predict(test_set_25dim)
rbf_predicted = rbf.predict(test_set_25dim)
neigh_mrx = confusion_matrix(test_label,neigh_predicted)
rbf_mrx = confusion_matrix(test_label,rbf_predicted)

print(neigh_rand, rbf_rand, "classification for true labels")
print(neigh_mrx)
print()
print(rbf_mrx)
print(accuracy_score(test_label,neigh_predicted), accuracy_score(test_label,rbf_predicted), " accuracy")
print(calc_precision(neigh_mrx),"\n",calc_precision(rbf_mrx), " precision")


neigh_predicted = neigh.predict(train_set_25dim)
rbf_predicted = rbf.predict(train_set_25dim)



plt.clf()
plt.scatter(train_set_2dim [:, 0], train_set_2dim [:, 1], 1.5 ,c= neigh_predicted, cmap="tab10")
plt.colorbar()
plt.savefig("scatter_2d_neigh_25dim.png")
plt.clf()
plt.scatter(train_set_2dim [:, 0], train_set_2dim [:, 1], 1.5 ,c= rbf_predicted, cmap="tab10")
plt.colorbar()
plt.savefig("scatter_2d_rbf_25dim.png")
plt.clf()

#print(calc_precision(neigh_mrx), " Precision")

#print(calc_precision(rbf_mrx), " Precision")












#
#
#
# the exact code given by ChatGPT
#
# imports ommited
#


# Step 1: Load the Fashion-MNIST dataset
fashion_mnist = datasets.fetch_openml('Fashion-MNIST', version=1, cache=True)

# Step 2: Data Preprocessing
X = fashion_mnist.data.astype('float32') / 255.0
y = fashion_mnist.target.astype('int')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Dimensionality Reduction using PCA
pca = decomposition.PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

# Step 4: Clustering using K-Means
kmeans = cluster.KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X_train_pca)

# Step 5: Classification using Decision Tree
tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, y_train)
y_pred = tree_classifier.predict(X_test)

# Step 6: Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)



# Step 7: Visualization
# Visualize the PCA reduced data with cluster assignments
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=clusters, cmap='viridis', s=20)
plt.title('PCA Reduced Data with K-Means Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Visualize confusion matrix
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)




#
#
# end of ChatGPT's code
#
#



#my own testing of gpt's clustering
print(metrics.rand_score(y_train,clusters))
