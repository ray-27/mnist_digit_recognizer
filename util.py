import matplotlib.pyplot as plt
from drive import upload

def plot_loss(loss_list,save=False):
    l = len(loss_list)
    epo = [i for i in range(l)]
    plt.plot(epo,loss_list)

    if save:
        upload(plt.savefig("output.jpg"),"loss and epoches graph")


def find_optimal_dim(n): #returns dimensions as a,b (a < b)
    # n is the number of graphs to be plotted

    #find factors
    factors = []
    for i in range(1,n+1):
        if(n % i == 0):
            factors.append(i)

    #now using two pointer we can find the minimum difference
    min_diff = 0

    l = len(factors)
    mid = l // 2
    pointer = mid
    while(mid != 0 or mid != l):

        val = factors[mid] * factors[pointer]

        if(val > n):
            pointer -= 1
        elif(val < n):
            pointer += 1
        elif(val == n):
            return min(factors[mid],factors[pointer]), max(factors[mid],factors[pointer])


    pass


#creating a function to print the number on a matplotlib pyplot
def show_img(X,y=-1):

    X = X.reshape(28,28)

    plt.imshow(X,cmap='gray')
    plt.title(y.numpy())

    pass


def show_multiple_img(X,y,range_max=1):
    range_min=0
    count = range_max - range_min
    n,m = find_optimal_dim(count)
    _, axs = plt.subplots(n, m, figsize=(8, 6))

    o = range_min #counter to next picture
    for i in range(n):
        for j in range(m):

            axs[i,j].imshow(X[o].reshape(28,28),cmap='gray')
            axs[i,j].set_title(y[o].numpy())
            axs[i,j].set_xticklabels([])
            axs[i,j].set_yticklabels([])
            o += 1

    plt.show()
