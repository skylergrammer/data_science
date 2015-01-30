import numpy as np
import matplotlib.pyplot as plt


def in_circle(x, y):

    # (y - h)**2 + (x - k)**2 = r**2
  
    circle_y = lambda x: np.sqrt(1 - np.power(x, 2))

    if y <= circle_y(x): 
        return True

    else:
        return False
  

def generate_random_xy():

    x = np.random.random()
    y = np.random.random()

    return (x, y)  

def main():

    inside = 0
    niters = 1000000
    pivals = []

    for iteration in range(niters):
        x, y = generate_random_xy()
        
        if in_circle(x, y):
            inside += 1

        pi = 4*float(inside) / float(iteration+1)
        pivals.append(pi)

    # plots
    fig = plt.figure()
    plt.plot([0,niters], [0,0], "k", ls="--", lw=2)
    plt.plot(range(niters), np.array(pivals)-np.pi, "r")
    plt.ylabel("$estimate - \pi$", fontsize=20)
    plt.xlabel("Iteration", fontsize=20)
    plt.xlim(0, niters)
    plt.ylim(-0.05, 0.05)
    plt.show()


main()
