#import numpy as np
from numpy import *

#logic behind the mathematics of error (or say cost function)
def compute_error_for_line_given_points(b,m,points):

    totalError = 0

    #for every pt
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]

        #get difference, square it and add it to the total
        totalError += (y - (m * x + b)) ** 2
    #equation to get the average
    return totalError/ float(len(points))


#funtion to run gradient descent for all the points
def gradient_descent_runner(points,start_b,start_m,learning_rate,num_iteration):
    #start b and m
    b = start_b
    m = start_m

    #gradient descent
    for i in range(num_iteration):
        b,m = step_gradient(b,m,array(points),learning_rate)
    return [b,m]

#function to find the gradient descent
def step_gradient(current_b,current_m,points,learningRate):

    #starting points of our gradient
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]

        #finding out partial derivative for each m and b to get directions for both
        b_gradient += -(2/N) * (y-((current_m * x) + current_b))
        m_gradient += -(2/N) * x * (y-((current_m * x) + current_b))

    #updating b and m
    new_b = current_b - (learningRate * b_gradient)
    new_m = current_m - (learningRate * m_gradient)
    return [new_b,new_m]


def run():
    #step1- collect data
    points = genfromtxt('data.csv', delimiter=',')

    #step2- define hyperparameters that is how fast our model should converge
    learning_rate = 0.0001

    #y = mx + b
    initial_b = 0
    initial_m = 0
    num_iteration = 2000

    #step3- train our model

    print("starting GD at b = {0}, m={1}, error = {2}".format(initial_b,initial_m, compute_error_for_line_given_points(initial_b,initial_m,points)))
    print("Running....")
    [b,m] = gradient_descent_runner(points,initial_b,initial_m,learning_rate,num_iteration)
    print("After {0} iterations b = {1}, m={2}, error = {3}".format(num_iteration,b, m, compute_error_for_line_given_points(b,m,points)))





if __name__ == '__main__':
    run()