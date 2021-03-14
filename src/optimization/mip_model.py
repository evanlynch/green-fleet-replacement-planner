import numpy as np
from sympy.utilities.iterables import multiset_permutations
import pandas as pd

numVehicles = 20


#TODO: Make a class

#create replacement schedules
#todo: will likely need a generalizable method for computing the maximum number of replacements possible
def make_replacement_schedules():
    oneReplacement = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    twoReplacements = np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
    threeReplacements = np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])

    oneReplacement = list(multiset_permutations(oneReplacement))
    twoReplacements = list(multiset_permutations(twoReplacements))
    threeReplacements = list(multiset_permutations(threeReplacements))

    replacementSchedules = np.array(oneReplacement+twoReplacements+threeReplacements)
    # replacementSchedules = np.array(twoReplacements+threeReplacements)

    #create set of schedules for each vehicle
    replacementSchedules = np.repeat(replacementSchedules[np.newaxis,:, :], numVehicles, axis=0)
    return replacementSchedules

def make_keep_schedules(replacementSchedules):
    """Opposite of replacement -- may want to use this at some point
    """
    return (replacementSchedules-1)*-1

def initialize_vehicle_age(keepSchedules,startingAge=0):
    """Gets the vehicle age building process started. Produces a matrix for the age of the vehicle according to the replacement schedule, which is later fixed by get_vehicle_age"""
    age = startingAge+np.cumsum(keepSchedules,axis=2)*keepSchedules
    age[age==startingAge] = 0 #fixes the fact that replaced vehicles start at 0 (if this wasn't here they would start at the starting age)
    return age

def get_vehicle_age(keepSchedules,numSchedules,age=None,k=1,startingAge=0):
    if k==1:
        age = initialize_vehicle_age(keepSchedules,startingAge)

    diff = np.diff(age,axis=2)
    diffMask = np.append(np.ones(shape=(numVehicles,numSchedules,1)),diff,axis=2)>1
    age[diffMask]=k

    if age[diffMask].size==0:
        return age
    else:
        return get_vehicle_age(keepSchedules,numSchedules,age,k=k+1)

#TODO: Allow mileage to vary randomly for all 15 years
#TODO: Allow mileage to vary randomly with own mean for each vehicle
def make_mileage(numVehicles,numSchedules):
    """creates mileage matrix for each year"""
    # vehicle_mileage = np.repeat(np.round(np.random.normal(loc=10000,scale=2000,size=(1, 15))),numSchedules,axis=0)
    annual_mileage = np.ones(shape=(numVehicles,numSchedules,15))
    annual_mileage[0,:,:]*=np.round(np.random.normal(loc=10000,scale=2000))
    annual_mileage[1,:,:]*=np.round(np.random.normal(loc=11000,scale=2000))
    return annual_mileage