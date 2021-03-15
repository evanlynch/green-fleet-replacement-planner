import numpy as np
from sympy.utilities.iterables import multiset_permutations
import pandas as pd
import datetime

#TODO: make years flexible
class MIP():
    def __init__(self,data):
        self.data = data
        self.num_vehicles = data.equipmentid.nunique()
        self.vehicle_idx = data.vehicle_idx.values
        self.current_vehicle_ages = np.array(data.current_age)
        self.replacement_schedules = self.make_replacement_schedules()
        _,self.num_schedules,self.num_years = self.replacement_schedules.shape

    #create replacement schedules
    #todo: will likely need a generalizable method for computing the maximum number of replacements possible
    def make_replacement_schedules(self):
        oneReplacement = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        twoReplacements = np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
        threeReplacements = np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0])

        oneReplacement = list(multiset_permutations(oneReplacement))
        twoReplacements = list(multiset_permutations(twoReplacements))
        threeReplacements = list(multiset_permutations(threeReplacements))

        replacementSchedules = np.array(oneReplacement+twoReplacements+threeReplacements)
        # replacementSchedules = np.array(twoReplacements+threeReplacements)

        #create set of schedules for each vehicle
        replacementSchedules = np.repeat(replacementSchedules[np.newaxis,:, :], self.num_vehicles, axis=0)
        return replacementSchedules

    def make_keep_schedules(self):
        """Opposite of replacement -- may want to use this at some point
        """
        return (self.replacement_schedules-1)*-1

    def initialize_vehicle_age(self,keepSchedules):
        """Gets the vehicle age building process started. Produces a matrix for the age of the vehicle according to the replacement schedule, which is later fixed by get_vehicle_age"""
        startingAge = np.repeat(self.current_vehicle_ages, self.num_schedules, axis=0).reshape(
            self.num_vehicles,self.num_schedules,1)
        age = startingAge+np.cumsum(keepSchedules,axis=2)*keepSchedules
        age[age==startingAge] = 0 #fixes the fact that replaced vehicles start at 0 (if this wasn't here they would start at the starting age)
        return age

    def get_vehicle_age(self,keepSchedules,age=None,k=1):
        if k==1:
            age = self.initialize_vehicle_age(keepSchedules)

        diff = np.diff(age,axis=2)
        diffMask = np.append(np.ones(shape=(self.num_vehicles,self.num_schedules,1)),diff,axis=2)>1
        age[diffMask]=k

        if age[diffMask].size==0:
            return age
        else:
            return self.get_vehicle_age(keepSchedules,age,k=k+1)

    #TODO: Allow mileage to vary randomly for all 15 years
    #TODO: Allow mileage to vary randomly with own mean for each vehicle
    def make_mileage(self):
        """Creates mileage matrix for each year. 
           Mileage is random every year, with mean of 2020 miles and std of overall inventory.
           Mileage is repeated for each schedule."""
        means = np.array(self.data.miles2020).reshape(-1,1)
        std_dev = self.data.miles2020.std()
        annual_mileage = np.round(np.random.normal(loc=means,scale=1000,size=(self.num_vehicles,15)))
        annual_mileage = np.repeat(annual_mileage[:,np.newaxis,:],self.num_schedules,axis=1)
        # vehicle_mileage = np.repeat(np.round(np.random.normal(loc=10000,scale=2000,size=(1, 15))),numSchedules,axis=0)
        # annual_mileage = np.ones(shape=(self.num_vehicles,numSchedules,15))
        # annual_mileage[0,:,:]*=np.round(np.random.normal(loc=10000,scale=std_dev))
        return annual_mileage

    #TODO: Fix to be the cumsum of annual mileage, allowing for odometer resets upon replacement.
    def make_odometer(self,annual_mileage,age):
        """Matrix showing what the odometer reading will be under each schedule."""
        odometer = annual_mileage*age
        return odometer

    #TODO: Make numSChedules an attribute of the class instead of passing it into every func
    #TODO: add in escalator %
    def get_acquisition_cost(self):
        acquisition = self.replacement_schedules.copy()
        acquisition = np.repeat(np.array(self.data.replacement_vehicle_purchaseprice),self.num_schedules).reshape(
            self.num_vehicles,self.num_schedules,1)*self.replacement_schedules
        return acquisition

    def get_vehicle_type_trackers(self):
        """Returns two matrices. One that tracks if in a givemn year a schedule implies the vehicle is still an ice, and then the opposite: whether or not in a given year a vehicle is now an EV"""
        firstReplacements = np.argmax(self.replacement_schedules[0]==1,axis=1) #gets index of year the vehicle is first replaced (ie it transitions from ICE to EV)
        is_ice = self.replacement_schedules.copy()
        is_ev = self.replacement_schedules.copy()
        
        for i in range(0,self.num_schedules): #there is most definitely a better way to do this in numpy but I took way too long researching
            is_ice[:,i,firstReplacements[i]:] = 0
            is_ice[:,i,:firstReplacements[i]] = 1

            is_ev[:,i,firstReplacements[i]:] = 1
            is_ev[:,i,:firstReplacements[i]] = 0
        return is_ice,is_ev
        
    def get_consumables(self,annual_mileage,is_ice,is_ev):
        """Will make this function more flexible later. Calculates fuel cost as if always ICE and always EV. And then applies to the schedules based on when the initial transition from ICE to EV occurs. """   
        #ICE
        #fuel $ = mileage/mpg*cpg
        cpg = 2.25
        mpg = 22
        fuel = annual_mileage/mpg*cpg

            #EV
        #fuel $ = mileage/mpeg * cpeg
        cpeg = 1.2
        mpeg = 100
        electricity = annual_mileage/mpeg*cpeg

        consumables = np.round(fuel*is_ice+(electricity*is_ev))
        return consumables

    def get_maintenance_cost(self,age,annual_mileage,odometer):
        """- ! because this is likely to change. For now I'm just going to treat as a linear regression with made up coeffs."""
        age_coef = .01
        mileage_coef = .2
        odometer_coef = .1
        maintenance = (age_coef*age)+(mileage_coef*annual_mileage)+(odometer_coef*odometer)
        return maintenance

    def get_emissions(self,annual_mileage,is_ice,is_ev):
        """Will make this function more flexible later. Calculates fuel cost as if always ICE and always EV. And then applies to the schedules based on when the initial transition from ICE to EV occurs. """   
        #calc: kg CO2/gallon * mileage/mpg
        
        ice_emission_factor = 2.421
        mpg = 22
        ice_emissions = ice_emission_factor*annual_mileage/mpg

        ev_emission_factor = 0
        mpge = 100
        ev_emissions = ev_emission_factor*annual_mileage/mpge

        emissions = np.round((ice_emissions*is_ice)+(ev_emissions*is_ev))
        return emissions

    #TODO: I feel like there should be more feasible schedules. Will need to come back to this. 
    def find_infeasible_schedules(self,odometer,age):
        """Generates a mask that is True for any schedule that is infeasible. These can be filtered out before running the model."""
        odometer_diff = np.diff(odometer)
        odometer_check = (odometer_diff>-150000) & (odometer_diff<=0)

        age_diff = np.diff(age)
        age_check = (age_diff>-6) & (age_diff<=0)#.any()

        both_check = odometer_check*age_check
        is_infeasible = both_check.any(axis=2)
        return is_infeasible



    #TODO: add number of charging stations to build and number of charging stations to operate as vars
