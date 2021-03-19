import numpy as np
from sympy.utilities.iterables import multiset_permutations
import pandas as pd
import datetime

class MIP_Inputs():
    def __init__(self,data,UI_params):
        self.data = data
        self.num_vehicles = data.equipmentid.nunique()
        self.vehicle_idx = data.vehicle_idx.values
        self.current_vehicle_ages = np.array(data.current_age)
        self.replacement_schedules = self.make_replacement_schedules()
        _,self.num_schedules,self.num_years = self.replacement_schedules.shape
        self.start_year = UI_params['planning_interval'][0]
        self.end_year = UI_params['planning_interval'][1]
        self.years = [t for t in range(self.start_year,self.end_year+1)]
        self.objective_weights = UI_params['objective_weights']
        self.budget_acquisition = UI_params['initial_procurement_budget']*np.ones(shape=(self.num_years)) 
        self.budget_operations = UI_params['initial_operations_budget']*np.ones(shape=(self.num_years))
        self.emissions_target_pct_of_base = UI_params['emissions_target_pct_of_base']
        self.emissions_baseline = UI_params['emissions_baseline']
        self.emissions_goal = self.get_emissions_goal()
        self.charging_station_cost = 5000

        self.keepSchedules = self.make_keep_schedules()
        self.age = self.get_vehicle_age()#[50:80]
        self.annual_mileage = self.make_mileage()
        self.odometer = self.make_odometer()
        self.acquisition = self.get_acquisition_cost()
        self.is_ice,self.is_ev = self.get_vehicle_type_trackers()
        self.consumables = self.get_consumables()
        self.emissions = self.get_emissions()
        self.maintenance = self.get_maintenance_cost()
        self.infeasible_filter = self.find_infeasible_schedules()

    def get_emissions_goal(self):
        return self.emissions_baseline*self.emissions_target_pct_of_base

    #TODO: make years flexible
    #TODO: will likely need a generalizable method for computing the maximum number of replacements possible
    def make_replacement_schedules(self):
        noReplacements = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        oneReplacement = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        twoReplacements = np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        threeReplacements = np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])

        noReplacements = list(multiset_permutations(noReplacements))
        oneReplacement = list(multiset_permutations(oneReplacement))
        twoReplacements = list(multiset_permutations(twoReplacements))
        threeReplacements = list(multiset_permutations(threeReplacements))

        replacementSchedules = np.array(noReplacements+oneReplacement+twoReplacements+threeReplacements)
        # replacementSchedules = np.array(twoReplacements+threeReplacements)

        #create set of schedules for each vehicle
        replacementSchedules = np.repeat(replacementSchedules[np.newaxis,:, :], self.num_vehicles, axis=0)
        return replacementSchedules

    def make_keep_schedules(self):
        """Opposite of replacement"""
        return (self.replacement_schedules-1)*-1

    def initialize_vehicle_age(self):
        """Gets the vehicle age building process started. Produces a matrix for the age of the vehicle according to the replacement schedule, which is later fixed by get_vehicle_age"""
        startingAge = np.repeat(self.current_vehicle_ages, self.num_schedules, axis=0).reshape(
            self.num_vehicles,self.num_schedules,1)
        age = startingAge+np.cumsum(self.keepSchedules,axis=2)*self.keepSchedules
        age[age==startingAge] = 0 #fixes the fact that replaced vehicles start at 0 (if this wasn't here they would start at the starting age)
        return age

    def get_vehicle_age(self,age=None,k=1):
        if k==1:
            age = self.initialize_vehicle_age()

        diff = np.diff(age,axis=2)
        diffMask = np.append(np.ones(shape=(self.num_vehicles,self.num_schedules,1)),diff,axis=2)>1
        age[diffMask]=k

        if age[diffMask].size==0:
            return age
        else:
            return self.get_vehicle_age(age,k=k+1)

    #TODO: Consider an alternative distro to normal bc of ngatives
    def make_mileage(self):
        """Creates mileage matrix for each year. 
           Mileage is random every year, with mean of 2020 miles and std of overall inventory.
           Mileage is repeated for each schedule."""
        means = np.array(self.data.miles2020).reshape(-1,1)
        std_dev = self.data.miles2020.std()
        annual_mileage = np.round(np.random.normal(loc=means,scale=std_dev,size=(self.num_vehicles,self.num_years)))
        annual_mileage = np.repeat(annual_mileage[:,np.newaxis,:],self.num_schedules,axis=1)
        annual_mileage[annual_mileage<0] = 1000
        
        # vehicle_mileage = np.repeat(np.round(np.random.normal(loc=10000,scale=2000,size=(1, 15))),numSchedules,axis=0)
        # annual_mileage = np.ones(shape=(self.num_vehicles,numSchedules,15))
        # annual_mileage[0,:,:]*=np.round(np.random.normal(loc=10000,scale=std_dev))
        return annual_mileage

    #TODO: Fix to be the cumsum of annual mileage, allowing for odometer resets upon replacement.
    def make_odometer(self):
        """Matrix showing what the odometer reading will be under each schedule."""
        odometer = self.annual_mileage*self.age
        return odometer

    def get_acquisition_cost(self):
        acquisition = self.replacement_schedules.copy()
        charging_station = self.charging_station_cost/2
        acquisition = np.repeat(np.array(self.data.replacement_vehicle_purchaseprice+charging_station),self.num_schedules).reshape(
            self.num_vehicles,self.num_schedules,1)*self.replacement_schedules
        return acquisition

    def get_vehicle_type_trackers(self):
        """Returns two matrices. One that tracks if in a givemn year a schedule implies the vehicle is still an ice, and then the opposite: whether or not in a given year a vehicle is now an EV"""
        firstReplacements = np.argmax(self.replacement_schedules[0]==1,axis=1) #gets index of year the vehicle is first replaced (ie it transitions from ICE to EV)
        firstReplacements[0] = self.num_years #first schedule is no replacement. This was originally 0, which meant no replacement is always ev, so this fixes to indicate never ev
        is_ice = self.replacement_schedules.copy()
        is_ev = self.replacement_schedules.copy()
        
        for i in range(0,self.num_schedules): #there is most definitely a better way to do this in numpy but I took way too long researching
            is_ice[:,i,firstReplacements[i]:] = 0
            is_ice[:,i,:firstReplacements[i]] = 1

            is_ev[:,i,firstReplacements[i]:] = 1
            is_ev[:,i,:firstReplacements[i]] = 0
        return is_ice,is_ev
        
    def get_consumables(self):
        """Will make this function more flexible later. Calculates fuel cost as if always ICE and always EV. And then applies to the schedules based on when the initial transition from ICE to EV occurs. """   
        #ICE
        #fuel $ = mileage/mpg*cpg
        cpg = 2.35
        fuel = np.round(self.annual_mileage/np.repeat(np.array(self.data.mpg2020),self.num_schedules).reshape(
            self.num_vehicles,self.num_schedules,1)*cpg)

        #EV
        #fuel $ = mileage/mpeg * cpeg
        cpeg = 1.16
        electricity = np.round(self.annual_mileage/np.repeat(np.array(self.data.replacement_vehicle_mpge),self.num_schedules).reshape(
            self.num_vehicles,self.num_schedules,1)*cpg)

        consumables = np.round(fuel*self.is_ice+(electricity*self.is_ev))
        return consumables

    def get_maintenance_cost(self):
        """- ! because this is likely to change. For now I'm just going to treat as a linear regression with made up coeffs."""
        maintenance = np.zeros(shape=(self.num_vehicles,self.num_schedules,self.num_years))
        maintenance[self.odometer<10000] = 1000
        maintenance[np.logical_and(self.odometer>=10000,self.odometer<150000)] = 1200
        return maintenance

    def get_emissions(self):
        """Will make this function more flexible later. Calculates fuel cost as if always ICE and always EV. And then applies to the schedules based on when the initial transition from ICE to EV occurs. """   
        #calc: kg CO2/gallon * mileage/mpg
        
        ice_emission_factor = 2.421
        ice_emissions = np.round(self.annual_mileage/np.repeat(np.array(self.data.mpg2020),self.num_schedules).reshape(
            self.num_vehicles,self.num_schedules,1)*ice_emission_factor)

        ev_emission_factor = 0
        ev_emissions = np.round(self.annual_mileage/np.repeat(np.array(self.data.replacement_vehicle_mpge),self.num_schedules).reshape(
            self.num_vehicles,self.num_schedules,1)*ev_emission_factor)

        emissions = np.round((ice_emissions*self.is_ice)+(ev_emissions*self.is_ev))
        return emissions

    def find_infeasible_schedules(self):
        """Generates a mask that is True for any schedule that is infeasible. These can be filtered out before running the model."""
        odometer_diff = np.diff(self.odometer)
        odometer_check = (odometer_diff>-150000) & (odometer_diff<=0)

        age_diff = np.diff(self.age)
        age_check = (age_diff>-6) & (age_diff<=0)#.any()

        both_check = odometer_check*age_check
        is_infeasible = both_check.any(axis=2)
        return is_infeasible

    # def get_valid_schedules(self):
    #     """"Returns a dict with each vehicle having a list of valid schedules."""
    #     validSchedules = list(self.consumables.keys())
    #     validSchedulesPerVehicle = pd.DataFrame(validSchedules).groupby(0)[1].unique().to_dict()
    #     return validSchedules,validSchedulesPerVehicle