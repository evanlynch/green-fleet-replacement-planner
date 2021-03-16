import numpy as np
from sympy.utilities.iterables import multiset_permutations
import pandas as pd
import datetime
import gurobipy as grb

#UI inputs. We can change the structure of this, but starting to form a JSON-esque format for inputs from 
#the front-end that we'll feed through the DataProjector before running the optimization

UI_params = {
    'initial_procurement_budget':1300000,
    'procurement_budget_rate':0.03,
    'initial_maintenance_budget':1000000,
    'maintenance_budget_rate':0.03,
    'planning_interval':[2022,2037],
    'emissions_baseline': 1000,#metric tons
    'emissions_target_pct_of_base':0.40,
    'min_miles_replacement_threshold':150000,#miles
    'min_vehicle_age_replacement_threshold':60,#years
    'max_vehicles_per_station':1000,
    'objective_weights':{'cost':0.01,'emissions':0.99},
}

#TODO: make years flexible
class MIP():
    def __init__(self,data,UI_params):
        self.data = data
        self.num_vehicles = data.equipmentid.nunique()
        self.vehicle_idx = data.vehicle_idx.values
        self.current_vehicle_ages = np.array(data.current_age)
        self.replacement_schedules = self.make_replacement_schedules()
        _,self.num_schedules,self.num_years = self.replacement_schedules.shape
        self.numDesiredSolutions = 3
        self.start_year = UI_params['planning_interval'][0]
        self.end_year = UI_params['planning_interval'][1]
        self.years = [t for t in range(self.start_year,self.end_year+1)]

    #create replacement schedules
    #todo: will likely need a generalizable method for computing the maximum number of replacements possible
    def make_replacement_schedules(self):
        oneReplacement = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        twoReplacements = np.array([1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        threeReplacements = np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0])

        oneReplacement = list(multiset_permutations(oneReplacement))
        twoReplacements = list(multiset_permutations(twoReplacements))
        threeReplacements = list(multiset_permutations(threeReplacements))

        replacementSchedules = np.array(oneReplacement+twoReplacements+threeReplacements)
        # replacementSchedules = np.array(twoReplacements+threeReplacements)

        #create set of schedules for each vehicle
        replacementSchedules = np.repeat(replacementSchedules[np.newaxis,:, :], self.num_vehicles, axis=0)
        return replacementSchedules

    def make_keep_schedules(self):
        """Opposite of replacement"""
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
    #TODO: Consider an alternative distro to normal bc of ngatives
    def make_mileage(self):
        """Creates mileage matrix for each year. 
           Mileage is random every year, with mean of 2020 miles and std of overall inventory.
           Mileage is repeated for each schedule."""
        means = np.array(self.data.miles2020).reshape(-1,1)
        std_dev = self.data.miles2020.std()
        annual_mileage = np.round(np.random.normal(loc=means,scale=1000,size=(self.num_vehicles,self.num_years)))
        annual_mileage = np.repeat(annual_mileage[:,np.newaxis,:],self.num_schedules,axis=1)
        annual_mileage[annual_mileage<0] = 1000
        
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
        cpg = 2.35
        fuel = np.round(annual_mileage/np.repeat(np.array(self.data.mpg2020),self.num_schedules).reshape(
            self.num_vehicles,self.num_schedules,1)*cpg)

        #EV
        #fuel $ = mileage/mpeg * cpeg
        cpeg = 1.2
        electricity = np.round(annual_mileage/np.repeat(np.array(self.data.replacement_vehicle_mpge),self.num_schedules).reshape(
            self.num_vehicles,self.num_schedules,1)*cpg)

        consumables = np.round(fuel*is_ice+(electricity*is_ev))
        return consumables

    def get_maintenance_cost(self,odometer):
        """- ! because this is likely to change. For now I'm just going to treat as a linear regression with made up coeffs."""
        maintenance = np.zeros(shape=(self.num_vehicles,self.num_schedules,self.num_years))
        maintenance[odometer<10000] = 1000
        maintenance[np.logical_and(odometer>=10000,odometer<150000)] = 1200
        return maintenance

    def get_emissions(self,annual_mileage,is_ice,is_ev):
        """Will make this function more flexible later. Calculates fuel cost as if always ICE and always EV. And then applies to the schedules based on when the initial transition from ICE to EV occurs. """   
        #calc: kg CO2/gallon * mileage/mpg
        
        ice_emission_factor = 2.421
        ice_emissions = np.round(annual_mileage/np.repeat(np.array(self.data.mpg2020),self.num_schedules).reshape(
            self.num_vehicles,self.num_schedules,1)*ice_emission_factor)

        ev_emission_factor = 0
        ev_emissions = np.round(annual_mileage/np.repeat(np.array(self.data.replacement_vehicle_mpge),self.num_schedules).reshape(
            self.num_vehicles,self.num_schedules,1)*ev_emission_factor)

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

    def get_valid_schedules(self,consumables):
        """"Returns a dict with each vehicle having a list of valid schedules."""
        validSchedules = list(consumables.keys())
        validSchedulesPerVehicle = pd.DataFrame(validSchedules).groupby(0)[1].unique().to_dict()
        return validSchedules,validSchedulesPerVehicle

    #TODO: add number of charging stations to build and number of charging stations to operate as vars
    #TODO: Eventually separate out into a few functions, if not it's own class
    def make_model(self,consumables,acquisition,maintenance,emissions,infeasible_filter):
        vehicles = [v for v in range(0,self.num_vehicles)]
        schedules = [s for s in range(0,self.num_schedules)]
        years = [t for t in range(0,self.num_years)]
        finalYear = max(years)

        c = {}
        a = {}
        m = {}
        e = {}
        for v in vehicles:
            for s in schedules:
                if not infeasible_filter[v,s]:
                    c[v,s] = consumables[v,s]
                    a[v,s] = acquisition[v,s]
                    m[v,s] = maintenance[v,s]
                    e[v,s] = emissions[v,s]

        consumables = c.copy()
        acquisition = a.copy()
        maintenance = m.copy()
        emissions = e.copy()

        validSchedules,validSchedulesPerVehicle = self.get_valid_schedules(consumables)

        #TODO: Add to init
        budget_acquisition = 1300000*np.ones(shape=(self.num_years)) 
        budget_operations = 1000000*np.ones(shape=(self.num_years))
        emissions_goal = 40000

        try: 
            m.reset()
            del m    
        except:
            None
            
        m = grb.Model('carnet')

        #TODO: Add to init  
        m.setParam('PoolSearchMode',2) #tell gurobi I want multiple solutions
        m.setParam('PoolSolutions',self.numDesiredSolutions) #number of solutions I want
        m.setParam('TimeLimit',30)

        x = {}
        for v in vehicles:
            for s in validSchedulesPerVehicle[v]:
                x[v,s] = m.addVar(vtype=grb.GRB.BINARY)
        penalty_budget = m.addVar(vtype=grb.GRB.CONTINUOUS,name='penalty_budget')
        penalty_emissions = m.addVar(vtype=grb.GRB.CONTINUOUS,name='penalty_emissions')

        #TODO: Add to init
        w = {'cost':0.70,'emissions':0.30}

        obj = m.setObjective(grb.quicksum(w['cost']*consumables[v,s][t]*x[v,s] for v,s in validSchedules for t in years) + 
                     grb.quicksum(w['emissions']*emissions[v,s][finalYear]*x[v,s] for v,s in validSchedules) +
                     1000000*(penalty_budget+penalty_emissions),grb.GRB.MINIMIZE)

        c1 = m.addConstrs((grb.quicksum(x[v,s] for s in validSchedulesPerVehicle[v])==1 for v in vehicles),'one_schedule_per_vehicle')
        c2 = m.addConstrs((grb.quicksum((consumables[v,s][t]+maintenance[v,s][t])*x[v,s] for v,s in validSchedules) <= budget_operations[t]+penalty_budget for t in years),'operations_budget')
        c3 = m.addConstrs((grb.quicksum(acquisition[v,s][t]*x[v,s] for v,s in validSchedules) <= budget_operations[t]+penalty_budget for t in years),'acquisition_budget')
        c4 = m.addConstr((grb.quicksum(emissions[v,s][finalYear]*x[v,s] for v,s in validSchedules) <= emissions_goal+penalty_emissions),'emissions_goal')

        m.optimize()

        return m,x,vehicles,validSchedulesPerVehicle
    
