
import gurobipy as grb
import pandas as pd
from .mip_inputs import MIP_Inputs

#TODO: Implement depreciation

class MIP_Model(MIP_Inputs): 
    def __init__(self,data,UI_params):
        super().__init__(data,UI_params)  
        self.numDesiredSolutions = 500
        self.solverTimeLimit = 30
        # self.acquisition = acquisition
        # self.consumables = consumables
        # self.maintenance = maintenance
        # self.emissions = emissions
        # self.infeasible_filter = infeasible_filter
        self.m = self.make_model()
        self.m.optimize()

    def make_model(self):
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
                if not self.infeasible_filter[v,s]:
                    c[v,s] = self.consumables[v,s]
                    a[v,s] = self.acquisition[v,s]
                    m[v,s] = self.maintenance[v,s]
                    e[v,s] = self.emissions[v,s]

        consumables = c.copy()
        acquisition = a.copy()
        maintenance = m.copy()
        emissions = e.copy()

        self.validSchedules = list(consumables.keys())
        self.validSchedulesPerVehicle = pd.DataFrame(self.validSchedules).groupby(0)[1].unique().to_dict()
        # self.validSchedules,self.validSchedulesPerVehicle = self.get_valid_schedules()

        try: 
            m.reset()
            del m    
        except:
            None
            
        m = grb.Model('carnet')

        m.setParam('PoolSearchMode',2) #tell gurobi I want multiple solutions
        m.setParam('PoolSolutions',self.numDesiredSolutions) #number of solutions I want
        m.setParam('TimeLimit',self.solverTimeLimit)

        x = {}
        for v in vehicles:
            for s in self.validSchedulesPerVehicle[v]:
                x[v,s] = m.addVar(vtype=grb.GRB.BINARY)
        penalty_budget = m.addVar(vtype=grb.GRB.CONTINUOUS,name='penalty_budget')
        penalty_emissions = m.addVar(vtype=grb.GRB.CONTINUOUS,name='penalty_emissions')        

        obj = m.setObjective(grb.quicksum(self.objective_weights['cost']*consumables[v,s][t]*x[v,s] for v,s in self.validSchedules for t in years) + 
                     1000000*(penalty_budget),grb.GRB.MINIMIZE)#penalty_emissions
                    #  grb.quicksum(self.objective_weights['emissions']*emissions[v,s][finalYear]*x[v,s] for v,s in validSchedules) +

        c1 = m.addConstrs((grb.quicksum(x[v,s] for s in self.validSchedulesPerVehicle[v])==1 for v in vehicles),'one_schedule_per_vehicle')
        c2 = m.addConstrs((grb.quicksum((consumables[v,s][t]+maintenance[v,s][t])*x[v,s] for v,s in self.validSchedules) <= self.budget_operations[t]+penalty_budget for t in years),'operations_budget')
        c3 = m.addConstrs((grb.quicksum(acquisition[v,s][t]*x[v,s] for v,s in self.validSchedules) <= self.budget_acquisition[t]+penalty_budget for t in years),'acquisition_budget')
        c4 = m.addConstr((grb.quicksum(emissions[v,s][finalYear]*x[v,s] for v,s in self.validSchedules) <= self.emissions_goal),'emissions_goal')

        # self.m = m
        self.x = x
        self.penalty_budget = penalty_budget
        self.penalty_emissions = penalty_emissions
        self.vehicles = vehicles

        return m#m,x,penalty_budget,penalty_emissions,vehicles,validSchedulesPerVehicle

