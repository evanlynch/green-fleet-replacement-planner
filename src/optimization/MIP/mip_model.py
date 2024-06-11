
import gurobipy as grb
import pandas as pd
import streamlit as st

from .mip_inputs import MIP_Inputs

@st.cache(hash_funcs={tuple:lambda _: None,grb.Model:lambda _: None,dict:lambda _: None},suppress_st_warning=True)
def carnet(inputs):
    vehicles = [v for v in range(0,inputs.num_vehicles)]
    schedules = [s for s in range(0,inputs.num_schedules)]
    years = [t for t in range(0,inputs.num_years)]
    finalYear = max(years)

    c = {}
    a = {}
    m = {}
    e = {}
    for v in vehicles:
        for s in schedules:
            if not inputs.infeasible_filter[v,s]:
                c[v,s] = inputs.consumables[v,s]
                a[v,s] = inputs.acquisition[v,s]
                m[v,s] = inputs.maintenance[v,s]
                e[v,s] = inputs.emissions[v,s]

    consumables = c.copy()
    acquisition = a.copy()
    maintenance = m.copy()
    emissions = e.copy()

    validSchedules = list(consumables.keys())
    validSchedulesPerVehicle = pd.DataFrame(validSchedules).groupby(0)[1].unique().to_dict()
    # self.validSchedules,self.validSchedulesPerVehicle = self.get_valid_schedules()

    try: 
        m.reset()
        del m    
    except:
        ooo = 0
    
    m = grb.Model('carnet')
    m.setParam('OutputFlag',1)
    m.setParam('PoolSearchMode',2) #tell gurobi I want multiple solutions
    m.setParam('PoolSolutions',inputs.numSolutions) #number of solutions I want
    m.setParam('TimeLimit',inputs.solverTimeLimit)

    x = {}
    for v in vehicles:
        for s in validSchedulesPerVehicle[v]:
            x[v,s] = m.addVar(vtype=grb.GRB.BINARY)
    penalty_budget = m.addVar(vtype=grb.GRB.CONTINUOUS,name='penalty_budget')
    penalty_emissions = m.addVar(vtype=grb.GRB.CONTINUOUS,name='penalty_emissions')        

    # obj = m.setObjective(grb.quicksum(inputs.objective_weights['cost']*(acquisition[v,s][t]+consumables[v,s][t]+maintenance[v,s][t])*x[v,s] for v,s in validSchedules for t in years) + 
    #                 100000000*penalty_budget + 10000*penalty_emissions,grb.GRB.MINIMIZE)
    cost_scale = sum(inputs.budget_acquisition+inputs.budget_operations)
    emissions_scale = inputs.emissions_baseline
    obj = m.setObjective(grb.quicksum(inputs.objective_weights['cost']*(((acquisition[v,s][t]+consumables[v,s][t]+maintenance[v,s][t]))*x[v,s]) +\
        inputs.objective_weights['emissions']*100*((emissions[v,s][finalYear])*x[v,s]) + 1000000000*(penalty_budget+penalty_emissions) for v,s in validSchedules for t in years
    ),grb.GRB.MINIMIZE)
    # obj = m.setObjective(grb.quicksum(maintenance[v,s][t]*x[v,s] + 1000000000*(penalty_budget+penalty_emissions) for v,s in validSchedules for t in years),grb.GRB.MINIMIZE)
    c1 = m.addConstrs((grb.quicksum(x[v,s] for s in validSchedulesPerVehicle[v])==1 for v in vehicles),'one_schedule_per_vehicle')
    c2 = m.addConstrs((grb.quicksum((consumables[v,s][t]+maintenance[v,s][t])*x[v,s] for v,s in validSchedules) <= inputs.budget_operations[t] + penalty_budget for t in years),'operations_budget')
    c3 = m.addConstrs((grb.quicksum(acquisition[v,s][t]*x[v,s] for v,s in validSchedules) <= inputs.budget_acquisition[t] + penalty_budget for t in years),'acquisition_budget')
    # c2 = m.addConstrs((grb.quicksum((consumables[v,s][t]+maintenance[v,s][t])*x[v,s] for v,s in validSchedules) <= inputs.budget_operations[t] + penalty_budget for t in years),'operations_budget')
    # c3 = m.addConstrs((grb.quicksum(acquisition[v,s][t]*x[v,s] for v,s in validSchedules) <= inputs.budget_acquisition[t] + penalty_budget for t in years),'acquisition_budget')
    c4 = m.addConstr((grb.quicksum(emissions[v,s][finalYear]*x[v,s] for v,s in validSchedules) <= inputs.emissions_goal + penalty_emissions),'emissions_goal')

    m.optimize()
    return (m,x,vehicles,penalty_budget,penalty_emissions,validSchedulesPerVehicle,validSchedules)#m,x,penalty_budget,penalty_emissions,vehicles,validSchedulesPerVehicle

class MIP_Model(MIP_Inputs): 
    def __init__(self,data,UI_params):
        super().__init__(data,UI_params)  
        self.numSolutions = 500
        self.solverTimeLimit = 30