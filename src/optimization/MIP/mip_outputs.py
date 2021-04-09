import numpy as np
import pandas as pd
import gurobipy as grb
import random
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from types import MappingProxyType


from .mip_model import MIP_Model

class MIP_Outputs(MIP_Model):

    # @st.cache(hash_funcs={MappingProxyType: lambda _: None})
    def __init__(self,data,UI_params,m,x,vehicles,penalty_budget,penalty_emissions,validSchedulesPerVehicle,validSchedules):
        super().__init__(data,UI_params)

        self.num_alternative_solutions = 3
        self.m = m
        self.x = x
        self.vehicles = vehicles
        self.penalty_budget = penalty_budget
        self.penalty_emissions = penalty_emissions
        self.validSchedulesPerVehicle = validSchedulesPerVehicle
        self.validSchedules = validSchedules
        self.numDesiredSolutions = 500

    # @st.cache
    def get_alternative_solutions(self):
        """Returns the detailed output options for all alternative solutions dumped in the solution pool."""
        #multiple solutions
        options = {}

        # options = ['A','B','C']
        for sol in range(0,self.numDesiredSolutions):
            try:
                schedules = []
                acquisition_costs = []
                mx_costs = []
                consumables_costs = []
                emissions_amts = []
                conversions = []
                # print()
                # print(f'Option: {sol}')
                self.m.setParam('SolutionNumber',sol)
                # print(self.m.PoolObjVal)
            #     print(f'obj:{m.getObjective().getValue()}')
                
                for v in self.vehicles:
                    for s in self.validSchedulesPerVehicle[v]:
                        if self.x[v,s].xn==1:
            #                 print(f'   Vehicle: {v+1} Schedule: {s} {mip.replacement_schedules[v,s]}')  
                            schedules.append([v]+[s]+list(self.replacement_schedules[v,s]))
                            acquisition_costs.append([v]+[s]+list(self.acquisition[v,s]))
                            mx_costs.append([v]+[s]+list(self.maintenance[v,s]))
                            consumables_costs.append([v]+[s]+list(self.consumables[v,s]))
                            emissions_amts.append([v]+[s]+list(self.emissions[v,s]))
                            conversions.append([v]+[s]+list(self.is_ev[v,s]))

                options[sol,'schedules'] = schedules
                options[sol,'acquisition_costs'] = acquisition_costs
                options[sol,'mx_costs'] = mx_costs
                options[sol,'consumables_costs'] = consumables_costs
                options[sol,'emissions_amts'] = emissions_amts
                options[sol,'conversions'] = conversions
                options[sol,'objective'] = self.m.PoolObjVal
            except:
                ooo = 0


        return options

    # @st.cache(hash_funcs={grb.Model: hash})
    def get_similarity_scores(self,alt_solutions):
        """Computes the cosine similarity of each solution relative to option A (first optimal solution)"""
        similarity_scores = []
        #get first optimal solution for comparison
        x = np.array(alt_solutions[0,'schedules'])[:,1:2].flatten().reshape(1,-1)

        for sol in range(0,self.numDesiredSolutions):
            
            try:
                y = np.array(alt_solutions[sol,'schedules'])[:,1:2].flatten().reshape(1,-1)
                similarity_scores.append((sol,cosine_similarity(x,y)[0][0]))
            except:
                ooo = 0
            
        similarity_scores = pd.DataFrame(similarity_scores,columns=['alt_sol_id','cosine_similarity'])

        return similarity_scores

    # @st.cache(hash_funcs={grb.Model: hash})
    def get_alt_sol_objs(self,alt_solutions):
        """Returns the objectives for alternative solutions"""
        objs = []
        for sol in range(0,self.numDesiredSolutions):
            try:
                self.m.setParam('SolutionNumber',sol)
                objs.append((sol,self.m.PoolObjVal/self.m.getObjective().getValue()))
            except:
                ooo = 0

        objs = pd.DataFrame(objs,columns=['alt_sol_id','obj'])
        return objs
    
    def get_total_cost(self,roadmapSelected,roadmap_lookup,options):
        acq_cost = pd.DataFrame(options[roadmap_lookup[roadmapSelected],'acquisition_costs'],columns=['vehicle_idx','solution_idx']+self.years)[self.years].sum().sum()
        mx_cost = pd.DataFrame(options[roadmap_lookup[roadmapSelected],'mx_costs'],columns=['vehicle_idx','solution_idx']+self.years)[self.years].sum().sum()
        con_cost = pd.DataFrame(options[roadmap_lookup[roadmapSelected],'consumables_costs'],columns=['vehicle_idx','solution_idx']+self.years)[self.years].sum().sum()
        return round((acq_cost+mx_cost+con_cost)/1000000,1)

    def get_final_year_emissions(self,roadmapSelected,roadmap_lookup,options):
        emiss = pd.DataFrame(options[roadmap_lookup[roadmapSelected],'emissions_amts'],columns=['vehicle_idx','solution_idx']+self.years)[self.end_year].sum().sum()
        return round(emiss,1)

    # @st.cache(hash_funcs={grb.Model: hash})
    # def select_alternative_solutions(self,alt_solutions,optimal_sol_obj):
    #     """Selects n solutions to return as alternatives"""
    #     alt_sol_sim_scores = self.get_similarity_scores(alt_solutions)
    #     alt_sol_objs = self.get_alt_sol_objs(alt_solutions,optimal_sol_obj)
    #     alt_sol_eval = alt_sol_sim_scores.merge(alt_sol_objs,on='alt_sol_id',how='left')
    #     # st.write(alt_sol_eval)

    #     sim_mask = alt_sol_eval.cosine_similarity <= 0.80
    #     obj_mask = alt_sol_eval.obj <= 5

    #     top_solutions = alt_sol_eval[(sim_mask)&(obj_mask)].sort_values('obj').sample(self.num_alternative_solutions-1) #get n-1 other different solutions
    #     top_solutions = pd.concat([alt_sol_eval[alt_sol_eval.alt_sol_id==0],top_solutions]) #append the optimal solution
    #     top_solutions.sort_values('alt_sol_id',inplace=True)
    #     top_solutions.reset_index(drop=True,inplace=True)
    #     st.write(top_solutions)
    #     selected_alternatives = {}
    #     for i in range(0,self.num_alternative_solutions):
    #         try:
    #             sol = top_solutions.iloc[i]['alt_sol_id']
    #             selected_alternatives[i,'schedules'] = alt_solutions[sol,'schedules']
    #             selected_alternatives[i,'acquisition_costs'] = alt_solutions[sol,'acquisition_costs']
    #             selected_alternatives[i,'mx_costs'] = alt_solutions[sol,'mx_costs']
    #             selected_alternatives[i,'consumables_costs'] = alt_solutions[sol,'consumables_costs']
    #             selected_alternatives[i,'emissions_amts'] = alt_solutions[sol,'emissions_amts']
    #             selected_alternatives[i,'conversions'] = alt_solutions[sol,'conversions']
    #             selected_alternatives[i,'objective'] = alt_solutions[sol,'objective']
    #         except:
    #             None
        
    #     return selected_alternatives