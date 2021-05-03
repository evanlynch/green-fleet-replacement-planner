import numpy as np
import pandas as pd
import gurobipy as grb
import random
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from types import MappingProxyType
import plotly.graph_objects as go
import plotly.express as px


from .mip_model import MIP_Model

roadmap_lookup = {'A':0,'B':1,'C':2}

class MIP_Outputs(MIP_Model):
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
        self.numSolutions = 500

    def get_alternative_solutions(self):
        """Returns the detailed output options for all alternative solutions dumped in the solution pool."""
        #multiple solutions
        options = {}

        # options = ['A','B','C']
        for sol in range(0,self.numSolutions):
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

    def get_similarity_scores(self,alt_solutions):
        """Computes the cosine similarity of each solution relative to option A (first optimal solution)"""
        similarity_scores = []
        #get first optimal solution for comparison
        x = np.array(alt_solutions[0,'schedules'])[:,1:2].flatten().reshape(1,-1)

        for sol in range(0,self.numSolutions):
            
            try:
                y = np.array(alt_solutions[sol,'schedules'])[:,1:2].flatten().reshape(1,-1)
                similarity_scores.append((sol,cosine_similarity(x,y)[0][0]))
            except Exception as e:
                ooo = 0
            
        similarity_scores = pd.DataFrame(similarity_scores,columns=['alt_sol_id','cosine_similarity'])

        return similarity_scores

    def get_alt_sol_objs(self,alt_solutions):
        """Returns the objectives for alternative solutions"""
        objs = []
        for sol in range(0,self.numSolutions):
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

@st.cache(hash_funcs={grb.Model:lambda _: None,dict:lambda _: None},suppress_st_warning=True)
def select_alternative_solutions(alt_solutions,outputs):
    """Selects n solutions to return as alternatives"""
    alt_sol_sim_scores = outputs.get_similarity_scores(alt_solutions)
    # print(alt_sol_sim_scores)
    alt_sol_objs = outputs.get_alt_sol_objs(alt_solutions)
    # print(alt_sol_objs.describe())
    alt_sol_eval = alt_sol_sim_scores.merge(alt_sol_objs,on='alt_sol_id',how='left')
    sim_mask = alt_sol_eval.cosine_similarity <= 0.90
    obj_mask = alt_sol_eval.obj <= 5
    # print(alt_sol_eval)
    top_solutions = alt_sol_eval.copy()
    # top_solutions = alt_sol_eval[(sim_mask)&(obj_mask)].sort_values('obj').sample(outputs.num_alternative_solutions-1,replace=False) #get n-1 other different solutions
    # print(top_solutions)
    top_solutions = pd.concat([alt_sol_eval[alt_sol_eval.alt_sol_id==0],top_solutions]) #append the optimal solution
    top_solutions.sort_values('alt_sol_id',inplace=True)
    top_solutions.reset_index(drop=True,inplace=True)
    # st.write(top_solutions)
    selected_alternatives = {}
    for i in range(0,outputs.num_alternative_solutions):
        try:
            sol = top_solutions.iloc[i]['alt_sol_id']
            selected_alternatives[i,'schedules'] = alt_solutions[sol,'schedules']
            selected_alternatives[i,'acquisition_costs'] = alt_solutions[sol,'acquisition_costs']
            selected_alternatives[i,'mx_costs'] = alt_solutions[sol,'mx_costs']
            selected_alternatives[i,'consumables_costs'] = alt_solutions[sol,'consumables_costs']
            selected_alternatives[i,'emissions_amts'] = alt_solutions[sol,'emissions_amts']
            selected_alternatives[i,'conversions'] = alt_solutions[sol,'conversions']
            selected_alternatives[i,'objective'] = alt_solutions[sol,'objective']
            # print(i,sol,selected_alternatives[i,'objective'])
        except:
            ooo = 0
    
    return selected_alternatives


def visualize_ev_vs_ice(roadmapSelected,options,inputs):
    ev_inv = pd.DataFrame(pd.DataFrame(options[roadmap_lookup[roadmapSelected],'conversions'],columns=['vehicle_idx','solution_idx']+inputs.years)[inputs.years].sum())

    ev_inv.columns = ['EV']
    ev_inv.EV = ev_inv.EV
    ev_inv['ICE'] = inputs.num_vehicles-ev_inv.EV
    fig = go.Figure()
    fig.add_trace(go.Bar(x=inputs.years,
                    y=ev_inv.EV,
                    name='EV/Hybrid',
                    marker_color='rgb(15,157,88)'
                    ))
    fig.add_trace(go.Bar(x=inputs.years,
                    y=ev_inv.ICE,
                    name='ICE',
                    marker_color='rgb(0,0,0)'
                    ))
    fig.update_layout(title={'text':'ICE vs EV/Hybrid Inventory'},xaxis=dict(tickmode='linear'))
    st.write(fig)
    return fig

def visualize_emissions(roadmapSelected,options,inputs):
    emiss = pd.DataFrame(pd.DataFrame(options[roadmap_lookup[roadmapSelected],'emissions_amts'],columns=['vehicle_idx','solution_idx']+inputs.years)[inputs.years].sum())
    emiss.columns = ['emissions']
    fig = px.line(x=inputs.years,y=emiss.emissions,title='Fleet Emissions (Metric Tons)',color_discrete_sequence=['#7D4444'])
    fig.update_traces(mode='markers+lines')
    fig.add_shape(type='line',
                y0=inputs.emissions_goal,
                y1=inputs.emissions_goal,
                x0=0,x1=1,
                line=dict(color='rgb(15,157,88)',dash='longdash'),
                xref='paper',
                yref='y')
    fig.update_layout(yaxis_range=[0,inputs.emissions_baseline])
    fig.update_layout(yaxis={'title': ''},xaxis=dict(title='',tickmode='linear'))
    st.write(fig)
    return fig

def visualize_aquisition_cost(roadmapSelected,options,inputs):
    acq_cost = pd.DataFrame(pd.DataFrame(options[roadmap_lookup[roadmapSelected],'acquisition_costs'],columns=['vehicle_idx','solution_idx']+inputs.years)[inputs.years].sum())

    acq_cost.columns = ['acq_cost']
    fig = go.Figure()
    fig.add_trace(go.Bar(x=inputs.years,
                    y=acq_cost.acq_cost,
                    name='Acquisition Cost',
                    marker_color='#F98807'
                    ))
    fig.add_trace(go.Line(x=inputs.years,
                    y=inputs.budget_acquisition,
                    name='Acquisition Budget',
                    marker_color='Black'
                    ))
    fig.update_layout(title={'text':'New Vehicle Acquisition Cost'},xaxis=dict(tickmode='linear'))
    st.write(fig)
    return fig

def visualize_operations_cost(roadmapSelected,options,inputs):
    mx_cost = pd.DataFrame(pd.DataFrame(options[roadmap_lookup[roadmapSelected],'mx_costs'],columns=['vehicle_idx','solution_idx']+inputs.years)[inputs.years].sum())
    mx_cost.columns = ['cost']
    mx_cost['cost_type'] = 'Maintenance'

    con_cost = pd.DataFrame(pd.DataFrame(options[roadmap_lookup[roadmapSelected],'consumables_costs'],columns=['vehicle_idx','solution_idx']+inputs.years)[inputs.years].sum())
    con_cost.columns = ['cost']
    con_cost['cost_type'] = 'Consumables'

    ops_cost = pd.concat([mx_cost,con_cost])

    # st.write(ops_cost)
    fig = px.bar(x=ops_cost.index,
                    y=ops_cost.cost,
                    color=ops_cost.cost_type
                    )

    fig.add_trace(go.Line(x=inputs.years,
                    y=inputs.budget_operations,
                    name='Operations Budget',
                    marker_color='Black',
                    ))
    fig.update_layout(yaxis={'title': ''},xaxis={'title':''},legend_title_text='')
    fig.update_layout(title={'text':'Fleet Operations Cost'},xaxis=dict(tickmode='linear'))
    st.write(fig)
    return fig

def get_summary(inputs,outputs,options,roadmap_lookup):
    #High level metrics
    total_cost = {}
    total_budget_util = {}
    final_year_emissions = {}
    emiss_pct_of_base = {}
    for r in ['A','B','C']:
        tot_cost = outputs.get_total_cost(r,roadmap_lookup,options)
        total_cost[r] = f'${tot_cost}M'
        total_budget_util[r] = f'{round(tot_cost/(np.sum(inputs.budget_acquisition+inputs.budget_operations)/1000000)*100,1)}%'

        emiss = outputs.get_final_year_emissions(r,roadmap_lookup,options)
        final_year_emissions[r] = f'{emiss} MT'
        emiss_pct_of_base[r] = f'{round(emiss/(inputs.emissions_baseline)*100,0)}%'

    st.subheader('Replacement Roadmap Summary')
    summary = pd.DataFrame([[total_cost['B'],total_cost['A'],total_cost['C']],
                            [total_budget_util['B'],total_budget_util['A'],total_budget_util['C']],
                            ['','',''],
                            [final_year_emissions['B'],final_year_emissions['A'],final_year_emissions['C']],
                            [emiss_pct_of_base['B'],emiss_pct_of_base['A'],emiss_pct_of_base['C']]],
            index=['Total Cost','Total Budget Utilization','',f'{inputs.end_year} Emissions (MT)',"Emissions % of Baseline"],columns=['A','B','C'])
    # summary = pd.DataFrame([[total_cost['B'],'$112.2M','$118.7M'],
    #                         [total_budget_util['B'],'46.2%','48.1%'],
    #                         ['','',''],
    #                         [final_year_emissions['B'],final_year_emissions['A'],final_year_emissions['C']],
    #                         [emiss_pct_of_base['B'],emiss_pct_of_base['A'],emiss_pct_of_base['C']]],
    #         index=['Total Cost','Total Budget Utilization','',f'{inputs.end_year} Emissions (MT)',"Emissions % of Baseline"],columns=['A','B','C'])
    return summary