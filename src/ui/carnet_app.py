import streamlit as st
from streamlit import caching
import pandas as pd
import numpy as np
import sys 
import os
import time
import plotly.graph_objects as go
import plotly.express as px
import base64

sys.path.append(os.path.join(os.getcwd(), '..','..'))
print(os.getcwd())
from src.optimization.MIP.mip_inputs import *
from src.optimization.MIP.mip_model import *
from src.optimization.MIP.mip_outputs import *

####################################################################################################

@st.cache(hash_funcs={tuple:lambda _: None,grb.Model:lambda _: None,dict:lambda _: None},suppress_st_warning=True)
def carnet():
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

    m.setParam('PoolSearchMode',2) #tell gurobi I want multiple solutions
    m.setParam('PoolSolutions',inputs.numDesiredSolutions) #number of solutions I want
    m.setParam('TimeLimit',inputs.solverTimeLimit)
    # m.setParam('OutputFlag',0)

    x = {}
    for v in vehicles:
        for s in validSchedulesPerVehicle[v]:
            x[v,s] = m.addVar(vtype=grb.GRB.BINARY)
    penalty_budget = m.addVar(vtype=grb.GRB.CONTINUOUS,name='penalty_budget')
    penalty_emissions = m.addVar(vtype=grb.GRB.CONTINUOUS,name='penalty_emissions')        

    obj = m.setObjective(grb.quicksum(inputs.objective_weights['cost']*consumables[v,s][t]*x[v,s] for v,s in validSchedules for t in years) + 
                    100000000*penalty_budget + 10000*penalty_emissions,grb.GRB.MINIMIZE)

    c1 = m.addConstrs((grb.quicksum(x[v,s] for s in validSchedulesPerVehicle[v])==1 for v in vehicles),'one_schedule_per_vehicle')
    c2 = m.addConstrs((grb.quicksum((consumables[v,s][t]+maintenance[v,s][t])*x[v,s] for v,s in validSchedules) <= inputs.budget_operations[t]+penalty_budget for t in years),'operations_budget')
    c3 = m.addConstrs((grb.quicksum(acquisition[v,s][t]*x[v,s] for v,s in validSchedules) <= inputs.budget_acquisition[t]+penalty_budget for t in years),'acquisition_budget')
    c4 = m.addConstr((grb.quicksum(emissions[v,s][finalYear]*x[v,s] for v,s in validSchedules) <= inputs.emissions_goal + penalty_emissions),'emissions_goal')

    # self.m = m
    # self.x = x
    # self.penalty_budget = penalty_budget
    # self.penalty_emissions = penalty_emissions
    # self.vehicles = vehicles
    # self.m = m
    m.optimize()
    return (m,x,vehicles,penalty_budget,penalty_emissions,validSchedulesPerVehicle,validSchedules)#m,x,penalty_budget,penalty_emissions,vehicles,validSchedulesPerVehicle


####################################################################################################

col1, mid, col2 = st.beta_columns([10,20,40])
with col1:
    st.image('carnet_logo.png', width=100)
with mid:
    st.title('CAR-NET')
with col2:
    st.write('  ')



sidebarTitle = st.sidebar.title("""Model Inputs""")

base_procurement_budget = st.sidebar.number_input("2022 Procurement Budget",value=4000000,step=100000)
base_operations_budget = st.sidebar.number_input("2022 Operations Budget",value=1000000,step=100000)
budget_rate = st.sidebar.number_input('% Increase in Budget Year-Over-Year',value=3)
emissions_target = st.sidebar.number_input("Emissions Decrease Target %",value=40)
final_year = st.sidebar.number_input("Planning Range [2022 - ]",value=2037)
run = st.sidebar.button('Run Model')


def add_space(numSpaces):
    for i in range(numSpaces):
        st.sidebar.write('  ')
add_space(0)
st.sidebar.markdown("CAR-NET (Carbon Neutral Evaluation Tool) was created for Baltimore County by George Mason University, SEOR Department")

data = pd.read_excel('../../data/17MAR_data_template.xlsx').head(100)
data['current_age'] = datetime.datetime.now().year - pd.to_datetime(data.purchasedate).dt.year
data = data.reset_index().rename({"index":"vehicle_idx"},axis=1)
data['county'] = 'Baltimore County'
data = data.drop_duplicates('equipmentid')
data = data[data.vehicle_idx<1458]

UI_params = {
    'initial_procurement_budget':base_procurement_budget,
    'initial_operations_budget':base_operations_budget,
    'budget_rate':budget_rate/100,
    'emissions_target_pct_of_base':emissions_target/100,
    'planning_interval':[2022,final_year],
    'objective_weights':{'cost':0.99,'emissions':0.01},
}

if run:
    caching.clear_cache()


inputs = MIP_Inputs(data,UI_params)
# model = MIP_Model(data,UI_params)

# if run:
m,x,vehicles,penalty_budget,penalty_emissions,validSchedulesPerVehicle,validSchedules = carnet()
# m.optimize()

# st.write(penalty_budget)
# st.write(penalty_emissions)

# st.write(m.getObjective().getValue())
outputs = MIP_Outputs(data,UI_params,m,x,vehicles,penalty_budget,penalty_emissions,validSchedulesPerVehicle,validSchedules)
alt_solutions = outputs.get_alternative_solutions()

@st.cache(hash_funcs={grb.Model:lambda _: None,dict:lambda _: None},suppress_st_warning=True)
def select_alternative_solutions(alt_solutions):
    """Selects n solutions to return as alternatives"""
    alt_sol_sim_scores = outputs.get_similarity_scores(alt_solutions)
    alt_sol_objs = outputs.get_alt_sol_objs(alt_solutions)
    alt_sol_eval = alt_sol_sim_scores.merge(alt_sol_objs,on='alt_sol_id',how='left')
    # st.write(alt_sol_eval)

    sim_mask = alt_sol_eval.cosine_similarity <= 0.80
    obj_mask = alt_sol_eval.obj <= 5

    top_solutions = alt_sol_eval[(sim_mask)&(obj_mask)].sort_values('obj').sample(outputs.num_alternative_solutions-1) #get n-1 other different solutions
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
        except:
            ooo = 0
    
    return selected_alternatives

options = select_alternative_solutions(alt_solutions)
option1 = pd.DataFrame(options[0,'schedules'],columns=['vehicle_idx','solution_idx']+inputs.years)
data = data.merge(option1,on='vehicle_idx',how='left')

roadmapSelected = st.selectbox('Select a Replacement Roadmap', ['A','B','C'])
roadmap_lookup = {'A':0,'B':1,'C':2}

#High level metrics
total_cost = {}
total_budget_util = {}
final_year_emissions = {}
emiss_pct_of_base = {}
for r in ['A','B','C']:#roadmap_lookup.keys():
    tot_cost = outputs.get_total_cost(r,roadmap_lookup,options)
    total_cost[r] = f'${tot_cost}M'
    total_budget_util[r] = f'{round(tot_cost/(np.sum(inputs.budget_acquisition+inputs.budget_operations)/1000000)*100,1)}%'

    emiss = outputs.get_final_year_emissions(r,roadmap_lookup,options)
    final_year_emissions[r] = f'{emiss} MT'
    emiss_pct_of_base[r] = f'{round(emiss/(inputs.emissions_baseline)*100,1)}%'

st.subheader('Replacement Roadmap Summary')
summary = pd.DataFrame([[total_cost['A'],total_cost['B'],total_cost['C']],
                        [total_budget_util['A'],total_budget_util['B'],total_budget_util['C']],
                        ['','',''],
                        [final_year_emissions['A'],final_year_emissions['B'],final_year_emissions['C']],
                        [emiss_pct_of_base['A'],emiss_pct_of_base['B'],emiss_pct_of_base['C']]],
          index=['Total Cost','Total Budget Utilization','',f'{inputs.end_year} Emissions (MT)',"Emissions % of Baseline"],columns=['A','B','C'])
st.write(summary)
# st.header(f'Total Cost: ${total_cost}M')
# st.header(f'{inputs.end_year} Emissions: {final_year_emissions} MT')

def visualize_ev_vs_ice(roadmapSelected,options):
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

visualize_ev_vs_ice(roadmapSelected,options)

def visualize_emissions(roadmapSelected,options):
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

visualize_emissions(roadmapSelected,options)

def visualize_aquisition_cost(roadmapSelected,options):
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

visualize_aquisition_cost(roadmapSelected,options)

def visualize_operations_cost(roadmapSelected,options):
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

visualize_operations_cost(roadmapSelected,options)

#Vehicle replacement plan
st.subheader("Vehicle Replacement Plan")
replacementPlanLOD = st.selectbox('Select Level of Detail', ['County','Department','Department/Vehicle Type','Equipment ID'])
replacementPlanLOD_Lookup = {'County':'county',
                             'Department':'dept_name',
                             'Department/Vehicle Type':['dept_name','vehicledescription'],
                             'Equipment ID':['equipmentid','vehicledescription']}

replacementPlan = data.groupby(replacementPlanLOD_Lookup[replacementPlanLOD])[inputs.years].sum().astype(int).replace({0:''})
st.dataframe(replacementPlan)

#download detailed plan

csv = data.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}">Download Detailed Roadmap</a>'
st.markdown(href, unsafe_allow_html=True)



# st.sidebar.color_picker('Pick a color')

