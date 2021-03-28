import streamlit as st
import pandas as pd
import numpy as np
import sys 
import os
import time
import plotly.graph_objects as go
import plotly.express as px
import base64

sys.path.append(os.path.join(os.getcwd(), '..','..'))

from src.optimization.MIP.mip_inputs import *
from src.optimization.MIP.mip_model import *
from src.optimization.MIP.mip_outputs import *

#TODO: See if we can remove +- on # inputs
#TODO: See if we can auto-format numbers (e.g. add commas to millions)
#TODO: Enable caching of all relevant solution information so I don't re-run the model everytime I click a button
#TODO: Design UI layout
#TODO: Implement high level layout
#TODO: Add logo and make title green

st.title("CAR-NET")
st.header("Carbon Neutral Evaluation Tool")


#INPUTS
sidebarTitle = st.sidebar.write("""# Model Inputs""")

base_procurement_budget = st.sidebar.number_input("Base Procurement Budget for 2021")
base_operations_budget = st.sidebar.number_input("Base Operations Budget for 2021")
budget_rate = st.sidebar.number_input('% Increase in Budget Year-Over-Year')
emissions_target = st.sidebar.number_input("Emissions Target %")
# run = st.sidebar.button('Run Model')
st.sidebar.color_picker('Pick a color')


def add_space(numSpaces):
    for i in range(numSpaces):
        st.sidebar.write('  ')
add_space(14)
st.sidebar.write(" Created for Baltimore County by George Mason University")

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
    'planning_interval':[2022,2037],
    # 'min_miles_replacement_threshold':150000,#miles
    # 'min_vehicle_age_replacement_threshold':6,#years
    'objective_weights':{'cost':0.70,'emissions':0.30},
}

inputs = MIP_Inputs(data,UI_params)
model = MIP_Model(data,UI_params)

# if run:
m,x,vehicles,penalty_budget,penalty_emissions,validSchedulesPerVehicle,validSchedules = model.make_model()
m.optimize()

# st.write(penalty_budget)
# st.write(penalty_emissions)

# st.write(m.getObjective().getValue())
outputs = MIP_Outputs(data,UI_params,m,x,vehicles,penalty_budget,penalty_emissions,validSchedulesPerVehicle,validSchedules)

optimal_obj,optimal_solution = outputs.get_optimal_solution()
alt_solutions = outputs.get_alternative_solutions()

selected_alternative_solutions = outputs.select_alternative_solutions(alt_solutions,optimal_obj)
options = selected_alternative_solutions
option1 = pd.DataFrame(options[0,'schedules'],columns=['vehicle_idx','solution_idx']+inputs.years)
data = data.merge(option1,on='vehicle_idx',how='left')

roadmapSelected = st.selectbox('Select a Roadmap', ['A (Optimal)','B','C'])
roadmap_lookup = {'A (Optimal)':0,'B':1,'C':2}

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
    fig.add_shape(type='line',
                y0=inputs.emissions_goal,
                y1=inputs.emissions_goal,
                x0=0,x1=1,
                line=dict(color='rgb(15,157,88)',dash='longdash'),
                xref='paper',
                yref='y')
    fig.update_layout(yaxis_range=[0,inputs.emissions_baseline])
    fig.update_layout(yaxis={'title': ''},xaxis={'title':''})
    st.write(fig)

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
    fig.update_layout(yaxis={'title': ''},xaxis={'title':''})
    fig.update_layout(title={'text':'Fleet Operations Cost'},xaxis=dict(tickmode='linear'))
    st.write(fig)

visualize_operations_cost(roadmapSelected,options)

#Vehicle replacement plan
st.subheader("Vehicle Replacement Plan")
replacementPlanLOD = st.selectbox('Select Level of Detail', ['County','Department','Department/Vehicle Type','Equipment ID'])
replacementPlanLOD_Lookup = {'County':'county',
                             'Department':'dept_name',
                             'Department/Vehicle Type':['dept_name','vehicledescription'],
                             'Equipment ID':'equipmentid'}

replacementPlan = data.groupby(replacementPlanLOD_Lookup[replacementPlanLOD])[inputs.years].sum().astype(int).replace({0:''})
st.dataframe(replacementPlan)

#download detailed plan
csv = data.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}">Download Detailed Roadmap</a>'
st.markdown(href, unsafe_allow_html=True)




