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
sys.path.append(os.path.join(os.getcwd(),'src'))
sys.path.append(os.path.join(os.getcwd(),'img'))

from optimization.MIP.mip_inputs import *
from optimization.MIP.mip_model import *
from optimization.MIP.mip_outputs import *

col1, mid, col2 = st.beta_columns([10,20,40])
with col1:
    st.image(logo, width=100)
with mid:
    st.title('CAR-NET')
with col2:
    st.write('  ')

#Sidebar
sidebarTitle = st.sidebar.title("""Model Inputs""")
base_procurement_budget = st.sidebar.number_input("2022 Procurement Budget",value=6500000,step=100000)
base_operations_budget = st.sidebar.number_input("2022 Operations Budget",value=6000000,step=100000)
budget_rate = st.sidebar.number_input('% Increase in Budget Year-Over-Year',value=3)
emissions_target = st.sidebar.number_input("Emissions Decrease Target %",value=40)
final_year = st.sidebar.number_input("Planning Range [2022 - ]",value=2037)
run = st.sidebar.button('Run Model')
add_space(0)
st.sidebar.markdown("CAR-NET (Carbon Neutral Evaluation Tool) was created for Baltimore County by George Mason University, SEOR Department")

#Parameters from front end to feed into the model
UI_params = {
    'initial_procurement_budget':base_procurement_budget,
    'initial_operations_budget':base_operations_budget,
    'budget_rate':budget_rate/100,
    'emissions_target_pct_of_base':emissions_target/100,
    'planning_interval':[2022,final_year],
    'objective_weights':{'cost':0.70,'emissions':0.30},
}

#the run button actually just clears the cache. The script will already be running anyway.
if run:
    caching.clear_cache()

#generate model inputs
inputs = MIP_Inputs(data,UI_params)

#run the model
m,x,vehicles,penalty_budget,penalty_emissions,validSchedulesPerVehicle,validSchedules = carnet(inputs)

#generate model outputs
outputs = MIP_Outputs(data,UI_params,m,x,vehicles,penalty_budget,penalty_emissions,validSchedulesPerVehicle,validSchedules)
alt_solutions = outputs.get_alternative_solutions()
options = select_alternative_solutions(alt_solutions,outputs)

roadmapSelected = st.selectbox('Select a Replacement Roadmap', ['A','B','C'])
roadmap_lookup = {'A':0,'B':1,'C':2}

#high level summary of all replacement plans
summary = get_summary(inputs,outputs,options,roadmap_lookup)
st.write(summary)

#high level visualizations for a given replacement plan
visualize_ev_vs_ice(roadmapSelected,options,outputs)
visualize_emissions(roadmapSelected,options,outputs)
visualize_aquisition_cost(roadmapSelected,options,outputs)
visualize_operations_cost(roadmapSelected,options,outputs)

#detailed vehicle replacement plan
st.subheader("Vehicle Replacement Plan")
plan = pd.DataFrame(options[roadmap_lookup[roadmapSelected],'schedules'],columns=['vehicle_idx','solution_idx']+inputs.years)
detailed_output = data.merge(plan,on='vehicle_idx',how='left')
replacementPlanLOD = st.selectbox('Select Level of Detail', ['Equipment ID','Department/Vehicle Type','Department','County'])
replacementPlanLOD_Lookup = {'County':'county',
                             'Department':'dept_name',
                             'Department/Vehicle Type':['dept_name','vehicledescription'],
                             'Equipment ID':['equipmentid','vehicledescription']}
replacementPlan = detailed_output.groupby(replacementPlanLOD_Lookup[replacementPlanLOD])[inputs.years].sum().astype(int).replace({0:''}).sort_values(inputs.years)
st.dataframe(replacementPlan)

#download detailed plan
csv = detailed_output.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}">Download Detailed Roadmap</a>'
st.markdown(href, unsafe_allow_html=True)