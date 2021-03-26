import streamlit as st
import pandas as pd
import numpy as np
import sys 
import os
import time
import plotly.graph_objects as go
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


st.title("CAR-NET")
st.header("Carbon Neutral Evaluation Tool")


#INPUTS
sidebarTitle = st.sidebar.write("""# Model Inputs""")

base_procurement_budget = st.sidebar.number_input("Base Procurement Budget for 2021")
base_operations_budget = st.sidebar.number_input("Base Operations Budget for 2021")
budget_rate = st.sidebar.number_input('% Increase in Budget Year-Over-Year')
emissions_baseline = st.sidebar.number_input("Emissions Baseline")
emissions_target = st.sidebar.number_input("Emissions Target %")
run = st.sidebar.button('Run Model')

def add_space(numSpaces):
    for i in range(numSpaces):
        st.sidebar.write('  ')
add_space(14)
st.sidebar.write(" Created for Baltimore County by George Mason University")

data = pd.read_excel('../../data/17MAR_data_template.xlsx').head(10)
data['current_age'] = datetime.datetime.now().year - pd.to_datetime(data.purchasedate).dt.year
data = data.reset_index().rename({"index":"vehicle_idx"},axis=1)
data['county'] = 'Baltimore County'
data = data.drop_duplicates('equipmentid')
data = data[data.vehicle_idx<1458]

UI_params = {
    'initial_procurement_budget':base_procurement_budget,
    'initial_operations_budget':base_operations_budget,
    'budget_rate':budget_rate/100,
    'emissions_baseline': emissions_baseline,#metric tons
    'emissions_target_pct_of_base':emissions_target/100,
    'planning_interval':[2022,2037],
    # 'min_miles_replacement_threshold':150000,#miles
    # 'min_vehicle_age_replacement_threshold':6,#years
    'objective_weights':{'cost':0.70,'emissions':0.30},
}


if run:

    # fff = funcc(235)

    # @st.cache
    inputs = MIP_Inputs(data,UI_params)
    MODEL = MIP_Model(data,UI_params)

try:
    # st.header(fff)
    st.write(inputs.make_replacement_schedules())
    st.header(MODEL.solverTimeLimit)
    outputs = MIP_Outputs(data,UI_params)

    optimal_obj,optimal_solution = outputs.get_optimal_solution()
    alt_solutions = outputs.get_alternative_solutions()
    selected_alternative_solutions = outputs.select_alternative_solutions(alt_solutions,optimal_obj)
    options = selected_alternative_solutions
    option1 = pd.DataFrame(options[0,'schedules'],columns=['vehicle_idx','solution_idx']+inputs.years)
    data = data.merge(option1,on='vehicle_idx',how='left')

    roadmapSelected = st.selectbox('Select a Roadmap', ['A (Optimal)','B','C'])
    roadmap_lookup = {'A (Optimal)':0,'B':1,'C':2}



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
    fig.update_layout(title={'text':'ICE vs EV/Hybrid Inventory Over Time'})
    st.write(fig)

    # data = [(1, 2, 3)]
    # When no file name is given, pandas returns the CSV as a string, nice.
    # df = pd.DataFrame(data, columns=["Col1", "Col2", "Col3"])
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download Detailed Roadmap</a>'
    st.markdown(href, unsafe_allow_html=True)


except Exception as e:
    st.write(e)



