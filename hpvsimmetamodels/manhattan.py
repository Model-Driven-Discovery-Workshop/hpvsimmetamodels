'''
Copyright 2023 IBM Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

'''

import json
import argparse
import hpvsim as hpv
import numpy as np

def run_model(parameters, pop_size=2e5):
    ##Country demographics
    pars = {}
    pop_scale = int(parameters["population"]/pop_size)

    pars["pop_scale"] = pop_scale
    pars["n_agents"] = pop_size # Population size

    results=[]
    verbose = 0

    # Define the parameters    
    pars["end"]=parameters["end"]
    pars["burnin"]=parameters["burnin"]
    pars["start"]=parameters["start"]
    pars["rand_seed"]=parameters["rand_seed"]

    # Define a series of interventions to screen, triage, assign treatment, and administer treatment
    screen      = hpv.routine_screening(start_year=parameters["start_year"], prob=parameters["screen_prob"], product='via', label='screen') # Routine screening
    to_triage   = lambda sim: sim.get_intervention('screen').outcomes['positive'] # Define who's eligible for triage
    triage      = hpv.routine_triage(eligibility=to_triage, prob=parameters["triage_prob"], product='hpv', label='triage') # Triage people
    to_treat    = lambda sim: sim.get_intervention('triage').outcomes['positive'] # Define who's eligible to be assigned treatment
    assign_tx   = hpv.routine_triage(eligibility=to_treat, prob=parameters["assign_tx_prob"], product='tx_assigner', label='assign_tx') # Assign treatment
    to_ablate   = lambda sim: sim.get_intervention('assign_tx').outcomes['ablation'] # Define who's eligible for ablation treatment
    ablation    = hpv.treat_num(eligibility=to_ablate, prob=parameters["ablation_prob"], product='ablation', label="ablation") # Administer ablation
    to_excise   = lambda sim: sim.get_intervention('assign_tx').outcomes['excision'] # Define who's eligible for excision
    excision    = hpv.treat_delay(eligibility=to_excise, prob=parameters["excision_prob"], product='excision', label="excision") # Administer excision

    # Create the sim with and without interventions
    sim = hpv.Sim(
        pars, 
        interventions = [screen, triage, assign_tx, ablation, excision], 
        verbose=verbose,
        label='Varing screen & treat scenario', 
        analyzers=[hpv.daly_computation(start=parameters["start_year"])])
    sim.run()
    a = sim.get_analyzers()[0]
    df = a.df
    discounted_cancers = np.array([i / (1+parameters["discountrate"]) ** t for t, i in enumerate(df['new_cancers'].values)])
    discounted_cancer_deaths = np.array([i / (1+parameters["discountrate"]) ** t for t, i in enumerate(df['new_cancer_deaths'].values)])

    avg_age_ca_death = df['av_age_cancer_deaths'].mean()
    avg_age_ca = df['av_age_cancers'].mean()
    ca_years = avg_age_ca_death - avg_age_ca
    yld = parameters['disutility_weight'] * ca_years * discounted_cancers
    yll = (parameters["expected_lifespan"] - avg_age_ca_death) * discounted_cancer_deaths
    dalys = yll + yld
    df['dalys']=dalys

    df['intervention_costs']= sim.get_intervention("screen").n_products_used.values[int(parameters["start_year"]-parameters["start"]):]*parameters["screen_cost"]+\
        sim.get_intervention("triage").n_products_used.values[int(parameters["start_year"]-parameters["start"]):]*parameters["triage_cost"]+\
        sim.get_intervention("assign_tx").n_products_used.values[int(parameters["start_year"]-parameters["start"]):]*parameters["assign_tx_cost"]+\
        sim.get_intervention("ablation").n_products_used.values[int(parameters["start_year"]-parameters["start"]):]*parameters["ablation_cost"]+\
        sim.get_intervention("excision").n_products_used.values[int(parameters["start_year"]-parameters["start"]):]*parameters["excision_cost"]

    for epoch in df.index:
        if epoch >= parameters["start_year"]:
            results.append({
                'epoch': epoch,
                'population': parameters["population"],
                'new_cancers': df["new_cancers"][epoch],
                'new_cancer_deaths': df["new_cancer_deaths"][epoch],
                'dalys': df["dalys"][epoch],
                'intervention_costs': df['intervention_costs'][epoch]
            })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', default="input.json", help='The JSON formatted file containing the parameters for the model')
    parser.add_argument('--output_data', default="output.json", help='The JSON formatted file containing the output for the model')
    args = parser.parse_args()

    input_data, output_data = args.input_data, args.output_data

    with open(input_data) as json_file:
        parms = json.load(json_file)
        results = run_model(parms[0])

    if output_data != "":
        with open(output_data, "w") as outfile:
            json.dump(results, outfile, indent=2)
