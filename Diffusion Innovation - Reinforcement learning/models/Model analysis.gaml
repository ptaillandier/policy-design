/**
* Name: Modelanalysis
* Based on the internal empty template. 
* Author: Patrick Taillandier
* Tags: 
*/


model Modelanalysis

import "Base_model.gaml"

/* Insert your model definition here */

experiment environmental_support until: end_simulation repeat: 50 type: batch keep_seed: true{
	parameter sensibilisation_level var: sensibilisation_level min: 0.0 max: 1.0 step: 0.2;
	parameter sensibilisation_proportion var: sensibilisation_proportion min: 0.0 max: 1.0 step: 0.2;
	init {
		new_budget_year <- #max_float;
		mode_batch <- true;
		name_file_save <-"environment_results.csv";
		save "id,seed,cycle,financial_help_level,training_level,training_proportion,sensibilisation_level,sensibilisation_proportion,adoption_rate,mean_intention" type: text  to: "environment_results.csv"; 
		gama.pref_parallel_simulations <- true;
		gama.pref_parallel_threads <- 25;
	}
	reflex write_end_result {
		list<float> vals_adoption <- simulations collect each.adoption_rate;
		list<float> vals_intention <- simulations collect each.mean_intention;
		
		write sample(sensibilisation_level)  +"," + sample(sensibilisation_proportion)+" adoption: " + mean(vals_adoption) +"(" + standard_deviation(vals_adoption) +") - intention:"+ mean(vals_intention) +"(" + standard_deviation(vals_intention) +")";
		
	
	}
}


experiment training_support until: end_simulation repeat: 50 type: batch keep_seed: true{
	parameter training_level var: training_level min: 0.0 max: 1.0 step: 0.2;
	parameter training_proportion var: training_proportion min: 0.0 max: 1.0 step: 0.2;
	init {
		new_budget_year <- #max_float;
		mode_batch <- true;
		name_file_save <-"training_results.csv";
		save "id,seed,cycle,financial_help_level,training_level,training_proportion,sensibilisation_level,sensibilisation_proportion,adoption_rate,mean_intention" type: text  to: "training_results.csv"; 
		gama.pref_parallel_simulations <- true;
		gama.pref_parallel_threads <- 25;
	}
	reflex write_end_result {
		list<float> vals_adoption <- simulations collect each.adoption_rate;
		list<float> vals_intention <- simulations collect each.mean_intention;
		
		write sample(training_level)  +"," + sample(training_proportion)+" adoption: " + mean(vals_adoption) +"(" + standard_deviation(vals_adoption) +") - intention:"+ mean(vals_intention) +"(" + standard_deviation(vals_intention) +")";
		
	}
}

experiment financial_help until:  end_simulation repeat: 50 type: batch keep_seed: true{
	parameter financial_help_level var: financial_help_level min: 0.0 max: 1.0 step: 0.1;
	init {
		new_budget_year <- #max_float;
		mode_batch <- true;
		gama.pref_parallel_simulations <- true;
		gama.pref_parallel_threads <- 25;
		
		name_file_save <-"financial_help_results.csv";
		save "id,seed,cycle,financial_help_level,training_level,training_proportion,sensibilisation_level,sensibilisation_proportion,adoption_rate,mean_intention" type: text  to: "financial_help_results.csv"; 
		
	}
	reflex write_end_result {
		list<float> vals_adoption <- simulations collect each.adoption_rate;
		list<float> vals_intention <- simulations collect each.mean_intention;
		
		write sample(financial_help_level) +" adoption: " + mean(vals_adoption) +"(" + standard_deviation(vals_adoption) +") - intention:"+ mean(vals_intention) +"(" + standard_deviation(vals_intention) +")";
		
	}
}

experiment test_interaction until:  end_simulation repeat: 4 type: batch keep_seed: true {
	parameter proba_interaction_day var: proba_interaction_day min: 0.0 max: 1.0 step: 0.2;
	init {
		mode_batch <- true;
		gama.pref_parallel_simulations <- true;
		gama.pref_parallel_threads <- 25;
		
		name_file_save <-"interaction_results.csv";
		save "id,seed,cycle,financial_help_level,training_level,training_proportion,sensibilisation_level,sensibilisation_proportion,adoption_rate,mean_intention" type: text  to: "interaction_results.csv"; 
		
	}
	
	reflex write_end_result {
		list<float> vals_adoption <- simulations collect each.adoption_rate;
		list<float> vals_intention <- simulations collect each.mean_intention;
		
		write sample(proba_interaction_day) +" adoption: " + mean(vals_adoption) +"(" + standard_deviation(vals_adoption) +") - intention:"+ mean(vals_intention) +"(" + standard_deviation(vals_intention) +")";
		
	}
}


experiment test_stochasticity until:  end_simulation repeat: 100 type: batch {
	init {
		mode_batch <- true;
		name_file_save <-"stochasticity_results.csv";
		gama.pref_parallel_simulations <- true;
		gama.pref_parallel_threads <- 25;
		
		save "id,seed,cycle,financial_help_level,training_level,training_proportion,sensibilisation_level,sensibilisation_proportion,adoption_rate,mean_intention" type: text  to: "stochasticity_results.csv"; 
		
	}
}