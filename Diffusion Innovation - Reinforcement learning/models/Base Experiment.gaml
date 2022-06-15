/**
* Name: BaseExperiment
* Based on the internal empty template. 
* Author: admin_ptaillandie
* Tags: 
*/


model BaseExperiment

import "Base_model.gaml"

experiment one_simulation_batch until: time >= end_simulation_after type: batch {
	init {
		mode_batch <- false;
	}
}

experiment one_simulation type: gui {
	action _init_ {
		create simulation with:(
			mode_batch:false,
			pause_sim:true
		);
	}
	output {
		display charts {
			chart "intention of farmers" memorize:false type: series  size: {1.0,0.5}{
				data "mean intention" value: mean_intention color: #blue marker: false;
				data "min intention" value:min_intention color: #red marker: false;
				data "max intention" value: max_intention color: #green marker: false;
				data "median intention" value: median_intention color: #magenta marker: false;
			
			}
			chart "percentage of adopters" memorize:false type: series size: {1.0,0.5} position: {0.0,0.5} {
				data "percentage of adopters" value: adoption_rate * 100.0  color: #green;
					
			}
		}
	}
}



