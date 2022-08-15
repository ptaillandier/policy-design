/**
* Name: Basemodel
* Based on the internal skeleton template. 
* Author: Patrick Taillandier
* Tags: 
*/

model Basemodel

import "Base_model.gaml"

global {
	
	int port;
	
	institution create_institution {
		create institution_tcp;
		return first(institution_tcp);
	}
	
	
	/*  ******* ACTIONS **********/

	action simulation_ending  {
		write "ending simulation";
		ask institution_tcp {
			//write "simulation sending last reward";
			do send_reward;
			//write "simulation sending end signal";
			do send_end;
			//write "simulation waiting for server's end signal";
			
			//when the server is over he sends a message to the simulation, needed to prevent connection reset exceptions
//			do read_line from:server;
			loop while: !has_more_message()  { 
				do fetch_message_from_network;
			}

		}
		//write "simulation pausing";
		do die;	
	}
	
	action game_over {
		write "ending simulation unexpectedly";
		do die;
	}
}


species institution_tcp parent: institution skills:[network] {
	
	bool at_least_one_policy;
	
	action other_things_init {
        write "other_things_init";
		at_least_one_policy		<- false;
		write "port " + port;
		do connect to:"localhost" port:port protocol:"tcp_client" raw:true;
		//do send_observations; //Initial observations sending for gym
		write "END other_things_init";
	}
	action thing_before_policy_selecting {
		//Sending the reward for the last policy choice
		write "send_reward";
		do send_reward;
		write "END send_reward";
		write "send_observations";
		do send_observations;
		write "END send_observations";
	}

	action send_end {
	        //budget restant, nb d'adoptant/taux, temps restant
		write "send_end";
		let observations <- "(" + budget + "," + adoption_rate + "," + (end_simulation_after - time) + ")" ;
		do send to:"localhost:" + port contents:observations+"END";
		
//		let sent 	<- send(server, observations+"END\n");
//		if (! sent) {
//			write "impossible d'envoyer le message de fin de simulation à : " + server;
//			ask world{
//				do game_over;				
//			}
//		}
	}	

	action send_reward {
		write "send_reward";
		if(at_least_one_policy) {
			//The reward = increment on percentage of new adopters
			//float reward 	<- previous_mean_intention != 0 ? (mean_intention - previous_mean_intention)/ previous_mean_intention : mean_intention ;
			let reward 	<- (adopters_nb - previous_adopters_nb)/number_farmers;
			write reward;
			do send to:"localhost:" + port contents:reward;
//			bool sent 	<- send(server, string(reward) + "\n");
//			if (! sent) {
//				write "impossible d'envoyer le reward " + reward + " à : " + server;
//				ask world{
//					do game_over;				
//				}
//			}
		}
		else {
			at_least_one_policy <- true;
		}
		previous_mean_intention <- mean_intention;
		previous_adoption_rate <- adoption_rate;
                previous_adopters_nb <- adopters_nb;
	}

	action send_observations {
		//budget restant, nb d'adoptant/taux, temps restant
		let observations <- "(" + budget + "," + adoption_rate + "," + (end_simulation_after - time) + ")" ;
		write "sending observations: " + observations;
		do send to:"localhost:"+port contents:observations;
		//let sent <-  send(server, observations);
		//if(!sent) {
//			write "Impossible d'envoyer les observations "+ observations + " au serveur " + server;
//			ask world{
//				do game_over;		
//			}
		//}
		
		
	}
	
	message wait_next_message {
		write "waiting for server to send data"; 
		loop while: !has_more_message()  { 
			do fetch_message_from_network;
		}
		return fetch_message();
	}
	
	action select_actions {
		write "select_actions";
		//Getting the actions from the server
		let msg <- wait_next_message();
		write "received " + msg;
				
		if(msg != nil) {
			string actions_msg <- msg.contents;
			write "actions_msg" + actions_msg;
			let actions 	<- replace(replace(actions_msg, "[", ""), "]","") split_with ",";
			let fin_support	<- float(actions[0]);
			let training_l	<- float(actions[1]);
			let training_p 	<- float(actions[2]);
			let envr_l		<- float(actions[3]);
			let envr_p		<- float(actions[4]);

			write actions_msg + " : " + fin_support + " " + training_l +","+training_p + " " + envr_l + "," + envr_p;
			
			
			do financial_support(fin_support);
			do training(training_l, training_p);
			do environmental_sensibilisation(envr_l, envr_p);
		}
		else {
			write "impossible de recevoir une politique du serveur ";
			ask world{
				do game_over;				
			}
		}
		
		
		
	}
	
} 


experiment one_simulation_batch until: time >= end_simulation_after type: batch {
	
	parameter "port" var:port init:0;
	
	init {
		mode_batch <- false;
	}
}

experiment one_simulation type: gui {
	 
	parameter "port" var:port init:0;
	 
	//output {
		//display charts {
		//	chart "intention of farmers" memorize:false type: series  size: {1.0,0.5}{
		//		data "mean intention" value: mean_intention color: #blue marker: false;
		//		data "min intention" value:min_intention color: #red marker: false;
		//		data "max intention" value: max_intention color: #green marker: false;
		//		data "median intention" value: median_intention color: #magenta marker: false;
			
		//	}
		//	chart "percentage of adopters" memorize:false type: series size: {1.0,0.5} position: {0.0,0.5} {
		//		data "percentage of adopters" value: adoption_rate * 100.0  color: #green;
					
		//	}
		//}
	//}
}
