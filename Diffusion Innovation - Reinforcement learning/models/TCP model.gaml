/**
* Name: Basemodel
* Based on the internal skeleton template. 
* Author: Patrick Taillandier
* Tags: 
*/

model Basemodel

global {
	/****** CONSTANTS **********/
	string FINANCIAL <- "financial";
	string ENVIRONMENTAL <- "environmental";
	string FARM_MANAGEMENT <- "farm management";
	
	string FINANCIAL_SUPPORT <- "financial support";
	string ENV_SENSIBILISATION <- "environmental sensibiliation";
	string TRAINING <- "training";
	string DO_NOTHING<- "do nothing";
	
	/****** PARAMETERS **********/
	float step <- 1 #day;
	float end_simulation_after <- 5 #year;
	int number_farmers <- 100;
	float new_budget_year <- 10.0;
	int nb_friends <- 5;
	float opinion_diff_accep <- 0.5;
	float opinion_speed_convergenge <- 0.1;
	float proba_interaction_day <- 0.1;
	
	
	list<string> topics <- [FINANCIAL, ENVIRONMENTAL, FARM_MANAGEMENT] ;
	list<string> possible_actions <- [FINANCIAL_SUPPORT, ENV_SENSIBILISATION, TRAINING,DO_NOTHING,TRAINING+"|" +FINANCIAL_SUPPORT, TRAINING+"|" +ENV_SENSIBILISATION, FINANCIAL_SUPPORT+"|" +ENV_SENSIBILISATION, FINANCIAL_SUPPORT+"|" + ENV_SENSIBILISATION+"|" + TRAINING] ;
	bool save_result_in_file <- false;
	/****** GLOBAL VARIBALES **********/
	
	int port;
	institution the_institution;
	float time_sim update: time_sim + step;
	/********* OUTPUT ***********/
	float adoption_rate <- 0.0 ;
	float min_intention <- 0.0 ;
	float max_intention <- 0.0 ;
	float mean_intention <- 0.0 ;
	float median_intention <- 0.0 ;
	
	list<float> adoption_rate_policy <- [] ;
	list<float> mean_intention_policy <- [] ;
	
	list<int> policy_id;
	list<float> adoption_rates;
	list<float> mean_intentions;
	list<float> num_adopters;
	list<float> mean_intentions_sim;
	list<float> time_s;
	
	bool mode_batch <- false;
	init {
		create institution {
			the_institution <- self;
			do initialize;
			
		}
		do create_farmers;
	}
	
	/*  ******* ACTIONS **********/

	
	action update_outputs(list<farmer> farmers) {
		adoption_rate <- (farmers count each.adoption) /length(farmers);
		mean_intention <- farmers mean_of (each.intention);
		min_intention <- farmers min_of (each.intention) ;
		max_intention <- farmers max_of (each.intention);
		median_intention <- median(farmers collect (each.intention)) ;
		
	}
	
	action create_farmers {
		create farmer number: number_farmers {
			loop topic over: topics{
				opinion_on_topics_init[topic] <- rnd(1.0);
				weights_for_topics[topic] <- rnd(1.0);
			}
			float sum_w <-sum(weights_for_topics.values);
			loop topic over:topics {
				weights_for_topics[topic] <- weights_for_topics[topic]/sum_w;
			}
			social_network <- nb_friends among (farmer - self);
			w_attitude <- rnd(1.0);
			w_social <- rnd(1.0);
			w_pbc <- rnd(1.0);
			technical_skill_init <- rnd(1.0);
			sum_w <- w_attitude + w_social + w_pbc;
			w_attitude <- w_attitude / sum_w;
			w_social <- w_social / sum_w;
			w_pbc <- w_pbc / sum_w;
			do initialize;
		}
	}
	
	reflex compute_stats  {
		do update_outputs(farmer as list);
	}
	
	

	reflex end_simulation when:(time >= end_simulation_after) {
		write "ending simulation";
		ask institution {
			write "simulation sending last reward";
			do send_reward;
			write "simulation sending end signal";
			do send_end;
			write "simulation waiting for server's end signal";
			do read_line from:server;//when the server is over he sends a message to the simulation, needed to prevent connexion reset exceptions
		}
		write "simulation pausing";
		do die;	
	}
}


species farmer schedules: shuffle(farmer) {
	float w_attitude ;
	float w_social ;
	float w_pbc ;
	float adoption_threshold <- gauss(0.8, 0.1) min: 0.0 max: 1.0;
	map<string,float> opinion_on_topics_init;
	
	map<string,float> opinion_on_topics;
	map<string,float> weights_for_topics;
	
	float attitude ;
	float social_norm ;
	float pbc;
	float intention ;
	list<farmer> social_network;
	bool adoption <- false;
	float technical_skill_init;
	
	float technical_skill <- 0.0;
	
	action initialize {
		opinion_on_topics <- copy(opinion_on_topics_init);
		adoption <- false;
		technical_skill <- technical_skill_init;
		
	}
	
	
	action exchange_with_other {
		if flip(proba_interaction_day) {
			string topic <- one_of(opinion_on_topics.keys);
			float x <-opinion_on_topics[topic];
			ask one_of(social_network) {
				float x_other <-opinion_on_topics[topic];
				if abs(x - x_other) < opinion_diff_accep and (x != x_other){
					myself.opinion_on_topics[topic] <- x + opinion_speed_convergenge * (x_other - x);
					opinion_on_topics[topic] <- x_other + opinion_speed_convergenge * (x - x_other);
					do compute_intention;
					ask myself {
						do compute_intention;
					}
				}
			}
		}
	}
	
	action compute_attitude {
		attitude <- 0.0;
		map<string, float> supp <- the_institution.support;
		loop topic over: topics {
			attitude <- attitude + (opinion_on_topics[topic] + (supp = nil ? 0.0 : supp[topic])) *  weights_for_topics[topic];
		}
	}
	action compute_social_norm {
		social_norm <- empty(social_network) ? 0.5 : (social_network count each.adoption)/length(social_network);
	}
	
	action compute_pbc {
		pbc <-  technical_skill;
	}
	action compute_intention {
		do compute_attitude;
		do compute_social_norm;
		do compute_pbc;
		intention <- attitude * w_attitude + social_norm * w_social + pbc * w_pbc;
		
	}
	action decision {
		if not adoption and (intention > adoption_threshold) {
			adoption <- true;
			ask the_institution {
				do give_financial_support;
			}
		}
	}
	
	reflex share_opinion {
		do exchange_with_other;
	}
	reflex decide when: every(#week){
		do compute_intention;
		do decision;
	}
}



species institution skills:[tcp] {
	
	float budget;
	map<string,float> support;
	int previous_adopters_nb;
	float previous_mean_intention;
	string server;
	bool at_least_one_policy;
	
	action initialize {
		support					<- [];
		budget 					<- 0.0;
		previous_adopters_nb 	<- 0;
		previous_mean_intention <- 0.0;
		at_least_one_policy		<- false;
		
		write "port " + port;
		server <- connect_to_server("localhost", port);
		
		
	}
	action select_policy {
		
		//Sending the reward for the last policy choice
		do send_reward;
		
		loop topic over: topics {
			support[topic] <- 0.0;
		}
		
		do send_observations;
		
		do select_actions ;
	}

	action send_end {
		let sent 	<- send(server, "END\n");
		if (! sent) {
			write "impossible d'envoyer le message de fin de simulation à : " + server;
		}
	}	

	action send_reward {
		if(at_least_one_policy) {
			//The reward = percentage of progression of mean_intention
			let reward 	<- previous_mean_intention != 0 ? (mean_intention - previous_mean_intention)/ previous_mean_intention : mean_intention ;
			let sent 	<- send(server, string(reward) + "\n");
			if (! sent) {
				write "impossible d'envoyer le reward " + reward + " à : " + server;
			}
		}
		else {
			at_least_one_policy <- true;
		}
		previous_mean_intention <- mean_intention;
	}

	action send_observations {
		//budget restant, nb d'adoptant/taux, temps restant
		let observations <- "(" + budget + "," + adoption_rate + "," + (end_simulation_after - time) + ")\n" ;
		let sent <- send(server, observations);
		if(!sent) {
			write "Impossible d'envoyer les observations "+ observations + " au serveur " + server;
		}
		
		
	}
	
	action select_actions {
		
		//Getting the actions from the server
		let actions_msg <- read_line(server);
		if(actions_msg != nil) {
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
			write "impossible de recevoir une politique du serveur " + server;
		}
		
		
		
	}
	action add_money {
		budget <- budget + new_budget_year;
	}
	
	action financial_support ( float level) {
		support[FINANCIAL] <- level;
	}
	
	action training (float level, float percent) {
		let number <- int(percent * length(farmer));
		if (budget > (number * level)) {
			ask number among farmer  {
				technical_skill <- technical_skill + level;
				opinion_on_topics[FARM_MANAGEMENT] <- opinion_on_topics[FARM_MANAGEMENT] + level; 
			}
			budget <- budget - (number * level);
		}
	}
	
	action environmental_sensibilisation (float level, float percent) {
		let number <- int(percent * length(farmer));
		if (budget > (number * level / 2.0)) {
			ask number among farmer  {
				opinion_on_topics[ENVIRONMENTAL] <- opinion_on_topics[ENVIRONMENTAL] + level; 
			}
			budget <- budget - (number * level) / 2.0;
		}
	}
	
	action give_financial_support {
		budget <- budget - support[FINANCIAL];
	}
	
	reflex receive_budget when: every(#year) {
		do add_money;
	}
	reflex choose_policy when: every(6 #month){
		do select_policy;
	}
} 


experiment one_simulation_batch until: time >= end_simulation_after type: batch {
	
	
	init {
		mode_batch <- false;
	}
}

experiment one_simulation type: gui {
	 
	 parameter "port" var:port init:0;
	 
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
