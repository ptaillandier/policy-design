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
	bool reinforcement_learning <- false;
	int nb_iterations_rl <- 10000;
	int nb_replications <- 1;
	float step <- 1 #day;
	float end_simulation_after <- 5 #year;
	int number_farmers <- 100;
	float new_budget_year <- 10.0;
	int nb_friends <- 5;
	float opinion_diff_accep <- 0.5;
	float opinion_speed_convergenge <- 0.1;
	float proba_interaction_day <- 0.1;
	
	
	float learning_rate <- 0.1;
	float discount_factor <- 0.5;
	
	list<string> topics <- [FINANCIAL, ENVIRONMENTAL, FARM_MANAGEMENT] ;
	list<string> possible_actions <- [FINANCIAL_SUPPORT, ENV_SENSIBILISATION, TRAINING,DO_NOTHING,TRAINING+"|" +FINANCIAL_SUPPORT, TRAINING+"|" +ENV_SENSIBILISATION, FINANCIAL_SUPPORT+"|" +ENV_SENSIBILISATION, FINANCIAL_SUPPORT+"|" + ENV_SENSIBILISATION+"|" + TRAINING] ;
	bool save_result_in_file <- false;
	bool display_action_chosen <- false;
	/****** GLOBAL VARIBALES **********/
	
	institution the_institution;
	float time_sim update: time_sim + step;
	bool end_of_run <- false update: false;
	
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
	
	
	init {
		create institution {
			the_institution <- self;
			if not reinforcement_learning {
				do initialize(0);
			}
		}
		if not reinforcement_learning {do create_farmers(0);}
		else {
			do learning;
		}
	}
	
	/*  ******* ACTIONS **********/
	action simulate(int i, list<farmer> farmers) {
	
		float time_sim_i <- 0.0;
		int week <- 0;
		int month <- 0;
		loop while:time_sim_i < end_simulation_after {
			time_sim_i <- time_sim_i + step;
			float week_ <- time_sim_i / #week;
			float month_ <- time_sim_i / #month;
			time_s[i] <- time_sim_i;
			do farmer_behavior_day(farmers);
			if (int(week_) > week) {
				week <- int(week_);
				do farmer_behavior_week(farmers);
			}
			if (int(month_) > month) {
				month <- int(month_);
				if (month mod 6) = 0 {
					num_adopters[i] <- farmers count each.adoption;
					mean_intentions_sim[i] <- farmers mean_of (each.intention);
					do institution_behavior_6month(i);
					ask farmers {
						do compute_intention;
					}
				}
				if (month mod 12) = 0 {
					do institution_behavior_year(i);
				}
				do farmer_behavior_week;
			}
		}
		ask the_institution {
			last_turn[i] <- true;
			map<string,int> new_state <- define_state(i);
			map<string,float> actions <- q[new_state];
			if (actions = nil) or empty(actions) {
				actions <- [];
				loop p over: possible_actions {
					actions[p] <- 1.0;
				}
			}
			q[new_state] <- actions;
			do update_q(i,new_state);
			
		}
		do update_outputs(farmers);
		
	}
	action farmer_behavior_day(list<farmer> farmers) {
		ask farmers {
			do exchange_with_other;
		}
	}
	action farmer_behavior_week(list<farmer> farmers) {
		ask farmers {
			do decision;
		}
	}
	action institution_behavior_6month (int i) {
		ask the_institution {
			do select_policy(i);
		}
	}
	action institution_behavior_year(int i) {
		ask the_institution {
			do add_money(i);
		}
	}
	
	action update_outputs(list<farmer> farmers) {
		adoption_rate <- (farmers count each.adoption) /length(farmers);
		mean_intention <- farmers mean_of (each.intention);
		min_intention <- farmers min_of (each.intention) ;
		max_intention <- farmers max_of (each.intention);
		median_intention <- median(farmers collect (each.intention)) ;
		if (reinforcement_learning) {
			adoption_rates << adoption_rate;
			mean_intentions <<mean_intention;
		}
	}
	
	action create_farmers(int i) {
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
			id <-i;
			do initialize;
		}
	}
	
	action reset_sim {
		time_sim <- 0.0;
		ask farmer {
			do initialize;
		}
		ask the_institution {
			budget <- [];
			support <- [];
			last_turn <- [];
			action_chosen <- [];
		}
		adoption_rates <- [];
		mean_intentions <- [];
		
		mean_intentions_sim <- [];
		num_adopters <- [];
	}
	
	reflex compute_stats when: not reinforcement_learning {
		do update_outputs(farmer as list);
	}
	
	action learning {
		int cpt <- 1;
		bool first_round <- true;
		loop times: nb_iterations_rl {
			float t <- machine_time;
			policy_id << cpt;
			do reset_sim;
			
			loop i from: 0 to: nb_replications -1{
				time_s << 0.0;
				if (first_round) {
					do create_farmers(i);
				}
				num_adopters << 0;
				ask the_institution {
					do initialize(i);
				}
			}
			first_round <- false;
			loop i from: 0 to: nb_replications -1{
				ask self parallel: true { 
					list<farmer> farmers <- farmer where (each.id = i);
					ask farmers {
						do compute_intention;
					}
					mean_intentions_sim << farmers mean_of each.intention ;
				
					ask the_institution{previous_mean_intention[i] <- farmers mean_of (each.intention);}
					do simulate(i, farmers);
				}
			}
			adoption_rate_policy << mean(adoption_rates);
			mean_intention_policy << mean(mean_intentions) ;
			string result <-  "" + cpt +" mean intention: " +  last(mean_intention_policy) with_precision 2 + " adoption percentage: " + (100.0 * (last(adoption_rate_policy))) with_precision 1 + "% computation time: " +((machine_time - t)/1000.0 with_precision 1) ;
			write result + (display_action_chosen ? (" _ " + the_institution.action_chosen[0]) : "");
			if save_result_in_file {
				save result to: "result.txt" rewrite: false;
			}
			cpt <- cpt + 1;	
		}
	}
	reflex end_simulation when: not reinforcement_learning and time_sim >= end_simulation_after {
		do pause;	
	}
}


species farmer schedules: container(reinforcement_learning ? [] : shuffle(farmer)) {
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
	int id;
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
		map<string, float> supp <- the_institution.support[id];
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
				do give_financial_support(myself.id);
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

species institution schedules: container(reinforcement_learning ? [] : shuffle(institution)){
	map<int,float> budget;
	map<int,map<string,float>> support;
	map<int,int> previous_mean_intention;
	map<map<string,int>, map<string,float>> q;
	map<string,int> current_state;
	string last_action <- "";
	list<bool> last_turn;
	list<string> action_chosen <- [];
	
	action initialize(int id) {
		support[id] <- [];
		budget[id] <- 0.0;
		previous_mean_intention[id] <- 0.0;
		last_turn<< false;
		
	}
	action select_policy(int id) {
		loop topic over: topics {
			support[id][topic] <- 0.0;
		}
		do select_actions ;
			
	}
	
	action update_q(int id, map<string,int> new_state) {
		float reward <- last_turn[id] ? num_adopters[id]/number_farmers : 0.0;
		float b <- q[current_state][last_action] ;
		q[current_state][last_action] <- (1 - learning_rate) * q[current_state][last_action] + learning_rate * (reward + discount_factor * max(q[new_state].values));
	}
	
	map<string,int> define_state(int id) {
		map<string,int> state_ <- [];
		state_["budget"] <- int(budget[id]/5);
		state_["num_adopters"] <- int(num_adopters[id]/ 10);
		state_["time"] <- time_s[id];
		return state_;
	}
	action select_actions(int id) {
		if (reinforcement_learning) {
			map<string,int> new_state <- define_state(id);
			map<string,float> actions <- q[new_state];
			if (actions = nil) or empty(actions) {
				actions <- [];
				loop p over: possible_actions {
					
					actions[p] <- 1.0;
				}
			}
			q[new_state] <- actions;
			
			if last_action != ""{
				do update_q(id,new_state);
			}
			
			//last_action <- shuffle(actions.keys) with_max_of (actions[each]);
		
			last_action <- (actions.keys)[rnd_choice(actions.values)];
			
			current_state <- new_state;
			if last_action != ""  {
				if display_action_chosen {
					if length(action_chosen) < (id + 1) {
						action_chosen <<"";
					}
					action_chosen[id] <- action_chosen[id] + " %% " + last_action;
				}
				list<string> acts_prim <- last_action split_with "|";
				loop act over: acts_prim {
					switch act {
						match FINANCIAL_SUPPORT {
							do financial_support(id,0.2);
						}
						match TRAINING {
							do training(id,0.2, 20);
						}
						match ENV_SENSIBILISATION {
							do environmental_sensibilisation(id,0.2, 100);
						}
					}
				}
			}
		} else {
			do financial_support(id,0.2);
			do training(id,0.2, 10);
			do environmental_sensibilisation(id,0.2, 100);
			ask farmer {
				do compute_intention;
			}
		}
		
	}
	action add_money(int id) {
		budget[id] <- budget[int(world)] + new_budget_year;
	}
	
	action financial_support (int id, float level) {
		support[id][FINANCIAL] <- level;
	}
	
	action training (int id, float level, int number) {
		if (budget[id] > (number * level)) {
			ask number among farmer  {
				technical_skill <- technical_skill + level;
				opinion_on_topics[FARM_MANAGEMENT] <- opinion_on_topics[FARM_MANAGEMENT] + level; 
			}
			budget[id] <- budget[id] - (number * level);
		}
		
	}
	
	action environmental_sensibilisation (int id,float level, int number) {
		if (budget[id] > (number * level / 10.0)) {
			ask number among farmer  {
				opinion_on_topics[ENVIRONMENTAL] <- opinion_on_topics[ENVIRONMENTAL] + level; 
			}
			budget[id] <- budget[id] - (number * level) / 10.0;
		}
	}
	
	action give_financial_support(int id) {
		budget[id] <- budget[id] - support[id][FINANCIAL];
	}
	
	reflex receive_budget when: every(#year) {
		do add_money(0);
	}
	reflex choose_policy when: every(6 #month){
		
		do select_policy(0);
	}
} 

experiment learn_policy type: gui autorun: true{
	action _init_ {
		create simulation with: (reinforcement_learning:true);
	}
}

experiment learn_policy_batch type: batch until: cycle > 0{
	action _init_ {
		create simulation with: (reinforcement_learning:true, nb_replications:20, save_result_in_file:true);
	}
}


experiment one_simulation type: gui {
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
