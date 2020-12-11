# Assessing the influence of different incentive structures on agent behaviour using reinforcement learning

> * Group Name: Team Potato
> * Group participants names: Anya Heider, Anna Huwyler, Bryan Yu ( By last names)
> * Project Title: Assessing the influence of different incentive structures onagent behaviour using reinforcement learning

## General Introduction

A better understanding of decision making and the reaction towards differentincentive structures could help to make more effective decision making. Computa-tional social science and reinforcement learning as one method in this scientific fieldyield the potential to model learning and the adaption of agents’ behaviour basedon maximising the reward of single actors.As a group, we hypothesized that if the story’s outcome resulted from incentivedriven decision making, then similar behavior could be simulated using computa-tional methods – specifically reinforcement learning and the reward optimizing agentswithin. Furthermore, if the results obtained were driven from changes in incentive,then would it be reasonable to expect to be able to observe similar behavior in re-ward optimizing agents

We begin with a motivating story from the annuals of business history ofincentive driven behavior in a large American corporation[3]. In the late 20thcen-tury, Federal Express(FedEx), an U.S. business offered the general public overnightparcel delivery. As the story goes - every night FedEx would fly their planes fromeach major city into a centralized location (i.e. planes from New York, Los Angeles,Seattle, Miami, etc... would all meet in Chicago). Once all the planes congregated,employees would then re-organize packages from the source onto the appropriateplanes according to the parcel’s final destination. Thus, it is obvious the integrity ofFedEx business depended on completing the delivery within the tight overnight win-dow. However, as with most business stories, FedEx experienced diﬀiculties meetingtheir goals as moving packages at the hub repeatedly took longer than expected.One day, someone in FedEx’s management got the bright idea to pay the employeesinvolved in the overnight delivery system not by the hour, but by that night’s suc-cessful delivery. That is, once packages were completely loaded onto the destinationplanes, everyone can go home! The problem vanished - getting packages onto thedestination plane was no longer a bottleneck


## The Model
It is beneficial to find an already existing environment with an easy access to the source code and possible features for the reinforcement learning agents. After some research on different environments and mini games, the team decided to implement the simulations in a already existing mini game called ”Collect Mineral Shards” inside the ”StarCraft II Learning Environment”[5]. In this mini game there are two agents in the environment and their goal is to navigate the map and collect the minerals.

This game has the advantages of an already existing library called pysc2 which allowed the experiment to use a standardized environment, allow for ready experi- ment replication, and allows the team to focus on the course concepts and not on programming and software technicalities.

The environment by de- fault spawns two marines and twenty mineral crystals. Each of the marines can be controlled by the agent. When the marines walk on top of a mineral crystal, as per Figure 3 (b), the mineral will disappear from the map and the mineral collection score will be incremented by 100. Each simulation is allowed 2 minutes which trans- lates into approximately 240 actions to be taken. On every step of the single-agent reinforcement learning controller, the agent can select one of two marines and one of four directions (up, down, left, right) to move. Upon termination of an episode determined by the 2 minute timer expiring, the environment is reset, the mineral collection counter set back to 0, and the marines and minerals are re-spawned on the map randomly.

Time Incentive introduces a penalty (negative reward) to agent for every action it takes. In theory, this reward design would incentivize the agents to gather as many minerals as possible with as few steps. However, we note that it may be possible to collect all the minerals on the screen without running out of time and as such, the agents can incur additional penalties from having nothing to do. The default value is chosen to be a negative penalty of -20 per step. The results for the parameter study of this parameter can be found in chapter 4.3.

Mineral Thresholding is thus introduced such that the time incentive rewards are terminated after a thresholding mineral has been collected. Afterwards, all min- erals collected contribute the full 100 value to the agent and additional steps incurs no additional costs. The default value is chosen to be 1000, if not stated otherwise. The results of parameter variation for this value can also be obtained from chapter 4.3. The group relates this design to the case study in that once the task at hand has been completed, the workers can go home. We note that we can improve on this design by terminating the episode and resetting the environment upon the collection of a threshold amount of mineral. We leave this to future experiment design.
Episodic Reward makes it such that the agent only receives reward indicators at the end of an simulation. That is, throughout the simulation the agent will get zero feedback. Upon completion of the simulation, the agent will receive one feed- back signal. Episodic rewards simulate the diﬀiculties of communicating intention with workers and serve as a baseline in which incremental rewards can be compared. When the episodic reward configuration is not applied, the environment will give the agent continuous feedback after every action taken.

Deep Q-Networks (DQN)[2] is one reinforcement learning algorithm popularized through its success in the Atari environment. DQN introduces two major changes to the traditional reinforcement learning architecture. Firstly, DQN applies a convolutional neural network to teach the agents to learn the control policies (as compared to tabular methods). Secondly, DQN introduces experience replay, a method which alleviates the problem of correlated data and non-stationary distribu- tions by randomly sampling from a pool of previous transitions.[2] An epsilon greedy strategy is applied which results in the agent not always taking the most probable actions but also occasionally some random actions.

Advantage Actor Critic (A2C) is a synchronous implementation of A3C[1]. A3C’s initial contribution was the idea of multiple workers in individual environments running in parallel but the updates would affect a shared model [1]. As a result, the risks of correlated data and non-stationary distributions could be alleviated with- out incurring the computational expenses of experience replay. Furthermore, A3C provided an additional contribution by demonstrating that similar levels of learning could be achieved without specialized hardware and in less time. Further research showed that the synchronous version achieved similar performance and as such, A2C is thus the synchronous version of A3C. That is, the update waits until every actor has completed its segment of experience before conducting the update step.

Proximal Policy Optimization (PPO) [4], a policy optimization method, takes advantage of the multiple workers in A2C and also introduces a trust region used to improve the actor. The main idea in PPO is that modifications should not cause too much differentiation from the existing policy. As such, a trust region is applied which constrains on the size of the updates.[4]. Furthermore, Schulman et al proposed the LCLIP.  While DQN has shown success in the discrete action space, it had challenges when faced with continuous control. PPO is expected to improve the performance in the continuous control space. Thus, we include PPO into our repertoire of algorithms to see how the triple advantages conferred from trust regions, multiple workers, and continuous control benefits may affect the agents’ learned behaviors and investigate the implications of restricting changes when attempting to optimize a process.

## Fundamental Questions
We hypothesize that the effectiveness of different incentive structures in causing theintended behavior change of agents can be modeled and evaluated by computationalmethods. In particular, we believe reinforcement learning, with it’s design of usingfeedback signals to teach an agent about an unknown environment, serves as an in-teresting framework with minimal assumptions in which to explore our hypothesis.


## Expected Results
Our initial interest in the multi-agent regime arises from the hypothesisthat it should be possible to capture emergent co-operative behavior driven solely byincentive structures. That is, in the single-controller reinforcement learning regime,one controller manipulates the two agents. As such, all rewards are attributed tothe single controller. We hypothesize that in the multi-agent regime, allocating oneself-serving controller per marine would lead to different behaviors as each agentwill be attributed a portion of the entire environmental rewards and as such, theirdecentralized decision making will be driven by their immediate, personal, rewards.Furthermore, we hypothesize that changes in reward structure would have prominenteffects. For example, if rewards were not shared between the agents(i.e. winner-takes-all), then it the agents were spawned near each other, we might fairly expectadversarial behavior such as where the agent learns behavior which actively pre-emptthe other agent from collecting minerals – by taking it first. As such, the pre-emptingagent will capture all the rewards of the episode.


## References 
* <a id="1">[1] </a> Volodymyr Mnih et al. “Asynchronous methods for deep reinforcement learning”. In:International conference on machine learning. 2016, pp. 1928–1937. 
* <a id="2">[2] </a> Volodymyr Mnih et al. “Playing atari with deep reinforcement learning”. In:arXivpreprint arXiv:1312.5602(2013).
* <a id="3">[3] </a> Charles T Munger.Poor Charlie’s Almanack: The Wit and Wisdom of Charles T.Munger. Donning Company, 2006.
* <a id="4">[4] </a> John Schulman et al. “Proximal policy optimization algorithms”. In:arXiv preprintarXiv:1707.06347(2017).
* <a id="5">[5] </a> Oriol Vinyals et al.StarCraft II: A New Challenge for Reinforcement Learning. 2017.
* <a id="6">[6] </a> Wikipedia.Q-Learning.url:https://en.wikipedia.org/wiki/Q-learning. (ac-cessed: 26.11.2020).
* <a id="7">[7] </a> Wikipedia.Reinforcement  Learning.url:https : / / en . wikipedia . org / wiki /Reinforcement_learning. (accessed: 24.11.2020).




## Research Methods
Reinforcement Learning, Incentive Structure, PySC2


## Other
N.A.
