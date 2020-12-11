# Assessing the influence of different incentive structures on agent behaviour using reinforcement learning

> * Group Name: Team Potato
> * Group participants names: Anya Heider, Anna Huwyler, Bryan Yu ( By last names)
> * Project Title: Assessing the influence of different incentive structures onagent behaviour using reinforcement learning

## General Introduction

A better understanding of decision making and the reaction towards differentincentive structures could help to make more effective decision making. Computa-tional social science and reinforcement learning as one method in this scientific fieldyield the potential to model learning and the adaption of agents’ behaviour basedon maximising the reward of single actors.As a group, we hypothesized that if the story’s outcome resulted from incentivedriven decision making, then similar behavior could be simulated using computa-tional methods – specifically reinforcement learning and the reward optimizing agentswithin. Furthermore, if the results obtained were driven from changes in incentive,then would it be reasonable to expect to be able to observe similar behavior in re-ward optimizing agents

We begin with a motivating story from the annuals of business history ofincentive driven behavior in a large American corporation[3]. In the late 20thcen-tury, Federal Express(FedEx), an U.S. business offered the general public overnightparcel delivery. As the story goes - every night FedEx would fly their planes fromeach major city into a centralized location (i.e. planes from New York, Los Angeles,Seattle, Miami, etc... would all meet in Chicago). Once all the planes congregated,employees would then re-organize packages from the source onto the appropriateplanes according to the parcel’s final destination. Thus, it is obvious the integrity ofFedEx business depended on completing the delivery within the tight overnight win-dow. However, as with most business stories, FedEx experienced diﬀiculties meetingtheir goals as moving packages at the hub repeatedly took longer than expected.One day, someone in FedEx’s management got the bright idea to pay the employeesinvolved in the overnight delivery system not by the hour, but by that night’s suc-cessful delivery. That is, once packages were completely loaded onto the destinationplanes, everyone can go home! The problem vanished - getting packages onto thedestination plane was no longer a bottleneck


## The Model


(Define dependent and independent variables you want to study. Say how you want to measure them.) (Why is your model a good abtraction of the problem you want to study?) (Are you capturing all the relevant aspects of the problem?)


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
