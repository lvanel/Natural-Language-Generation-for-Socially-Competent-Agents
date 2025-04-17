<h1>Code Associated with Natural Language Generation for Socially Competent Agents</h1>
Our overarching objective is to develop a method for producing more appropriate responses in task-oriented discussions. Our technological challenge is how to dynamically consider the evolution of emotional and conversational states during a task-oriented dialogue for the next speaker-turn generation.

As this thesis exists in an industrial context, where both transparency and controllability are crucial aspects, we propose a 2-step architecture that includes an explicit planning step. This yields a sequence of socio-emotional labels that will condition the generation to output an accurate response. 

Our results show that multi-label planning provides better results, beyond providing control and visibility lacking in its end-to-end counterparts.

<h2>Experiment 1</h2>
Code associated with the experiment led on DailyDialog, an English Open-Domain dataset, including:
* Sequence of Labels Planning module
* Generation
* Rerank and Response Selection
* Human and Automatic Evaluation

<h2>Experiment 2 and 3</h2>
Code associated with the experiment led on DATA-SERGe, a French Task-Oriented dataset, including:
* Prompt-based Generation
* Human and Automatic Evaluation

<h2>Coming soon...</h2>
* Official public release of DATA-SERGe
