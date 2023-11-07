Challenges


- [ ] First Try Machine learning model 
  - [ ] Dataset creation
    - [ ] Incremental (id * 100000 + event_id) 
    - [X] Drop the ID column
    - [X] Remove Move from in Categorical Columns - completed
    - [X] Change the text_change column
    - [X] Change the Events - idea to aggregate all the punctuations
    - [ ] Change the Socring system - divide the score by the number of events
    - [X] Drop the Event column - To make the columns independent
    - [ ] Make the score some incremental average or something (Should do something like this!)
  - [ ] Exploratory Data analysis 
  - [ ] Machine learning Model
    - [X] Make the grid Search pipeline
    - [ ] Make the random search pipeline
    - [ ] Add Optuna for optimization
    - [X] Create prediction function
    - [ ] Create for each model params and model instance
- [ ] Try a Deep Learning type model to learn  sequential data
