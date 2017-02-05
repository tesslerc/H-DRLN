# Distilled Multi Skill Agent Process
__This is in development stage__

For further explanation see the [website](http://chentessler.wixsite.com/hdrlnminecraft).

This code requires a lot of refactoring and cleanup.
You let your "teachers" play for N time steps and record the data (Input: state, Output: Q values).
Then with this code, load the data and train the new distilled "student".

Currently trains using the MSE loss function, need to swap it to KL-Divergence.
