https://github.com/Musteva

# Projectile Simulation with Sway Correction

## About This Project

So, this is my **first GitHub project**, and honestly, it's a pretty big one for me. I'm still warming up to the whole coding thing, but I wanted to challenge myself with something more than just basic scripts. This project simulates projectile motion in 3D, factoring in things like gravity, air resistance, and wind.

It also introduces a sway effect mid-flight, meaning the projectile gets thrown off course a little, and then the program tries to correct its velocity to hit the original target anyway. I thought this would be a fun and useful way to explore numerical simulations and physics-based programming.

## What It Does

- Generates random environment and projectile properties
- Simulates normal projectile motion without disturbances
- Introduces a random mid-flight disturbance (sway)
- Tries to correct the projectile's path so it still reaches the original target
- Visualizes both the unaltered and corrected trajectories in 3D
- Prints out a summary comparing the two flight paths

## Process

Since this was a big project for me, I picked up a lot along the way. Things like:

- How to structure a larger Python program
- How numerical simulations work (RK4 method, physics calculations, etc.)
- Handling randomness in simulations
- Creating 3D plots to visualize data

## Future Improvements

I know there's still a lot to improve. Some ideas I have:

- Make the sway correction more accurate
- Add more customization options for the projectile and environment
- Maybe even a simple UI for adjusting parameters
- Better handling of edge cases in the simulation

## Feedback

Like I said, I’m still pretty new to all of this, so if you have any suggestions or spot something that could be done better, feel free!


## A Question for the Senior Devs
I usually use a float assigned to executemain and then an if statement to run the main function instead of coding directly inside it. It just feels more organized to me that way. Is this actually a better approach, or could it cause issues down the line?
