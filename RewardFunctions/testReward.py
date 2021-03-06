import reward as rw

params = {"all_wheels_on_track" : True,
          "x" : -5,
          "y" : -2.4,
          "progress" : 5,
          "speed" : 2.5,
          "track_width" : 0.6096,
          "steering_angle" : 2.5,
          "steps": 10
          }


reward = rw.reward_function(params)
print("************************")
print("Total Reward: ")
print(reward)
