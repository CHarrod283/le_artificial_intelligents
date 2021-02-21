import reward as rw

params = {"all_wheels_on_track" : True,
          "x" : -7.14731106,
          "y" : 0.484933341,
          "progress" : 5,
          "speed" : 2.894987373062093,
          "track_width" : 0.6096,
          "steering_angle" : -1.6046855759720993,
          "steps": 10
          }

print(rw.reward_function(params))
