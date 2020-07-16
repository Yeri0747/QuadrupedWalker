import pybullet as p
import pybullet_data as pd
import time
import numpy as np
import math
from random import shuffle, random, sample, randint, randrange, uniform, choice
from copy import deepcopy
	
    
#Cromosomas optimos:

#chromosome = [39.41703576233556, 0.8989694917227745, 0.749419610293473, 0.2679782111215405, 0.23065743756081544, 0.48050881036384585, -0.29821284549477056, 0.11885371024919522, -0.26558370234246254]
#chromosome = [26.59971258599214, 0.8989694917227745, 0.749419610293473, 0.2679782111215405, 0.23065743756081544, 0.48050881036384585, -0.29821284549477056, 0.11885371024919522, -0.26558370234246254]
#chromosome = [0.4832155975338474, 0.9310372934578574, 0.1867101438396955, 0.5028930107751859, 0.4275985234159625, 0.6884386078722258, 0.13119643954830962, 0.44417521156731077, 1.595032686960912]
#chromosome =[0.2854544486993065, 0.29855770197724085, 0.13331365711138532, 0.17117016823244877, 0.3493912322839895, 0.13069744579647835, 0.4070149047638687, 0.2209553318630051, 1.3913343740305235]
#chromosome =[0.39734498297289855, 0.1966451248849927, 0.31097380564160576, 0.11922817209590084, 0.2677015435014718, -0.5737406725978437, -0.3528223590153008, -0.30138117539588516, 1.3052308104748103]
#chromosome =[0.44105732719256974, 0.6361992782472478, 0.03368516187296319, 0.5041603118515686, 0.5999105203833784, 0.5063423389408941, -0.34815672156454036, 0.2933282926931722, 1.9576956938954737]
#chromosome=[0.20531928762221935, 0.9622239493000148, 0.7048469893098965, 0.9268951359932678, 0.18752262891682925, 0.2611445708061151, 0.9919111384288433, 0.7303574245036215, 1.3018090061840648]
chromosome=[0.38090872947934046, 0.7591541827172383, 0.29131566280885723, 0.1, 0.5243374805672234, 0.8076584354644847, 0.2345219961502294, 0.3185293810419183, 1]



p.connect(p.GUI)
p.setGravity(0,0,-9.8)
p.setAdditionalSearchPath(pd.getDataPath())
floor = p.loadURDF("plane.urdf")
startPos = [0,0,0.4]
robot = p.loadURDF("mini_cheetah/mini_cheetah.urdf",startPos)
numJoints = p.getNumJoints(robot)
print(numJoints)
#positions, orientation = pybullet.getBasePositionAndOrientation(robot)
p.changeVisualShape(robot,-1,rgbaColor=[1,1,1,1])
sup=[1,5,9,13]
inf=[2,6,10,14]

last_base_position = [0,0,0.4]
startPos = [0,0,0.4]
#dt = 1./30.
dt = 1/60
p.setTimeStep(dt)
#sum_reward = 0
force=100
sum_reward = 0
sum_speed = 0
rango = 700
t_aceleriación=400
for step_counter in range(rango):     
    
    t = step_counter * chromosome[0]
        #t=step_counter*math.pi/20
    if step_counter<t_aceleriación:
        f=chromosome[8]*step_counter/t_aceleriación
    else:
        f=chromosome[8]

    a1 = f*chromosome[1]*math.sin(t)
    a2 = f*chromosome[2]*math.sin(t + chromosome[5]*math.pi)
    a3 = f*chromosome[3]*math.sin(t + chromosome[6]*math.pi)
    a4 = f*chromosome[4]*math.sin(t + chromosome[7]*math.pi)

    for j in range (numJoints):
        if(j==sup[0])or(j==sup[1]):
            pos=a1
        elif (j==sup[2])or(j==sup[3]):
            pos=a2
        elif (j==inf[0])or(j==inf[1]):
            pos=a3
        elif (j==inf[2])or(j==inf[3]):
            pos=a4
        else:
            pos=0
        p.setJointMotorControl2(robot,j,p.POSITION_CONTROL,pos,force=force)

    current_base_position, orientation = p.getBasePositionAndOrientation(robot)
    forward_reward = abs(current_base_position[0] - last_base_position[0])
    drift_reward = -abs(current_base_position[1] - last_base_position[1])
    shake_reward = -abs(current_base_position[2] - last_base_position[2])
    last_base_position = current_base_position
        #if (current_base_position[2]>0.2)and(abs(current_base_position[0])>0.1):
    if (current_base_position[2]>0.2):
        reward = (20 * forward_reward + 5 * drift_reward + 5 * shake_reward)
    else:
        reward = -0.5
    sum_reward += reward
    sum_speed = abs(current_base_position[0])/(rango*0.002)     
    p.stepSimulation()
    time.sleep(dt)

fitness = np.zeros(2) # objetivos
fitness[0] = sum_reward
fitness[1] = sum_speed 

print('Cromosoma: {}'.format(chromosome))
print('Fitness: {}, Velocidad: {}'.format(fitness[0], fitness[1]))