## Establishes connection with hebi modules and creates hexapod and imu groups


import numpy as np
import hebi
from time import sleep


# import rospkg


def setup_modules():
    # Get Names of All the Modules
    bases = ['L1_J1_Base', 'L2_J1_Base', 'L3_J1_Base', 'L4_J1_Base', 'L5_J1_Base', 'L6_J1_Base']
    names = ['L1_J1_Base', 'L1_J2_Shoulder', 'L1_J3_Elbow', 'L2_J1_Base', 'L2_J2_Shoulder',
             'L2_J3_Elbow', 'L3_J1_Base', 'L3_J2_Shoulder', 'L3_J3_Elbow', 'L4_J1_Base',
             'L4_J2_Shoulder', 'L4_J3_Elbow', 'L5_J1_Base', 'L5_J2_Shoulder',
             'L5_J3_Elbow', 'L6_J1_Base', 'L6_J2_Shoulder', 'L6_J3_Elbow']

    HebiLookup = hebi.Lookup()
    imu = HebiLookup.get_group_from_names(['*'], bases)
    hexapod = HebiLookup.get_group_from_names('*', names)

    while imu == None or hexapod == None or imu.size != 6 or hexapod.size != 18:
        print('Waiting for modules')
        imu = HebiLookup.get_group_from_names('*', bases)
        hexapod = HebiLookup.get_group_from_names('*', names)

    print('Found {} modules in shoulder group, {} in robot.'.format(imu.size, hexapod.size))

    # Set the Gains (Multiple Times)

    gains_command = hebi.GroupCommand(hexapod.size)
    gains_command.read_gains('/home/marmot/Sood/legged_gym/legged_gym/scripts/robot_setup/setupFiles/gains18.xml')
    for i in range(3):
        hexapod.send_command(gains_command)
        sleep(0.1)
    hexapod.command_lifetime = 5;
    hexapod.feedback_frequency = 100;

    return imu, hexapod


if __name__ == "__main__":
    imu, hexapod = setup_modules()
