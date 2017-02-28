import numpy as np

c, s = np.cos(np.radians(-90)), np.sin(np.radians(-90))
R_x_clockwise_90 = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(1, 0, 0, 0, c, -s, 0, s, c))
R_y_clockwise_90 = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(c, 0, -s, 0, 1, 0, s, 0, c))
R_z_clockwise_90 = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(c, -s, 0, s, c, 0, 0, 0, 1))
c, s = np.cos(np.radians(90)), np.sin(np.radians(90))
R_x_anti_90 = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(1, 0, 0, 0, c, -s, 0, s, c))
R_y_anti_90 = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(c, 0, -s, 0, 1, 0, s, 0, c))
R_z_anti_90 = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(c, -s, 0, s, c, 0, 0, 0, 1))

vec = np.matrix([1, 0, 0]).transpose()

print(R_z_clockwise_90.dot(R_z_clockwise_90.dot(vec)))
print(R_y_clockwise_90.dot(vec))
# print(R_z_clockwise_90.dot(vec))
