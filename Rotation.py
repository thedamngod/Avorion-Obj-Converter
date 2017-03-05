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
vec2 = np.matrix([1, 0, 0])

temp_result = (R_z_clockwise_90*R_y_clockwise_90*vec).tolist()
res = [np.round(temp_result[0][0], 10), np.round(temp_result[1][0], 10), np.round(temp_result[2][0], 10)]
print(res)
# print(R_z_clockwise_90.dot(vec))
