import numpy as np


class ACIQ(object):

    def __init__(self, num_bits):
        super(ACIQ, self).__init__()
        self.num_bits = num_bits

    def get_alpha_gaus(self, min, max, size):
        alpha_gaus = [None, None, 1.710635, 2.151593, 2.559136, 2.936201, 3.286914, 3.615114,
                      3.924035, 4.216331, 4.494167, 4.759313, 5.013188, 5.257151, 5.491852, 5.719160,
                      5.938345, 6.150141, 6.356593, 6.560495, 6.752936, 6.931921, 7.106395, 7.350340,
                      7.482915, 7.691728, 7.668494, 7.583591, 7.583591, 8.326501, 8.171210, 8.171210]
        gaussian_const = (0.5 * 0.35) * (1 + (np.pi * np.log(4)) ** 0.5)
        sigma = ((max - min) * gaussian_const) / ((2 * np.log(size)) ** 0.5)

        alpha_opt = alpha_gaus[31] if self.num_bits > 31 else alpha_gaus[self.num_bits]
        return alpha_opt * sigma

    def get_alpha_gaus_direct(self, sigma):
        alpha_gaus = [None, None, 1.710635, 2.151593, 2.559136, 2.936201, 3.286914, 3.615114,
                      3.924035, 4.216331, 4.494167, 4.759313, 5.013188, 5.257151, 5.491852, 5.719160,
                      5.938345, 6.150141, 6.356593, 6.560495, 6.752936, 6.931921, 7.106395, 7.350340,
                      7.482915, 7.691728, 7.668494, 7.583591, 7.583591, 8.326501, 8.171210, 8.171210]
        alpha_opt = alpha_gaus[31] if self.num_bits > 31 else alpha_gaus[self.num_bits]
        return alpha_opt * sigma