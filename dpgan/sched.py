
D_STEPS_RATE_OPTIONS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

class ConstDStepScheduler():
    def __init__(self, d_steps_rate, beta):
        self.d_steps_rate = d_steps_rate
        self.beta = beta

        self.ema = None
        self.d_steps_taken = 0

    def g_step(self, d_acc):
        if self.ema is None:
            self.ema = d_acc
        else:
            self.ema = self.ema*self.beta + (1-self.beta)*d_acc
        self.d_steps_taken = 0

    def d_step(self):
        self.d_steps_taken+=1

    def get_d_steps_rate(self):
        return self.d_steps_rate

    def is_g_step_time(self):
        return self.d_steps_taken == self.get_d_steps_rate()



class DStepScheduler():

    def __init__(self, d_steps_rate_init, grace, thresh, beta):
        assert d_steps_rate_init in D_STEPS_RATE_OPTIONS
        self.d_steps_rate_index = D_STEPS_RATE_OPTIONS.index(d_steps_rate_init)

        self.thresh = thresh
        self.beta = beta
        self.grace = grace

        self.ema = None
        self.g_steps_taken = 0
        self.d_steps_taken = 0

    def g_step(self, d_acc):
        if self.ema is None:
            self.ema = d_acc
        else:
            self.ema = self.ema*self.beta + (1-self.beta)*d_acc

        self.g_steps_taken+=1
        self.d_steps_taken=0

        if self.g_steps_taken >= self.grace and self.ema <= self.thresh:
            self.d_steps_rate_index = min(len(D_STEPS_RATE_OPTIONS)-1, self.d_steps_rate_index+1)
            self.g_steps_taken = 0

    def d_step(self):
        self.d_steps_taken+=1

    def is_g_step_time(self):
        return self.d_steps_taken == self.get_d_steps_rate()

    def get_d_steps_rate(self):
        return D_STEPS_RATE_OPTIONS[self.d_steps_rate_index]
