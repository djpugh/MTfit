import time


class TaskTest(object):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self):
        print('Test 1 {}'.format(self.a*self.b))
        return self.a*self.b


class TaskTest2(object):

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self):
        print('Test 2 {}'.format(self.a+self.b))
        time.sleep(self.b*0.5)
        return self.a+self.b
