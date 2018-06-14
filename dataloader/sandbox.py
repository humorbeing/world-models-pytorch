class a(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __call__(self, me):
        g = me * self.x + self.y
        return g


aa = a(3, 3)
print(aa(5))