import time
import pymunk

start = time.time_ns()

for i in range(100000):
    b = pymunk.Body(1, 10)
    b.position = 1.0, 2.0
    b.angle = 3.0
    t = b.position.x + b.position.y + b.angle

end = time.time_ns()
print((end - start) / 1_000_000_000)
