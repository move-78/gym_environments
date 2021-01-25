import pymunk as pm
from pymunk.pygame_util import DrawOptions


def create_mocap_circle(space: pm.Space, position=(0, 0, 0), radius=50, color=DrawOptions.shape_static_color):
    mocap = pm.Circle(space.static_body, radius=radius)
    mocap.filter = pm.ShapeFilter(mask=0)
    mocap.color = color
    mocap.body.position = tuple(position)
    return mocap

