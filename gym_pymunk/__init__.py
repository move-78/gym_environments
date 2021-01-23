from gym.envs.registration import register

register(id="BallReacher-v0",
         entry_point="gym_pymunk.envs.ball_reacher_env:BallReacherEnv")
