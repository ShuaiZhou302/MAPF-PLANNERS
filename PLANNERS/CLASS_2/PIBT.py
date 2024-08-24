import heapq
import math
from collections import deque
import random
import numpy as np


class Pibt:
    def __init__(self, graph, start_points, end_points, dis_table):
        self.graph = graph
        self.v_init = start_points
        self.v_f = end_points
        self.rng = np.random.default_rng(0)  # tier-breaking policy
        if dis_table is None:
            self.dis_table = self.generate_distable()
        else:
            self.dis_table = dis_table
        # initialize joint policy
        self.joint_policy = [self.v_init]
        #initialize priority
        self.agents = []
        gap = 1 / (len(start_points) + 1)  # gap to make initial distinct priority
        for i in range(len(start_points)):
            self.agents.append(agent(i, 1 - i * gap))  # initialize policy
        # maintain a priority queue

    def generate_distable(self):
        """
        Bfs based
        set up the h value table and policy table for each agent to refer
        Useful for joint node h value and policy
        """
        # use BFS as the heuristic function value also the optimal policy for M*
        dis_Table = []
        policy_Table = []
        for i in range(len(self.v_init)):
            tmp = deque()
            tmp.append(self.v_f[i])
            inf = float('inf')
            dist_table = [[inf for _ in range(self.graph.width)] for _ in
                          range(self.graph.height)]  # initialize distance
            policy_table = [[None for _ in range(self.graph.width)] for _ in range(self.graph.height)]
            dist_table[self.v_f[i][0]][self.v_f[i][1]] = 0  # set initial point distance to 0
            policy_table[self.v_f[i][0]][self.v_f[i][1]] = self.v_f[i]  # set end point policy as itself
            while len(tmp) > 0:
                curr = tmp.pop()
                for neigh in self.graph.neighbors[curr]:
                    if dist_table[neigh[0]][neigh[1]] > dist_table[curr[0]][curr[1]] + 1:  # update neighbour's distance
                        dist_table[neigh[0]][neigh[1]] = dist_table[curr[0]][curr[1]] + 1
                        policy_table[neigh[0]][neigh[1]] = curr  # map the policy of neigh to the optimal path for M*
                        tmp.append(neigh)

            dis_Table.append(dist_table)
            policy_Table.append(policy_table)

        return dis_Table

    def PIBT(self,agent1, agent2, Sfrom, Sto):  # maybe the t is the value needs to be passed in
        # sort neighbour and it self by distance to neighbour.
        C = [Sfrom[agent1.id]] + self.graph.neighbors[Sfrom[agent1.id]]
        self.rng.shuffle(C)
        C.sort(key=lambda coord: self.dis_table[agent1.id][coord[0]][coord[1]])  #
        conti = False
        for v in C:

            if self.checkoccupied(v,Sto):
                continue
            if agent2 is not None and Sfrom[agent2.id] == v:
                continue  # break
            # check first two if
            Sto[agent1.id] = v
            a2 = self.maypush(v, Sfrom, Sto)
            if a2 is not None:
                if not self.PIBT(a2, agent1, Sfrom, Sto):
                    continue
            return True
        Sto[agent1.id] = Sfrom[agent1.id]
        return False

    def solve(self):
        while True:
            Sfrom = self.joint_policy[-1]
            Sto = [None]*len(Sfrom)

            if Sfrom == self.v_f:
                #print("done!")
                self.calculateCost()
                return self.joint_policy
            # update priority
            for a in self.agents:
                if Sfrom[a.id] == self.v_f[a.id]:
                    a.priority = a.init_pri
                else:
                    a.priority = a.priority + 1
            self.agents = sorted(self.agents, key=lambda agent: agent.priority, reverse=True)
            for ag in self.agents:
                if Sto[ag.id] is None:
                    self.PIBT(ag, None, Sfrom, Sto)

            if Sto in self.joint_policy:
                # Fail case
                return None
            self.joint_policy.append(tuple(Sto))

    def checkoccupied(self, v, Sto):
        for agent in self.agents:
            if Sto[agent.id] == v:
                return True
        return False

    def maypush(self, v, Sfrom, Sto):
        for agent in self.agents:
            if Sfrom[agent.id] == v and Sto[agent.id] is None:
                return agent
        return None

    def calculateCost(self):
        Sfrom = self.joint_policy[0]
        cost = 0
        for i in range(1,len(self.joint_policy)):
          Sto = self.joint_policy[i]
          for index, v in enumerate(Sto):
              if v == Sfrom[index] == self.v_f[index]:
                  continue
              cost += 1
          Sfrom = Sto
        print(f"the cost is {cost}")
        return cost
class agent:
    def __init__(self, id, priority):
        self.id = id
        self.priority = priority
        self.init_pri = priority  # when reach goal come back to this

    def set_priority(self, pri):
        self.priority = pri

    def __lt__(self, other):
        return self.priority < other.priority


"""
IDK WHAT WE GOT HERE
def check_collision(agents):
    # 遍历第一个代理的路径
    for i in range(len(agents[0].path)):
        # 获取当前位置的坐标
        current_coord = agents[0].path[i]

        # 遍历其他代理，检查是否有冲突
        for j in range(1, len(agents)):
            # 获取其他代理在相同位置的坐标
            other_coord = agents[j].path[i]

            # 检查是否有冲突
            if current_coord == other_coord:
                print(f"Collision detected at index {i} between agents {0} and {j}")
                # 如果只需要检测第一个冲突，可以直接返回
                return True

    # 如果没有冲突，返回False
    return False
collision_detected = check_collision(agents)
if not collision_detected:
    print("No collision detected")
"""
