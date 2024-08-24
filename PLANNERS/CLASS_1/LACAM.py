"""
LACAM star without pibt "swap" version, IJCAI 2023
initially from  Keisuke Okumura
"""

import heapq
import time
from collections import deque
import queue
import numpy as np


class Node:
    """
    Joint node of search process
    """

    def __init__(self, Config, parent, g, h, dis_set):
        # 因为每个node的config都是定的 就按照距离goal的目标来定顺序  到了goal的最后，没到的在前
        self.tree = queue.Queue()
        self.tree.put(Connode(float('inf'), None, None))
        self.config = Config  # a set contains all the agents places
        self.parent = parent
        self.neigh = set()
        self.g = g
        self.f = self.g + h
        self.order = []
        self.rng = np.random.default_rng(0)  # tier-breaking policy
        for i in range(len(Config)):
            dist = dis_set[i][Config[i][0]][Config[i][1]]
            self.order.append({"agent": i, "config": Config[i], "dist": dist})
        self.order.sort(key=lambda x: -x['dist'])

    def updateG(self, g, h):
        self.g = g
        self.f = g + h

    def __lt__(self, other):
        # 定义对象之间的比较规则
        return self.f < other.f

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.config == other.config
        return False

    def __hash__(self) -> int:
        return self.config.__hash__()


class Connode:
    """
    Constrain node for constrain tree
    actually, its more like a action tree
    """

    def __init__(self, agent, config, parent):
        # the agent is set to be a specific value and it is not easy to
        self.agent = agent  # the agent serial number :agent的序号
        self.config = config  # agent 不准去的位置
        self.parent = parent  # a new set to keep children
        if parent is None:
            self.d = 0
        else:
            self.d = parent.d + 1


class LaCAM:
    def __init__(self, graph, start, goal, runtime):
        """
        This is a anytime algorithm, a runtime limit is required, otherwise you will have to wait to the end of 2*n
        that is what you are not gonna want.
        """
        self._graph = graph
        self._start = start
        self._goal = goal
        self._dis_set = self.generate_distable()
        self.rng = np.random.default_rng(0)
        self._Open = deque([])
        self._Explored = {}
        self._runtime = runtime
        self._ngoal = None
        self._ninit = Node(start, None, 0, self.h(start), self._dis_set)
        self._Open.appendleft(self._ninit)
        self._Explored[self._ninit.config] = self._ninit

    def generate_distable(self):
        """
        Bfs based
        set up the h value table and policy table for each agent to refer
        Useful for joint node h value and policy
        """
        # use BFS as the heuristic function value also the optimal policy for M*
        dis_Table = []
        policy_Table = []
        for i in range(len(self._start)):
            tmp = deque()
            tmp.append(self._goal[i])
            inf = float('inf')
            dist_table = [[inf for _ in range(self._graph.width)] for _ in range(self._graph.height)]  # initialize distance
            #policy_table = [[None for _ in range(self._graph.width)] for _ in range(self._graph.height)]
            dist_table[self._goal[i][0]][self._goal[i][1]] = 0  # set initial point distance to 0
            #policy_table[self._goal[i][0]][self._goal[i][1]] = self._goal[i]  # set end point policy as itself
            while len(tmp) > 0:
                curr = tmp.pop()
                for neigh in self._graph.neighbors[curr]:
                    if dist_table[neigh[0]][neigh[1]] > dist_table[curr[0]][curr[1]] + 1:  # update neighbour's distance
                        dist_table[neigh[0]][neigh[1]] = dist_table[curr[0]][curr[1]] + 1
                        #policy_table[neigh[0]][neigh[1]] = curr  # map the policy of neigh to the optimal path for M*
                        tmp.append(neigh)

            dis_Table.append(dist_table)
            #policy_Table.append(policy_table)

        return dis_Table

    def h(self, config):
        cost = 0
        for index, coord in enumerate(config):
            cost += self._dis_set[index][coord[0]][coord[1]]
        return cost

    def low_level_expansion(self, node, con):
        if con.d < len(self._start):  # double check 了
            dict = node.order[con.d - 1]  # for example: constrain in 1, but actually its position in dict is 0
            v = dict['config']
            all = self._graph.neighbors[v] + [v]
            self.rng.shuffle(all)
            for u in all:
                Cnew = Connode(dict['agent'], u, con)
                node.tree.put(Cnew)

    def configuration_generator(self, node, con):  # PIBT-LaCAM generator
        # use number to represents the agents which is sequenced by
        curr_con = con.parent
        constrain = [con]
        # occupied = [[None for _ in range(graph.width)] for _ in range(graph.height)]
        while curr_con is not None:
            constrain.append(curr_con)  # add config for each agent
            curr_con = curr_con.parent
        Qto = [None] * len(node.config)
        con_agent = set()  # 记录有哪些已经设置好了位置
        for con in constrain:  # 这里不应该规避安排的  应该全部放进去
            if con.d == 0:
                continue
            if con.config in Qto:  # 假如有vertex conflict
                # print("vertex conflict")
                return None
            # elif con.config in node.config and (Qto[node.config.index(con.config)] is None):   # 这个修改不确定  但是很明显不能swap
            #    continue
            elif con.config in node.config:  # 检测swap conflict
                agent1 = node.config.index(con.config)
                agent2 = con.agent
                if Qto[agent1] == node.config[agent2]:
                    # print("swap conflict")
                    return None
            Qto[con.agent] = con.config
            con_agent.add(con.agent)

        # node.Config is Qfrom
        for agent, config in enumerate(node.config):
            # agent = node.config.index(config)
            # if agent in con_agent:
            #    continue
            if Qto[agent] is None:
                self.PIBT(agent, None, constrain, Qto, node.config)
        if None in Qto:
            return None
        # check switch collisiion
        for i in range(len(Qto)):
            for j in range(len(node.config)):
                if i == j:
                    continue
                # check switch collisiion
                if (Qto[i] == node.config[j]) and (Qto[j] == node.config[i]):
                    "oh no swap"
                    return None
                # check vertex collision
                if Qto[i] == Qto[j]:
                    # print("oh no vertex")
                    return None
        # every thing great
        return tuple(Qto)

    def PIBT(self, agent, agent_to, constrain, Qto, Qfrom):
        C = [Qfrom[agent]] + self._graph.neighbors[Qfrom[agent]]
        C.sort(key=lambda coord: self._dis_set[agent][coord[0]][coord[1]])
        for v in C:
            conti = False
            for i in range(len(Qto)):
                if i == agent:
                    continue
                if Qto[i] == v:
                    conti = True
                    break
                if agent_to is not None and Qfrom[agent_to] == v:
                    conti = True
                    break
            if conti:
                continue
            Qto[agent] = v
            for j in range(len(Qto)):
                if j == agent:
                    continue
                if (Qfrom[j] == v) and (Qto[j] is None):
                    if not self.PIBT(j, agent, constrain, Qto, Qfrom):
                        conti = True
                        break
                # if (Qfrom[j] == v) and (Qto[j] == Qfrom[agent]):
                #    conti = True
                #    break
            if conti:
                continue
            return True
        Qto[agent] = Qfrom[agent]
        return False

    def cost(self, Qfrom, Qto):  # take tw0 nodes
        cost = 0
        for i in range(len(Qfrom)):
            if not (self._goal[i] == Qfrom[i] == Qto[i]):
                cost += 1
        return cost

    def backtrack(self, Ngoal, path):
        path.append(Ngoal)
        N_curr = Ngoal
        # print(Ngoal.g)
        while (N_curr.parent is not None) and N_curr is not self._ninit:
            path.append(N_curr.parent)
            # print(N_curr.parent.g)
            N_curr = N_curr.parent
        return path.reverse()

    def solve(self):
        start_time = time.time()
        while len(self._Open) > 0:
            node = self._Open[0]

            cut = time.time()
            if (cut - start_time) >= self._runtime:
                break

            if self._ngoal is None and node.config == self._goal:
                self._ngoal = node

            if self._ngoal is not None and node.f >= self._ngoal.g:
                self._Open.popleft()
                continue

            if node.tree.qsize() == 0:
                self._Open.popleft()
                continue
            con = node.tree.get()

            if con.d < len(self._start):
                self.low_level_expansion(node, con)

            Qnew = self.configuration_generator(node, con)

            if Qnew is None:
                continue
            elif self._Explored.get(Qnew) is not None:
                n_known = self._Explored.get(Qnew)
                node.neigh.add(n_known)
                self._Open.appendleft(n_known)
                D = []
                heapq.heappush(D, (node.g, node))

                while len(D) > 0:
                    Nfrom_g, Nfrom = heapq.heappop(D)
                    for Nto in Nfrom.neigh:
                        g = Nfrom_g + self.cost(Nfrom.config, Nto.config)
                        if g < Nto.g:
                            Nto.g = g
                            Nto.f = Nto.g + self.h(Nto.config)
                            Nto.parent = Nfrom
                            heapq.heappush(D, (g, Nto))
                            if self._ngoal is not None and Nto.f < self._ngoal.g:
                                self._Open.appendleft(Nto)
            else:
                Nnew = Node(Qnew, node, node.g + self.cost(node.config, Qnew), self.h(Qnew), self._dis_set)
                self._Open.appendleft(Nnew)
                self._Explored[Qnew] = Nnew
                node.neigh.add(Nnew)

        if (self._ngoal is not None) and len(self._Open) == 0:
            end_t = time.time()
            path = []
            self.backtrack(self._ngoal, path)
            path_visual = []
            for i in range(len(path)):
                path_visual.append(path[i].config)
                set1 = set(path[i].config)
                if len(set1) < len(path[i].config):
                    print(f"Collision at {i}th config")
            print(path_visual)
            print(self._ngoal.g)
            print(f"Runtime: {end_t - start_time}s")
        elif self._ngoal is not None:
            end_t = time.time()
            path = []
            self.backtrack(self._ngoal, path)
            path_visual = []
            for i in range(len(path)):
                path_visual.append(path[i].config)
                set1 = set(path[i].config)
                if len(set1) < len(path[i].config):
                    print(f"Collision at {i}th config")
            print(path_visual)
            print(self._ngoal.g)
            print(f"Runtime: {end_t - start_time}s")
        elif len(self._Open) == 0:
            print("No solution")
        else:
            print("You are such a failure")
