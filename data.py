import os
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset

from utils import loadmat

# MATLAB Constants (offset by 1)
BR_STATUS = 11 - 1
RATE_A = 6 - 1
BR_R = 3 - 1
BR_X = 4 - 1
PF = 14 - 1
QF = 15 - 1
PT = 16 - 1
QT = 17 - 1
BUS_TYPE = 2 - 1
VM = 8 - 1
VMIN = 13 - 1
GEN_STATUS = 8 - 1
PG = 2 - 1
QG = 3 - 2
PMAX = 9 - 1
QMAX = 4 - 1
QMIN = 5 - 1


class ACCFM(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        assert name in ['case39', 'case118']
        self.name = name

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name)

    @property
    def raw_file_names(self):
        if self.name == 'case39':
            profile = 'data/accfm/case39/raw/init_pf_case39_reduced_rate_accfm.mat'
            power_flow = 'data/accfm/case39/raw/results_pf_case39_nearest_1_10_6000_accfm.mat'
            cascade_simulation = 'data/accfm/case39/raw/results_case39_nearest_1_10_6000_accfm.mat'
        elif self.name == 'case118':
            profile = 'data/accfm/case118/raw/init_pf_case118_accfm.mat'
            power_flow = 'data/accfm/case118/raw/results_pf_case118_nearest_1_40_6000_accfm.mat'
            cascade_simulation = 'data/accfm/case118/raw/results_case118_nearest_1_40_6000_accfm.mat'
        else:
            return [None, None, None]

        return [profile, power_flow, cascade_simulation]

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []

        # Load initial profile
        profile = loadmat(self.raw_file_names[0])
        profile = profile['result']
        network = profile['pf']
        branch_labels = np.array(profile['networks_after_cascade']['branch'])[:, BR_STATUS]
        bus_type = np.array(profile['networks_after_cascade']['bus'])[:, BUS_TYPE]
        gen_labels = np.array(profile['networks_after_cascade']['gen'])[:, GEN_STATUS]

        data = self._get_single_network(network,
                                        branch_labels,
                                        bus_type)
        # data.y = F.one_hot(torch.tensor(y, dtype=torch.long), num_classes=2).float()

        initial_pf = {'bus': data.x,
                      'branch': data.edge_attr}

        data_list.append(data)

        # Load branch status
        data = loadmat(self.raw_file_names[2])
        data = data['result']
        data.keys()

        # verify uniqueness of scenarios
        unique_scen = [[x] if isinstance(x, int) else x for x in data['scenarios']]
        unique_scen = list(map(list, set(map(tuple, map(set, unique_scen)))))

        assert len(data['scenarios']) == len(unique_scen)

        networks_after_cascade = data['networks_after_cascade']
        events = data['scenarios']
        buses_tripped_in_scenario = np.array(data['tripped_buses_in_scenario'])
        gen_tripped_in_scenario = np.array(data['tripped_gens_in_scenario'])
        branch_after_cascade = []
        bus_after_cascade = []
        gen_after_cascade = []
        tripped_buses = []
        tripped_gens = []
        num_off = []
        scenario_indx = []

        num_branches = len(data['network']['branch'])

        for i, scenario in enumerate(networks_after_cascade):
            if isinstance(scenario, dict):
                num = np.sum(1 - np.array(scenario['branch'])[:, BR_STATUS], dtype=int)

                if isinstance(events[i], list) and num > len(events[i]):
                    num_off.append(num)
                    branch_after_cascade.append(np.array(scenario['branch'])[:, BR_STATUS])
                    bus_after_cascade.append(np.array(scenario['bus'])[:, BUS_TYPE])
                    gen_after_cascade.append(np.array(scenario['gen'])[:, GEN_STATUS])
                    tripped_buses.append(buses_tripped_in_scenario[i])
                    tripped_gens.append(gen_tripped_in_scenario[i])
                    scenario_indx.append(i)

        scenario_indx = scenario_indx[0:5000]  # select only 5000 simulations
        branch_after_cascade = branch_after_cascade[0:5000]
        bus_after_cascade = bus_after_cascade[0:5000]
        gen_after_cascade = gen_after_cascade[0:5000]
        tripped_buses = tripped_buses[0:5000]
        tripped_gens = tripped_gens[0:5000]
        num_off = num_off[0:5000]

        # Load features from powerflow solutions
        networks = loadmat(self.raw_file_names[1])
        networks = networks['result']['pf']
        # networks = loadmat(networks)

        for i, indx in enumerate(scenario_indx):
            network = networks[indx]

            data = self._get_single_network(network,
                                            branch_after_cascade[i],
                                            bus_after_cascade[i])

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def _get_single_network(self, network, branch_labels, bus_type):
        bus = np.array(network['bus'])
        gen = np.array(network['gen'])
        branch = np.array(network['branch'])

        # Bus features
        load_pq = bus[:, 2:4]  # real and reactive power demand at buses
        before_bus_type = bus[:, BUS_TYPE]  # bus type: 1:PQ, 2:PV, 3:REF
        voltage_ratio = (bus[:, VM] / bus[:, VMIN])
        bus = bus[:, 7:9]  # index 0 is bus_id; voltage magnitude and voltage angle
        bus = np.concatenate((before_bus_type.reshape((len(before_bus_type), -1)),
                              voltage_ratio.reshape((len(voltage_ratio), -1)),
                              bus), axis=1)

        # Generator features
        gen_idx = (gen[:, 0] - 1).astype(int)  # offset for matlab 1-based indexing
        pg_ratio = gen[:, PG] / (gen[:, PMAX] + 1e-5)
        qmin = gen[:, QMIN]
        qmax = gen[:, QMAX]
        qmin_ratio = -(gen[:, QG] - qmin) / (np.absolute(qmin) * 0.1 + 1e-5)
        qmax_ratio = (gen[:, QG] - qmax) / (np.absolute(qmax) * 0.1 + 1e-5)
        gen_pq = gen[:, [PG, QG]]  # index 0 is bus_id, > 8: all 0s; real and reactive power generation
        gen = np.concatenate((pg_ratio.reshape((len(pg_ratio), -1)),
                             qmin_ratio.reshape((len(qmin_ratio), -1)),
                             qmax_ratio.reshape((len(qmax_ratio), -1))), axis=1)

        net_pq = np.zeros((bus.shape[0], gen_pq.shape[1]))
        gen_dummy = np.zeros((bus.shape[0], gen.shape[1]))
        # dummy_labels = np.zeros(bus_labels.shape)
        for j, idx in enumerate(gen_idx):
            net_pq[idx] = gen_pq[j]
            gen_dummy[idx] = gen[j]
        apparent_pq = np.subtract(net_pq[:, 0:2], load_pq)  # compute net real and reactive power

        # combine bus and gen to get node features
        bus = np.concatenate((bus,
                              apparent_pq,
                              gen_dummy), axis=1)

        # Bus labels
        bus_type[bus_type == 4] = 0  # NONE = 4; NONE type bus is isolated and turned off
        bus_type[bus_type > 0] = 1
        # for j, idx in enumerate(gen_idx):
        #     bus_type[idx] = gen_status[j]  # Assign gen status after cascade to corresponding buses

        # Edges (directed)
        row = (branch[:, 0]).astype(int) - 1  # offset for matlab 1-based indexing
        col = (branch[:, 1]).astype(int) - 1

        # Branch features
        branch_status = branch[:, BR_STATUS]
        rate_a = branch[:, RATE_A] * 2.02
        branch_mask = branch_status > 0
        flow = branch[:, [PF, QF, PT, QT]]
        branch = branch[:, [BR_R, BR_X, RATE_A, PF, QF, PT, QT]]

        bus = torch.tensor(bus, dtype=torch.float32)
        flow = torch.tensor(flow, dtype=torch.float64)
        branch = torch.tensor(branch, dtype=torch.float64)
        branch_status = torch.tensor(branch_status, dtype=torch.float64)

        # Matpower powerflow solutions may result in NaNs for Qd in buses
        # and columns 13-16 in branch
        # Replace NaNs with 0
        bus = torch.nan_to_num(bus, nan=0.)
        flow = torch.nan_to_num(flow, nan=0.)

        # Compute meaningful features
        absolute_branch_flow = torch.add(
            torch.sqrt(torch.add(torch.square(flow[:, 0]), torch.square(flow[:, 1]))),
            torch.sqrt(torch.add(torch.square(flow[:, 2]), torch.square(flow[:, 3])))
        ).view(len(rate_a), -1)
        branch_flow_ratio = torch.divide(absolute_branch_flow,
                                         torch.tensor(rate_a, dtype=torch.float64).view(len(rate_a), -1))

        # Replace NaNs with 0
        bus = torch.nan_to_num(bus, nan=0.)
        branch = torch.nan_to_num(branch, nan=0.)

        # Add status as a feature
        # Add flow ratio as a feature
        # Convert to float32 from float64
        branch = torch.cat((branch_status.reshape(-1, 1).float(), branch_flow_ratio.float(), branch.float()), dim=1)

        edge_index = torch.tensor([row, col], dtype=torch.long)

        row = np.concatenate((row, col), axis=0)
        col = np.concatenate((np.arange(edge_index.shape[1]), np.arange(edge_index.shape[1])), axis=0)
        node_edge_index = torch.tensor([row, col], dtype=torch.long)

        # 0-1 Normalization
        bus = torch.div((bus - bus.min(0, keepdim=True)[0]),
                        (bus.max(0, keepdim=True)[0] - bus.min(0, keepdim=True)[0]))
        branch = torch.div((branch - branch.min(0, keepdim=True)[0]),
                           (branch.max(0, keepdim=True)[0] - branch.min(0, keepdim=True)[0]))

        # Branch labels
        # branch_labels = branch_labels[branch_mask]
        edge_y = F.one_hot(torch.tensor(branch_labels, dtype=torch.long), num_classes=2).float()
        y = F.one_hot(torch.tensor(bus_type, dtype=torch.long), num_classes=2).float()

        data = Data(x=bus, edge_index=edge_index, edge_attr=branch, y=y)
        data.edge_label = edge_y
        data.node_edge_index = node_edge_index

        return data

    def __repr__(self):
        return f'accfm-{self.name}()'


def load_dataset(name):
    path = os.path.join('data', 'accfm')
    data = ACCFM(path, name)

    return data
