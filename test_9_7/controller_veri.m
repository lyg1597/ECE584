% /* An example of verifying a continuous nonlinear NNCS */
% / FFNN controller
[weights, bias] = parameter();
n = 2;
Layers = [];
for i=1:n - 1
    L = LayerS(weights{1, i}, bias{i, 1}, 'poslin');
    Layers = [Layers L];
end
L = LayerS(weights{1, n}, bias{n, 1}, 'purelin');
Layers = [Layers L];
L = LayerS([1 0;0 1],[0;pi/3],'poslin');
Layers = [Layers L];
L = LayerS([-1 0;0 -1],[100;2*pi/3],'poslin');
Layers = [Layers L];
L = LayerS([-1 0; 0 -1],[100;pi/3],'purelin');
Layers = [Layers L];

controller = FFNNS(Layers); 
% /* car model
Tr = 0.01; % reachability time step for the plant
Tc = 0.01; % control period of the plant
% output matrix
C = [-1 0 0 1 0 0 0 0;0 -1 0 0 1 0 0 0; 0 0 0 0 0 0 1 0; 0 0 0 0 0 0 0 1]; % output matrix
car = NonLinearODE(8, 2, @car_dynamics_modify, Tr, Tc, C);
% /* system
ncs = NonlinearNNCS(controller, car); 

% /* ranges of initial set of states of the plant
lb = [-0.1; -0.6; pi/2; 0; 0; pi/2;1;0];
ub = [0.1; -0.4; pi/2; 0; 0; pi/2;1;0];

% /* reachability parameters
reachPRM.init_set = Star(lb, ub);
reachPRM.ref_input = [];
reachPRM.numSteps = 2;
reachPRM.reachMethod = 'approx-star';
reachPRM.numCores = 4;
% /* usafe region: x1 - x4 <= 1.4 * v_ego + 10
unsafe_mat = [1 0 0 0 0 0 0 0];
unsafe_vec = -10;
U = HalfSpace(unsafe_mat, unsafe_vec);
%U = HalfSpace([-1 0 0 0 0 0], -20);
% /* verify the system
[safe, counterExamples, verifyTime] = ncs.verify(reachPRM, U);

%% plot 2d output sets
figure; 
map_mat = [1 0 0 0 0 0 0 0; 0 1 0 0 0 0 0 0];
map_vec = [];
ncs.plotOutputReachSets('blue', map_mat, map_vec);
title('Actual Distance versus. Ego car speed');
