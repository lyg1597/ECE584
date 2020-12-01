function [dx]=discrete_car_dynamics(t,x,u,T)
% Note that t and T is required for reachability analysis
T = [];

Lr = 2;
Lf = 2;

% lead car dynamics
if vr > 100
    vr = 100;
elseif vr < -0
    vr = -0;
end

if delta > np.pi/3 
    delta = np.pi/3;
elseif delta < -np.pi/3
    delta = -np.pi/3;
end

beta = arctan(Lr/(Lr+Lf) * sin(delta)/np.cos(delta));

dx(1,1) = x(1) + 0.01 * vr * cos(x(3)+beta);                    % x
dx(2,1) = x(2) + 0.01 * vr * sin(x(3)+beta);                    % y
dx(3,1) = x(3) + 0.01 * vr/Lr * sin(beta);                      % theta
dx(4,1) = 0;                                                    % x_ref
dx(5,1) = x(5) + 0.5;                                           % y_ref
dx(6,1) = 0;                                                    % theta_ref
dx(7,1) = cos(0-(x(3) + 0.01 * vr/Lr * sin(beta)));             % error_theta_cos
dx(8,1) = sin(0-(x(3) + 0.01 * vr/Lr * sin(beta)));             % error_theta_sin

end