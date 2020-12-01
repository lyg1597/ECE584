w = [-27.0365 -0.0068 -1.8390;-0.0086 -0.0005 1.7351];
b = [-5.1557;-5.4321];
x = -15.231525210174928;
y = 1.5917592854076972;
theta = pi;
Lr = 2;
Lf = 2;

trajectory = [x,y,theta];

for i=1:300
    ref = [-15+(i-1)*0.05;0;0];
    data = ref-trajectory(i,:)';
    u = w*data+b;
    vr = u(1);
    delta = u(2);
    
    if vr>0
        vr = 0;
    elseif vr<-30
        vr = -30;
    end
    
    if delta>pi/4
        delta = pi/4;
    elseif delta<-pi/4
        delta = -pi/4;
    end
    
    delta = atan(Lr/(Lr+Lf)*sin(delta)/cos(delta));
    new_x = trajectory(i,1)+0.01*vr*cos(trajectory(i,3)+delta);
    new_y = trajectory(i,2)+0.01*vr*sin(trajectory(i,3)+delta);
    new_theta = trajectory(i,3)+0.01*vr/Lr*sin(delta);

    trajectory = [trajectory;[new_x new_y new_theta]];
    disp([i, vr, delta, new_y, new_theta]);
end

plot(trajectory(:,1),trajectory(:,2))