w1 = [1 0;0 1];
b1 = [0;pi/3];
w2 = [-1 0;0 -1];
b2 = [100;2*pi/3];
w3 = [-1 0; 0 -1];
b3 = [100;pi/3];

input = [(200+200)*rand()+(-200);-pi/4]; 
output = w1*input+b1;
output(output<0) = 0;
output = w2*output+b2;
output(output<0) = 0;
output = w3*output+b3;
disp(input)
disp(output)