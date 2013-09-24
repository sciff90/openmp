clear all
order = 3;
fs = 400;
fc = 20;
fnorm = fc*2/fs;
dt = 1/fs;
t1 = 1.0;
t0 = 0.0;
t = t0:dt:t1;
num_samples = (t1-t0)/(dt)+1;
u = randn(num_samples,1);
%u = ones(num_samples,1);
[b,a] = butter(order,fnorm);
D = filter(b,a,u);

%Write u to file
fileID = fopen('../data/u.dat','w');
fprintf(fileID,'%g\n',u);
fclose(fileID);

%Write D to file
fileID = fopen('../data/D.dat','w');
fprintf(fileID,'%g\n',D);
fclose(fileID);

%Write t to file
fileID = fopen('../data/t.dat','w');
fprintf(fileID,'%g\n',t);
fclose(fileID);

%Write Parameters
fileID = fopen('../data/params.dat','w');
fprintf(fileID,'(Order,fs,fc,fnorm,dt,t1,t0,num_samples)\n');
fprintf(fileID,'%d\n',order);
fprintf(fileID,'%g\n',[fs fc fnorm dt t1 t0 ]);
fprintf(fileID,'%d\n',num_samples);
fclose(fileID);
b
a
