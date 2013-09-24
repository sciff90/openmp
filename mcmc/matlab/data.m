fid = fopen('../data/a.dat');
A = fscanf(fid,'%f %f %f %f',[4 inf]);
A = A';
A( ~any(A,2), : ) = [];  %rows
A( :, ~any(A,1) ) = [];  %columns
fclose(fid);

fid = fopen('../data/b.dat');
B = fscanf(fid,'%f %f %f %f',[4 inf]);
B = B';
B( ~any(B,2), : ) = [];  %rows
B( :, ~any(B,1) ) = [];  %columns
fclose(fid);

figure(1)
subplot(2,3,1)
hist(A(:,1),1000);

subplot(2,3,2)
hist(A(:,2),1000);

subplot(2,3,3)
hist(A(:,3),1000);

subplot(2,3,4)
hist(B(:,1),1000);

subplot(2,3,5)
hist(B(:,2),1000);

subplot(2,3,6)
hist(B(:,3),1000);
