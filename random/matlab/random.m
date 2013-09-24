fid = fopen('../data/normal.dat');
A = fscanf(fid, '%g', [1 inf]);
fclose(fid);
hist(A,1000)
