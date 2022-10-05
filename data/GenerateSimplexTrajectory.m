outFol = 'SimplexDynamics3';
landmarks = [0,1,0; ...
    0.1, 0.6, 0.3; ...
    0.2, 0.2, 0.6; ...
    0.6, 0.15, 0.25; ...
    0.8, 0.1, 0.1; ...
    1, 0, 0];
tLandmark = linspace(0,1,6);
nSamp = 20001;
t = linspace(0,1,nSamp);
out = zeros(nSamp,3);

out(:,1) = spline(tLandmark, landmarks(:,1), t);
out(:,2) = spline(tLandmark, landmarks(:,2), t);
out(:,3) = spline(tLandmark, landmarks(:,3), t);

for i =1:t
    out(i,:) = out(i,:)/sum(out(i,:));
end

X=out;

save('simplex.mat', 'X');
%figure(3);clf
%[h,hg,htick]=terplot;
%hter = ternaryc(out(:,1),out(:,2),out(:,3));

%% lininterp
v = 0.9;
v0 = 0.05;
vh = (v+v0)/2;
landmarks = [v, v0, v0; ...
    v, v0, v0; ...
    vh, vh, v0; ...
    v0, v, v0; ...
    v0, v, v0; ...
    v0, vh, vh; ...
    v0, v0, v; ...
    v0, v0, v; ...
    vh, v0, vh; ...
    v, v0, v0];
tLandmark = linspace(0,1,10);
nSamp = 20001;
t = linspace(0,1,nSamp);
out = zeros(nSamp,3);

out(:,1) = interp1(tLandmark, landmarks(:,1), t);
out(:,2) = interp1(tLandmark, landmarks(:,2), t);
out(:,3) = interp1(tLandmark, landmarks(:,3), t);

X = out;

save('linear.mat', 'X');

%% linearSpline
landmarks = [1,0,0; ...
    1,0,0; ...
    0.45, 0.45, 0.1; ...
    0, 1, 0; ...
    0, 1, 0; ...
    0.1, 0.45, 0.45; ...
    0, 0, 1; ...
    0, 0, 1; ...
    0.45, 0.1, 0.45; ...
    1, 0, 0; ...
    1, 0, 0];
tLandmark = linspace(0,1,11);
nSamp = 20000*11/10;
t = linspace(0,1,nSamp);
out = zeros(nSamp,3);

out(:,1) = spline(tLandmark, landmarks(:,1), t);
out(:,2) = spline(tLandmark, landmarks(:,2), t);
out(:,3) = spline(tLandmark, landmarks(:,3), t);
out = out(1:nSamp*10/11+1,:);

out(find(out>1))=1;
out(find(out<0))=0;

for i =1:t
    out(i,:) = out(i,:)/sum(out(i,:));
end

X=out;

save('vertex.mat', 'X');
